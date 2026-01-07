//
//  NodeMLXCore.swift
//  NodeMLXCore
//
//  Main entry point for LLM inference without mlx-swift-lm dependency.
//
//  Copyright © 2026 Sebastian Software GmbH. All rights reserved.
//

import Foundation
import Hub
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Tokenizers

// MARK: - Public API

/// Main interface for LLM operations
public class LLMEngine {
    private var model: (any LLMModel)?
    private var tokenizer: HFTokenizer?
    private var modelDirectory: URL?

    public init() {}

    // MARK: - Model Loading

    /// Load a model from HuggingFace Hub
    public func loadModel(
        modelId: String,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws {
        // Download model files
        let hub = HubApi()
        let repo = Hub.Repo(id: modelId)

        let directory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "*.json", "tokenizer*", "vocab*", "merges*"],
            progressHandler: { progress in
                progressHandler?(Float(progress.fractionCompleted))
            }
        )

        self.modelDirectory = directory

        // Detect architecture from config
        let configPath = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]

        guard let modelType = configDict["model_type"] as? String else {
            throw LLMEngineError.invalidConfig("model_type not found in config.json")
        }

        guard let architecture = ModelArchitecture.from(modelType: modelType) else {
            throw LLMEngineError.unsupportedModel("Unsupported model type: \(modelType)")
        }

        // Create model
        let model = try ModelFactory.createModel(
            modelDirectory: directory,
            architecture: architecture
        )

        // Load weights
        let weights = try loadWeights(from: directory)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Apply weights to model
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights))

        self.model = model

        // Load tokenizer
        self.tokenizer = try await HFTokenizer(modelDirectory: directory)
    }

    // MARK: - Generation

    /// Generate text from a prompt
    public func generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) throws -> GenerationResult {
        guard let model = model else {
            throw LLMEngineError.modelNotLoaded
        }
        guard let tokenizer = tokenizer else {
            throw LLMEngineError.tokenizerNotLoaded
        }

        // Encode prompt
        let inputTokens = tokenizer.encode(prompt)
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0) // Add batch dimension

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Generation loop
        for _ in 0..<maxTokens {
            // Forward pass
            let logits = model(inputArray)

            // Get logits for last token
            let lastLogits = logits[0..., -1, 0...]

            // Sample next token
            let nextToken = sampleToken(
                logits: lastLogits,
                temperature: temperature,
                topP: topP
            )

            // Check for EOS
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }

            generatedTokens.append(nextToken)

            // Prepare next input
            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
        }

        let endTime = Date()
        let duration = Float(endTime.timeIntervalSince(startTime))
        let tokensPerSecond = duration > 0 ? Float(generatedTokens.count) / duration : 0

        // Decode generated tokens
        let generatedText = tokenizer.decode(generatedTokens)

        return GenerationResult(
            text: generatedText,
            tokenCount: generatedTokens.count,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Generate text with streaming callback
    public func generateStream(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        onToken: @escaping (String) -> Bool // Return false to stop
    ) throws -> GenerationResult {
        guard let model = model else {
            throw LLMEngineError.modelNotLoaded
        }
        guard let tokenizer = tokenizer else {
            throw LLMEngineError.tokenizerNotLoaded
        }

        let inputTokens = tokenizer.encode(prompt)
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0)

        let startTime = Date()
        var generatedTokens: [Int] = []

        for _ in 0..<maxTokens {
            let logits = model(inputArray)
            let lastLogits = logits[0..., -1, 0...]

            let nextToken = sampleToken(
                logits: lastLogits,
                temperature: temperature,
                topP: topP
            )

            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }

            generatedTokens.append(nextToken)

            // Stream the token
            let tokenText = tokenizer.decode([nextToken])
            if !onToken(tokenText) {
                break // User requested stop
            }

            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
        }

        let endTime = Date()
        let duration = Float(endTime.timeIntervalSince(startTime))
        let tokensPerSecond = duration > 0 ? Float(generatedTokens.count) / duration : 0

        return GenerationResult(
            text: tokenizer.decode(generatedTokens),
            tokenCount: generatedTokens.count,
            tokensPerSecond: tokensPerSecond
        )
    }

    // MARK: - Cleanup

    /// Unload the model from memory
    public func unload() {
        model = nil
        tokenizer = nil
        modelDirectory = nil
    }

    // MARK: - Private Helpers

    private func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]

        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        )!

        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let fileWeights = try loadArrays(url: url)
                for (key, value) in fileWeights {
                    weights[key] = value
                }
            }
        }

        if weights.isEmpty {
            throw LLMEngineError.weightsNotFound
        }

        return weights
    }

    private func sampleToken(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        if temperature == 0 {
            // Greedy decoding
            return Int(argMax(logits, axis: -1).item(Int32.self))
        }

        // Temperature scaling
        let scaledLogits = logits / temperature
        let probs = softmax(scaledLogits, axis: -1)

        // Top-p sampling
        if topP > 0 && topP < 1 {
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = take(probs, sortedIndices, axis: -1)

            // Find cutoff
            var cumSum: Float = 0
            var cutoffIndex = sortedProbs.count - 1

            for i in stride(from: sortedProbs.count - 1, through: 0, by: -1) {
                cumSum += sortedProbs[i].item(Float.self)
                if cumSum >= topP {
                    cutoffIndex = i
                    break
                }
            }

            // Zero out tokens below threshold
            var maskedProbs = probs
            for i in 0..<cutoffIndex {
                let idx = Int(sortedIndices[i].item(Int32.self))
                maskedProbs[idx] = MLXArray(Float(0))
            }

            // Renormalize
            let probSum = sum(maskedProbs)
            maskedProbs = maskedProbs / probSum

            return Int(MLXRandom.categorical(maskedProbs).item(Int32.self))
        }

        return Int(MLXRandom.categorical(probs).item(Int32.self))
    }
}

// MARK: - Types

public struct GenerationResult {
    public let text: String
    public let tokenCount: Int
    public let tokensPerSecond: Float

    public init(text: String, tokenCount: Int, tokensPerSecond: Float) {
        self.text = text
        self.tokenCount = tokenCount
        self.tokensPerSecond = tokensPerSecond
    }
}

public enum LLMEngineError: Error, LocalizedError {
    case modelNotLoaded
    case tokenizerNotLoaded
    case invalidConfig(String)
    case unsupportedModel(String)
    case weightsNotFound

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "No model loaded. Call loadModel() first."
        case .tokenizerNotLoaded:
            return "No tokenizer loaded."
        case .invalidConfig(let msg):
            return "Invalid config: \(msg)"
        case .unsupportedModel(let msg):
            return "Unsupported model: \(msg)"
        case .weightsNotFound:
            return "No weights found in model directory."
        }
    }
}

// MARK: - Convenience

/// Quick generation without managing engine lifecycle
public func quickGenerate(
    modelId: String,
    prompt: String,
    maxTokens: Int = 256
) async throws -> String {
    let engine = LLMEngine()
    try await engine.loadModel(modelId: modelId)
    let result = try engine.generate(prompt: prompt, maxTokens: maxTokens)
    engine.unload()
    return result.text
}
