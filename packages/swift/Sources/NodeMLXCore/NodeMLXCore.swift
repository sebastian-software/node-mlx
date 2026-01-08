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
        var model = try ModelFactory.createModel(
            modelDirectory: directory,
            architecture: architecture
        )

        // Load weights first
        let weights = try loadWeights(from: directory)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed - use dynamic quantization based on weight presence
        if let quantizationConfig = configDict["quantization"] as? [String: Any],
           let groupSize = quantizationConfig["group_size"] as? Int,
           let bits = quantizationConfig["bits"] as? Int {
            // Quantize modules that have .scales weights
            // The filter returns (groupSize, bits) if the module should be quantized
            quantize(model: model) { path, module in
                if sanitizedWeights["\(path).scales"] != nil {
                    return (groupSize, bits)
                } else {
                    return nil
                }
            }
        }

        // Apply weights to model
        model.update(parameters: ModuleParameters.unflattened(sanitizedWeights))

        // Force evaluation of weights to ensure they're loaded to GPU
        eval(model)

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

        // Apply chat template to format prompt correctly for the model
        let inputTokens: [Int]
        do {
            inputTokens = try tokenizer.applyChatTemplate(userMessage: prompt)
        } catch {
            // Fallback to raw encoding if chat template fails
            inputTokens = tokenizer.encode(prompt)
        }
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0) // Add batch dimension

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Create KV cache for efficient generation
        var cache: [KVCache]? = model.newCache()

        // Process prompt (prefill) - all tokens at once
        var logits = model(inputArray, cache: &cache)
        var lastLogits = logits[0, logits.dim(1) - 1]
        eval(lastLogits)

        // Sample first token
        var nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)

        // Generation loop - one token at a time with cached context
        for _ in 0..<maxTokens {
            // Check for EOS
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }

            generatedTokens.append(nextToken)

            // Prepare next input - just the single new token
            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)

            // Forward pass with cache - only processes new token
            logits = model(inputArray, cache: &cache)
            lastLogits = logits[0, 0] // Single token output

            // Async eval for pipelining
            eval(lastLogits)

            // Sample next token
            nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)
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

        // Apply chat template to format prompt correctly for the model
        let inputTokens: [Int]
        do {
            inputTokens = try tokenizer.applyChatTemplate(userMessage: prompt)
        } catch {
            // Fallback to raw encoding if chat template fails
            inputTokens = tokenizer.encode(prompt)
        }
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0)

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Create KV cache for efficient generation
        var cache: [KVCache]? = model.newCache()

        // Process prompt (prefill) - all tokens at once
        var logits = model(inputArray, cache: &cache)
        var lastLogits = logits[0, logits.dim(1) - 1]
        eval(lastLogits)

        // Sample first token
        var nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)

        // Generation loop with KV cache
        for _ in 0..<maxTokens {
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }

            generatedTokens.append(nextToken)

            // Stream the token
            let tokenText = tokenizer.decode([nextToken])
            if !onToken(tokenText) {
                break // User requested stop
            }

            // Prepare next input - just the single new token
            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)

            // Forward pass with cache - only processes new token
            logits = model(inputArray, cache: &cache)
            lastLogits = logits[0, 0] // Single token output
            eval(lastLogits)

            nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)
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
            // Greedy decoding - no randomness
            let token = argMax(logits, axis: -1)
            eval(token)
            return Int(token.item(Int32.self))
        }

        // Temperature scaling and convert to probabilities
        let temp = MLXArray(temperature)
        var logitsFloat = logits
        if logitsFloat.dtype == .bfloat16 {
            logitsFloat = logitsFloat.asType(.float32)
        }
        let probs = softmax(logitsFloat / temp, axis: -1)

        // For top-p sampling, use the mlx-swift-lm approach
        if topP > 0 && topP < 1 {
            let topPArray = MLXArray(topP)

            // Sort in ascending order (lowest first)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = take(probs, sortedIndices, axis: -1)

            // Cumulative sum (from lowest to highest)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            // Keep only tokens where cumulative prob > (1 - topP)
            // This keeps the top-p highest probability tokens
            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topPArray),
                sortedProbs,
                MLXArray.zeros(like: sortedProbs)
            )

            // Sample using log probabilities (avoid numerical issues)
            let sortedToken = MLXRandom.categorical(log(topProbs))
            eval(sortedToken)

            // Map back to original index
            let originalIdx = sortedIndices[Int(sortedToken.item(Int32.self))]
            eval(originalIdx)
            return Int(originalIdx.item(Int32.self))
        }

        // Simple temperature sampling without top-p
        let token = MLXRandom.categorical(probs)
        eval(token)
        return Int(token.item(Int32.self))
    }

    /// Debug function to print top-5 logits and their decoded tokens
    private func debugPrintTopLogits(_ logits: MLXArray, tokenizer: HFTokenizer, label: String) {
        // Get top-5 indices
        let topK = 5
        let sortedIndices = argSort(-logits, axis: -1)  // Descending
        eval(sortedIndices)

        print("[DEBUG \(label)] Top-\(topK) predictions:")
        for i in 0..<topK {
            let idx = Int(sortedIndices[i].item(Int32.self))
            let logit = logits[idx].item(Float.self)
            let token = tokenizer.decode([idx])
            print("  \(i+1). token=\(idx) logit=\(String(format: "%.2f", logit)) '\(token)'")
        }

        // Also print logits stats
        let minLogit = MLX.min(logits).item(Float.self)
        let maxLogit = MLX.max(logits).item(Float.self)
        let meanLogit = mean(logits).item(Float.self)
        print("[DEBUG \(label)] Logits stats: min=\(String(format: "%.2f", minLogit)) max=\(String(format: "%.2f", maxLogit)) mean=\(String(format: "%.2f", meanLogit))")
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
