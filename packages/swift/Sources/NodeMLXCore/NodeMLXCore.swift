// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Core Swift implementation for node-mlx.
// Provides the main integration point between Node.js and MLX.

import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Generation Result

/// Result of text generation.
public struct GenerationResult: Sendable {
    /// The generated text.
    public let text: String

    /// Number of tokens generated.
    public let tokenCount: Int

    /// Tokens per second.
    public let tokensPerSecond: Float

    /// Time to first token in seconds.
    public let timeToFirstToken: Double

    /// Total generation time in seconds.
    public let totalTime: Double
}

// MARK: - LLM Engine

/// Main engine for loading and running language models.
///
/// This class manages model loading, tokenization, and generation,
/// providing a high-level API for the Node.js bindings.
public class LLMEngine {
    private var model: (any LLMModel)?
    private var tokenizer: HFTokenizer?
    private var modelPath: String?

    /// Whether a model is currently loaded.
    public var isLoaded: Bool { model != nil }

    /// Whether this is a vision-language model (VLM).
    public var isVLM: Bool { false } // Not implemented yet

    /// Creates an empty engine.
    public init() {}

    /// Loads a model from HuggingFace Hub or local directory.
    ///
    /// - Parameter modelId: HuggingFace model ID or local path
    /// - Throws: Error if model cannot be loaded
    public func loadModel(modelId: String) async throws {
        // Check if it's a local path
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: modelId) {
            try await loadModelFromPath(modelId)
        } else {
            // Download from HuggingFace Hub
            let hubApi = HubApi()
            let repo = Hub.Repo(id: modelId)
            let localPath = try await hubApi.snapshot(from: repo, matching: ["*.json", "*.safetensors"])
            try await loadModelFromPath(localPath.path)
        }
    }

    /// Loads a model from a local directory.
    ///
    /// - Parameter path: Path to model directory containing config.json and weights
    /// - Throws: Error if model cannot be loaded
    private func loadModelFromPath(_ path: String) async throws {
        let url = URL(fileURLWithPath: path)

        // Load configuration
        let configPath = url.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw LLMEngineError.invalidConfig("Cannot parse config.json")
        }

        // Detect architecture
        guard let architecture = ModelFactory.detectArchitecture(from: config) else {
            let modelType = config["model_type"] as? String ?? "unknown"
            throw LLMEngineError.unsupportedModel("Unsupported model type: \(modelType)")
        }

        // Create model
        let newModel = try ModelFactory.createModel(architecture: architecture, config: config)

        // Load weights
        let weights = try loadWeights(from: url, config: config)

        // Sanitize weight keys
        let sanitizedWeights = newModel.sanitize(weights: weights)

        // Handle quantization
        if let quantConfig = config["quantization"] as? [String: Any],
           let groupSize = quantConfig["group_size"] as? Int,
           let bits = quantConfig["bits"] as? Int
        {
            quantize(model: newModel, predicate: { weightPath, _ in
                // Check if this weight has quantization scales
                if sanitizedWeights["\(weightPath).scales"] != nil {
                    return (groupSize, bits, .affine)
                }
                return nil
            })
        }

        // Apply weights
        try newModel.update(parameters: ModuleParameters.unflattened(sanitizedWeights))
        eval(newModel.parameters())

        // Load tokenizer
        let newTokenizer = try await HFTokenizer(path: path)

        model = newModel
        tokenizer = newTokenizer
        modelPath = path
    }

    /// Generates text from a prompt.
    ///
    /// - Parameters:
    ///   - prompt: Input text
    ///   - config: Generation configuration
    ///   - onToken: Optional callback for streaming tokens
    /// - Returns: Generated text
    public func generate(
        prompt: String,
        config: GenerationConfig = GenerationConfig(),
        onToken: ((String) -> Bool)? = nil
    ) throws -> String {
        guard let model, let tokenizer else {
            throw LLMEngineError.modelNotLoaded
        }

        // Encode prompt
        let inputIds = tokenizer.encode(text: prompt)

        // Set up stop tokens
        var genConfig = config
        if let eosId = tokenizer.eosTokenId {
            genConfig.stopTokens.insert(eosId)
        }

        // Generate tokens
        let generatedIds = NodeMLXCore.generate(
            model: model,
            inputIds: inputIds,
            config: genConfig,
            onToken: onToken.map { callback in
                { tokenId in
                    let text = tokenizer.decode(tokens: [tokenId])
                    return callback(text)
                }
            }
        )

        // Decode result
        return tokenizer.decode(tokens: generatedIds)
    }

    /// Generates text with streaming and returns detailed result.
    ///
    /// - Parameters:
    ///   - prompt: Input text
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature
    ///   - topP: Nucleus sampling threshold
    ///   - repetitionPenalty: Penalty for repeated tokens (optional)
    ///   - repetitionContextSize: Context size for repetition penalty
    ///   - onToken: Callback for each generated token
    /// - Returns: Generation result with timing information
    public func generateStream(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        repetitionPenalty: Float? = nil,
        repetitionContextSize _: Int = 20,
        onToken: @escaping (String) -> Bool
    ) throws -> GenerationResult {
        guard let model, let tokenizer else {
            throw LLMEngineError.modelNotLoaded
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?

        // Encode prompt
        let inputIds = tokenizer.encode(text: prompt)

        // Set up config
        var config = GenerationConfig(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: repetitionPenalty ?? 1.0
        )
        if let eosId = tokenizer.eosTokenId {
            config.stopTokens.insert(eosId)
        }

        // Generate tokens
        let generatedIds = NodeMLXCore.generate(
            model: model,
            inputIds: inputIds,
            config: config,
            onToken: { tokenId in
                if firstTokenTime == nil {
                    firstTokenTime = CFAbsoluteTimeGetCurrent()
                }
                let text = tokenizer.decode(tokens: [tokenId])
                return onToken(text)
            }
        )

        let endTime = CFAbsoluteTimeGetCurrent()
        let totalTime = endTime - startTime
        let timeToFirst = (firstTokenTime ?? endTime) - startTime

        return GenerationResult(
            text: tokenizer.decode(tokens: generatedIds),
            tokenCount: generatedIds.count,
            tokensPerSecond: generatedIds.count > 0 ? Float(generatedIds.count) / Float(totalTime) : 0,
            timeToFirstToken: timeToFirst,
            totalTime: totalTime
        )
    }

    /// Generates text with an image (VLM).
    ///
    /// - Note: VLM support is not yet implemented.
    public func generateStreamWithImage(
        prompt _: String,
        imagePath _: String,
        maxTokens _: Int,
        temperature _: Float,
        topP _: Float,
        repetitionPenalty _: Float? = nil,
        repetitionContextSize _: Int = 20,
        onToken _: @escaping (String) -> Bool
    ) throws -> GenerationResult {
        throw LLMEngineError.unsupportedModel("VLM support not yet implemented")
    }

    /// Unloads the current model.
    public func unload() {
        model = nil
        tokenizer = nil
        modelPath = nil
    }
}

// MARK: - Weight Loading

/// Loads model weights from a directory.
///
/// Supports both safetensors and npz formats.
private func loadWeights(from url: URL, config _: [String: Any]) throws -> [String: MLXArray] {
    // Find weight files
    let fileManager = FileManager.default
    let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)

    // Prefer safetensors
    let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }
    let npzFiles = contents.filter { $0.pathExtension == "npz" }

    var weights: [String: MLXArray] = [:]

    if !safetensorFiles.isEmpty {
        // Load all safetensor files
        for file in safetensorFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let fileWeights = try MLX.loadArrays(url: file)
            for (key, value) in fileWeights {
                weights[key] = value
            }
        }
    } else if !npzFiles.isEmpty {
        // Load first npz file
        if let npzFile = npzFiles.first {
            weights = try MLX.loadArrays(url: npzFile)
        }
    } else {
        throw LLMEngineError.weightsNotFound
    }

    return weights
}

// MARK: - Error Types

/// Errors that can occur during LLM engine operations.
public enum LLMEngineError: Error, LocalizedError {
    case modelNotLoaded
    case invalidConfig(String)
    case unsupportedModel(String)
    case weightsNotFound
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "No model is loaded"
        case let .invalidConfig(msg):
            "Invalid configuration: \(msg)"
        case let .unsupportedModel(msg):
            "Unsupported model: \(msg)"
        case .weightsNotFound:
            "No weight files found in model directory"
        case let .generationFailed(msg):
            "Generation failed: \(msg)"
        }
    }
}

// MARK: - Quantization Helper

/// Quantizes model layers that have corresponding scale weights.
private func quantize(
    model: Module,
    predicate: (String, Module) -> (Int, Int, QuantizationMode)?
) {
    model.update(modules: ModuleChildren.unflattened(
        model.leafModules().flattened().compactMap { path, module in
            guard let (groupSize, bits, mode) = predicate(path, module) else {
                return nil
            }
            if let linear = module as? Linear {
                return (path, QuantizedLinear(linear, groupSize: groupSize, bits: bits, mode: mode))
            }
            return nil
        }
    ))
}
