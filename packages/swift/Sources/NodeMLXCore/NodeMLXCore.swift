// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Core Swift implementation for node-mlx.
// Provides the main integration point between Node.js and MLX.

import Foundation
import Hub
import MLX
import MLXNN

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

    /// Creates an empty engine.
    public init() {}

    /// Loads a model from a local directory.
    ///
    /// - Parameter path: Path to model directory containing config.json and weights
    /// - Throws: Error if model cannot be loaded
    public func loadModel(path: String) async throws {
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
            quantize(model: newModel) { weightPath, _ in
                // Check if this weight has quantization scales
                if sanitizedWeights["\(weightPath).scales"] != nil {
                    return (groupSize, bits, .affine)
                }
                return nil
            }
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
