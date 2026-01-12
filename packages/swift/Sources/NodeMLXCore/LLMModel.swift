// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Core LLM model protocol and factory for node-mlx.

import Foundation
import MLX
import MLXNN

// MARK: - Type Aliases for Compatibility

/// Type alias for backward compatibility with generated models.
/// The generated models use KVCache as a protocol/type constraint.
public typealias KVCache = KVCacheProtocol

/// Simple KV cache - the default implementation used by generated models.
public typealias KVCacheSimple = StandardKVCache

// MARK: - LLM Model Protocol

/// Protocol that all language models must conform to.
///
/// This defines the common interface for forward passes, caching,
/// and weight loading across all model architectures.
public protocol LLMModel: Module {
    /// Vocabulary size for the model
    var vocabularySize: Int { get }

    /// Number of transformer layers
    var numLayers: Int { get }

    /// Number of key-value heads per layer
    var numKVHeads: Int { get }

    /// Dimension of each attention head
    var headDim: Int { get }

    /// Forward pass with optional cache
    func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCacheProtocol]?) -> MLXArray

    /// Creates a new cache for generation
    func newCache() -> [any KVCacheProtocol]

    /// Sanitizes weight keys for this model architecture
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

// MARK: - Model Architecture Registry

/// Supported model architectures
public enum ModelArchitecture: String, CaseIterable {
    case llama
    case qwen3
    case phi3
    case gemma3
    case gemma3n
    case mistral
    case mistral3
    case smollm3
    case gptoss = "gpt_oss"

    /// Creates architecture from HuggingFace model_type string.
    public init?(modelType: String) {
        let normalized = modelType.lowercased().replacingOccurrences(of: "-", with: "_")

        // Try direct match first
        if let arch = ModelArchitecture(rawValue: normalized) {
            self = arch
            return
        }

        // Handle aliases and variations
        switch normalized {
        case "qwen", "qwen2", "qwen2.5", "qwen25":
            self = .qwen3 // All Qwen models use Qwen3 architecture now
        case "llama2", "llama3", "llama3.1", "llama3.2":
            self = .llama
        case "phi-3", "phi_3":
            self = .phi3
        case "gemma-3", "gemma_3":
            self = .gemma3
        case "gemma-3n", "gemma_3n":
            self = .gemma3n
        case "mistral-3", "mistral_3":
            self = .mistral3
        case "smollm-3", "smollm_3":
            self = .smollm3
        case "gptoss", "gpt-oss":
            self = .gptoss
        default:
            return nil
        }
    }
}

// MARK: - Model Factory

/// Factory for creating model instances from configurations.
public enum ModelFactory {
    /// Creates a model instance from a configuration dictionary.
    ///
    /// - Parameters:
    ///   - architecture: The model architecture to create
    ///   - config: JSON configuration dictionary
    /// - Returns: Instantiated model
    /// - Throws: DecodingError if configuration is invalid
    public static func createModel(
        architecture: ModelArchitecture,
        config: [String: Any]
    ) throws -> any LLMModel {
        let jsonData = try JSONSerialization.data(withJSONObject: config)
        let decoder = JSONDecoder()

        switch architecture {
        case .llama:
            let cfg = try decoder.decode(LlamaConfiguration.self, from: jsonData)
            return LlamaModel(cfg)

        case .qwen3:
            let cfg = try decoder.decode(Qwen3Configuration.self, from: jsonData)
            return Qwen3Model(cfg)

        case .phi3:
            let cfg = try decoder.decode(Phi3Configuration.self, from: jsonData)
            return Phi3Model(cfg)

        case .gemma3:
            let cfg = try decoder.decode(Gemma3Configuration.self, from: jsonData)
            return Gemma3Model(cfg)

        case .gemma3n:
            let cfg = try decoder.decode(Gemma3nConfiguration.self, from: jsonData)
            return Gemma3nModel(cfg)

        case .mistral:
            let cfg = try decoder.decode(MistralConfiguration.self, from: jsonData)
            return MistralModel(cfg)

        case .mistral3:
            let cfg = try decoder.decode(Mistral3Configuration.self, from: jsonData)
            return Mistral3Model(cfg)

        case .smollm3:
            let cfg = try decoder.decode(SmolLM3Configuration.self, from: jsonData)
            return SmolLM3Model(cfg)

        case .gptoss:
            let cfg = try decoder.decode(GptOSSConfiguration.self, from: jsonData)
            return GptOSSModel(cfg)
        }
    }

    /// Detects the model architecture from a configuration dictionary.
    ///
    /// - Parameter config: JSON configuration dictionary
    /// - Returns: Detected architecture, or nil if unknown
    public static func detectArchitecture(from config: [String: Any]) -> ModelArchitecture? {
        // Try model_type field first
        if let modelType = config["model_type"] as? String {
            return ModelArchitecture(modelType: modelType)
        }

        // Try architectures array
        if let architectures = config["architectures"] as? [String],
           let first = architectures.first
        {
            // Parse architecture name (e.g., "LlamaForCausalLM" -> "llama")
            let normalized = first
                .replacingOccurrences(of: "ForCausalLM", with: "")
                .replacingOccurrences(of: "Model", with: "")
                .lowercased()
            return ModelArchitecture(modelType: normalized)
        }

        return nil
    }
}
