//
//  LLMModel.swift
//  NodeMLXCore
//
//  Protocol defining the interface for language models.
//
//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).
//  See: https://github.com/ml-explore/mlx-swift-lm
//

import Foundation
import MLX
import MLXNN

// MARK: - LLM Model Protocol

/// Protocol that all language models must conform to
public protocol LLMModel: Module {
    /// Vocabulary size of the model
    var vocabularySize: Int { get }

    /// Number of transformer layers
    var numLayers: Int { get }

    /// Forward pass with KV cache for efficient generation
    func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray

    /// Forward pass without cache (for simple models)
    func callAsFunction(_ inputIds: MLXArray) -> MLXArray

    /// Create a new KV cache for this model
    func newCache() -> [KVCache]

    /// Whether this model supports KV caching
    var supportsCache: Bool { get }

    /// Sanitize weight keys during loading (optional override)
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

// MARK: - Default Implementations

public extension LLMModel {
    /// Default sanitize implementation (no-op)
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    /// Default cache creation
    func newCache() -> [KVCache] {
        createLayerCaches(numLayers: numLayers)
    }

    /// Default: models don't support cache
    var supportsCache: Bool { false }

    /// Default cache implementation - falls back to non-cached version
    func callAsFunction(_ inputIds: MLXArray, cache _: inout [KVCache]?) -> MLXArray {
        // Default: ignore cache and call simple version
        callAsFunction(inputIds)
    }
}

// MARK: - Model Registry

/// Supported model architectures
public enum ModelArchitecture: String, CaseIterable {
    case llama
    case phi3
    case gemma3
    case gemma3vlm // Gemma 3 with vision
    case gemma3n
    case qwen2
    case qwen3
    case mistral
    case mistral3 // Ministral 3 / Mistral 3
    case smollm3 // SmolLM 3
    case gptOss // GPT-OSS MoE model

    /// Get architecture from model_type in config.json
    public static func from(modelType: String) -> ModelArchitecture? {
        let normalized = modelType.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")

        // Direct matches first (order matters - more specific first)
        if normalized == "llama" { return .llama }
        if normalized == "phi3" { return .phi3 }
        if normalized == "gemma3n" || normalized == "gemma3ntext" { return .gemma3n }
        if normalized == "gemma3" || normalized == "gemma3text" { return .gemma3 }
        if normalized == "qwen3" { return .qwen3 } // Check qwen3 before qwen2
        if normalized == "qwen2" { return .qwen2 }
        if normalized == "mistral3" || normalized == "ministral3" { return .mistral3 } // Check mistral3 before mistral
        if normalized == "mistral" { return .mistral }
        if normalized == "smollm3" { return .smollm3 }
        if normalized == "gptoss" { return .gptOss }

        // Partial matches
        for arch in allCases {
            let archNormalized = arch.rawValue
                .replacingOccurrences(of: "_", with: "")
                .replacingOccurrences(of: "-", with: "")
            if normalized.contains(archNormalized) {
                return arch
            }
        }
        return nil
    }

    /// Check if this is a VLM architecture
    public var isVLM: Bool {
        switch self {
        case .gemma3vlm:
            true
        default:
            false
        }
    }
}

// MARK: - Model Factory

/// Create a model instance from config and weights
public enum ModelFactory {
    public enum ModelError: Error {
        case unsupportedArchitecture(String)
        case configLoadFailed(String)
        case weightLoadFailed(String)
    }

    /// Create model from downloaded directory
    public static func createModel(
        modelDirectory: URL,
        architecture: ModelArchitecture
    ) throws -> any LLMModel {
        switch architecture {
        case .phi3:
            let config = try loadConfig(Phi3Configuration.self, from: modelDirectory)
            return Phi3Model(config)
        case .llama:
            let config = try loadConfig(LlamaConfiguration.self, from: modelDirectory)
            return LlamaModel(config)
        case .gemma3n:
            let config = try loadConfig(Gemma3nConfiguration.self, from: modelDirectory)
            return Gemma3nModel(config)
        case .qwen2:
            let config = try loadConfig(Qwen2Configuration.self, from: modelDirectory)
            return Qwen2Model(config)
        case .qwen3:
            let config = try loadConfig(Qwen3Configuration.self, from: modelDirectory)
            return Qwen3Model(config)
        case .gemma3:
            // Gemma 3 uses standard transformer architecture with some Gemma-specific features
            let config = try loadConfig(Gemma3Configuration.self, from: modelDirectory)
            return Gemma3Model(config)
        case .gemma3vlm:
            // Gemma 3 Vision-Language Model
            let config = try loadConfig(Gemma3VLMConfiguration.self, from: modelDirectory)
            return Gemma3VLMModel(config)
        case .mistral:
            let config = try loadConfig(MistralConfiguration.self, from: modelDirectory)
            return MistralModel(config)
        case .mistral3:
            let config = try loadConfig(Mistral3Configuration.self, from: modelDirectory)
            return Mistral3Model(config)
        case .smollm3:
            let config = try loadConfig(Smollm3Configuration.self, from: modelDirectory)
            return Smollm3Model(config)
        case .gptOss:
            let config = try loadConfig(GptOSSConfiguration.self, from: modelDirectory)
            return GptOSSModel(config)
        }
    }

    private static func loadConfig<T: Decodable>(_: T.Type, from directory: URL) throws -> T {
        let configPath = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configPath)
        return try JSONDecoder().decode(T.self, from: data)
    }

    /// Detect if a model is a VLM by checking for vision_config in config.json
    public static func detectVLM(modelDirectory: URL) -> Bool {
        let configPath = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return false
        }

        // VLM configs have vision_config
        return json["vision_config"] != nil
    }

    /// Get architecture, automatically detecting VLM
    public static func detectArchitecture(modelDirectory: URL) throws -> ModelArchitecture {
        let configPath = modelDirectory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configPath)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String
        else {
            throw ModelError.configLoadFailed("Missing model_type in config.json")
        }

        // Gemma3n needs special handling for config in text_config
        if modelType.lowercased().contains("gemma3n") {
            return .gemma3n
        }

        // Check for VLM first
        if let visionConfig = json["vision_config"] as? [String: Any] {
            // Check if vision is disabled (skip_vision: true indicates text-only quantized from VLM)
            let skipVision = visionConfig["skip_vision"] as? Bool ?? false
            if !skipVision {
                // It's a VLM - check which type
                if modelType.lowercased().contains("gemma") {
                    return .gemma3vlm
                }
                // Add other VLM types here as needed
            }
        }

        // Fall back to text-only architecture detection
        guard let arch = ModelArchitecture.from(modelType: modelType) else {
            throw ModelError.unsupportedArchitecture(modelType)
        }

        return arch
    }
}
