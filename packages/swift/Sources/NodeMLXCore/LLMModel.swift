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

extension LLMModel {
    /// Default sanitize implementation (no-op)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }

    /// Default cache creation
    public func newCache() -> [KVCache] {
        return createLayerCaches(numLayers: numLayers)
    }

    /// Default: models don't support cache
    public var supportsCache: Bool { false }

    /// Default cache implementation - falls back to non-cached version
    public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        // Default: ignore cache and call simple version
        return callAsFunction(inputIds)
    }
}

// MARK: - Model Registry

/// Supported model architectures
public enum ModelArchitecture: String, CaseIterable {
    case llama = "llama"
    case phi3 = "phi3"
    case gemma3 = "gemma3"
    case gemma3n = "gemma3n"
    case qwen2 = "qwen2"
    case mistral = "mistral"

    /// Get architecture from model_type in config.json
    public static func from(modelType: String) -> ModelArchitecture? {
        let normalized = modelType.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")

        // Direct matches first (order matters - gemma3n before gemma3)
        if normalized == "llama" { return .llama }
        if normalized == "phi3" { return .phi3 }
        if normalized == "gemma3n" { return .gemma3n }
        if normalized == "gemma3" || normalized == "gemma3text" { return .gemma3 }
        if normalized == "qwen2" { return .qwen2 }
        if normalized == "mistral" { return .mistral }

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
        case .gemma3:
            // Gemma 3 uses standard transformer architecture with some Gemma-specific features
            let config = try loadConfig(Gemma3Configuration.self, from: modelDirectory)
            return Gemma3Model(config)
        case .mistral:
            let config = try loadConfig(MistralConfiguration.self, from: modelDirectory)
            return MistralModel(config)
        }
    }

    private static func loadConfig<T: Decodable>(_ type: T.Type, from directory: URL) throws -> T {
        let configPath = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configPath)
        return try JSONDecoder().decode(T.self, from: data)
    }
}
