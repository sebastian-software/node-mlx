//
//  ModelLoader.swift
//  NodeMLXCore
//
//  Downloads and loads MLX models from HuggingFace Hub.
//
//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).
//  See: https://github.com/ml-explore/mlx-swift-lm
//

import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Model Loading Errors

public enum ModelLoaderError: Error, LocalizedError {
    case downloadFailed(String)
    case configNotFound(String)
    case weightsNotFound(String)
    case unsupportedArchitecture(String)
    case weightLoadingFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .downloadFailed(msg): "Download failed: \(msg)"
        case let .configNotFound(msg): "Config not found: \(msg)"
        case let .weightsNotFound(msg): "Weights not found: \(msg)"
        case let .unsupportedArchitecture(msg): "Unsupported architecture: \(msg)"
        case let .weightLoadingFailed(msg): "Weight loading failed: \(msg)"
        }
    }
}

// MARK: - Model Configuration (from config.json)

public struct ModelConfig: Codable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let numKeyValueHeads: Int?
    public let intermediateSize: Int?
    public let vocabSize: Int?
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Float?
    public let rmsNormEps: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
    }
}

// MARK: - Model Loader

public class ModelLoader {
    private let hub: HubApi

    public init() {
        hub = HubApi()
    }

    /// Download a model from HuggingFace Hub
    /// Returns the local directory URL containing the model files
    public func download(
        modelId: String,
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let repo = Hub.Repo(id: modelId)

        // Download safetensors and config files
        let patterns = ["*.safetensors", "*.json"]

        do {
            let modelDir = try await hub.snapshot(
                from: repo,
                matching: patterns,
                progressHandler: progressHandler ?? { _ in }
            )
            return modelDir
        } catch Hub.HubClientError.authorizationRequired {
            throw ModelLoaderError.downloadFailed("Model requires authentication: \(modelId)")
        } catch {
            throw ModelLoaderError.downloadFailed("\(error)")
        }
    }

    /// Load configuration from config.json
    public func loadConfig(from modelDir: URL) throws -> ModelConfig {
        let configURL = modelDir.appendingPathComponent("config.json")

        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ModelLoaderError.configNotFound(configURL.path)
        }

        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(ModelConfig.self, from: data)
        return config
    }

    /// Load weights from safetensors files
    public func loadWeights(from modelDir: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]

        let enumerator = FileManager.default.enumerator(
            at: modelDir,
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
            throw ModelLoaderError.weightsNotFound(modelDir.path)
        }

        return weights
    }

    /// Get the model architecture type from config
    public func getModelType(from modelDir: URL) throws -> String {
        let config = try loadConfig(from: modelDir)
        guard let modelType = config.modelType else {
            throw ModelLoaderError.configNotFound("model_type not found in config.json")
        }
        return modelType
    }
}

// MARK: - Weight Utilities

/// Sanitize weight keys (remove common prefixes, handle quantization)
public func sanitizeWeights(_ weights: [String: MLXArray], prefix: String = "model.") -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
        var newKey = key

        // Remove common prefixes
        if newKey.hasPrefix(prefix) {
            newKey = String(newKey.dropFirst(prefix.count))
        }

        sanitized[newKey] = value
    }

    return sanitized
}
