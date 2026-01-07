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

    /// Number of KV heads per layer (for cache allocation)
    var kvHeads: [Int] { get }

    /// Prepare input for the model (embed tokens)
    func prepare(_ input: MLXArray) -> MLXArray

    /// Forward pass: compute logits from embeddings
    func callAsFunction(_ inputs: MLXArray, cache: [[any KVCache]]?) -> MLXArray

    /// Sanitize weight keys during loading
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

// MARK: - Default Implementations

extension LLMModel {
    /// Default sanitize implementation (no-op)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }
}

// MARK: - Model Output

/// Output from a single forward pass
public struct ModelOutput {
    /// Logits for next token prediction (shape: [batch, vocab_size])
    public let logits: MLXArray

    /// Hidden states (optional, for advanced use)
    public let hiddenStates: MLXArray?

    public init(logits: MLXArray, hiddenStates: MLXArray? = nil) {
        self.logits = logits
        self.hiddenStates = hiddenStates
    }
}

// MARK: - Model Registry

/// Registry of supported model architectures
public enum ModelArchitecture: String, CaseIterable {
    case llama = "llama"
    case qwen2 = "qwen2"
    case phi3 = "phi3"
    case gemma = "gemma"
    case gemma2 = "gemma2"
    case mistral = "mistral"
    case gptOss = "gpt-oss"

    /// Get architecture from model_type in config.json
    public static func from(modelType: String) -> ModelArchitecture? {
        let normalized = modelType.lowercased()
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "-", with: "")

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

