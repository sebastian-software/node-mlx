// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared protocols for LLM model configurations.
// These enable generic implementations of common model components.

import Foundation
import MLX

// MARK: - Base Configuration Protocol

/// Common configuration properties shared by all transformer models.
public protocol BaseModelConfiguration: Decodable, Sendable {
    var hiddenSize: Int { get }
    var numHiddenLayers: Int { get }
    var numAttentionHeads: Int { get }
    var numKeyValueHeads: Int { get }
    var intermediateSize: Int { get }
    var vocabSize: Int { get }
    var headDim: Int { get }
    var rmsNormEps: Float { get }
    var ropeTheta: Float { get }
    var maxPositionEmbeddings: Int { get }
    var attentionBias: Bool { get }
    var mlpBias: Bool { get }
    var ropeScaling: [String: StringOrNumber]? { get }
}

// MARK: - Sliding Window Configuration

/// Configuration for models with sliding window attention (Mistral, etc.)
public protocol SlidingWindowConfiguration: BaseModelConfiguration {
    var slidingWindow: Int { get }
    var slidingWindowPattern: Int { get }

    /// Check if a layer uses global attention
    func isGlobalLayer(_ layerIdx: Int) -> Bool
}

public extension SlidingWindowConfiguration {
    func isGlobalLayer(_ layerIdx: Int) -> Bool {
        (layerIdx % slidingWindowPattern) == (slidingWindowPattern - 1)
    }
}

// MARK: - MoE Configuration

/// Configuration for Mixture of Experts models (GPT-OSS, etc.)
public protocol MoEConfiguration: BaseModelConfiguration {
    var numLocalExperts: Int { get }
    var numExpertsPerTok: Int { get }
}

// MARK: - Configuration Decoding Helper

/// Helper struct for decoding model configurations from JSON.
/// Handles both top-level and nested text_config patterns.
public struct ConfigDecoder<Keys: CodingKey> {
    private let container: KeyedDecodingContainer<Keys>
    private let textConfigKey: Keys?

    public init(container: KeyedDecodingContainer<Keys>, textConfigKey: Keys? = nil) {
        self.container = container
        self.textConfigKey = textConfigKey
    }

    /// Decode a value, trying text_config first if available, then top-level.
    public func decode<T: Decodable>(_ key: Keys, default defaultValue: T? = nil) throws -> T {
        // Try nested text_config first
        if let textKey = textConfigKey,
           let nested = try? container.nestedContainer(keyedBy: Keys.self, forKey: textKey),
           let value = try? nested.decode(T.self, forKey: key)
        {
            return value
        }

        // Try top-level
        if let value = try? container.decode(T.self, forKey: key) {
            return value
        }

        // Use default if provided
        if let defaultValue {
            return defaultValue
        }

        throw DecodingError.keyNotFound(
            key,
            DecodingError.Context(codingPath: [], debugDescription: "Missing \(key)")
        )
    }
}
