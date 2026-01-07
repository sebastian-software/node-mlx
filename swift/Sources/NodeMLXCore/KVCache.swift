//
//  KVCache.swift
//  NodeMLXCore
//
//  Key-Value cache for efficient autoregressive generation.
//
//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).
//  See: https://github.com/ml-explore/mlx-swift-lm
//

import Foundation
import MLX

// MARK: - KV Cache Protocol

/// Protocol for key-value caches used in transformer attention
public protocol KVCache {
    /// Update the cache with new keys and values
    /// Returns the full keys and values including cached content
    mutating func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Current sequence length in the cache
    var offset: Int { get }

    /// Reset the cache to empty state
    mutating func reset()
}

// MARK: - Simple KV Cache

/// Simple KV cache that grows unbounded
/// Good for short sequences, but memory grows with sequence length
public struct KVCacheSimple: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public private(set) var offset: Int = 0

    public init() {}

    public mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            // Concatenate along sequence dimension (axis 2)
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }

        offset = keys!.dim(2)

        return (keys!, values!)
    }

    public mutating func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

// MARK: - Rotating KV Cache

/// Rotating KV cache with fixed maximum size
/// Overwrites old entries when full, preserving first few tokens (for BOS etc.)
public struct RotatingKVCache: KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public let maxSize: Int
    public let keepFirst: Int

    public private(set) var offset: Int = 0

    public init(maxSize: Int, keepFirst: Int = 4) {
        self.maxSize = maxSize
        self.keepFirst = keepFirst
    }

    public mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let newSeqLen = newKeys.dim(2)

        if let existingKeys = keys, let existingValues = values {
            let currentLen = existingKeys.dim(2)

            if currentLen + newSeqLen <= maxSize {
                // Still have room, just concatenate
                keys = concatenated([existingKeys, newKeys], axis: 2)
                values = concatenated([existingValues, newValues], axis: 2)
            } else {
                // Need to rotate: keep first `keepFirst` tokens, then most recent
                let spaceNeeded = currentLen + newSeqLen - maxSize

                if spaceNeeded < currentLen - keepFirst {
                    // Remove from middle (after keepFirst)
                    let removeEnd = keepFirst + spaceNeeded

                    // Keep: [0:keepFirst] + [removeEnd:] + new
                    let firstPart = existingKeys[0..., 0..., 0..<keepFirst, 0...]
                    let laterPart = existingKeys[0..., 0..., removeEnd..., 0...]

                    keys = concatenated([firstPart, laterPart, newKeys], axis: 2)

                    let firstPartV = existingValues[0..., 0..., 0..<keepFirst, 0...]
                    let laterPartV = existingValues[0..., 0..., removeEnd..., 0...]
                    values = concatenated([firstPartV, laterPartV, newValues], axis: 2)
                } else {
                    // Too much to remove, just keep first and new
                    let firstPart = existingKeys[0..., 0..., 0..<keepFirst, 0...]
                    keys = concatenated([firstPart, newKeys], axis: 2)

                    let firstPartV = existingValues[0..., 0..., 0..<keepFirst, 0...]
                    values = concatenated([firstPartV, newValues], axis: 2)
                }
            }
        } else {
            keys = newKeys
            values = newValues
        }

        offset += newSeqLen

        return (keys!, values!)
    }

    public mutating func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

// MARK: - Cache Factory

/// Create appropriate cache based on parameters
public func createKVCache(maxSize: Int? = nil) -> any KVCache {
    if let maxSize = maxSize {
        return RotatingKVCache(maxSize: maxSize)
    } else {
        return KVCacheSimple()
    }
}

