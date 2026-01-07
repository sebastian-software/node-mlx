//
//  KVCache.swift
//  NodeMLXCore
//
//  Key-Value cache for efficient autoregressive generation.
//

import Foundation
import MLX

// MARK: - KVCache Protocol

/// Protocol for key-value caches used in transformer attention
public protocol KVCacheProtocol {
    /// Current sequence length in cache
    var sequenceLength: Int { get }

    /// Maximum sequence length
    var maxSequenceLength: Int { get }

    /// Update cache with new keys and values
    mutating func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Reset the cache
    mutating func reset()
}

// MARK: - Simple KVCache

/// Simple key-value cache implementation
public struct KVCache: KVCacheProtocol {
    private var keys: MLXArray?
    private var values: MLXArray?
    public let maxSequenceLength: Int

    public init(maxSequenceLength: Int = 4096) {
        self.maxSequenceLength = maxSequenceLength
    }

    public var sequenceLength: Int {
        keys?.dim(2) ?? 0
    }

    public mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            // Concatenate with existing cache
            self.keys = concatenated([existingKeys, newKeys], axis: 2)
            self.values = concatenated([existingValues, newValues], axis: 2)
        } else {
            // Initialize cache
            self.keys = newKeys
            self.values = newValues
        }

        // Truncate if exceeds max length
        if sequenceLength > maxSequenceLength {
            let start = sequenceLength - maxSequenceLength
            self.keys = self.keys?[0..., 0..., start..., 0...]
            self.values = self.values?[0..., 0..., start..., 0...]
        }

        return (self.keys!, self.values!)
    }

    public mutating func reset() {
        keys = nil
        values = nil
    }
}

// MARK: - Rotating KVCache

/// KVCache with rotating buffer for efficient memory usage
public struct RotatingKVCache: KVCacheProtocol {
    private var keys: MLXArray?
    private var values: MLXArray?
    private var position: Int = 0
    public let maxSequenceLength: Int

    public init(maxSequenceLength: Int = 4096) {
        self.maxSequenceLength = maxSequenceLength
    }

    public var sequenceLength: Int {
        min(position, maxSequenceLength)
    }

    public mutating func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = newKeys.dim(2)

        if keys == nil {
            // Initialize with zeros
            let shape = [newKeys.dim(0), newKeys.dim(1), maxSequenceLength, newKeys.dim(3)]
            self.keys = MLXArray.zeros(shape, dtype: newKeys.dtype)
            self.values = MLXArray.zeros(shape, dtype: newValues.dtype)
        }

        // Calculate position in rotating buffer
        let startPos = position % maxSequenceLength
        let endPos = (position + seqLen) % maxSequenceLength

        if endPos > startPos {
            // Simple case: no wrap-around
            // Note: MLX doesn't support in-place updates easily,
            // so this is simplified
            self.keys = concatenated([keys![0..., 0..., 0..<startPos, 0...], newKeys], axis: 2)
            self.values = concatenated([values![0..., 0..., 0..<startPos, 0...], newValues], axis: 2)
        }

        position += seqLen

        // Return full cache for attention
        return (keys!, values!)
    }

    public mutating func reset() {
        keys = nil
        values = nil
        position = 0
    }
}

// MARK: - Multi-Layer Cache

/// Cache manager for multiple transformer layers
public class LayerKVCaches {
    public var caches: [KVCache]

    public init(numLayers: Int, maxSequenceLength: Int = 4096) {
        self.caches = (0..<numLayers).map { _ in
            KVCache(maxSequenceLength: maxSequenceLength)
        }
    }

    public subscript(index: Int) -> KVCache {
        get { caches[index] }
        set { caches[index] = newValue }
    }

    public func reset() {
        for i in 0..<caches.count {
            caches[i].reset()
        }
    }
}

