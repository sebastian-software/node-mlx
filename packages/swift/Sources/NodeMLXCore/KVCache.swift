// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/cache.py
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - KVCache Protocol

/// Protocol for Key/Value caches used in transformer attention layers.
///
/// All cache implementations share a common interface for updating and querying
/// cached key/value pairs. The cache abstracts away the storage strategy
/// (simple, rotating, quantized) from the attention mechanism.
public protocol KVCache: AnyObject {
    /// Current number of cached tokens
    var offset: Int { get }

    /// Current cached keys and values (for KV-sharing scenarios like Gemma3n)
    var state: (keys: MLXArray, values: MLXArray)? { get }

    /// Whether this cache can be trimmed (removed from end)
    var isTrimmable: Bool { get }

    /// Update cache with new key/value pairs and return full cached sequence.
    ///
    /// - Parameters:
    ///   - keys: New keys to cache, shape [B, H, S, D]
    ///   - values: New values to cache, shape [B, H, S, D]
    /// - Returns: Tuple of (allKeys, allValues) including cached entries
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Remove tokens from the end of the cache.
    ///
    /// - Parameter n: Number of tokens to trim
    /// - Returns: Actual number of tokens trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create attention mask for this cache's current state.
    ///
    /// - Parameters:
    ///   - queryLength: Number of query tokens (N)
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of explicit mask array
    /// - Returns: Mask mode for scaled dot product attention
    func makeMask(
        queryLength: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode
}

// MARK: - Default Implementations

public extension KVCache {
    var isTrimmable: Bool { false }

    func trim(_: Int) -> Int { 0 }
}

// MARK: - Causal Mask Creation

/// Creates a causal attention mask with optional sliding window.
///
/// The mask ensures that each position can only attend to previous positions
/// (and itself). With a window size, attention is further limited to the
/// most recent `windowSize` positions.
///
/// - Parameters:
///   - n: Number of query positions
///   - offset: Offset into the sequence (for cached keys)
///   - windowSize: Optional sliding window size
/// - Returns: Boolean mask array of shape [1, 1, N, offset+N]
public func createCausalMask(
    n: Int,
    offset: Int = 0,
    windowSize: Int? = nil
) -> MLXArray {
    // Row indices: positions in full sequence [0, offset+n)
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    // Column indices: query positions [offset, offset+n)
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds

    // Reshape for broadcasting: linds [N, 1], rinds [1, offset+N]
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]

    // Causal: each position attends to itself and earlier positions
    var mask = linds .>= rinds

    // Sliding window: limit attention to recent positions
    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    return mask
}

/// Creates attention mask based on hidden state and cache.
///
/// Convenience function that delegates to the cache's mask creation
/// or falls back to default behavior when no cache is present.
///
/// - Parameters:
///   - h: Hidden state tensor, shape [B, N, ...]
///   - cache: Optional KV cache
///   - windowSize: Optional sliding window size
///   - returnArray: Force return of explicit mask array
/// - Returns: Mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's implementation if available
    if let cache {
        return cache.makeMask(queryLength: n, windowSize: windowSize, returnArray: returnArray)
    }

    // No cache: simple causal mask
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

// MARK: - KVCacheSimple

/// Standard KV cache with grow-in-place allocation strategy.
///
/// This is the default cache for most transformer models. It grows the internal
/// buffer in steps (default 256 tokens) to balance memory efficiency with
/// allocation overhead.
///
/// ## Usage
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.updateAndFetch(keys: newKeys, values: newValues)
/// ```
///
/// ## Memory Strategy
/// The cache pre-allocates buffer space in chunks of `step` tokens.
/// When the buffer fills, a new chunk is concatenated. This avoids
/// per-token allocation overhead while keeping memory bounded.
public class KVCacheSimple: KVCache {
    // MARK: - Configuration

    /// Buffer growth step size (tokens)
    public static let step = 256

    // MARK: - State

    public private(set) var offset: Int = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    // MARK: - Initialization

    public init() {}

    // MARK: - KVCache Protocol

    public var state: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values, offset > 0 else { return nil }
        return (k[.ellipsis, ..<offset, 0...], v[.ellipsis, ..<offset, 0...])
    }

    public var isTrimmable: Bool { true }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let numNewTokens = newKeys.dim(2)
        let step = Self.step

        // Check if we need to grow the buffer
        let needsGrowth: Bool = {
            guard let currentKeys = keys else { return true }
            return (prev + numNewTokens) > currentKeys.dim(2)
        }()

        if needsGrowth {
            let B = newKeys.dim(0)
            let nKVHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)

            // Calculate new buffer size (rounded up to step)
            let nSteps = (step + numNewTokens - 1) / step
            let kShape = [B, nKVHeads, nSteps * step, kHeadDim]
            let vShape = [B, nKVHeads, nSteps * step, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if var currentKeys = keys, var currentValues = values {
                // Trim to actual content if not aligned to step boundary
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                keys = concatenated([currentKeys, newK], axis: 2)
                values = concatenated([currentValues, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }

        // Write new tokens into buffer
        offset += numNewTokens
        keys![.ellipsis, prev ..< offset, 0...] = newKeys
        values![.ellipsis, prev ..< offset, 0...] = newValues

        // Return valid portion
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }

    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public func makeMask(
        queryLength n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // Single token: no mask needed
        if n == 1 {
            return .none
        }

        // Multi-token: check if explicit array is needed
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }

    // MARK: - Additional Operations

    /// Reset cache to empty state
    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

// MARK: - RotatingKVCache

/// Rotating KV cache for sliding window attention.
///
/// This cache maintains a fixed-size buffer that rotates once full.
/// It's essential for models like Mistral and GPT-OSS that use
/// sliding window attention to limit memory usage.
///
/// ## How It Works
/// 1. Cache grows normally until reaching `maxSize`
/// 2. Once full, new tokens overwrite oldest tokens (after `keep` positions)
/// 3. The `keep` parameter preserves attention sinks at the start
///
/// ## Usage
/// ```swift
/// let cache = RotatingKVCache(maxSize: 4096, keep: 4)
/// ```
public class RotatingKVCache: KVCache {
    // MARK: - Configuration

    /// Buffer growth step size
    public static let step = 256

    /// Maximum cache size (sliding window)
    public let maxSize: Int

    /// Number of initial positions to preserve (attention sinks)
    public let keep: Int

    // MARK: - State

    public private(set) var offset: Int = 0
    private var keys: MLXArray?
    private var values: MLXArray?
    private var idx: Int = 0

    // MARK: - Initialization

    /// Create a rotating cache with specified window size.
    ///
    /// - Parameters:
    ///   - maxSize: Maximum number of tokens to cache
    ///   - keep: Number of initial tokens to always preserve (default: 0)
    public init(maxSize: Int, keep: Int = 0) {
        self.maxSize = maxSize
        self.keep = keep
    }

    // MARK: - KVCache Protocol

    public var state: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        return (temporalOrder(k), temporalOrder(v))
    }

    public var isTrimmable: Bool {
        offset < maxSize
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        // Single token: use efficient in-place update with rotation
        if newKeys.dim(2) == 1 {
            return updateInPlace(keys: newKeys, values: newValues)
        }
        // Multi-token (prompt): use concatenation strategy
        return updateConcat(keys: newKeys, values: newValues)
    }

    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    public func makeMask(
        queryLength n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            // Multi-token case
            let actualWindowSize = windowSize ?? maxSize
            let cappedOffset = min(maxSize - 1, offset)

            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        }

        // Single token case
        guard let windowSize else {
            return .none
        }

        // Need mask when window < maxSize and cache has wrapped
        if offset >= windowSize, maxSize > windowSize {
            var currentIdx = idx
            if currentIdx >= maxSize {
                currentIdx = 0
            }

            let maskSize = offset < maxSize ? offset + 1 : maxSize
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            let rolledMask = roll(mask, shift: currentIdx + 1)

            return .array(rolledMask)
        }

        return .none
    }

    // MARK: - Private Helpers

    /// Trim array and optionally append new content
    private func trim(_ array: MLXArray, by trimSize: Int, append: MLXArray? = nil) -> MLXArray {
        var parts: [MLXArray] = []

        if trimSize > 0 {
            // Keep preserved tokens + everything after trim point
            parts = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            parts = [array]
        }

        if let append {
            parts.append(append)
        }

        return concatenated(parts, axis: 2)
    }

    /// Rearrange cache into temporal order for state access
    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        let size = array.dim(2)

        if idx == size {
            return array
        } else if idx < offset {
            // Cache has wrapped: reorder [keep, idx+keep..., keep..idx]
            return concatenated([
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, idx..., 0...],
                array[.ellipsis, keep ..< idx, 0...],
            ], axis: 2)
        } else {
            // Cache hasn't wrapped: just slice to valid region
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    /// Update with concatenation (for multi-token prompts)
    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if keys == nil {
            keys = newKeys
            values = newValues
        } else {
            // Restore temporal order before modification
            keys = temporalOrder(keys!)
            values = temporalOrder(values!)
            idx = keys!.dim(2)

            // Trim to maintain max size (allow temporary growth of S-1)
            let trimSize = idx - maxSize + 1
            keys = trim(keys!, by: trimSize, append: newKeys)
            values = trim(values!, by: trimSize, append: newValues)
        }

        offset += newKeys.dim(2)
        idx = keys!.dim(2)

        return (keys!, values!)
    }

    /// Update in-place with rotation (for single tokens during generation)
    private func updateInPlace(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = newKeys.dim(0)
        let nKVHeads = newKeys.dim(1)
        let S = newKeys.dim(2)
        let kHeadDim = newKeys.dim(3)
        let vHeadDim = newValues.dim(3)
        let prev = offset
        let step = Self.step

        // Grow buffer if needed (before hitting maxSize)
        if keys == nil || (prev >= keys!.dim(2) && keys!.dim(2) < maxSize) {
            let newSize = min(step, maxSize - prev)
            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if let currentKeys = keys, let currentValues = values {
                keys = concatenated([currentKeys, newK], axis: 2)
                values = concatenated([currentValues, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
            idx = prev
        }

        // Trim if we've exceeded maxSize
        let trimSize = keys!.dim(2) - maxSize
        if trimSize > 0 {
            keys = trim(keys!, by: trimSize)
            values = trim(values!, by: trimSize)
            idx = maxSize
        }

        // Rotate: wrap around after preserved tokens
        if idx == maxSize {
            idx = keep
        }

        // Write new token
        keys![.ellipsis, idx ..< (idx + S), 0...] = newKeys
        values![.ellipsis, idx ..< (idx + S), 0...] = newValues
        offset += S
        idx += S

        // Return valid portion
        if offset < maxSize {
            return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
        }
        return (keys!, values!)
    }
}

// MARK: - QuantizedKVCache

/// Quantized KV cache for memory-efficient long contexts.
///
/// This cache stores keys and values in quantized format (default 8-bit),
/// reducing memory usage by 4x compared to float16. Essential for
/// processing long documents or conversations.
///
/// ## Trade-offs
/// - Pro: ~4x memory reduction
/// - Con: Slight quality degradation from quantization
///
/// ## Usage
/// ```swift
/// let cache = QuantizedKVCache(groupSize: 64, bits: 8)
/// ```
public class QuantizedKVCache: KVCache {
    // MARK: - Configuration

    /// Buffer growth step size
    public static let step = 256

    /// Quantization group size
    public let groupSize: Int

    /// Bits per value (4 or 8)
    public let bits: Int

    // MARK: - State

    public private(set) var offset: Int = 0

    /// Quantized keys: (quantized, scales, biases)
    private var keys: (MLXArray, MLXArray, MLXArray)?

    /// Quantized values: (quantized, scales, biases)
    private var values: (MLXArray, MLXArray, MLXArray)?

    // MARK: - Initialization

    /// Create a quantized cache.
    ///
    /// - Parameters:
    ///   - groupSize: Number of values per quantization group (default: 64)
    ///   - bits: Bits per quantized value, 4 or 8 (default: 8)
    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
    }

    // MARK: - KVCache Protocol

    public var state: (keys: MLXArray, values: MLXArray)? {
        // Note: Returns quantized format - caller must handle dequantization
        guard let k = keys, let v = values else { return nil }
        if offset == k.0.dim(2) {
            return (k.0, v.0)
        }
        return (k.0[.ellipsis, ..<offset, 0...], v.0[.ellipsis, ..<offset, 0...])
    }

    public var isTrimmable: Bool { true }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = newKeys.dim(0)
        let nKVHeads = newKeys.dim(1)
        let numNewTokens = newKeys.dim(2)
        let kHeadDim = newKeys.dim(3)
        let vHeadDim = newValues.dim(3)
        let prev = offset
        let step = Self.step

        // Check if we need to grow the buffer
        let needsGrowth: Bool = {
            guard let currentKeys = keys else { return true }
            return (prev + numNewTokens) > currentKeys.0.dim(2)
        }()

        if needsGrowth {
            let elPerInt = 8 * MemoryLayout<UInt32>.size / bits
            let newSteps = (step + numNewTokens - 1) / step * step
            let shape: [Int] = [B, nKVHeads, newSteps]

            func initQuant(dim: Int) -> (MLXArray, MLXArray, MLXArray) {
                (
                    MLXArray.zeros(shape + [dim / elPerInt], dtype: .uint32),
                    MLXArray.zeros(shape + [dim / groupSize], dtype: newKeys.dtype),
                    MLXArray.zeros(shape + [dim / groupSize], dtype: newKeys.dtype)
                )
            }

            func expandQuant(_ x: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
                func expand(_ arr: MLXArray) -> MLXArray {
                    let newArr = MLXArray.zeros(shape + [arr.dim(-1)], dtype: arr.dtype)
                    return concatenated([arr, newArr], axis: 2)
                }
                return (expand(x.0), expand(x.1), expand(x.2))
            }

            if var currentKeys = keys, var currentValues = values {
                // Trim to actual content if not step-aligned
                if prev % step != 0 {
                    func trimToOffset(_ x: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
                        (x.0[.ellipsis, ..<prev, 0...], x.1[.ellipsis, ..<prev, 0...], x.2[.ellipsis, ..<prev, 0...])
                    }
                    currentKeys = trimToOffset(currentKeys)
                    currentValues = trimToOffset(currentValues)
                }
                keys = expandQuant(currentKeys)
                values = expandQuant(currentValues)
            } else {
                keys = initQuant(dim: kHeadDim)
                values = initQuant(dim: vHeadDim)
            }
        }

        offset += numNewTokens

        // Quantize new tokens (affine mode always produces biases)
        let qKeysResult = MLX.quantized(newKeys, groupSize: groupSize, bits: bits, mode: .affine)
        let qValuesResult = MLX.quantized(newValues, groupSize: groupSize, bits: bits, mode: .affine)

        // Write into buffer
        keys!.0[.ellipsis, prev ..< offset, 0...] = qKeysResult.wq
        keys!.1[.ellipsis, prev ..< offset, 0...] = qKeysResult.scales
        keys!.2[.ellipsis, prev ..< offset, 0...] = qKeysResult.biases!
        values!.0[.ellipsis, prev ..< offset, 0...] = qValuesResult.wq
        values!.1[.ellipsis, prev ..< offset, 0...] = qValuesResult.scales
        values!.2[.ellipsis, prev ..< offset, 0...] = qValuesResult.biases!

        // Return valid portion (still quantized - caller handles SDPA)
        func slice(_ x: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
            (x.0[.ellipsis, ..<offset, 0...], x.1[.ellipsis, ..<offset, 0...], x.2[.ellipsis, ..<offset, 0...])
        }

        // Note: This returns the first component only for interface compatibility
        // Real quantized SDPA requires all three components
        let slicedKeys = slice(keys!)
        let slicedValues = slice(values!)
        return (slicedKeys.0, slicedValues.0)
    }

    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public func makeMask(
        queryLength n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    // MARK: - Quantized Access

    /// Get full quantized state for quantized SDPA.
    ///
    /// - Returns: Tuple of ((keys, scales, biases), (values, scales, biases))
    public var quantizedState: ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray))? {
        guard let k = keys, let v = values else { return nil }
        if offset == k.0.dim(2) {
            return (k, v)
        }
        func slice(_ x: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
            (x.0[.ellipsis, ..<offset, 0...], x.1[.ellipsis, ..<offset, 0...], x.2[.ellipsis, ..<offset, 0...])
        }
        return (slice(k), slice(v))
    }
}

// MARK: - Factory Functions

/// Create caches for all layers of a model.
///
/// - Parameters:
///   - numLayers: Number of transformer layers
///   - maxKVSize: Optional maximum cache size (enables rotating cache)
/// - Returns: Array of caches, one per layer
public func createLayerCaches(numLayers: Int, maxKVSize: Int? = nil) -> [KVCache] {
    if let maxKVSize {
        (0 ..< numLayers).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
    } else {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }
}
