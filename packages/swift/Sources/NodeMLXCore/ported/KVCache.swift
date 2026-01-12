// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/cache.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Causal Mask Creation

/// Creates a causal attention mask for autoregressive decoding.
///
/// The mask ensures that each position can only attend to itself and previous positions.
///
/// - Parameters:
///   - n: Query sequence length
///   - offset: Number of previously cached tokens
///   - windowSize: Optional sliding window size for local attention
/// - Returns: Causal mask as MLXArray with shape [n, offset + n]
public func createCausalMask(
    n: Int,
    offset: Int = 0,
    windowSize: Int? = nil
) -> MLXArray {
    // Row indices: [0, 1, ..., n-1] + offset
    let rowIndices = MLXArray(Int32(offset) ..< Int32(offset + n))
        .reshaped([n, 1])

    // Column indices: [0, 1, ..., offset + n - 1]
    let colIndices = MLXArray(0 ..< Int32(offset + n))
        .reshaped([1, offset + n])

    // Causal: can only attend to current and previous positions
    var mask = rowIndices .>= colIndices

    // Optional window constraint: can only attend within window
    if let windowSize {
        let windowMask = rowIndices .< (colIndices + Int32(windowSize))
        mask = logicalAnd(mask, windowMask)
    }

    return mask
}

/// Creates an attention mask appropriate for the given parameters.
///
/// - Parameters:
///   - n: Query sequence length
///   - offset: Cache offset (number of previously cached tokens)
///   - returnArray: If true, always returns array mask; if false, may return "causal" string
///   - windowSize: Optional sliding window size
/// - Returns: Mask mode for MLXFast scaled dot product attention
public func createAttentionMask(
    n: Int,
    offset: Int,
    returnArray: Bool = false,
    windowSize: Int? = nil
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    // Single token generation with no window constraint - no mask needed
    if n == 1 && windowSize == nil {
        return .none
    }

    if returnArray || windowSize != nil {
        return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
    } else {
        return .causal
    }
}

// MARK: - KVCache Protocol

/// Protocol for all KV cache implementations.
///
/// Caches store key-value pairs from previous forward passes to enable
/// efficient autoregressive generation without recomputing attention
/// over the entire sequence.
public protocol KVCacheProtocol: AnyObject {
    /// Updates the cache with new keys/values and returns the full sequence.
    ///
    /// - Parameters:
    ///   - keys: New keys to add, shape [B, H, S, D]
    ///   - values: New values to add, shape [B, H, S, D]
    /// - Returns: Tuple of (allKeys, allValues) including new and cached entries
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Number of cached tokens.
    var offset: Int { get }

    /// Current cached keys and values (for KV-sharing scenarios like Gemma3n).
    /// Returns nil if no cache exists yet.
    var state: (keys: MLXArray, values: MLXArray)? { get }

    /// Whether this cache can be trimmed.
    var isTrimmable: Bool { get }

    /// Trim the cache by removing the last n tokens.
    /// - Returns: Actual number of tokens trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create attention mask for the current cache state.
    ///
    /// - Parameters:
    ///   - queryLength: Length of the query sequence
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: If true, always returns array mask
    /// - Returns: Mask mode for scaled dot product attention
    func makeMask(
        queryLength: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode
}

// MARK: - Default Protocol Implementation

public extension KVCacheProtocol {
    var isTrimmable: Bool { true }

    func makeMask(
        queryLength: Int,
        windowSize: Int? = nil,
        returnArray: Bool = false
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        createAttentionMask(
            n: queryLength,
            offset: offset,
            returnArray: returnArray,
            windowSize: windowSize
        )
    }
}

// MARK: - KVCache

/// Standard KV cache with grow-in-place strategy for efficient memory use.
///
/// Uses a step-based allocation strategy to avoid frequent reallocations.
/// The internal buffer grows in steps of `step` (256) tokens.
///
/// Ported from: mlx_lm/models/cache.py::KVCache
public final class StandardKVCache: KVCacheProtocol {
    /// Growth step size for buffer allocation
    public static let step = 256

    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0

    public init() {}

    /// Returns the current cached keys and values.
    public var state: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values, offset > 0 else { return nil }
        return (k[.ellipsis, ..<offset, 0...], v[.ellipsis, ..<offset, 0...])
    }

    /// Updates the cache with new key/value pairs and returns the full sequence.
    ///
    /// Uses a grow-in-place strategy: the internal buffer grows in steps of
    /// `Self.step` (256) to avoid frequent reallocations.
    ///
    /// - Parameters:
    ///   - keys: New keys to add, shape [B, H, S, D]
    ///   - values: New values to add, shape [B, H, S, D]
    /// - Returns: Tuple of (allKeys, allValues) including new and cached entries
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let numSteps = newKeys.dim(2)

        // Check if we need to grow the buffer
        if keys == nil || (prev + numSteps) > keys!.dim(2) {
            let batchSize = newKeys.dim(0)
            let numKvHeads = newKeys.dim(1)
            let keyHeadDim = newKeys.dim(3)
            let valueHeadDim = newValues.dim(3)

            // Calculate new buffer size (round up to step boundary)
            let nBufferSteps = (Self.step + numSteps - 1) / Self.step
            let bufferSize = nBufferSteps * Self.step

            let kShape = [batchSize, numKvHeads, bufferSize, keyHeadDim]
            let vShape = [batchSize, numKvHeads, bufferSize, valueHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if let existingKeys = keys, let existingValues = values {
                // Trim existing buffer if not aligned to step
                var trimmedKeys = existingKeys
                var trimmedValues = existingValues
                if prev % Self.step != 0 {
                    trimmedKeys = existingKeys[.ellipsis, ..<prev, 0...]
                    trimmedValues = existingValues[.ellipsis, ..<prev, 0...]
                }
                keys = concatenated([trimmedKeys, newK], axis: 2)
                values = concatenated([trimmedValues, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }

        // Update offset and assign new values
        offset += numSteps
        keys![.ellipsis, prev ..< offset, 0...] = newKeys
        values![.ellipsis, prev ..< offset, 0...] = newValues

        // Return sliced view up to current offset
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Converts this cache to a quantized version.
    ///
    /// - Parameters:
    ///   - groupSize: Quantization group size (default: 64)
    ///   - bits: Bits per weight (default: 4)
    /// - Returns: New QuantizedKVCache with quantized contents
    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        let quantCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantCache.offset = offset
        if let k = keys, let v = values {
            quantCache.keys = MLX.quantized(k, groupSize: groupSize, bits: bits)
            quantCache.values = MLX.quantized(v, groupSize: groupSize, bits: bits)
        }
        return quantCache
    }
}

// MARK: - Compatibility Alias

/// Alias for backward compatibility - use StandardKVCache directly for new code.
@available(*, deprecated, renamed: "StandardKVCache")
public typealias SimpleKVCache = StandardKVCache

// MARK: - RotatingKVCache

/// Rotating KV cache for sliding window attention.
///
/// Maintains a fixed-size window of the most recent tokens, with optional
/// "attention sinks" (kept tokens at the beginning) for stability.
///
/// Ported from: mlx_lm/models/cache.py::RotatingKVCache
public final class RotatingKVCache: KVCacheProtocol {
    /// Growth step size for buffer allocation
    public static let step = 256

    /// Number of initial tokens to keep as attention sinks
    public let keep: Int

    /// Maximum cache size (sliding window size)
    public let maxSize: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0

    /// Internal write index for rotation
    private var idx: Int = 0

    /// Creates a rotating KV cache.
    ///
    /// - Parameters:
    ///   - maxSize: Maximum number of tokens to keep in cache
    ///   - keep: Number of initial tokens to preserve as attention sinks (default: 0)
    public init(maxSize: Int, keep: Int = 0) {
        self.maxSize = maxSize
        self.keep = keep
    }

    /// Returns the current cached keys and values in temporal order.
    public var state: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values else { return nil }
        let reorderedK = temporalOrder(k)
        let reorderedV = temporalOrder(v)
        return (reorderedK, reorderedV)
    }

    // MARK: - Private Helpers

    /// Trims the cache and optionally appends new values.
    private func trimBuffer(_ trimSize: Int, _ v: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []

        if trimSize > 0 {
            // Keep the "sink" tokens and skip trimmed portion
            toCat.append(v[.ellipsis, ..<keep, 0...])
            toCat.append(v[.ellipsis, (trimSize + keep)..., 0...])
        } else {
            toCat.append(v)
        }

        if let append {
            toCat.append(append)
        }

        return concatenated(toCat, axis: 2)
    }

    /// Rearranges cache into temporal order.
    private func temporalOrder(_ v: MLXArray) -> MLXArray {
        if idx == v.dim(2) {
            v
        } else if idx < offset {
            // Cache has rotated - reorder
            concatenated([
                v[.ellipsis, ..<keep, 0...],
                v[.ellipsis, idx..., 0...],
                v[.ellipsis, keep ..< idx, 0...],
            ], axis: 2)
        } else {
            v[.ellipsis, ..<idx, 0...]
        }
    }

    /// Update using concatenation (for multi-token input).
    private func updateConcat(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if keys == nil {
            keys = newKeys
            values = newValues
        } else {
            // Reorder to temporal order to preserve context
            keys = temporalOrder(keys!)
            values = temporalOrder(values!)
            idx = keys!.dim(2)

            // Calculate trim size (keep at least maxSize context)
            let trimSize = idx - maxSize + 1
            keys = trimBuffer(trimSize, keys!, append: newKeys)
            values = trimBuffer(trimSize, values!, append: newValues)
        }

        offset += newKeys.dim(2)
        idx = keys!.dim(2)
        return (keys!, values!)
    }

    /// Update in-place (for single-token generation).
    private func updateInPlace(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let batchSize = newKeys.dim(0)
        let numKvHeads = newKeys.dim(1)
        let seqLen = newKeys.dim(2)
        let keyHeadDim = newKeys.dim(3)
        let valueHeadDim = newValues.dim(3)

        let prev = offset

        // Grow cache if needed (up to maxSize)
        if keys == nil || (prev >= keys!.dim(2) && keys!.dim(2) < maxSize) {
            let newSize = min(Self.step, maxSize - prev)
            let kShape = [batchSize, numKvHeads, newSize, keyHeadDim]
            let vShape = [batchSize, numKvHeads, newSize, valueHeadDim]

            let newK = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: newValues.dtype)

            if let existingKeys = keys, let existingValues = values {
                keys = concatenated([existingKeys, newK], axis: 2)
                values = concatenated([existingValues, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = keys!.dim(2) - maxSize
        if trimSize > 0 {
            keys = trimBuffer(trimSize, keys!)
            values = trimBuffer(trimSize, values!)
            idx = maxSize
        }

        // Rotate index when we hit max size
        if idx == maxSize {
            idx = keep
        }

        // Assign new values at current position
        keys![.ellipsis, idx ..< (idx + seqLen), 0...] = newKeys
        values![.ellipsis, idx ..< (idx + seqLen), 0...] = newValues
        offset += seqLen
        idx += seqLen

        // Return current valid portion
        if offset < maxSize {
            return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
        }
        return (keys!, values!)
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if keys.dim(2) == 1 {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }

    public var isTrimmable: Bool {
        offset < maxSize
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    public func makeMask(
        queryLength n: Int,
        windowSize: Int? = nil,
        returnArray: Bool = false
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            let effectiveWindowSize = windowSize ?? maxSize
            let effectiveOffset = min(maxSize - 1, offset)
            if effectiveOffset + n > effectiveWindowSize || returnArray {
                return .array(createCausalMask(n: n, offset: effectiveOffset, windowSize: effectiveWindowSize))
            } else {
                return .causal
            }
        } else {
            // Single token generation
            guard let windowSize else {
                return .none
            }

            // May need mask when window < maxSize
            if offset >= windowSize, maxSize > windowSize {
                var maskIdx = idx
                if maskIdx >= maxSize {
                    maskIdx = 0
                }

                let maskSize = offset < maxSize ? offset + 1 : maxSize
                var mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
                mask = MLX.roll(mask, shift: maskIdx + 1)
                return .array(mask)
            }
            return .none
        }
    }
}

// MARK: - QuantizedKVCache

/// Quantized KV cache for reduced memory usage.
///
/// Stores keys and values in quantized format (default: 8-bit) to reduce
/// memory footprint for long context windows.
///
/// Ported from: mlx_lm/models/cache.py::QuantizedKVCache
public final class QuantizedKVCache: KVCacheProtocol {
    /// Growth step size for buffer allocation
    public static let step = 256

    /// Quantized keys: tuple of (quantized, scales, biases)
    public var keys: (MLXArray, MLXArray, MLXArray?)?

    /// Quantized values: tuple of (quantized, scales, biases)
    public var values: (MLXArray, MLXArray, MLXArray?)?

    public var offset: Int = 0

    /// Quantization group size
    public let groupSize: Int

    /// Bits per quantized value
    public let bits: Int

    /// Creates a quantized KV cache.
    ///
    /// - Parameters:
    ///   - groupSize: Number of values per quantization group (default: 64)
    ///   - bits: Bits per quantized value (default: 8)
    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
    }

    /// Returns dequantized keys and values for KV-sharing scenarios.
    public var state: (keys: MLXArray, values: MLXArray)? {
        guard let k = keys, let v = values, offset > 0 else { return nil }
        let dequantK = MLX.dequantized(k.0, scales: k.1, biases: k.2, groupSize: groupSize, bits: bits)
        let dequantV = MLX.dequantized(v.0, scales: v.1, biases: v.2, groupSize: groupSize, bits: bits)
        return (dequantK[.ellipsis, ..<offset, 0...], dequantV[.ellipsis, ..<offset, 0...])
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let batchSize = newKeys.dim(0)
        let numKvHeads = newKeys.dim(1)
        let numSteps = newKeys.dim(2)
        let keyHeadDim = newKeys.dim(3)
        let valueHeadDim = newValues.dim(3)

        let prev = offset

        // Calculate elements per int for this bit width
        let elPerInt = 8 * MemoryLayout<UInt32>.size / bits

        // Check if we need to grow buffers
        if keys == nil || (prev + numSteps) > keys!.0.dim(2) {
            let newSteps = (Self.step + numSteps - 1) / Self.step * Self.step
            let shape = [batchSize, numKvHeads, newSteps]

            func initQuant(dim: Int) -> (MLXArray, MLXArray, MLXArray?) {
                (
                    MLXArray.zeros(shape + [dim / elPerInt], dtype: .uint32),
                    MLXArray.zeros(shape + [dim / groupSize], dtype: newKeys.dtype),
                    MLXArray.zeros(shape + [dim / groupSize], dtype: newKeys.dtype)
                )
            }

            func expandQuant(_ x: (MLXArray, MLXArray, MLXArray?)) -> (MLXArray, MLXArray, MLXArray?) {
                func expand(_ arr: MLXArray) -> MLXArray {
                    let newArr = MLXArray.zeros(shape + [arr.dim(-1)], dtype: arr.dtype)
                    return concatenated([arr, newArr], axis: 2)
                }
                return (expand(x.0), expand(x.1), x.2.map { expand($0) })
            }

            if keys != nil {
                // Trim if not aligned
                if prev % Self.step != 0 {
                    func trimToOffset(_ x: (MLXArray, MLXArray, MLXArray?)) -> (MLXArray, MLXArray, MLXArray?) {
                        (
                            x.0[.ellipsis, ..<prev, 0...],
                            x.1[.ellipsis, ..<prev, 0...],
                            x.2.map { $0[.ellipsis, ..<prev, 0...] }
                        )
                    }
                    keys = trimToOffset(keys!)
                    values = trimToOffset(values!)
                }
                keys = expandQuant(keys!)
                values = expandQuant(values!)
            } else {
                keys = initQuant(dim: keyHeadDim)
                values = initQuant(dim: valueHeadDim)
            }
        }

        offset += numSteps

        // Quantize new keys and values
        let (qk, sk, bk) = MLX.quantized(newKeys, groupSize: groupSize, bits: bits)
        let (qv, sv, bv) = MLX.quantized(newValues, groupSize: groupSize, bits: bits)

        // Assign to buffers
        keys!.0[.ellipsis, prev ..< offset, 0...] = qk
        keys!.1[.ellipsis, prev ..< offset, 0...] = sk
        if let bk {
            keys!.2![.ellipsis, prev ..< offset, 0...] = bk
        }

        values!.0[.ellipsis, prev ..< offset, 0...] = qv
        values!.1[.ellipsis, prev ..< offset, 0...] = sv
        if let bv {
            values!.2![.ellipsis, prev ..< offset, 0...] = bv
        }

        // Return sliced quantized arrays
        let returnKeys = (
            keys!.0[.ellipsis, ..<offset, 0...],
            keys!.1[.ellipsis, ..<offset, 0...],
            keys!.2.map { $0[.ellipsis, ..<offset, 0...] }
        )
        let returnValues = (
            values!.0[.ellipsis, ..<offset, 0...],
            values!.1[.ellipsis, ..<offset, 0...],
            values!.2.map { $0[.ellipsis, ..<offset, 0...] }
        )

        // Note: Returns quantized tuples, not dequantized arrays
        // The caller must handle dequantization if needed
        return (returnKeys.0, returnValues.0)
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }
}

// MARK: - Factory Functions

/// Creates prompt caches for a model.
///
/// Defers to the model's `makeCache()` if available, otherwise creates
/// default KVCache instances for each layer.
///
/// - Parameters:
///   - numLayers: Number of transformer layers
///   - maxKvSize: If provided, creates RotatingKVCache with this max size
/// - Returns: Array of cache instances, one per layer
public func makePromptCache(
    numLayers: Int,
    maxKvSize: Int? = nil
) -> [any KVCacheProtocol] {
    if let maxKvSize {
        (0 ..< numLayers).map { _ in
            RotatingKVCache(maxSize: maxKvSize, keep: 4)
        }
    } else {
        (0 ..< numLayers).map { _ in StandardKVCache() }
    }
}

/// Returns the maximum cache length across all caches.
public func cacheLength(_ cache: [any KVCacheProtocol]) -> Int {
    cache.map(\.offset).max() ?? 0
}

/// Checks if all caches in the list can be trimmed.
public func canTrimPromptCache(_ cache: [any KVCacheProtocol]) -> Bool {
    cache.allSatisfy(\.isTrimmable)
}

/// Trims all caches by the specified number of tokens.
///
/// - Returns: Actual number of tokens trimmed (from first cache)
@discardableResult
public func trimPromptCache(_ cache: [any KVCacheProtocol], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    return cache.map { $0.trim(numTokens) }.first ?? 0
}
