// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for ported/KVCache.swift

import MLX
import XCTest

@testable import NodeMLXCore

final class KVCacheTests: XCTestCase {
    // MARK: - StandardKVCache Tests

    func testStandardKVCacheInitialState() {
        let cache = StandardKVCache()
        XCTAssertEqual(cache.offset, 0)
        XCTAssertNil(cache.state)
        XCTAssertTrue(cache.isTrimmable)
    }

    func testStandardKVCacheUpdate() {
        let cache = StandardKVCache()

        // Create test tensors: [batch, heads, seq, dim]
        let keys = MLXArray.ones([1, 4, 8, 64])
        let values = MLXArray.ones([1, 4, 8, 64])

        let (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(updatedKeys.shape, [1, 4, 8, 64])
        XCTAssertEqual(updatedValues.shape, [1, 4, 8, 64])
    }

    func testStandardKVCacheMultipleUpdates() {
        let cache = StandardKVCache()

        // First update
        let keys1 = MLXArray.ones([1, 4, 5, 64])
        let values1 = MLXArray.ones([1, 4, 5, 64])
        _ = cache.update(keys: keys1, values: values1)
        XCTAssertEqual(cache.offset, 5)

        // Second update (simulating single token generation)
        let keys2 = MLXArray.ones([1, 4, 1, 64])
        let values2 = MLXArray.ones([1, 4, 1, 64])
        let (updatedKeys, updatedValues) = cache.update(keys: keys2, values: values2)

        XCTAssertEqual(cache.offset, 6)
        // Keys should include all 6 tokens
        XCTAssertEqual(updatedKeys.dim(2), 6)
        XCTAssertEqual(updatedValues.dim(2), 6)
    }

    func testStandardKVCacheState() {
        let cache = StandardKVCache()

        // Initial state should be nil
        XCTAssertNil(cache.state)

        // After update, state should contain the cached values
        let keys = MLXArray.ones([1, 4, 3, 64])
        let values = MLXArray.zeros([1, 4, 3, 64])
        _ = cache.update(keys: keys, values: values)

        let state = cache.state
        XCTAssertNotNil(state)
        XCTAssertEqual(state?.keys.dim(2), 3)
        XCTAssertEqual(state?.values.dim(2), 3)
    }

    func testStandardKVCacheTrim() {
        let cache = StandardKVCache()

        let keys = MLXArray.ones([1, 4, 10, 64])
        let values = MLXArray.ones([1, 4, 10, 64])
        _ = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 10)

        let trimmed = cache.trim(3)
        XCTAssertEqual(trimmed, 3)
        XCTAssertEqual(cache.offset, 7)
    }

    func testStandardKVCacheMakeMask() {
        let cache = StandardKVCache()

        // Single token, no window - should return .none
        let mask1 = cache.makeMask(queryLength: 1, windowSize: nil, returnArray: false)
        if case .none = mask1 {
            // Expected
        } else {
            XCTFail("Expected .none mask for single token")
        }

        // Multiple tokens - should return .causal
        let mask2 = cache.makeMask(queryLength: 5, windowSize: nil, returnArray: false)
        if case .causal = mask2 {
            // Expected
        } else {
            XCTFail("Expected .causal mask for multiple tokens")
        }

        // Force array return
        let mask3 = cache.makeMask(queryLength: 5, windowSize: nil, returnArray: true)
        if case .array = mask3 {
            // Expected
        } else {
            XCTFail("Expected .array mask when returnArray=true")
        }
    }

    // MARK: - RotatingKVCache Tests

    func testRotatingKVCacheInitialState() {
        let cache = RotatingKVCache(maxSize: 512, keep: 4)
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.maxSize, 512)
        XCTAssertEqual(cache.keep, 4)
        XCTAssertNil(cache.state)
    }

    func testRotatingKVCacheUpdate() {
        let cache = RotatingKVCache(maxSize: 16, keep: 2)

        let keys = MLXArray.ones([1, 4, 8, 64])
        let values = MLXArray.ones([1, 4, 8, 64])

        let (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(updatedKeys.dim(2), 8)
        XCTAssertEqual(updatedValues.dim(2), 8)
    }

    func testRotatingKVCacheRotation() {
        let cache = RotatingKVCache(maxSize: 8, keep: 2)

        // Fill initial buffer
        let keys1 = MLXArray.ones([1, 4, 6, 64])
        let values1 = MLXArray.ones([1, 4, 6, 64])
        _ = cache.update(keys: keys1, values: values1)
        XCTAssertEqual(cache.offset, 6)

        // Add more tokens - should start rotating
        let keys2 = MLXArray.ones([1, 4, 4, 64])
        let values2 = MLXArray.ones([1, 4, 4, 64])
        let (updatedKeys, _) = cache.update(keys: keys2, values: values2)

        // Should be at max size after rotation
        XCTAssertEqual(cache.offset, 10)
        // Output should be capped at maxSize
        XCTAssertLessThanOrEqual(updatedKeys.dim(2), cache.maxSize)
    }

    func testRotatingKVCacheTrimmable() {
        let cache = RotatingKVCache(maxSize: 16, keep: 2)

        // Before reaching maxSize, should be trimmable
        let keys = MLXArray.ones([1, 4, 8, 64])
        let values = MLXArray.ones([1, 4, 8, 64])
        _ = cache.update(keys: keys, values: values)

        XCTAssertTrue(cache.isTrimmable)
    }

    // MARK: - QuantizedKVCache Tests

    func testQuantizedKVCacheInitialState() {
        let cache = QuantizedKVCache(groupSize: 64, bits: 8)
        XCTAssertEqual(cache.offset, 0)
        XCTAssertEqual(cache.groupSize, 64)
        XCTAssertEqual(cache.bits, 8)
        XCTAssertNil(cache.state)
    }

    func testQuantizedKVCacheUpdate() {
        let cache = QuantizedKVCache(groupSize: 64, bits: 8)

        // Create test tensors with dimensions divisible by groupSize
        let keys = MLXArray.ones([1, 4, 8, 64])
        let values = MLXArray.ones([1, 4, 8, 64])

        let (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 8)
        // Dequantized output should match original dimensions
        XCTAssertEqual(updatedKeys.shape, [1, 4, 8, 64])
        XCTAssertEqual(updatedValues.shape, [1, 4, 8, 64])
    }

    // MARK: - Helper Function Tests

    func testCreateCausalMask() {
        // Test basic causal mask
        let mask = createCausalMask(n: 4, offset: 0)
        XCTAssertEqual(mask.shape, [4, 4])

        // Upper triangle should be 0 (masked)
        // Lower triangle + diagonal should be 1 (visible)
        let maskArray = mask.asArray(Float.self)
        XCTAssertEqual(maskArray[0], 0) // [0,0] - visible (self)
        XCTAssertEqual(maskArray[1], Float.leastNormalMagnitude) // [0,1] - masked (future)
    }

    func testCreateCausalMaskWithOffset() {
        // Test causal mask with offset (continuing generation)
        let mask = createCausalMask(n: 1, offset: 5)
        // Single token at position 5 should see all 6 positions (0-5)
        XCTAssertEqual(mask.shape, [1, 6])
    }

    func testCreateCausalMaskWithWindow() {
        // Test causal mask with sliding window
        let mask = createCausalMask(n: 4, offset: 0, windowSize: 2)
        XCTAssertEqual(mask.shape, [4, 4])
        // Window should limit visibility
    }

    func testCreateAttentionMask() {
        // Single token, no window - should be .none
        let mask1 = createAttentionMask(n: 1, offset: 0, windowSize: nil)
        if case .none = mask1 {
            // Expected
        } else {
            XCTFail("Expected .none for single token")
        }

        // Multiple tokens - should be .causal
        let mask2 = createAttentionMask(n: 5, offset: 0, returnArray: false, windowSize: nil)
        if case .causal = mask2 {
            // Expected
        } else {
            XCTFail("Expected .causal for multiple tokens")
        }

        // With window - should be .array
        let mask3 = createAttentionMask(n: 5, offset: 0, windowSize: 3)
        if case .array = mask3 {
            // Expected
        } else {
            XCTFail("Expected .array with window constraint")
        }
    }

    // MARK: - Prompt Cache Helper Tests

    func testCacheLength() {
        let cache = StandardKVCache()
        let keys = MLXArray.ones([1, 4, 10, 64])
        let values = MLXArray.ones([1, 4, 10, 64])
        _ = cache.update(keys: keys, values: values)

        let length = cacheLength([cache])
        XCTAssertEqual(length, 10)
    }

    func testCanTrimPromptCache() {
        let cache = StandardKVCache()
        XCTAssertTrue(canTrimPromptCache([cache]))
    }

    func testTrimPromptCache() {
        let cache = StandardKVCache()
        let keys = MLXArray.ones([1, 4, 10, 64])
        let values = MLXArray.ones([1, 4, 10, 64])
        _ = cache.update(keys: keys, values: values)

        let trimmed = trimPromptCache([cache], numTokens: 3)
        XCTAssertEqual(trimmed, 3)
        XCTAssertEqual(cache.offset, 7)
    }
}
