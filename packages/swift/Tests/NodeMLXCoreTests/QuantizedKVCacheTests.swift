//
//  QuantizedKVCacheTests.swift
//  NodeMLXCoreTests
//
//  Additional tests for KVCache implementations - edge cases and advanced scenarios
//

import MLX
import MLXFast
@testable import NodeMLXCore
import XCTest

class AdditionalKVCacheTests: XCTestCase {
    // MARK: - KVCacheSimple Edge Cases

    func testKVCacheSimpleLargeSequence() {
        let cache = KVCacheSimple()

        // Test with sequence larger than default step size (256)
        let keys = MLXArray.ones([1, 4, 300, 64])
        let values = MLXArray.ones([1, 4, 300, 64])

        let (ck, cv) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 300)
        XCTAssertEqual(ck.dim(2), 300)
        XCTAssertEqual(cv.dim(2), 300)
    }

    func testKVCacheSimpleMultipleResets() {
        let cache = KVCacheSimple()

        // First batch
        let keys1 = MLXArray.ones([1, 4, 50, 64])
        let values1 = MLXArray.ones([1, 4, 50, 64])
        _ = cache.update(keys: keys1, values: values1)
        XCTAssertEqual(cache.offset, 50)

        // Reset and start fresh
        cache.reset()
        XCTAssertEqual(cache.offset, 0)

        // New batch after reset
        let keys2 = MLXArray.ones([1, 4, 100, 64])
        let values2 = MLXArray.ones([1, 4, 100, 64])
        let (ck2, _) = cache.update(keys: keys2, values: values2)

        XCTAssertEqual(cache.offset, 100)
        XCTAssertEqual(ck2.dim(2), 100)
    }

    func testKVCacheSimpleDifferentHeadDims() {
        let cache = KVCacheSimple()

        // Keys and values can have different head dimensions
        let keys = MLXArray.ones([1, 8, 16, 64]) // 8 heads, dim 64
        let values = MLXArray.ones([1, 8, 16, 128]) // 8 heads, dim 128

        let (ck, cv) = cache.update(keys: keys, values: values)

        XCTAssertEqual(ck.dim(3), 64, "Key dimension should be preserved")
        XCTAssertEqual(cv.dim(3), 128, "Value dimension should be preserved")
    }

    func testKVCacheSimpleWithBatchSize() {
        let cache = KVCacheSimple()

        // Test with batch size > 1
        let keys = MLXArray.ones([4, 8, 16, 64]) // batch=4
        let values = MLXArray.ones([4, 8, 16, 64])

        let (ck, cv) = cache.update(keys: keys, values: values)

        XCTAssertEqual(ck.dim(0), 4, "Batch dimension should be preserved")
        XCTAssertEqual(cv.dim(0), 4)
    }

    // MARK: - RotatingKVCache Edge Cases

    func testRotatingKVCacheKeepParameter() {
        // Test that 'keep' tokens are preserved during rotation
        let cache = RotatingKVCache(maxSize: 100, keep: 10, step: 50)

        // Fill cache past rotation point
        for _ in 0 ..< 3 {
            let keys = MLXArray.ones([1, 4, 50, 64])
            let values = MLXArray.ones([1, 4, 50, 64])
            _ = cache.update(keys: keys, values: values)
        }

        // Offset should track total tokens seen
        XCTAssertEqual(cache.offset, 150)
    }

    func testRotatingKVCacheSingleTokenUpdates() {
        let cache = RotatingKVCache(maxSize: 100, keep: 0)

        // Initial batch
        let initKeys = MLXArray.ones([1, 4, 50, 64])
        let initValues = MLXArray.ones([1, 4, 50, 64])
        _ = cache.update(keys: initKeys, values: initValues)

        // Single token updates (typical for generation)
        for i in 0 ..< 60 {
            let keys = MLXArray.ones([1, 4, 1, 64])
            let values = MLXArray.ones([1, 4, 1, 64])
            let (ck, _) = cache.update(keys: keys, values: values)

            if cache.offset <= 100 {
                XCTAssertEqual(ck.dim(2), 51 + i, "Cache should grow until maxSize")
            }
        }

        // Final offset
        XCTAssertEqual(cache.offset, 110)
    }

    func testRotatingKVCacheMaxSizeProperty() {
        let cache = RotatingKVCache(maxSize: 512)
        XCTAssertEqual(cache.maxSize, 512)
    }

    // MARK: - MakeMask Tests

    func testKVCacheSimpleMakeMaskSingleToken() {
        let cache = KVCacheSimple()

        // Add some data to the cache
        let keys = MLXArray.ones([1, 4, 10, 64])
        let values = MLXArray.ones([1, 4, 10, 64])
        _ = cache.update(keys: keys, values: values)

        // Single token should return no mask
        let mask = cache.makeMask(n: 1, windowSize: nil, returnArray: false)

        switch mask {
        case .none:
            break // Expected
        default:
            XCTFail("Single token should return .none mask")
        }
    }

    func testKVCacheSimpleMakeMaskMultiToken() {
        let cache = KVCacheSimple()

        let mask = cache.makeMask(n: 10, windowSize: nil, returnArray: false)

        switch mask {
        case .causal:
            break // Expected for multi-token without window
        default:
            XCTFail("Multi-token should return .causal mask")
        }
    }

    func testKVCacheSimpleMakeMaskWithWindow() {
        let cache = KVCacheSimple()

        // Add data
        let keys = MLXArray.ones([1, 4, 50, 64])
        let values = MLXArray.ones([1, 4, 50, 64])
        _ = cache.update(keys: keys, values: values)

        // Request mask with window size smaller than sequence
        let mask = cache.makeMask(n: 20, windowSize: 10, returnArray: true)

        switch mask {
        case let .array(arr):
            // Should have the mask array
            XCTAssertGreaterThan(arr.size, 0)
        default:
            XCTFail("Should return array mask when window size specified")
        }
    }

    func testRotatingKVCacheMakeMaskAfterRotation() {
        let cache = RotatingKVCache(maxSize: 50, keep: 5)

        // Fill past rotation
        let keys = MLXArray.ones([1, 4, 100, 64])
        let values = MLXArray.ones([1, 4, 100, 64])
        _ = cache.update(keys: keys, values: values)

        // Mask after rotation
        let mask = cache.makeMask(n: 10, windowSize: 30, returnArray: true)

        switch mask {
        case .array:
            break // Expected
        case .causal:
            break // Also acceptable
        default:
            XCTFail("Should return array or causal mask")
        }
    }

    // MARK: - createLayerCaches Tests

    func testCreateLayerCachesDefaultType() {
        let caches = createLayerCaches(numLayers: 32)

        XCTAssertEqual(caches.count, 32)
        XCTAssertTrue(caches[0] is KVCacheSimple)
        XCTAssertTrue(caches[31] is KVCacheSimple)
    }

    func testCreateLayerCachesWithMaxKVSize() {
        let caches = createLayerCaches(numLayers: 24, maxKVSize: 4096)

        XCTAssertEqual(caches.count, 24)

        // All should be RotatingKVCache
        for cache in caches {
            XCTAssertTrue(cache is RotatingKVCache, "Should create RotatingKVCache when maxKVSize specified")
        }
    }

    // MARK: - createCausalMask Tests

    func testCreateCausalMaskBasic() {
        let mask = createCausalMask(n: 4, offset: 0)
        eval(mask)

        // Should be a lower triangular matrix
        XCTAssertEqual(mask.shape, [4, 4])

        // Check diagonal and below are true
        XCTAssertTrue(mask[0, 0].item(Bool.self))
        XCTAssertTrue(mask[1, 1].item(Bool.self))
        XCTAssertTrue(mask[2, 2].item(Bool.self))
        XCTAssertTrue(mask[3, 3].item(Bool.self))

        // Check above diagonal is false
        XCTAssertFalse(mask[0, 1].item(Bool.self))
        XCTAssertFalse(mask[0, 3].item(Bool.self))
    }

    func testCreateCausalMaskWithOffset() {
        let mask = createCausalMask(n: 2, offset: 3)
        eval(mask)

        // With offset 3, new tokens (positions 3,4) can attend to old (0,1,2) and themselves
        XCTAssertEqual(mask.shape, [2, 5]) // 2 new tokens, 5 total positions

        // First new token (pos 3) can see positions 0-3
        XCTAssertTrue(mask[0, 0].item(Bool.self))
        XCTAssertTrue(mask[0, 3].item(Bool.self))
        XCTAssertFalse(mask[0, 4].item(Bool.self)) // Can't see future
    }

    func testCreateCausalMaskWithWindowSize() {
        let mask = createCausalMask(n: 4, offset: 0, windowSize: 2)
        eval(mask)

        XCTAssertEqual(mask.shape, [4, 4])

        // With window size 2, each position can only see 2 previous positions
        // Position 3 can see positions 2 and 3 (not 0 and 1)
        XCTAssertFalse(mask[3, 0].item(Bool.self), "Position 3 should not see position 0 with window=2")
        XCTAssertFalse(mask[3, 1].item(Bool.self), "Position 3 should not see position 1 with window=2")
        XCTAssertTrue(mask[3, 2].item(Bool.self), "Position 3 should see position 2")
        XCTAssertTrue(mask[3, 3].item(Bool.self), "Position 3 should see itself")
    }

    // MARK: - createAttentionMask Function Tests

    func testCreateAttentionMaskNilCache() {
        let h = MLXArray.ones([1, 10, 64]) // seq_len = 10

        let mask = createAttentionMask(h: h, cache: nil, windowSize: nil, returnArray: false)

        switch mask {
        case .causal:
            break // Expected for seq > 1
        default:
            XCTFail("Should return causal mask for multi-token without cache")
        }
    }

    func testCreateAttentionMaskSingleToken() {
        let h = MLXArray.ones([1, 1, 64]) // seq_len = 1

        let mask = createAttentionMask(h: h, cache: nil, windowSize: nil, returnArray: false)

        switch mask {
        case .none:
            break // Expected for single token
        default:
            XCTFail("Should return .none mask for single token")
        }
    }

    func testCreateAttentionMaskWithCache() {
        let cache = KVCacheSimple()

        // Add some data
        let keys = MLXArray.ones([1, 4, 20, 64])
        let values = MLXArray.ones([1, 4, 20, 64])
        _ = cache.update(keys: keys, values: values)

        let h = MLXArray.ones([1, 5, 64]) // New 5 tokens

        let mask = createAttentionMask(h: h, cache: cache, windowSize: nil, returnArray: false)

        // Should delegate to cache.makeMask
        switch mask {
        case .causal:
            break // Expected
        default:
            XCTFail("Should return causal mask from cache")
        }
    }

    // MARK: - Data Type Preservation Tests

    func testKVCachePreservesDtype() {
        let cache = KVCacheSimple()

        // Test with float16
        let keys = MLXArray.ones([1, 4, 10, 64]).asType(.float16)
        let values = MLXArray.ones([1, 4, 10, 64]).asType(.float16)

        let (ck, cv) = cache.update(keys: keys, values: values)

        XCTAssertEqual(ck.dtype, .float16, "Cache should preserve key dtype")
        XCTAssertEqual(cv.dtype, .float16, "Cache should preserve value dtype")
    }

    func testKVCacheWithBFloat16() {
        let cache = KVCacheSimple()

        let keys = MLXArray.ones([1, 4, 10, 64]).asType(.bfloat16)
        let values = MLXArray.ones([1, 4, 10, 64]).asType(.bfloat16)

        let (ck, cv) = cache.update(keys: keys, values: values)

        XCTAssertEqual(ck.dtype, .bfloat16)
        XCTAssertEqual(cv.dtype, .bfloat16)
    }
}
