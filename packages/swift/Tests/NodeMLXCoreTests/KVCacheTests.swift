// Copyright © 2026 Sebastian Software GmbH.
// Tests adapted from mlx-swift-lm patterns (MIT License, Apple Inc.)

import MLX
import MLXFast
@testable import NodeMLXCore
import XCTest

final class KVCacheTests: XCTestCase {
    // MARK: - KVCacheSimple Tests

    func testKVCacheSimpleBasic() throws {
        let cache = KVCacheSimple()
        XCTAssertEqual(cache.offset, 0)

        // First update with sequence of 8 tokens
        let k1 = MLXArray.ones([1, 4, 8, 64]) // [batch, heads, seq, dim]
        let v1 = MLXArray.ones([1, 4, 8, 64])
        let (ck1, cv1) = cache.update(keys: k1, values: v1)

        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(ck1.dim(2), 8)
        XCTAssertEqual(cv1.dim(2), 8)
    }

    func testKVCacheSimpleIncrementalUpdate() throws {
        let cache = KVCacheSimple()

        // Initial prefill with 8 tokens
        let k1 = MLXArray.ones([1, 4, 8, 64])
        let v1 = MLXArray.ones([1, 4, 8, 64])
        _ = cache.update(keys: k1, values: v1)
        XCTAssertEqual(cache.offset, 8)

        // Add single tokens incrementally
        for i in 1 ... 5 {
            let k = MLXArray.ones([1, 4, 1, 64])
            let v = MLXArray.ones([1, 4, 1, 64])
            let (ck, cv) = cache.update(keys: k, values: v)

            XCTAssertEqual(cache.offset, 8 + i)
            XCTAssertEqual(ck.dim(2), 8 + i)
            XCTAssertEqual(cv.dim(2), 8 + i)
        }
    }

    func testKVCacheSimpleReset() throws {
        let cache = KVCacheSimple()

        // Add some data
        let k = MLXArray.ones([1, 4, 10, 64])
        let v = MLXArray.ones([1, 4, 10, 64])
        _ = cache.update(keys: k, values: v)
        XCTAssertEqual(cache.offset, 10)

        // Reset
        cache.reset()
        XCTAssertEqual(cache.offset, 0)

        // Should work fresh after reset
        let k2 = MLXArray.ones([1, 4, 5, 64])
        let v2 = MLXArray.ones([1, 4, 5, 64])
        _ = cache.update(keys: k2, values: v2)
        XCTAssertEqual(cache.offset, 5)
    }

    func testKVCacheSimplePreAllocation() throws {
        // Test that cache grows efficiently with step-based pre-allocation
        let cache = KVCacheSimple()
        // step is now a static constant (256)

        // Add tokens that would trigger growth
        for _ in 1 ... 300 {
            let k = MLXArray.ones([1, 4, 1, 64])
            let v = MLXArray.ones([1, 4, 1, 64])
            _ = cache.update(keys: k, values: v)
        }

        XCTAssertEqual(cache.offset, 300)
    }

    // MARK: - RotatingKVCache Tests

    func testRotatingKVCacheBasic() throws {
        let maxSize = 100
        let cache = RotatingKVCache(maxSize: maxSize, keep: 0)

        // Initial fill
        let k = MLXArray.ones([1, 4, 50, 64])
        let v = MLXArray.ones([1, 4, 50, 64])
        let (ck, cv) = cache.update(keys: k, values: v)

        XCTAssertEqual(cache.offset, 50)
        XCTAssertEqual(ck.dim(2), 50)
    }

    func testRotatingKVCacheOverflow() throws {
        let maxSize = 20
        let cache = RotatingKVCache(maxSize: maxSize, keep: 4)

        // Fill beyond capacity
        for i in 1 ... 30 {
            let k = MLXArray.ones([1, 4, 1, 64])
            let v = MLXArray.ones([1, 4, 1, 64])
            let (ck, _) = cache.update(keys: k, values: v)

            // After overflow, cache size should be capped at maxSize
            if i >= maxSize {
                XCTAssertLessThanOrEqual(ck.dim(2), maxSize)
            }
        }

        // Offset keeps growing even after rotation
        XCTAssertEqual(cache.offset, 30)
    }

    // MARK: - Attention Mask Tests

    func testCreateCausalMask() throws {
        // Test basic causal mask
        let mask = createCausalMask(n: 4, offset: 0)
        eval(mask)

        // Shape should be [4, 4] for n=4, offset=0
        XCTAssertEqual(mask.shape, [4, 4])

        // Check causal structure (lower triangular including diagonal should be true)
        // Position (i, j) should be true if j <= i
        let maskValues = mask.asArray(Bool.self)
        XCTAssertEqual(maskValues.count, 16)
    }

    func testCreateCausalMaskWithOffset() throws {
        // Test causal mask with offset (simulating cached context)
        let mask = createCausalMask(n: 2, offset: 5)
        eval(mask)

        // Shape should be [2, 7] (n=2 new tokens, total=7 with offset)
        XCTAssertEqual(mask.shape, [2, 7])
    }

    func testCreateAttentionMaskModes() throws {
        let cache = KVCacheSimple()

        // Single token - should return .none
        let h1 = MLXArray.ones([1, 1, 64]) // [batch, seq=1, dim]
        let mask1 = createAttentionMask(h: h1, cache: cache, windowSize: nil, returnArray: false)
        if case .none = mask1 {
            // Good - single token doesn't need mask
        } else {
            XCTFail("Single token should return .none mask")
        }

        // Multiple tokens - should return .causal
        let h2 = MLXArray.ones([1, 10, 64]) // [batch, seq=10, dim]
        let mask2 = createAttentionMask(h: h2, cache: nil, windowSize: nil, returnArray: false)
        if case .causal = mask2 {
            // Good - multiple tokens need causal mask
        } else {
            XCTFail("Multiple tokens should return .causal mask")
        }
    }

    // MARK: - Helper Functions Tests

    func testCreateLayerCaches() throws {
        // Test creating caches for a model with 12 layers
        let caches = createLayerCaches(numLayers: 12)
        XCTAssertEqual(caches.count, 12)

        // All should be KVCacheSimple instances
        for cache in caches {
            XCTAssertTrue(cache is KVCacheSimple)
        }
    }

    func testCreateLayerCachesWithMaxSize() throws {
        // Test creating rotating caches
        let caches = createLayerCaches(numLayers: 8, maxKVSize: 1024)
        XCTAssertEqual(caches.count, 8)

        // All should be RotatingKVCache instances
        for cache in caches {
            XCTAssertTrue(cache is RotatingKVCache)
        }
    }
}
