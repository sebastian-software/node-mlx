// Copyright © 2026 Sebastian Software GmbH.
// Tests adapted from mlx-swift-lm patterns (MIT License, Apple Inc.)

import XCTest
import MLX
import MLXFast
@testable import NodeMLXCore

final class AttentionUtilsTests: XCTestCase {

    // MARK: - attentionWithCacheUpdate Tests

    func testAttentionWithoutCache() throws {
        let B = 1  // Batch
        let H = 4  // Heads
        let L = 8  // Sequence length
        let D = 64 // Head dimension

        let queries = MLXArray.ones([B, H, L, D])
        let keys = MLXArray.ones([B, H, L, D])
        let values = MLXArray.ones([B, H, L, D])
        let scale: Float = 1.0 / sqrt(Float(D))

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale,
            mask: .causal
        )
        eval(output)

        XCTAssertEqual(output.shape, [B, H, L, D])
    }

    func testAttentionWithCache() throws {
        let B = 1
        let H = 4
        let D = 64

        let cache = KVCacheSimple()

        // Initial prefill with 8 tokens
        let L1 = 8
        let q1 = MLXArray.ones([B, H, L1, D])
        let k1 = MLXArray.ones([B, H, L1, D])
        let v1 = MLXArray.ones([B, H, L1, D])
        let scale: Float = 1.0 / sqrt(Float(D))

        let output1 = attentionWithCacheUpdate(
            queries: q1,
            keys: k1,
            values: v1,
            cache: cache,
            scale: scale,
            mask: .causal
        )
        eval(output1)

        XCTAssertEqual(output1.shape, [B, H, L1, D])
        XCTAssertEqual(cache.offset, L1)

        // Incremental generation - single token
        let L2 = 1
        let q2 = MLXArray.ones([B, H, L2, D])
        let k2 = MLXArray.ones([B, H, L2, D])
        let v2 = MLXArray.ones([B, H, L2, D])

        let output2 = attentionWithCacheUpdate(
            queries: q2,
            keys: k2,
            values: v2,
            cache: cache,
            scale: scale,
            mask: .none  // No mask needed for single token with cache
        )
        eval(output2)

        XCTAssertEqual(output2.shape, [B, H, L2, D])
        XCTAssertEqual(cache.offset, L1 + L2)
    }

    func testAttentionGQA() throws {
        // Test Grouped Query Attention (different number of query heads vs KV heads)
        let B = 1
        let qHeads = 8  // Query heads
        let kvHeads = 2  // KV heads (GQA ratio = 4)
        let L = 4
        let D = 64

        let queries = MLXArray.ones([B, qHeads, L, D])
        let keys = MLXArray.ones([B, kvHeads, L, D])
        let values = MLXArray.ones([B, kvHeads, L, D])
        let scale: Float = 1.0 / sqrt(Float(D))

        // MLXFast.scaledDotProductAttention handles GQA automatically
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale,
            mask: .causal
        )
        eval(output)

        // Output should have same shape as queries
        XCTAssertEqual(output.shape, [B, qHeads, L, D])
    }

    func testAttentionWithDifferentMasks() throws {
        let B = 1
        let H = 4
        let L = 8
        let D = 64

        let q = MLXArray.ones([B, H, L, D])
        let k = MLXArray.ones([B, H, L, D])
        let v = MLXArray.ones([B, H, L, D])
        let scale: Float = 1.0 / sqrt(Float(D))

        // Test .none mask
        let out1 = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: nil, scale: scale, mask: .none)
        eval(out1)
        XCTAssertEqual(out1.shape, [B, H, L, D])

        // Test .causal mask
        let out2 = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: nil, scale: scale, mask: .causal)
        eval(out2)
        XCTAssertEqual(out2.shape, [B, H, L, D])

        // Test .array mask
        let maskArray = createCausalMask(n: L, offset: 0)
        let out3 = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: nil, scale: scale, mask: .array(maskArray))
        eval(out3)
        XCTAssertEqual(out3.shape, [B, H, L, D])
    }

    // MARK: - Performance Tests

    func testAttentionPerformance() throws {
        // Test that attention is reasonably fast
        let B = 1
        let H = 32  // Realistic number of heads
        let L = 512 // Realistic sequence length
        let D = 64

        let q = MLXArray.ones([B, H, L, D])
        let k = MLXArray.ones([B, H, L, D])
        let v = MLXArray.ones([B, H, L, D])
        let scale: Float = 1.0 / sqrt(Float(D))

        let start = Date()
        let output = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: nil, scale: scale, mask: .causal)
        eval(output)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(output.shape, [B, H, L, D])
        XCTAssertLessThan(elapsed, 2.0, "Attention should complete in under 2 seconds")

        print("Attention with L=\(L), H=\(H) took \(elapsed * 1000)ms")
    }
}

