//
//  PerformanceTests.swift
//  Test MLXFast performance
//

import XCTest
import MLX
import MLXFast
@testable import NodeMLXCore

final class PerformanceTests: XCTestCase {

    func testScaledDotProductAttention() throws {
        // Simple attention test
        let q = MLXArray.ones([1, 4, 8, 64])  // [batch, heads, seq, dim]
        let k = MLXArray.ones([1, 4, 8, 64])
        let v = MLXArray.ones([1, 4, 8, 64])

        print("Testing MLXFast.scaledDotProductAttention...")
        let start = Date()
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 0.125,
            mask: .causal
        )
        eval(result)
        let elapsed = Date().timeIntervalSince(start)

        print("Shape: \(result.shape)")
        print("Time: \(elapsed) s")

        XCTAssertEqual(result.shape, [1, 4, 8, 64])
        XCTAssertLessThan(elapsed, 1.0, "Attention should be fast!")
    }

    func testRoPE() throws {
        let x = MLXArray.ones([1, 4, 8, 64])

        print("Testing MLXFast.RoPE...")
        let start = Date()
        let result = MLXFast.RoPE(
            x,
            dimensions: 64,
            traditional: false,
            base: 10000.0,
            scale: 1.0,
            offset: 0
        )
        eval(result)
        let elapsed = Date().timeIntervalSince(start)

        print("Shape: \(result.shape)")
        print("Time: \(elapsed) s")

        XCTAssertEqual(result.shape, [1, 4, 8, 64])
        XCTAssertLessThan(elapsed, 0.5, "RoPE should be fast!")
    }

    func testKVCache() throws {
        var cache = KVCacheSimple()

        // First update
        let k1 = MLXArray.ones([1, 4, 8, 64])
        let v1 = MLXArray.ones([1, 4, 8, 64])
        let (ck1, cv1) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(ck1.dim(2), 8, "Cache should have 8 positions")
        XCTAssertEqual(cache.offset, 8)

        // Second update (single token)
        let k2 = MLXArray.ones([1, 4, 1, 64])
        let v2 = MLXArray.ones([1, 4, 1, 64])
        let (ck2, cv2) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(ck2.dim(2), 9, "Cache should have 9 positions")
        XCTAssertEqual(cache.offset, 9)

        print("KV Cache works correctly!")
    }
}

