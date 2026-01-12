// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Performance tests for MLX operations.

import MLX
import MLXFast
@testable import NodeMLXCore
import XCTest

final class PerformanceTests: XCTestCase {
    func testScaledDotProductAttention() throws {
        // Simple attention test
        let q = MLXArray.ones([1, 4, 8, 64]) // [batch, heads, seq, dim]
        let k = MLXArray.ones([1, 4, 8, 64])
        let v = MLXArray.ones([1, 4, 8, 64])

        let start = Date()
        let result = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: 0.125,
            mask: .causal
        )
        eval(result)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(result.shape, [1, 4, 8, 64])
        XCTAssertLessThan(elapsed, 1.0, "Attention should be fast!")
    }

    func testRoPE() throws {
        let x = MLXArray.ones([1, 4, 8, 64])

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

        XCTAssertEqual(result.shape, [1, 4, 8, 64])
        XCTAssertLessThan(elapsed, 0.5, "RoPE should be fast!")
    }

    func testKVCache() throws {
        let cache = StandardKVCache()

        // First update
        let k1 = MLXArray.ones([1, 4, 8, 64])
        let v1 = MLXArray.ones([1, 4, 8, 64])
        let (ck1, _) = cache.update(keys: k1, values: v1)
        XCTAssertEqual(ck1.dim(2), 8, "Cache should have 8 positions")
        XCTAssertEqual(cache.offset, 8)

        // Second update (single token)
        let k2 = MLXArray.ones([1, 4, 1, 64])
        let v2 = MLXArray.ones([1, 4, 1, 64])
        let (ck2, _) = cache.update(keys: k2, values: v2)
        XCTAssertEqual(ck2.dim(2), 9, "Cache should have 9 positions")
        XCTAssertEqual(cache.offset, 9)
    }

    func testMatmulPerformance() throws {
        // Test basic matmul performance
        let a = MLXArray.ones([256, 512])
        let b = MLXArray.ones([512, 256])

        let start = Date()
        let result = matmul(a, b)
        eval(result)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(result.shape, [256, 256])
        XCTAssertLessThan(elapsed, 0.5, "Matmul should be fast!")
    }

    func testSoftmaxPerformance() throws {
        let x = MLXArray.ones([1, 32, 128, 128])

        let start = Date()
        let result = softmax(x, axis: -1)
        eval(result)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(result.shape, [1, 32, 128, 128])
        XCTAssertLessThan(elapsed, 0.5, "Softmax should be fast!")
    }
}
