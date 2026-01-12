// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for attention mask creation.
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: tests/test_models.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import MLX
import XCTest

@testable import NodeMLXCore

final class MaskTests: XCTestCase {
    // MARK: - Basic Causal Mask Tests

    func testBasicCausalMask() {
        // A basic causal mask should be lower triangular
        let mask = createCausalMask(n: 4, offset: 0)

        // Row 0: [T, F, F, F]
        // Row 1: [T, T, F, F]
        // Row 2: [T, T, T, F]
        // Row 3: [T, T, T, T]

        XCTAssertEqual(mask.shape, [4, 4])

        // First row: only first position visible
        XCTAssertTrue(mask[0, 0].item(Bool.self))
        XCTAssertFalse(mask[0, 1].item(Bool.self))

        // Last row: all positions visible
        XCTAssertTrue(mask[3, 0].item(Bool.self))
        XCTAssertTrue(mask[3, 3].item(Bool.self))
    }

    func testCausalMaskWithOffset() {
        // With offset, the mask should account for cached keys
        let mask = createCausalMask(n: 3, offset: 2)

        // Shape should be [3, 5] (3 queries, 5 keys = 2 cached + 3 new)
        XCTAssertEqual(mask.shape, [3, 5])

        // First query can see all 3 keys (positions 0, 1, 2)
        XCTAssertTrue(mask[0, 0].item(Bool.self))
        XCTAssertTrue(mask[0, 1].item(Bool.self))
        XCTAssertTrue(mask[0, 2].item(Bool.self))
        XCTAssertFalse(mask[0, 3].item(Bool.self))
        XCTAssertFalse(mask[0, 4].item(Bool.self))
    }

    // MARK: - Window Mask Tests

    func testMaskWithWindow() {
        // Test sliding window attention mask
        let mask = createCausalMask(n: 5, offset: 0, windowSize: 3)

        // With window size 3, each position can see at most 3 positions
        // Row 0: [T, F, F, F, F] -> sum = 1
        // Row 1: [T, T, F, F, F] -> sum = 2
        // Row 2: [T, T, T, F, F] -> sum = 3
        // Row 3: [F, T, T, T, F] -> sum = 3
        // Row 4: [F, F, T, T, T] -> sum = 3

        let expectedSums = [1, 2, 3, 3, 3]
        for (i, expected) in expectedSums.enumerated() {
            let rowSum = mask[i].asType(.int32).sum().item(Int.self)
            XCTAssertEqual(rowSum, expected, "Row \(i) should have \(expected) visible positions")
        }
    }

    func testMaskWithWindowAndOffset() {
        // Test sliding window with offset
        let mask = createCausalMask(n: 5, offset: 1, windowSize: 3)

        // Shape should be [5, 6] (5 queries, 1 cached + 5 new)
        XCTAssertEqual(mask.shape, [5, 6])

        // First query at offset 1 can see positions 0 and 1 (within window)
        // Expected sums: [2, 3, 3, 3, 3]
        let expectedSums = [2, 3, 3, 3, 3]
        for (i, expected) in expectedSums.enumerated() {
            let rowSum = mask[i].asType(.int32).sum().item(Int.self)
            XCTAssertEqual(rowSum, expected, "Row \(i) should have \(expected) visible positions")
        }
    }

    func testMaskWithWindowLargerOffset() {
        // With larger offset, window should be fully utilized
        let mask = createCausalMask(n: 5, offset: 2, windowSize: 3)

        // Shape: [5, 7]
        XCTAssertEqual(mask.shape, [5, 7])

        // All positions should see exactly 3 keys (window is full)
        let expectedSums = [3, 3, 3, 3, 3]
        for (i, expected) in expectedSums.enumerated() {
            let rowSum = mask[i].asType(.int32).sum().item(Int.self)
            XCTAssertEqual(rowSum, expected, "Row \(i) should have \(expected) visible positions")
        }
    }

    // MARK: - Edge Cases

    func testSingleTokenMask() {
        let mask = createCausalMask(n: 1, offset: 0)
        XCTAssertEqual(mask.shape, [1, 1])
        XCTAssertTrue(mask[0, 0].item(Bool.self))
    }

    func testSingleTokenWithOffset() {
        let mask = createCausalMask(n: 1, offset: 5)
        XCTAssertEqual(mask.shape, [1, 6])
        // Single query can see all 6 positions
        let rowSum = mask[0].asType(.int32).sum().item(Int.self)
        XCTAssertEqual(rowSum, 6)
    }

    func testWindowSizeOne() {
        let mask = createCausalMask(n: 4, offset: 0, windowSize: 1)

        // Each position can only see itself
        for i in 0 ..< 4 {
            let rowSum = mask[i].asType(.int32).sum().item(Int.self)
            XCTAssertEqual(rowSum, 1, "Row \(i) should only see 1 position")
        }
    }
}
