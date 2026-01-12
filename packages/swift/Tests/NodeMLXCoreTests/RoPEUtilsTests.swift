// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for ported/RoPEUtils.swift

import MLX
import MLXNN
import XCTest

@testable import NodeMLXCore

final class RoPEUtilsTests: XCTestCase {
    // MARK: - StandardRoPE Tests

    func testStandardRoPEInitialization() {
        let rope = StandardRoPE(dims: 64)
        XCTAssertNotNil(rope)
    }

    func testStandardRoPEApply() {
        let rope = StandardRoPE(dims: 64, base: 10000.0)

        // Create test input: [batch, heads, seq, dim]
        let input = MLXArray.ones([1, 4, 8, 64])

        // Apply RoPE at offset 0
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testStandardRoPEWithOffset() {
        let rope = StandardRoPE(dims: 64)

        let input = MLXArray.ones([1, 4, 1, 64])

        // Apply at different offsets
        let output1 = rope(input, offset: 0)
        let output2 = rope(input, offset: 10)

        // Outputs should be different due to different positions
        XCTAssertEqual(output1.shape, output2.shape)
        // Note: We can't easily compare values, but shapes should match
    }

    // MARK: - Llama3RoPE Tests

    func testLlama3RoPEInitialization() {
        let config: [String: Any] = [
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        ]
        let rope = Llama3RoPE(
            dims: 64,
            maxPositionEmbeddings: 8192,
            base: 500_000.0,
            scalingConfig: config
        )
        XCTAssertNotNil(rope)
    }

    func testLlama3RoPEApply() {
        let config: [String: Any] = [
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        ]
        let rope = Llama3RoPE(
            dims: 64,
            maxPositionEmbeddings: 8192,
            base: 500_000.0,
            scalingConfig: config
        )

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }

    // MARK: - SuScaledRoPE Tests

    func testSuScaledRoPEInitialization() {
        let longFactor = [Float](repeating: 1.0, count: 32)
        let rope = SuScaledRoPE(
            dims: 64,
            maxPositionEmbeddings: 131_072,
            originalMaxPositionEmbeddings: 4096,
            longFactor: longFactor
        )
        XCTAssertNotNil(rope)
    }

    func testSuScaledRoPEApply() {
        // Create with proper long factor (one per dimension pair)
        let longFactor = [Float](repeating: 1.0, count: 32) // 64 dims / 2
        let rope = SuScaledRoPE(
            dims: 64,
            maxPositionEmbeddings: 131_072,
            originalMaxPositionEmbeddings: 4096,
            longFactor: longFactor
        )

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }

    // MARK: - YarnRoPE Tests

    func testYarnRoPEInitialization() {
        let rope = YarnRoPE(
            dims: 64,
            traditional: false,
            base: 10000.0,
            scalingFactor: 1.0
        )
        XCTAssertNotNil(rope)
    }

    func testYarnRoPEApply() {
        let rope = YarnRoPE(
            dims: 64,
            traditional: false,
            base: 10000.0,
            scalingFactor: 1.0
        )

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }

    // MARK: - Factory Function Tests

    func testInitializeRopeDefault() {
        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: nil
        )

        XCTAssertTrue(rope is StandardRoPE)
    }

    func testInitializeRopeLlama3() {
        let config: [String: Any] = [
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        ]

        let rope = initializeRope(
            dims: 64,
            base: 500_000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 131_072
        )

        XCTAssertTrue(rope is Llama3RoPE)
    }

    func testInitializeRopeYarn() {
        let config: [String: Any] = [
            "type": "yarn",
            "factor": 2.0,
            "attention_factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 8192
        )

        XCTAssertTrue(rope is YarnRoPE)
    }

    func testInitializeRopeSuScaled() {
        // Note: "su" type maps to "longrope" in initializeRope
        let config: [String: Any] = [
            "type": "longrope",
            "long_factor": [Float](repeating: 1.0, count: 32),
            "original_max_position_embeddings": 4096,
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 131_072
        )

        XCTAssertTrue(rope is SuScaledRoPE)
    }

    func testInitializeRopeLongRope() {
        let config: [String: Any] = [
            "type": "longrope",
            "long_factor": [Float](repeating: 1.0, count: 32),
            "original_max_position_embeddings": 4096,
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 131_072
        )

        XCTAssertTrue(rope is SuScaledRoPE)
    }

    // MARK: - Edge Cases

    func testRoPEWithSmallDimensions() {
        let rope = StandardRoPE(dims: 8)
        let input = MLXArray.ones([1, 1, 4, 8])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testRoPEWithLargeOffset() {
        let rope = StandardRoPE(dims: 64)
        let input = MLXArray.ones([1, 4, 1, 64])

        // Large offset simulating long context
        let output = rope(input, offset: 10000)

        XCTAssertEqual(output.shape, input.shape)
    }

    func testRoPEWithBatchSize() {
        let rope = StandardRoPE(dims: 64)
        let input = MLXArray.ones([4, 8, 16, 64]) // batch=4

        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape)
    }
}
