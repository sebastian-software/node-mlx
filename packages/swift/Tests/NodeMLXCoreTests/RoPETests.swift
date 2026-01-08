//
//  RoPETests.swift
//  NodeMLXCoreTests
//
//  Tests for Rotary Position Embedding implementations
//

import XCTest
import MLX
import MLXNN
@testable import NodeMLXCore

class RoPETests: XCTestCase {

    // MARK: - initializeRope Factory Tests

    func testInitializeRopeDefault() {
        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: 2048
        )

        XCTAssertTrue(rope is RoPE, "Default should create standard RoPE")
    }

    func testInitializeRopeLinear() {
        let config: [String: StringOrNumber] = [
            "type": .string("linear"),
            "factor": .float(2.0)
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 2048
        )

        XCTAssertTrue(rope is RoPE, "Linear should create standard RoPE with scale")
    }

    func testInitializeRopeLlama3() {
        let config: [String: StringOrNumber] = [
            "type": .string("llama3"),
            "factor": .float(8.0),
            "low_freq_factor": .float(1.0),
            "high_freq_factor": .float(4.0),
            "original_max_position_embeddings": .int(8192)
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 131072
        )

        XCTAssertTrue(rope is Llama3RoPE, "llama3 type should create Llama3RoPE")
    }

    func testInitializeRopeYarn() {
        let config: [String: StringOrNumber] = [
            "type": .string("yarn"),
            "factor": .float(16.0),
            "original_max_position_embeddings": .int(4096),
            "beta_fast": .float(32.0),
            "beta_slow": .float(1.0),
            "mscale": .float(1.0),
            "mscale_all_dim": .float(0.0)
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 65536
        )

        XCTAssertTrue(rope is YarnRoPE, "yarn type should create YarnRoPE")
    }

    func testInitializeRopeLongrope() {
        // LongRoPE requires short_factor and long_factor arrays
        let config: [String: StringOrNumber] = [
            "type": .string("longrope"),
            "original_max_position_embeddings": .int(4096),
            "short_factor": .floats(Array(repeating: 1.0, count: 32)),
            "long_factor": .floats(Array(repeating: 2.0, count: 32))
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 131072
        )

        XCTAssertTrue(rope is SuScaledRoPE, "longrope type should create SuScaledRoPE")
    }

    func testInitializeRopeMrope() {
        let config: [String: StringOrNumber] = [
            "type": .string("mrope"),
            "mrope_section": .ints([16, 24, 24])
        ]

        let rope = initializeRope(
            dims: 64,
            base: 10000.0,
            traditional: false,
            scalingConfig: config,
            maxPositionEmbeddings: 2048
        )

        // MRoPE falls back to standard RoPE
        XCTAssertTrue(rope is RoPE, "mrope type should create standard RoPE")
    }

    // MARK: - Llama3RoPE Tests

    func testLlama3RoPEOutput() {
        let config: [String: StringOrNumber] = [
            "factor": .float(8.0),
            "low_freq_factor": .float(1.0),
            "high_freq_factor": .float(4.0),
            "original_max_position_embeddings": .int(8192)
        ]

        let rope = Llama3RoPE(
            dims: 64,
            maxPositionEmbeddings: 131072,
            traditional: false,
            base: 10000.0,
            scalingConfig: config
        )

        // Test input: [batch=1, seq=4, heads=8, dims=64]
        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testLlama3RoPEWithOffset() {
        let config: [String: StringOrNumber] = [
            "factor": .float(8.0),
            "low_freq_factor": .float(1.0),
            "high_freq_factor": .float(4.0),
            "original_max_position_embeddings": .int(8192)
        ]

        let rope = Llama3RoPE(
            dims: 64,
            maxPositionEmbeddings: 131072,
            traditional: false,
            base: 10000.0,
            scalingConfig: config
        )

        let input = MLXArray.ones([1, 1, 8, 64])

        // Same input at different offsets should produce different outputs
        let output0 = rope(input, offset: 0)
        let output100 = rope(input, offset: 100)
        eval(output0, output100)

        // Check that outputs differ
        let diff = abs(output0 - output100)
        let maxDiff = MLX.max(diff).item(Float.self)
        XCTAssertGreaterThan(maxDiff, 0.01, "Different offsets should produce different embeddings")
    }

    // MARK: - YarnRoPE Tests

    func testYarnRoPEOutput() {
        let rope = YarnRoPE(
            dimensions: 64,
            traditional: false,
            maxPositionEmbeddings: 65536,
            base: 10000.0,
            scalingFactor: 16.0,
            originalMaxPositionEmbeddings: 4096,
            betaFast: 32.0,
            betaSlow: 1.0,
            mscale: 1.0,
            mscaleAllDim: 0.0
        )

        // Test input
        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testYarnRoPEWithMscale() {
        // Test with mscale != 1.0
        let rope = YarnRoPE(
            dimensions: 64,
            traditional: false,
            maxPositionEmbeddings: 65536,
            base: 10000.0,
            scalingFactor: 16.0,  // > 1 will activate mscale
            originalMaxPositionEmbeddings: 4096,
            betaFast: 32.0,
            betaSlow: 1.0,
            mscale: 0.707,  // Custom mscale
            mscaleAllDim: 0.0
        )

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    // MARK: - SuScaledRoPE Tests

    func testSuScaledRoPEShortContext() {
        let rope = SuScaledRoPE(
            dimensions: 64,
            base: 10000.0,
            maxPositionEmbeddings: 131072,
            originalMaxPositionEmbeddings: 4096,
            shortFactor: Array(repeating: 1.0, count: 32),
            longFactor: Array(repeating: 2.0, count: 32)
        )

        // Short context (within original max)
        let input = MLXArray.ones([1, 100, 8, 64])  // seq=100 < 4096
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testSuScaledRoPELongContext() {
        let rope = SuScaledRoPE(
            dimensions: 64,
            base: 10000.0,
            maxPositionEmbeddings: 131072,
            originalMaxPositionEmbeddings: 4096,
            shortFactor: Array(repeating: 1.0, count: 32),
            longFactor: Array(repeating: 2.0, count: 32)
        )

        // Long context (beyond original max using offset)
        let input = MLXArray.ones([1, 100, 8, 64])
        let output = rope(input, offset: 5000)  // 100 + 5000 > 4096

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testSuScaledRoPEDifferentContextLengths() {
        let rope = SuScaledRoPE(
            dimensions: 64,
            base: 10000.0,
            maxPositionEmbeddings: 131072,
            originalMaxPositionEmbeddings: 4096,
            shortFactor: Array(repeating: 1.0, count: 32),
            longFactor: Array(repeating: 3.0, count: 32)
        )

        let input = MLXArray.ones([1, 100, 8, 64])

        // Short vs long context should produce different outputs
        let outputShort = rope(input, offset: 0)     // seq_len = 100 < 4096
        let outputLong = rope(input, offset: 4000)   // seq_len = 4100 > 4096
        eval(outputShort, outputLong)

        let diff = abs(outputShort - outputLong)
        let maxDiff = MLX.max(diff).item(Float.self)

        // Should use different frequency factors
        XCTAssertGreaterThan(maxDiff, 0.001, "Short and long context should produce different embeddings")
    }

    // MARK: - Standard RoPE Reference Tests

    func testStandardRoPEBasic() {
        let rope = RoPE(dimensions: 64, traditional: false, base: 10000.0, scale: 1.0)

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testStandardRoPEWithScale() {
        let rope = RoPE(dimensions: 64, traditional: false, base: 10000.0, scale: 0.5)

        let input = MLXArray.ones([1, 4, 8, 64])
        let output = rope(input, offset: 0)

        XCTAssertEqual(output.shape, input.shape, "Output shape should match input shape")
    }

    func testRoPETraditionalMode() {
        // Traditional mode uses different rotation formula
        let ropeTraditional = RoPE(dimensions: 64, traditional: true, base: 10000.0, scale: 1.0)
        let ropeModern = RoPE(dimensions: 64, traditional: false, base: 10000.0, scale: 1.0)

        let input = MLXArray.ones([1, 4, 8, 64]) * 0.5  // Non-trivial values

        let outputTraditional = ropeTraditional(input, offset: 10)
        let outputModern = ropeModern(input, offset: 10)
        eval(outputTraditional, outputModern)

        // Traditional and modern modes should produce different results
        let diff = abs(outputTraditional - outputModern)
        let maxDiff = MLX.max(diff).item(Float.self)

        XCTAssertGreaterThan(maxDiff, 0.001, "Traditional and modern RoPE should differ")
    }
}

