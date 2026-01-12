// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for ported/SwitchLayers.swift

import MLX
import MLXNN
import XCTest

@testable import NodeMLXCore

final class SwitchLayersTests: XCTestCase {
    // MARK: - Helper Function Tests

    // Note: gatherSort and scatterUnsort are internal helper functions
    // that are tested implicitly through the SwitchGLU tests.
    // Direct testing requires very specific input formats.

    // MARK: - SwitchLinear Tests

    func testSwitchLinearInitialization() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 4,
            bias: false
        )

        XCTAssertNotNil(layer)
    }

    func testSwitchLinearForward() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 4,
            bias: false
        )

        // Input: [batch*seq, topK, 1, inputDims]
        let x = MLXArray.ones([8, 2, 1, 64])
        // Expert indices: [batch*seq, topK]
        let indicesData: [Int32] = [0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 2, 1, 3]
        let indices = MLXArray(indicesData, [8, 2])

        let output = layer(x, indices: indices)

        // Output should be [batch*seq, topK, 1, outputDims]
        XCTAssertEqual(output.shape, [8, 2, 1, 128])
    }

    func testSwitchLinearWithBias() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 4,
            bias: true
        )

        let x = MLXArray.ones([4, 2, 1, 64])
        let indicesData: [Int32] = [0, 1, 2, 3, 0, 1, 2, 3]
        let indices = MLXArray(indicesData, [4, 2])

        let output = layer(x, indices: indices)

        XCTAssertEqual(output.shape, [4, 2, 1, 128])
    }

    // MARK: - SwitchGLU Tests

    func testSwitchGLUInitialization() {
        let glu = SwitchGLU(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4,
            bias: false
        )

        XCTAssertNotNil(glu)
    }

    func testSwitchGLUForward() {
        let glu = SwitchGLU(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4,
            bias: false
        )

        // Input: [batch, seq, hidden]
        let x = MLXArray.ones([2, 8, 64])
        // Expert indices: [batch, seq, topK]
        let indices = MLXArray([Int32](repeating: 0, count: 16) + [Int32](repeating: 1, count: 16)).reshaped([2, 8, 2])

        let output = glu(x, indices: indices)

        // Output should be [batch, seq, topK, hidden]
        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 8)
        XCTAssertEqual(output.dim(-1), 64)
    }

    // MARK: - SwiGLU Activation Tests

    func testSwiGLU() {
        let x = MLXArray.ones([4, 64])
        let gate = MLXArray.ones([4, 64])

        let output = swiGLU(x, gate: gate)

        XCTAssertEqual(output.shape, x.shape)
    }

    // MARK: - SwiGLUSwitchGLU Tests (GPT-OSS)

    func testSwiGLUSwitchGLUInitialization() {
        let glu = SwiGLUSwitchGLU(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4,
            bias: true
        )

        XCTAssertNotNil(glu)
    }

    func testSwiGLUSwitchGLUForward() {
        let glu = SwiGLUSwitchGLU(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4,
            bias: true
        )

        let x = MLXArray.ones([2, 8, 64])
        let indices = MLXArray([Int32](repeating: 0, count: 16) + [Int32](repeating: 1, count: 16)).reshaped([2, 8, 2])

        let output = glu(x, indices: indices)

        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 8)
        XCTAssertEqual(output.dim(-1), 64)
    }

    // MARK: - SwitchMLP Tests

    func testSwitchMLPInitialization() {
        let mlp = SwitchMLP(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4
        )

        XCTAssertNotNil(mlp)
    }

    func testSwitchMLPForward() {
        let mlp = SwitchMLP(
            inputDims: 64,
            hiddenDims: 256,
            numExperts: 4,
            activation: gelu
        )

        let x = MLXArray.ones([2, 8, 64])
        let indices = MLXArray([Int32](repeating: 0, count: 16) + [Int32](repeating: 1, count: 16)).reshaped([2, 8, 2])

        let output = mlp(x, indices: indices)

        XCTAssertEqual(output.dim(0), 2)
        XCTAssertEqual(output.dim(1), 8)
        XCTAssertEqual(output.dim(-1), 64)
    }

    // MARK: - MoE Tensor Conversion Tests

    func testConvertMoePackedTensors() {
        // Create mock packed tensors
        let blocks = MLXArray.ones([4, 64, 256])
        let scales = MLXArray.ones([4, 64, 4])

        let result = convertMoePackedTensors(blocks: blocks, scales: scales)

        // Result should maintain expert dimension
        XCTAssertEqual(result.dim(0), 4)
    }

    // MARK: - Edge Cases

    func testSwitchLinearSingleExpert() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 1,
            bias: false
        )

        let x = MLXArray.ones([4, 1, 1, 64])
        let indicesData: [Int32] = [0, 0, 0, 0]
        let indices = MLXArray(indicesData, [4, 1])

        let output = layer(x, indices: indices)

        XCTAssertEqual(output.shape, [4, 1, 1, 128])
    }

    func testSwitchLinearManyExperts() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 16,
            bias: false
        )

        let x = MLXArray.ones([8, 4, 1, 64])
        // Expert indices between 0-15
        let indicesData: [Int32] = (0 ..< 32).map { Int32($0 % 16) }
        let indices = MLXArray(indicesData, [8, 4])

        let output = layer(x, indices: indices)

        XCTAssertEqual(output.shape, [8, 4, 1, 128])
    }
}
