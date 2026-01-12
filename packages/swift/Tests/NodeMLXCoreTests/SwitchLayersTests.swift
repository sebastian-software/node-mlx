// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// Tests for SwitchLayers (MoE infrastructure)
// SPDX-License-Identifier: MIT

import MLX
import MLXNN
@testable import NodeMLXCore
import XCTest

final class SwitchLayersTests: XCTestCase {
    // MARK: - SwitchLinear Tests

    func testSwitchLinearBasic() {
        let numExperts = 8
        let inputDims = 64
        let outputDims = 128

        let layer = SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            bias: true
        )

        XCTAssertEqual(layer.inputDims, inputDims)
        XCTAssertEqual(layer.outputDims, outputDims)
        XCTAssertEqual(layer.numExperts, numExperts)
        XCTAssertNotNil(layer.bias)
    }

    func testSwitchLinearNoBias() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 8,
            bias: false
        )

        XCTAssertNil(layer.bias)
    }

    func testSwitchLinearForward() {
        let numExperts = 4
        let batchSeq = 8
        let inputDims = 32
        let outputDims = 64

        let layer = SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            bias: true
        )

        // Input: [batch*seq, 1, 1, inputDims] after expansion
        let x = MLXArray.ones([batchSeq, 1, 1, inputDims])
        // Expert indices for each token
        let indices = MLXArray([Int32(0), 1, 2, 3, 0, 1, 2, 3])

        let output = layer(x, indices, sortedIndices: false)
        eval(output)

        // Output should have shape [batchSeq, 1, topK, outputDims]
        XCTAssertEqual(output.dim(0), batchSeq)
        XCTAssertEqual(output.dim(-1), outputDims)
    }

    func testSwitchLinearSortedIndices() {
        let layer = SwitchLinear(
            inputDims: 32,
            outputDims: 64,
            numExperts: 4,
            bias: true
        )

        let x = MLXArray.ones([4, 1, 1, 32])
        // Pre-sorted indices
        let indices = MLXArray([Int32(0), 1, 2, 3])

        let output = layer(x, indices, sortedIndices: true)
        eval(output)

        XCTAssertEqual(output.dim(-1), 64)
    }

    // MARK: - QuantizedSwitchLinear Tests

    func testQuantizedSwitchLinearCreation() {
        let layer = SwitchLinear(
            inputDims: 64,
            outputDims: 128,
            numExperts: 8,
            bias: true
        )

        let quantized = layer.toQuantized(groupSize: 64, bits: 4, mode: .affine)

        XCTAssertTrue(quantized is QuantizedSwitchLinear)
    }

    func testQuantizedSwitchLinearForward() {
        let numExperts = 4
        let inputDims = 64
        let outputDims = 128

        let layer = SwitchLinear(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            bias: true
        )

        guard let quantized = layer.toQuantized(groupSize: 64, bits: 4, mode: .affine) as? QuantizedSwitchLinear else {
            XCTFail("Failed to create QuantizedSwitchLinear")
            return
        }

        let x = MLXArray.ones([4, 1, 1, inputDims])
        let indices = MLXArray([Int32(0), 1, 2, 3])

        let output = quantized(x, indices, sortedIndices: false)
        eval(output)

        XCTAssertEqual(output.dim(-1), outputDims)
    }

    // MARK: - SwitchGLU Tests

    func testSwitchGLUBasic() {
        let inputDims = 64
        let hiddenDims = 256
        let numExperts = 8

        let glu = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            activation: MLXNN.silu,
            bias: false
        )

        XCTAssertEqual(glu.inputDims, inputDims)
        XCTAssertEqual(glu.hiddenDims, hiddenDims)
        XCTAssertEqual(glu.numExperts, numExperts)
    }

    func testSwitchGLUForward() {
        let inputDims = 32
        let hiddenDims = 64
        let numExperts = 4
        let batchSeq = 8
        let topK = 2

        let glu = SwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            activation: MLXNN.silu,
            bias: false
        )

        // Input shape: [batchSeq, inputDims]
        let x = MLXArray.ones([batchSeq, inputDims])
        // Expert indices for each token: [batchSeq, topK]
        let indicesFlat: [Int32] = [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3, 2, 0, 3, 1]
        let indices = MLXArray(indicesFlat).reshaped([batchSeq, topK])

        let output = glu(x, indices)
        eval(output)

        // Output should have same shape as input but with topK experts
        XCTAssertEqual(output.dim(0), batchSeq)
        XCTAssertEqual(output.dim(1), topK)
        XCTAssertEqual(output.dim(2), inputDims)
    }

    // MARK: - SwitchMLP Tests

    func testSwitchMLPBasic() {
        let inputDims = 64
        let hiddenDims = 256
        let numExperts = 8

        let mlp = SwitchMLP(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            activation: gelu,
            bias: false
        )

        XCTAssertEqual(mlp.inputDims, inputDims)
        XCTAssertEqual(mlp.hiddenDims, hiddenDims)
        XCTAssertEqual(mlp.numExperts, numExperts)
    }

    func testSwitchMLPForward() {
        let inputDims = 32
        let hiddenDims = 64
        let numExperts = 4
        let batchSeq = 8
        let topK = 2

        let mlp = SwitchMLP(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            activation: gelu,
            bias: false
        )

        let x = MLXArray.ones([batchSeq, inputDims])
        let indicesFlat: [Int32] = [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3, 2, 0, 3, 1]
        let indices = MLXArray(indicesFlat).reshaped([batchSeq, topK])

        let output = mlp(x, indices)
        eval(output)

        XCTAssertEqual(output.dim(0), batchSeq)
        XCTAssertEqual(output.dim(1), topK)
        XCTAssertEqual(output.dim(2), inputDims)
    }

    // MARK: - GPT-OSS SwiGLU Tests

    func testGptOssSwiGLUBasic() {
        let xLinear = MLXArray([Float(1.0), 2.0, 3.0, 4.0])
        let xGlu = MLXArray([Float(0.5), 1.0, 1.5, 2.0])

        let output = gptOssSwiGLU(xLinear, xGlu)
        eval(output)

        XCTAssertEqual(output.shape, xLinear.shape)
    }

    func testGptOssSwiGLUClipping() {
        // Test that values are clipped
        let xLinear = MLXArray([Float(10.0), -10.0]) // Exceeds limit=7.0
        let xGlu = MLXArray([Float(10.0), 10.0]) // Exceeds limit=7.0

        let output = gptOssSwiGLU(xLinear, xGlu, alpha: 1.702, limit: 7.0)
        eval(output)

        // Output should be bounded due to clipping
        let maxVal = MLX.max(abs(output)).item(Float.self)
        XCTAssertLessThan(maxVal, 100.0, "Output should be bounded due to clipping")
    }

    func testCompiledGptOssSwiGLU() {
        let compiledFn = compiledGptOssSwiGLU()

        let xLinear = MLXArray([Float(1.0), 2.0, 3.0])
        let xGlu = MLXArray([Float(0.5), 1.0, 1.5])

        let output = compiledFn(xLinear, xGlu)
        eval(output)

        XCTAssertEqual(output.shape, xLinear.shape)
    }

    // MARK: - SwiGLUSwitchGLU (GPT-OSS specific) Tests

    func testSwiGLUSwitchGLUBasic() {
        let inputDims = 64
        let hiddenDims = 256
        let numExperts = 8

        let glu = SwiGLUSwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            bias: false
        )

        XCTAssertEqual(glu.inputDims, inputDims)
        XCTAssertEqual(glu.hiddenDims, hiddenDims)
        XCTAssertEqual(glu.numExperts, numExperts)
    }

    func testSwiGLUSwitchGLUForward() {
        let inputDims = 32
        let hiddenDims = 64
        let numExperts = 4
        let batchSeq = 8
        let topK = 2

        let glu = SwiGLUSwitchGLU(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            bias: false
        )

        let x = MLXArray.ones([batchSeq, inputDims])
        let indicesFlat: [Int32] = [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3, 2, 0, 3, 1]
        let indices = MLXArray(indicesFlat).reshaped([batchSeq, topK])

        let output = glu(x, indices)
        eval(output)

        XCTAssertEqual(output.dim(0), batchSeq)
        XCTAssertEqual(output.dim(1), topK)
        XCTAssertEqual(output.dim(2), inputDims)
    }

    // MARK: - Helper Function Tests

    func testGatherSortBasic() {
        // Create x: [4, 2, 1]
        let xData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let x = MLXArray(xData).reshaped([4, 2, 1])

        // Indices: [4, 2]
        let indicesData: [Int32] = [3, 1, 0, 2, 2, 0, 1, 3]
        let indices = MLXArray(indicesData).reshaped([4, 2])

        let (sortedX, sortedIndices, invOrder) = gatherSort(x: x, indices: indices)
        eval(sortedX, sortedIndices, invOrder)

        // Sorted indices should be in order
        XCTAssertGreaterThan(sortedX.size, 0)
        XCTAssertGreaterThan(sortedIndices.size, 0)
        XCTAssertGreaterThan(invOrder.size, 0)
    }

    func testScatterUnsortBasic() {
        let x = MLXArray([Float(1.0), 2.0, 3.0, 4.0]).reshaped([4, 1])
        let invOrder = MLXArray([Int32(2), 0, 3, 1])

        let result = scatterUnsort(x: x, invOrder: invOrder, shape: nil)
        eval(result)

        XCTAssertEqual(result.shape, x.shape)
    }

    func testScatterUnsortWithShape() {
        let x = MLXArray([Float(1.0), 2.0, 3.0, 4.0]).reshaped([4, 1])
        let invOrder = MLXArray([Int32(2), 0, 3, 1])

        let result = scatterUnsort(x: x, invOrder: invOrder, shape: [2, 2])
        eval(result)

        XCTAssertEqual(result.dim(0), 2)
        XCTAssertEqual(result.dim(1), 2)
    }
}
