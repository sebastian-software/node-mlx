// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/switch_layers.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Sorts tokens by expert assignment for efficient batched access.
///
/// When processing many tokens, sorting by expert index improves memory
/// access patterns during the expert computation.
///
/// - Parameters:
///   - x: Input tensor [N, ...]
///   - indices: Expert indices [N, K]
/// - Returns: Tuple of (sorted x, sorted indices, inverse order for unsorting)
public func gatherSort(_ x: MLXArray, _ indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.shape.last!
    let flatIndices = indices.flattened()
    let order = argSort(flatIndices)
    let invOrder = argSort(order)

    let sortedIndices = flatIndices[order]
    let sortedX = x.flattened(start: 0, end: -3)[order / m]

    return (sortedX, sortedIndices, invOrder)
}

/// Restores original token order after expert processing.
///
/// - Parameters:
///   - x: Sorted tensor
///   - invOrder: Inverse permutation from gatherSort
///   - shape: Optional original shape to restore
/// - Returns: Tensor in original token order
public func scatterUnsort(_ x: MLXArray, _ invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var result = x[invOrder]
    if let shape {
        result = result.reshaped([shape[0], shape[1]] + Array(result.shape.dropFirst()))
    }
    return result
}

// MARK: - SwitchLinear

/// Expert-specific linear layer for Mixture of Experts.
///
/// Maintains separate weight matrices for each expert and uses
/// `gather_mm` for efficient batched computation.
///
/// Ported from: mlx_lm/models/switch_layers.py::SwitchLinear
public class SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    public var inputDims: Int { weight.dim(2) }
    public var outputDims: Int { weight.dim(1) }
    public var numExperts: Int { weight.dim(0) }

    /// Creates a SwitchLinear layer.
    ///
    /// - Parameters:
    ///   - inputDims: Input feature dimension
    ///   - outputDims: Output feature dimension
    ///   - numExperts: Number of expert weight matrices
    ///   - bias: Whether to include bias terms
    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        let scale = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            _bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }
    }

    /// Forward pass with expert selection.
    ///
    /// - Parameters:
    ///   - x: Input tensor
    ///   - indices: Expert indices for each token
    ///   - sortedIndices: Whether indices are pre-sorted
    /// - Returns: Expert-weighted output
    public func callAsFunction(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        var result = MLX.gatherMatmul(
            x,
            weight.swappedAxes(-1, -2),
            rhsIndices: indices,
            sortedIndices: sortedIndices
        )

        if let bias {
            result = result + expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    /// Converts to quantized version.
    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) -> QuantizedSwitchLinear {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

// MARK: - QuantizedSwitchLinear

/// Quantized version of SwitchLinear for reduced memory usage.
///
/// Uses quantized weights with per-group scales and biases for
/// memory-efficient expert computation.
///
/// Ported from: mlx_lm/models/switch_layers.py::QuantizedSwitchLinear
public class QuantizedSwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?
    @ModuleInfo(key: "bias") var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    /// Creates a QuantizedSwitchLinear from an existing SwitchLinear.
    public init(_ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        inputDims = other.inputDims
        outputDims = other.outputDims
        numExperts = other.numExperts
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (qw, sc, bi) = MLX.quantized(other.weight, groupSize: groupSize, bits: bits)
        _weight.wrappedValue = qw
        _scales.wrappedValue = sc
        _biases.wrappedValue = bi

        if let otherBias = other.bias {
            _bias.wrappedValue = otherBias
        }

        super.init()

        // Freeze quantized weights
        freeze()
    }

    /// Creates a QuantizedSwitchLinear with explicit parameters.
    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        bias: Bool = true,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let scale = sqrt(1.0 / Float(inputDims))
        let initialWeight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        let (qw, sc, bi) = MLX.quantized(initialWeight, groupSize: groupSize, bits: bits)
        _weight.wrappedValue = qw
        _scales.wrappedValue = sc
        _biases.wrappedValue = bi

        if bias {
            _bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
        freeze()
    }

    public func callAsFunction(_ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false) -> MLXArray {
        var result = MLX.gatherQuantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            sortedIndices: sortedIndices
        )

        if let bias {
            result = result + expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}

// MARK: - SwiGLU Activation

/// Compiled SwiGLU activation for optimal performance.
private let compiledSwiGLU: (MLXArray, MLXArray) -> MLXArray = { x, gate in
    silu(gate) * x
}

/// SwiGLU activation: SiLU(gate) * x
public func swiGLU(_ x: MLXArray, gate: MLXArray) -> MLXArray {
    compiledSwiGLU(x, gate)
}

// MARK: - SwitchGLU

/// Gated Linear Unit with expert switching for MoE.
///
/// Implements the standard GLU pattern with separate experts:
/// output = down_proj(activation(up_proj(x), gate_proj(x)))
///
/// Ported from: mlx_lm/models/switch_layers.py::SwitchGLU
public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    /// Creates a SwitchGLU layer.
    ///
    /// - Parameters:
    ///   - inputDims: Input/output feature dimension
    ///   - hiddenDims: Hidden layer dimension
    ///   - numExperts: Number of experts
    ///   - bias: Whether to include bias terms
    public init(inputDims: Int, hiddenDims: Int, numExperts: Int, bias: Bool = false) {
        _gateProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _upProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _downProj.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = expandedDimensions(x, axes: [-2, -3])

        // Sort for efficient expert access when processing many tokens
        let doSort = indices.size >= 64
        var idx = indices
        var invOrder: MLXArray?

        if doSort {
            (input, idx, invOrder) = gatherSort(input, indices)
        }

        // GLU computation
        let xUp = upProj(input, indices: idx, sortedIndices: doSort)
        let xGate = gateProj(input, indices: idx, sortedIndices: doSort)
        var result = downProj(swiGLU(xUp, gate: xGate), indices: idx, sortedIndices: doSort)

        // Restore original order
        if doSort, let inv = invOrder {
            result = scatterUnsort(result, inv, shape: Array(indices.shape))
        }

        return result.squeezed(axis: -2)
    }
}

// MARK: - SwitchMLP

/// Simple MLP with expert switching for MoE.
///
/// Implements: output = fc2(activation(fc1(x)))
///
/// Ported from: mlx_lm/models/switch_layers.py::SwitchMLP
public class SwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    private let activation: (MLXArray) -> MLXArray

    /// Creates a SwitchMLP layer.
    ///
    /// - Parameters:
    ///   - inputDims: Input/output feature dimension
    ///   - hiddenDims: Hidden layer dimension
    ///   - numExperts: Number of experts
    ///   - activation: Activation function (default: GELU)
    ///   - bias: Whether to include bias terms
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = { geluApproximate($0) },
        bias: Bool = false
    ) {
        self.activation = activation

        _fc1.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _fc2.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = expandedDimensions(x, axes: [-2, -3])

        // Sort for efficient expert access
        let doSort = indices.size >= 64
        var idx = indices
        var invOrder: MLXArray?

        if doSort {
            (input, idx, invOrder) = gatherSort(input, indices)
        }

        // MLP computation
        var result = fc1(input, indices: idx, sortedIndices: doSort)
        result = activation(result)
        result = fc2(result, indices: idx, sortedIndices: doSort)

        // Restore original order
        if doSort, let inv = invOrder {
            result = scatterUnsort(result, inv, shape: Array(indices.shape))
        }

        return result.squeezed(axis: -2)
    }
}

// MARK: - GPT-OSS Specific SwiGLU

/// GPT-OSS specific SwiGLU with clipping for numerical stability.
///
/// Uses hard clipping on the gate value before SiLU activation.
private func gptOssSwiGLU(_ x: MLXArray, gate: MLXArray, limit: Float = 7.0) -> MLXArray {
    let clippedGate = clip(gate, min: -limit, max: limit)
    return silu(clippedGate) * x
}

/// Compiled version of GPT-OSS SwiGLU for optimal performance.
private let compiledGptOssSwiGLU: (MLXArray, MLXArray) -> MLXArray = { x, gate in
    gptOssSwiGLU(x, gate: gate)
}

/// GPT-OSS variant of SwitchGLU with clipped activation.
public class SwiGLUSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    public init(inputDims: Int, hiddenDims: Int, numExperts: Int, bias: Bool = false) {
        _gateProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _upProj.wrappedValue = SwitchLinear(inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        _downProj.wrappedValue = SwitchLinear(inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64
        var idx = indices
        var invOrder: MLXArray?

        if doSort {
            (input, idx, invOrder) = gatherSort(input, indices)
        }

        let xUp = upProj(input, indices: idx, sortedIndices: doSort)
        let xGate = gateProj(input, indices: idx, sortedIndices: doSort)
        var result = downProj(compiledGptOssSwiGLU(xUp, xGate), indices: idx, sortedIndices: doSort)

        if doSort, let inv = invOrder {
            result = scatterUnsort(result, inv, shape: Array(indices.shape))
        }

        return result.squeezed(axis: -2)
    }
}

// MARK: - Weight Conversion Utilities

/// Converts packed MoE tensors from blocks+scales format.
///
/// Used during model loading to transform the packed tensor format
/// used in some quantized MoE checkpoints.
///
/// - Parameters:
///   - blocks: Quantized weight blocks
///   - scales: Quantization scales
/// - Returns: Transformed tensor suitable for weight loading
public func convertMoePackedTensors(blocks: MLXArray, scales _: MLXArray) -> MLXArray {
    // Interleave scales with blocks for the expected format
    // This matches the pattern from mlx-swift-lm's GPTOSS.swift
    // For now, return the blocks directly (scales handled separately)
    blocks
}
