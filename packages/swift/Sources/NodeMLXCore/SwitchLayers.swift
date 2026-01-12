// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/switch_layers.py
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Gather and sort tokens by expert index for efficient batched computation.
///
/// When processing many tokens with different expert assignments, sorting
/// them by expert index allows for more efficient memory access patterns.
///
/// - Parameters:
///   - x: Input tensor to sort
///   - indices: Expert indices for each token
/// - Returns: Tuple of (sorted_x, sorted_indices, inverse_order)
public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let flatIndices = indices.flattened()
    let order = argSort(flatIndices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        flatIndices[order],
        inverseOrder
    )
}

/// Unsort tokens back to original order after expert processing.
///
/// - Parameters:
///   - x: Sorted output from experts
///   - invOrder: Inverse order from gatherSort
///   - shape: Optional shape to unflatten to
/// - Returns: Tensor with original token order
public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var result = x[invOrder]
    if let shape {
        result = unflatten(result, axis: 0, shape: shape)
    }
    return result
}

// MARK: - SwitchLinear

/// Linear layer with expert-specific weights for Mixture of Experts.
///
/// Each expert has its own weight matrix. The layer performs batched
/// matrix multiplication selecting the appropriate expert for each input.
///
/// Shape:
/// - weight: [num_experts, output_dims, input_dims]
/// - bias: [num_experts, output_dims] (optional)
public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

    // MARK: - Initialization

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            _bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initialize with pre-computed weights (for quantization).
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        _weight.wrappedValue = weight
        _bias.wrappedValue = bias

        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = weight.swappedAxes(-1, -2)
        var result = MLX.gatherMatmul(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    // MARK: - Quantization

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

// MARK: - QuantizedSwitchLinear

/// Quantized version of SwitchLinear for memory-efficient MoE.
///
/// Stores weights in quantized format (4 or 8 bits) with per-group
/// scales and biases for dequantization.
public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    // MARK: - Initialization

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode
        )

        _scales.wrappedValue = scales
        _biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias
        )

        freeze()
    }

    // MARK: - Forward

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMatmul(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}

// MARK: - SwitchGLU

/// Gated Linear Unit with expert routing for MoE models.
///
/// Combines three SwitchLinear projections with an activation function:
/// - gate_proj: For gating signal
/// - up_proj: For value signal
/// - down_proj: Output projection
///
/// Output = down_proj(activation(gate_proj(x)) * up_proj(x))
public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let activation: (MLXArray) -> MLXArray

    // MARK: - Initialization

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias
        )
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias
        )
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias
        )

        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // Sort tokens by expert for efficient batched computation
        let doSort = indices.size > 64
        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort
        )

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchMLP

/// Simple MLP with expert routing (without gating).
///
/// Uses two SwitchLinear projections with an activation:
/// Output = fc2(activation(fc1(x)))
public class SwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int
    public let activation: (MLXArray) -> MLXArray

    // MARK: - Initialization

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = gelu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        _fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias
        )
        _fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias
        )

        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64
        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        x = fc1(x, idx, sortedIndices: doSort)
        x = activation(x)
        x = fc2(x, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - GPT-OSS Custom SwiGLU

/// GPT-OSS custom SwiGLU activation with clipping.
///
/// This variant includes value clipping for numerical stability:
/// ```
/// x_glu = clip(x_glu, max=limit)
/// x_linear = clip(x_linear, min=-limit, max=limit)
/// glu_scaled = alpha * x_glu
/// sig = sigmoid(glu_scaled)
/// out_glu = x_glu * sig
/// return out_glu * (x_linear + 1)
/// ```
public func gptOssSwiGLU(
    _ xLinear: MLXArray,
    _ xGlu: MLXArray,
    alpha: Float = 1.702,
    limit: Float = 7.0
) -> MLXArray {
    let clippedGlu = clip(xGlu, max: MLXArray(limit))
    let clippedLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    let gluScaled = alpha * clippedGlu
    let sig = sigmoid(gluScaled)
    let outGlu = clippedGlu * sig

    return outGlu * (clippedLinear + 1)
}

/// Compiled version for better performance.
public func compiledGptOssSwiGLU() -> @Sendable (MLXArray, MLXArray) -> MLXArray {
    compile(shapeless: true) { xLinear, xGlu in
        gptOssSwiGLU(xLinear, xGlu)
    }
}

// MARK: - SwiGLUSwitchGLU (GPT-OSS)

/// SwitchGLU with GPT-OSS custom SwiGLU activation.
///
/// Used in GPT-OSS MoE models which require the clipped SwiGLU variant.
public class SwiGLUSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    public let inputDims: Int
    public let hiddenDims: Int
    public let numExperts: Int

    // MARK: - Initialization

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias
        )
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias
        )
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias
        )

        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64
        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            compiledGptOssSwiGLU()(xUp, xGate),
            idx,
            sortedIndices: doSort
        )

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return x.squeezed(axis: -2)
    }
}
