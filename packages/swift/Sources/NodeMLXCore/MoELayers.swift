//
//  MoELayers.swift
//  NodeMLXCore
//
//  Mixture of Experts (MoE) layers for GPT-OSS and similar architectures.
//
//  Based on patterns from mlx-lm switch_layers.py and gpt_oss.py:
//  - https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py
//  - https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gpt_oss.py
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Custom SwiGLU Activation for GPT-OSS

/// GPT-OSS uses a modified SwiGLU with specific parameters
/// ```python
/// def swiglu(x_linear, x_glu, alpha=1.702, limit=7.0):
///     x_glu = clip(x_glu, max=limit)
///     x_linear = clip(x_linear, min=-limit, max=limit)
///     glu_scaled = alpha * x_glu
///     sig = sigmoid(glu_scaled)
///     out_glu = x_glu * sig
///     return out_glu * (x_linear + 1)
/// ```
public func gptOssSwiGLU(
    _ xLinear: MLXArray,
    _ xGlu: MLXArray,
    alpha: Float = 1.702,
    limit: Float = 7.0
) -> MLXArray {
    // Clip inputs
    let clippedGlu = clip(xGlu, max: MLXArray(limit))
    let clippedLinear = clip(xLinear, min: MLXArray(-limit), max: MLXArray(limit))

    // Scaled sigmoid gate
    let gluScaled = clippedGlu * alpha
    let sig = sigmoid(gluScaled)
    let outGlu = clippedGlu * sig

    // Apply to linear with +1 bias
    return outGlu * (clippedLinear + 1)
}

// MARK: - Expert Router

/// Routes tokens to top-k experts based on learned gating
public class MoERouter: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let hiddenSize: Int
    let numExperts: Int
    let topK: Int

    public init(hiddenSize: Int, numExperts: Int, topK: Int, bias: Bool = true) {
        self.hiddenSize = hiddenSize
        self.numExperts = numExperts
        self.topK = topK

        _weight.wrappedValue = MLXArray.zeros([numExperts, hiddenSize])
        if bias {
            _bias.wrappedValue = MLXArray.zeros([numExperts])
        } else {
            _bias.wrappedValue = nil
        }
    }

    /// Forward pass returns (weights, indices) for top-k experts per token
    public func callAsFunction(_ x: MLXArray) -> (weights: MLXArray, indices: MLXArray) {
        // x: [batch, seq, hidden] -> logits: [batch, seq, numExperts]
        var logits = matmul(x, weight.T)
        if let b = bias {
            logits = logits + b
        }

        // Get top-k experts using argPartition
        // argPartition partitions so that the k largest are at the end
        let kth = numExperts - topK
        let partitionedIndices = argPartition(logits, kth: kth, axis: -1)

        // Take the last topK indices (the largest)
        let indices = partitionedIndices[.ellipsis, kth...]

        // Gather the corresponding logit values
        let topKLogits = takeAlong(logits, indices, axis: -1)

        // Softmax over selected experts to get weights
        let weights = softmax(topKLogits, axis: -1)

        return (weights, indices)
    }
}

// MARK: - SwitchGLU Expert Layer

/// A single GLU expert with gate, up, and down projections
public class GLUExpert: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let useCustomSwiGLU: Bool

    public init(
        inputDims: Int,
        hiddenDims: Int,
        bias: Bool = false,
        useCustomSwiGLU: Bool = true
    ) {
        self.useCustomSwiGLU = useCustomSwiGLU
        _gateProj.wrappedValue = Linear(inputDims, hiddenDims, bias: bias)
        _upProj.wrappedValue = Linear(inputDims, hiddenDims, bias: bias)
        _downProj.wrappedValue = Linear(hiddenDims, inputDims, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)

        let hidden: MLXArray = if useCustomSwiGLU {
            // GPT-OSS style activation
            gptOssSwiGLU(up, gate)
        } else {
            // Standard SwiGLU
            silu(gate) * up
        }

        return downProj(hidden)
    }
}

// MARK: - SwitchGLU (Batched Expert MoE)

/// SwitchGLU implements batched expert computation for MoE layers.
///
/// This matches the Python mlx-lm SwitchGLU implementation which uses
/// batched operations for efficient expert computation.
///
/// Weight structure:
/// - experts.gate_proj.weight: [num_experts, hidden_dims, input_dims]
/// - experts.up_proj.weight: [num_experts, hidden_dims, input_dims]
/// - experts.down_proj.weight: [num_experts, input_dims, hidden_dims]
public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: MLXArray
    @ModuleInfo(key: "up_proj") var upProj: MLXArray
    @ModuleInfo(key: "down_proj") var downProj: MLXArray

    // Bias tensors (optional)
    @ModuleInfo(key: "gate_proj_bias") var gateProjBias: MLXArray?
    @ModuleInfo(key: "up_proj_bias") var upProjBias: MLXArray?
    @ModuleInfo(key: "down_proj_bias") var downProjBias: MLXArray?

    let numExperts: Int
    let inputDims: Int
    let hiddenDims: Int
    let useBias: Bool
    let useCustomSwiGLU: Bool

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        useCustomSwiGLU: Bool = true
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        useBias = bias
        self.useCustomSwiGLU = useCustomSwiGLU

        // Initialize expert weights: [num_experts, out_features, in_features]
        _gateProj.wrappedValue = MLXArray.zeros([numExperts, hiddenDims, inputDims])
        _upProj.wrappedValue = MLXArray.zeros([numExperts, hiddenDims, inputDims])
        _downProj.wrappedValue = MLXArray.zeros([numExperts, inputDims, hiddenDims])

        if bias {
            _gateProjBias.wrappedValue = MLXArray.zeros([numExperts, hiddenDims])
            _upProjBias.wrappedValue = MLXArray.zeros([numExperts, hiddenDims])
            _downProjBias.wrappedValue = MLXArray.zeros([numExperts, inputDims])
        } else {
            _gateProjBias.wrappedValue = nil
            _upProjBias.wrappedValue = nil
            _downProjBias.wrappedValue = nil
        }
    }

    /// Forward pass routes tokens to selected experts and computes weighted output
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch * seq, hidden]
    ///   - indices: Expert indices for each token [batch * seq, topK]
    /// - Returns: Expert output [batch * seq, hidden]
    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        // Get the selected expert weights for each token
        // indices: [tokens, topK] -> gather from [numExperts, hiddenDims, inputDims]
        let selectedGate = gateProj[indices] // [tokens, topK, hiddenDims, inputDims]
        let selectedUp = upProj[indices]
        let selectedDown = downProj[indices] // [tokens, topK, inputDims, hiddenDims]

        // Expand x for broadcasting: [tokens, 1, inputDims, 1]
        let xExpanded = x[.ellipsis, .newAxis, 0..., .newAxis]

        // Batched matmul: [tokens, topK, hiddenDims, inputDims] @ [tokens, 1, inputDims, 1]
        // Result: [tokens, topK, hiddenDims, 1] -> squeeze to [tokens, topK, hiddenDims]
        var gateOut = squeezed(matmul(selectedGate, xExpanded), axis: -1)
        var upOut = squeezed(matmul(selectedUp, xExpanded), axis: -1)

        // Apply bias if present
        if let gateBias = gateProjBias, let upBias = upProjBias {
            let selectedGateBias = gateBias[indices] // [tokens, topK, hiddenDims]
            let selectedUpBias = upBias[indices]
            gateOut = gateOut + selectedGateBias
            upOut = upOut + selectedUpBias
        }

        // Apply activation
        let hidden: MLXArray = if useCustomSwiGLU {
            gptOssSwiGLU(upOut, gateOut)
        } else {
            silu(gateOut) * upOut
        }

        // Down projection: [tokens, topK, inputDims, hiddenDims] @ [tokens, topK, hiddenDims, 1]
        let hiddenExpanded = hidden[.ellipsis, .newAxis]
        var output = squeezed(matmul(selectedDown, hiddenExpanded), axis: -1) // [tokens, topK, inputDims]

        if let downBias = downProjBias {
            let selectedDownBias = downBias[indices]
            output = output + selectedDownBias
        }

        return output
    }
}

// MARK: - Full MoE MLP Layer

/// Complete MoE MLP layer with router and experts
/// This is used in GPT-OSS style models where each decoder layer has an MoE MLP
public class MoEMLP: Module {
    @ModuleInfo(key: "router") var router: MoERouter
    @ModuleInfo(key: "experts") var experts: SwitchGLU

    let numExperts: Int
    let topK: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numExperts: Int,
        topK: Int,
        bias: Bool = true,
        useCustomSwiGLU: Bool = true
    ) {
        self.numExperts = numExperts
        self.topK = topK

        _router.wrappedValue = MoERouter(
            hiddenSize: hiddenSize,
            numExperts: numExperts,
            topK: topK,
            bias: bias
        )
        _experts.wrappedValue = SwitchGLU(
            inputDims: hiddenSize,
            hiddenDims: intermediateSize,
            numExperts: numExperts,
            bias: bias,
            useCustomSwiGLU: useCustomSwiGLU
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let batchSeq = shape.dropLast().reduce(1, *)
        let hidden = shape.last!

        // Flatten to [batch * seq, hidden]
        let xFlat = x.reshaped([batchSeq, hidden])

        // Get routing weights and expert indices
        let (weights, indices) = router(xFlat)

        // Get expert outputs [batch * seq, topK, hidden]
        let expertOutput = experts(xFlat, indices: indices)

        // Weighted sum of expert outputs
        // weights: [batch * seq, topK] -> [batch * seq, topK, 1]
        let weightsExpanded = weights[.ellipsis, .newAxis]
        let weightedOutput = sum(expertOutput * weightsExpanded, axis: 1) // [batch * seq, hidden]

        // Reshape back to original shape
        return weightedOutput.reshaped(shape)
    }
}
