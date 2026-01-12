// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Generic Sparse MLP layer with gelu_topk activation.
// Used by Gemma3n and similar architectures.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Sparse MLP

/// MLP layer with optional sparse gelu_topk activation.
///
/// The gelu_topk activation zeros out activations below a dynamic threshold
/// computed from the input statistics, enabling more efficient sparse computation.
///
/// Usage in generated models:
/// ```swift
/// typealias Gemma3nMLP = SparseMLP<Gemma3nConfiguration>
/// ```
public class SparseMLP<C: SparseMLPConfiguration>: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public let activationSparsity: Float
    public let stdMultiplier: Float?

    public init(_ config: C, layerIdx: Int = 0) {
        let intermediateSize = config.intermediateSize(forLayer: layerIdx)
        _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)

        // Get activation sparsity for this layer
        if layerIdx < config.activationSparsityPattern.count {
            activationSparsity = config.activationSparsityPattern[layerIdx]
        } else {
            activationSparsity = 0.0
        }

        // Precompute std multiplier for gelu_topk if sparsity > 0
        if activationSparsity > 0 {
            // sqrt(2) * erfinv(2 * sparsity - 1)
            stdMultiplier = Float(sqrt(2.0)) * MathUtils.erfinv(2.0 * activationSparsity - 1.0)
        } else {
            stdMultiplier = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateOutput = gateProj(x)
        let activations: MLXArray

        if let stdMult = stdMultiplier, activationSparsity > 0 {
            // gelu_topk: sparse activation
            let inputMean = mean(gateOutput, axis: -1, keepDims: true)
            let inputStd = sqrt(mean((gateOutput - inputMean).pow(2), axis: -1, keepDims: true))
            let cutoffX = inputMean + inputStd * stdMult
            activations = geluApproximate(maximum(MLXArray(Float(0)), gateOutput - cutoffX))
        } else {
            activations = geluApproximate(gateOutput)
        }

        return downProj(activations * upProj(x))
    }
}
