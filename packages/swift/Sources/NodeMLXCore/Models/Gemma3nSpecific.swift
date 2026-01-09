//
//  Gemma3nSpecific.swift
//  NodeMLXCore
//
//  Gemma 3n specific architecture components.
//  RMSNorm, Embeddings, Attention, and MLP layers.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RMS Norm without scale (for v_norm)

/// RMSNorm without learnable scale weights - used for value normalization
class RMSNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-5) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // RMSNorm without weight parameter
        let variance = mean(x.pow(2), axis: -1, keepDims: true)
        return x * rsqrt(variance + eps)
    }
}

// MARK: - Gemma3n RMSNorm (Gemma-style with 1+weight)

/// RMSNorm with Gemma-style (1 + weight) scaling
class Gemma3nRMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        // Initialize to zeros - forward uses (1 + weight)
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

// MARK: - Standard RMSNorm for Gemma 3n

class Gemma3nStandardRMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Gemma3n Attention

class Gemma3nAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma3nStandardRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma3nStandardRMSNorm
    @ModuleInfo(key: "v_norm") var vNorm: RMSNoScale

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let isKVSharedLayer: Bool
    let rope: RoPE

    init(_ config: Gemma3nConfiguration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.isSliding = !config.isGlobalLayer(layerIdx)
        self.isKVSharedLayer = config.isKVSharedLayer(layerIdx)

        // Gemma3n uses scale=1.0
        self.scale = 1.0

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        _qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        _oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)

        _qNorm.wrappedValue = Gemma3nStandardRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Gemma3nStandardRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _vNorm.wrappedValue = RMSNoScale(eps: config.rmsNormEps)

        let ropeBase = isSliding ? config.ropeLocalBaseFreq : config.ropeTheta
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)
    }

    /// Forward pass with optional shared KV cache
    /// - Parameters:
    ///   - hiddenStates: Input tensor
    ///   - mask: Attention mask
    ///   - cache: KV cache (for non-shared layers, or the shared cache for shared layers)
    ///   - sharedCache: If this is a KV-shared layer, this contains the pre-computed KV
    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?,
        sharedKV: (keys: MLXArray, values: MLXArray, offset: Int)? = nil
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        var keys: MLXArray
        var values: MLXArray
        var offset: Int

        if isKVSharedLayer, let shared = sharedKV {
            // For KV-shared layers, use the pre-computed KV from the designated cache
            keys = shared.keys
            values = shared.values
            offset = shared.offset
        } else {
            // Compute KV for this layer
            offset = cache?.offset ?? 0

            keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
            keys = kNorm(keys)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: offset)

            values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)

            if let c = cache {
                (keys, values) = c.update(keys: keys, values: values)
            }
        }

        queries = rope(queries, offset: offset)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return oProj(outputReshaped)
    }
}

// MARK: - Gemma3n MLP

class Gemma3nMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let activationSparsity: Float
    let stdMultiplier: Float?

    init(_ config: Gemma3nConfiguration, layerIdx: Int = 0) {
        let intermediateSize = config.intermediateSize(forLayer: layerIdx)
        _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)

        // Get activation sparsity for this layer
        if layerIdx < config.activationSparsityPattern.count {
            self.activationSparsity = config.activationSparsityPattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }

        // Precompute std multiplier for gelu_topk if sparsity > 0
        if activationSparsity > 0 {
            // sqrt(2) * erfinv(2 * sparsity - 1)
            // For sparsity=0.95: erfinv(0.9) ≈ 1.163, so stdMultiplier ≈ 1.645
            self.stdMultiplier = Float(sqrt(2.0)) * Self.erfinv(2.0 * activationSparsity - 1.0)
        } else {
            self.stdMultiplier = nil
        }
    }

    /// Approximate inverse error function
    private static func erfinv(_ x: Float) -> Float {
        // Approximation for erfinv
        let a: Float = 0.147
        let sign: Float = x < 0 ? -1 : 1
        let x2 = x * x
        let lnTerm = log(1 - x2)
        let term1 = 2 / (Float.pi * a) + lnTerm / 2
        let term2 = lnTerm / a
        return sign * sqrt(sqrt(term1 * term1 - term2) - term1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
