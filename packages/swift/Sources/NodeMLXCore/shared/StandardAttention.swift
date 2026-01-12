// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Standard Multi-Head Attention implementation shared by most transformer models.

import Foundation
import MLX
import MLXFast
import MLXNN

/// Standard Multi-Head Attention with Grouped Query Attention (GQA) support.
///
/// This implementation is shared by Llama, Qwen, Mistral, Phi, and other
/// transformer models that use the standard attention pattern.
///
/// Features:
/// - Grouped Query Attention (GQA) via numKVHeads < numHeads
/// - Rotary Position Embedding (RoPE)
/// - KV-Cache support for efficient generation
/// - Uses MLXFast for optimized attention computation
public class StandardAttention<Config: BaseModelConfiguration>: Module {
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rope: RoPE

    public init(_ config: Config) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1.0 / Foundation.sqrt(Float(headDim))

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let attnBias = config.attentionBias

        _qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: attnBias)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)
        _oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: attnBias)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])
        var keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
        var values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])

        // Transpose for attention: [B, heads, L, headDim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE with cache offset
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update cache
        if let c = cache {
            (keys, values) = c.update(keys: keys, values: values)
        }

        // Attention using MLXFast (handles GQA automatically)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [B, heads, L, headDim] -> [B, L, hidden]
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        return oProj(outputReshaped)
    }
}
