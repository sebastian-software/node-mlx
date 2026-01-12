// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Generic Fused QKV Attention layer for models using qkv_proj.
// Used by Phi3, Phi4, and similar architectures.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Fused QKV Attention

/// Attention layer using a single fused qkv_proj projection.
///
/// This is more efficient than separate q/k/v projections as it requires
/// only one matrix multiplication instead of three.
///
/// Usage in generated models:
/// ```swift
/// typealias Phi3Attention = FusedQKVAttention<Phi3Configuration>
/// ```
public class FusedQKVAttention<C: AttentionConfiguration>: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rope: RoPE

    public init(_ config: C) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = config.attentionScale ?? (1.0 / sqrt(Float(headDim)))

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let opSize = qDim + 2 * kvDim

        _qkvProj.wrappedValue = Linear(config.hiddenSize, opSize, bias: false)
        _oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCacheProtocol?
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        let qkv = qkvProj(hiddenStates)
        let queryPos = numHeads * headDim
        let kvPos = queryPos + numKVHeads * headDim

        var queries = qkv[0..., 0..., ..<queryPos].reshaped([B, L, numHeads, headDim])
        var keys = qkv[0..., 0..., queryPos ..< kvPos].reshaped([B, L, numKVHeads, headDim])
        var values = qkv[0..., 0..., kvPos...].reshaped([B, L, numKVHeads, headDim])

        // Transpose for attention: [B, heads, L, headDim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE with cache offset
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update cache (class-based protocol, reference is modified in place)
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
