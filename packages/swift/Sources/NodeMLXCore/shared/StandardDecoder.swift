// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Standard Decoder Layer implementation shared by most transformer models.

import MLX
import MLXFast
import MLXNN

/// Standard Pre-Norm Decoder Layer used by most modern LLMs.
///
/// Architecture:
/// 1. LayerNorm → Self-Attention → Residual
/// 2. LayerNorm → MLP → Residual
///
/// This "pre-norm" architecture (normalize before the operation) is
/// standard in Llama, Qwen, Mistral, etc.
public class StandardDecoderLayer<Config: BaseModelConfiguration>: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: StandardAttention<Config>
    @ModuleInfo(key: "mlp") public var mlp: StandardMLP<Config>
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: RMSNorm

    public init(_ config: Config, layerIdx _: Int = 0) {
        _selfAttn.wrappedValue = StandardAttention(config)
        _mlp.wrappedValue = StandardMLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {
        // 1. Pre-norm + Self-attention
        let normed = inputLayernorm(hiddenStates)
        let attnOut = selfAttn(normed, mask: mask, cache: &cache)
        var h = hiddenStates + attnOut

        // 2. Pre-norm + MLP
        let mlpNormed = postAttentionLayernorm(h)
        let mlpOut = mlp(mlpNormed)
        h = h + mlpOut
        return h
    }
}
