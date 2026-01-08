//
//  Llama.swift
//  NodeMLXCore
//
//  Llama model implementation with proper RoPE and KV cache support.
//
//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).
//  See: https://github.com/ml-explore/mlx-swift-lm
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Wrapper that accepts either Int or [Int] for eos_token_id
public struct FlexibleIntArray: Codable, Sendable {
    public let values: [Int]

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let single = try? container.decode(Int.self) {
            values = [single]
        } else if let array = try? container.decode([Int].self) {
            values = array
        } else {
            values = []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }
}

public struct LlamaConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 4096
    public var numHiddenLayers: Int = 32
    public var intermediateSize: Int = 11008
    public var numAttentionHeads: Int = 32
    public var numKeyValueHeads: Int? = nil
    public var vocabSize: Int = 32000
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10000
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var headDim: Int? = nil
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var tieWordEmbeddings: Bool = false
    public var modelType: String = "llama"

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case headDim = "head_dim"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "llama"
    }

    // Computed property for effective KV heads
    public var effectiveNumKVHeads: Int {
        numKeyValueHeads ?? numAttentionHeads
    }

    // Computed property for head dimension
    public var effectiveHeadDim: Int {
        headDim ?? (hiddenSize / numAttentionHeads)
    }
}

// MARK: - LlamaAttention

class LlamaAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ config: LlamaConfiguration) {
        let hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.effectiveNumKVHeads
        self.headDim = config.effectiveHeadDim
        self.scale = 1.0 / sqrt(Float(headDim))

        // Initialize RoPE with scaling support
        let ropeScale: Float
        if let ropeScaling = config.ropeScaling,
           ropeScaling["type"] == .string("linear"),
           let factor = ropeScaling["factor"]?.asFloat() {
            ropeScale = 1 / factor
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: ropeScale
        )

        let hasBias = config.attentionBias
        self._qProj.wrappedValue = Linear(hiddenSize, numHeads * headDim, bias: hasBias)
        self._kProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: hasBias)
        self._vProj.wrappedValue = Linear(hiddenSize, numKVHeads * headDim, bias: hasBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: hasBias)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape: [B, L, H*D] -> [B, L, H, D]
        queries = queries.reshaped([B, L, numHeads, headDim])
        keys = keys.reshaped([B, L, numKVHeads, headDim])
        values = values.reshaped([B, L, numKVHeads, headDim])

        // Apply RoPE with cache offset
        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Update KV cache
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Transpose for attention: [B, L, H, D] -> [B, H, L, D]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Use MLXFast.scaledDotProductAttention for GQA support
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back: [B, H, L, D] -> [B, L, H*D]
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])

        return oProj(outputReshaped)
    }
}

// MARK: - LlamaMLP

class LlamaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: LlamaConfiguration) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize
        let hasBias = config.mlpBias
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: hasBias)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: hasBias)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: hasBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - LlamaDecoderLayer

class LlamaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaAttention
    let mlp: LlamaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: LlamaConfiguration) {
        self._attention.wrappedValue = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Self-attention with residual
        let residual1 = x
        let h1 = inputLayerNorm(x)
        let attnOut = attention(h1, mask: mask, cache: cache)
        let h2 = residual1 + attnOut

        // MLP with residual
        let residual2 = h2
        let h3 = postAttentionLayerNorm(h2)
        let mlpOut = mlp(h3)
        return residual2 + mlpOut
    }
}

// MARK: - LlamaModelInner

public class LlamaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [LlamaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ config: LlamaConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { _ in
            LlamaDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)

        // Create causal mask
        let mask: MLXArray?
        if h.dim(1) > 1 {
            let offset = cache?[0].offset ?? 0
            mask = createCausalMask(n: h.dim(1), offset: offset)
        } else {
            mask = nil
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - LlamaModel

public class LlamaModel: Module, LLMModel {
    public let vocabularySize: Int
    public let numLayers: Int
    public var supportsCache: Bool { true }

    public let model: LlamaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    private let config: LlamaConfiguration

    public init(_ config: LlamaConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.numLayers = config.numHiddenLayers
        self.model = LlamaModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Create new KV caches for generation
    public func newCache() -> [KVCache] {
        return createLayerCaches(numLayers: numLayers)
    }

    /// Forward pass without cache
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let h = model(inputIds, cache: nil)
        return lmHead(h)
    }

    /// Forward pass with KV cache support
    public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        let h = model(inputIds, cache: cache)
        return lmHead(h)
    }

    /// Sanitize weight keys during model loading
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            // Remove unused precomputed rotary freqs
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }

        // Handle tied embeddings
        if config.tieWordEmbeddings {
            if sanitized["lm_head.weight"] == nil,
               let embedWeight = sanitized["model.embed_tokens.weight"] {
                sanitized["lm_head.weight"] = embedWeight
            }
        }

        return sanitized
    }
}
