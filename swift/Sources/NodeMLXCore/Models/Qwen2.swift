//
//  Qwen2.swift
//  NodeMLXCore
//
//  Qwen2 model implementation using helper classes from mlx-swift-lm.
//  Auto-generated structure with robust attention and cache handling.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Qwen2 configuration - matches mlx-swift-lm's Qwen2Configuration
public struct Qwen2Configuration: Codable, Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var kvHeads: Int
    public var ropeTheta: Float = 1_000_000
    public var ropeTraditional: Bool = false
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    // Computed properties
    public var headDim: Int {
        hiddenSize / attentionHeads
    }
}

// MARK: - Qwen2Attention

class Qwen2Attention: Module {
    let args: Qwen2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: Qwen2Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        // Qwen2 always uses bias for Q/K/V projections
        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        // Initialize RoPE
        let ropeScale: Float
        if let ropeScaling = args.ropeScaling,
           ropeScaling["type"] == .string("linear"),
           let factor = ropeScaling["factor"]?.asFloat() {
            ropeScale = 1 / factor
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE with cache offset
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        // Use the robust attentionWithCacheUpdate from mlx-swift-lm
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - Qwen2MLP

class Qwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Qwen2TransformerBlock

class Qwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2Attention
    let mlp: Qwen2MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Qwen2Configuration) {
        _attention.wrappedValue = Qwen2Attention(args)
        self.mlp = Qwen2MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

// MARK: - Qwen2ModelInner

public class Qwen2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen2TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen2Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0..<args.hiddenLayers).map { _ in
            Qwen2TransformerBlock(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first, windowSize: nil, returnArray: false)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Qwen2Model

public class Qwen2Model: Module, LLMModel {
    public let vocabularySize: Int
    public let numLayers: Int
    public let kvHeads: [Int]

    public let model: Qwen2ModelInner
    let configuration: Qwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.numLayers = args.hiddenLayers
        self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    /// This model supports KV caching
    public var supportsCache: Bool { true }

    /// Create new KV cache for this model
    public func newCache() -> [KVCache] {
        return createLayerCaches(numLayers: numLayers)
    }

    /// Forward pass: compute logits from input token IDs with KV cache
    public func callAsFunction(_ inputs: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Forward pass without cache (for compatibility)
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var noCache: [KVCache]? = nil
        return callAsFunction(inputIds, cache: &noCache)
    }

    /// Sanitize weight keys during model loading
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // Remove unused precomputed rotary freqs
        return weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}
