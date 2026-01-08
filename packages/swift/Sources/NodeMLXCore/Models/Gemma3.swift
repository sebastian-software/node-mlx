//
//  Gemma3.swift
//  NodeMLXCore
//
//  Gemma 3 Text Model implementation (model_type: gemma3_text).
//  This is a standard transformer architecture similar to Llama.
//  NOT to be confused with Gemma 3n which has AltUp/Laurel complexity.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Gemma3Configuration: Decodable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeLocalTheta: Float  // Lower theta for sliding window layers
    public var maxPositionEmbeddings: Int
    public var slidingWindow: Int     // Window size for sliding attention layers
    public var slidingWindowPattern: Int  // Every Nth layer is global (full) attention

    public var modelType: String?

    /// Check if a layer is a global attention layer
    public func isGlobalLayer(_ layerIdx: Int) -> Bool {
        // Every Nth layer is global, where N = slidingWindowPattern
        // Pattern starts from layer 0, so layer 5 (6th) is global when pattern=6
        return slidingWindowPattern > 0 && (layerIdx + 1) % slidingWindowPattern == 0
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeLocalTheta = "rope_local_base_freq"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case modelType = "model_type"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Helper to get value from text_config or top level
        func getValue<T: Decodable>(_ key: CodingKeys, type: T.Type, default defaultValue: T? = nil) throws -> T {
            if let textContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
                if let value = try? textContainer.decode(T.self, forKey: key) {
                    return value
                }
            }
            if let value = try? container.decode(T.self, forKey: key) {
                return value
            }
            if let defaultValue = defaultValue {
                return defaultValue
            }
            throw DecodingError.keyNotFound(key, DecodingError.Context(
                codingPath: container.codingPath,
                debugDescription: "Key '\(key.stringValue)' not found in config"
            ))
        }

        func getOptionalValue<T: Decodable>(_ key: CodingKeys, type: T.Type) -> T? {
            if let textContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
                if let value = try? textContainer.decode(T.self, forKey: key) {
                    return value
                }
            }
            return try? container.decode(T.self, forKey: key)
        }

        hiddenSize = try getValue(.hiddenSize, type: Int.self)
        numHiddenLayers = try getValue(.numHiddenLayers, type: Int.self)
        intermediateSize = try getValue(.intermediateSize, type: Int.self)

        // Gemma 3 models have consistent head_dim of 256
        headDim = getOptionalValue(.headDim, type: Int.self) ?? 256

        // num_attention_heads might not be in VLM configs - calculate from hidden_size/head_dim
        // Gemma 3 sizes: 1B=4 heads, 4B=10 heads, 12B=16 heads, 27B=32 heads
        if let heads = getOptionalValue(.numAttentionHeads, type: Int.self) {
            numAttentionHeads = heads
        } else {
            numAttentionHeads = hiddenSize / headDim
        }

        // num_key_value_heads defaults to 1 for Gemma 3 (extreme GQA)
        numKeyValueHeads = getOptionalValue(.numKeyValueHeads, type: Int.self) ?? 1

        // vocab_size for Gemma 3 is 262144
        vocabSize = getOptionalValue(.vocabSize, type: Int.self) ?? 262144

        rmsNormEps = getOptionalValue(.rmsNormEps, type: Float.self) ?? 1e-6
        ropeTheta = getOptionalValue(.ropeTheta, type: Float.self) ?? 1000000.0
        ropeLocalTheta = getOptionalValue(.ropeLocalTheta, type: Float.self) ?? 10000.0
        maxPositionEmbeddings = getOptionalValue(.maxPositionEmbeddings, type: Int.self) ?? 32768
        slidingWindow = getOptionalValue(.slidingWindow, type: Int.self) ?? 512
        slidingWindowPattern = getOptionalValue(.slidingWindowPattern, type: Int.self) ?? 6
        modelType = getOptionalValue(.modelType, type: String.self)
    }
}

// MARK: - RMS Norm (Gemma style - uses MLXFast.rmsNorm with 1+weight)

class Gemma3RMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        // Initialize to zeros - will be (1 + weight) in forward
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma uses (1 + weight) scaling
        return MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}


// MARK: - Attention

class Gemma3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma3RMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE
    let isGlobal: Bool
    let slidingWindow: Int?

    init(_ config: Gemma3Configuration, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.isGlobal = config.isGlobalLayer(layerIdx)
        self.slidingWindow = isGlobal ? nil : config.slidingWindow

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)

        // Gemma 3 applies RMSNorm to Q and K after projection
        self._qNorm.wrappedValue = Gemma3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma3RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Different RoPE theta for sliding vs global layers
        let ropeBase = isGlobal ? config.ropeTheta : config.ropeLocalTheta
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Project to Q, K, V and reshape
        var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])
        var keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
        var values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])

        // Apply RMSNorm to Q and K (Gemma-specific)
        queries = qNorm(queries)
        keys = kNorm(keys)

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

// MARK: - MLP

class Gemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma3Configuration) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma uses GELU instead of SiLU
        return downProj(gelu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

class Gemma3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma3RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma3RMSNorm
    // Gemma 3 has additional pre/post feedforward norms
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma3RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma3RMSNorm

    let isGlobal: Bool
    let slidingWindow: Int?

    init(_ config: Gemma3Configuration, layerIdx: Int) {
        self.isGlobal = config.isGlobalLayer(layerIdx)
        self.slidingWindow = isGlobal ? nil : config.slidingWindow

        self._selfAttn.wrappedValue = Gemma3Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma3MLP(config)
        self._inputLayernorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {
        // 1. Pre-norm + Self-attention
        let normed = inputLayernorm(hiddenStates)
        let attnOut = selfAttn(normed, mask: mask, cache: &cache)
        let attnNormed = postAttentionLayernorm(attnOut)
        var h = hiddenStates + attnNormed

        // 2. Pre-norm + MLP
        let mlpIn = preFeedforwardLayernorm(h)
        let mlpOut = mlp(mlpIn)
        let mlpNormed = postFeedforwardLayernorm(mlpOut)
        h = h + mlpNormed

        return h
    }
}

// MARK: - Inner Model

class Gemma3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma3RMSNorm

    let numLayers: Int
    let hiddenSize: Int
    let slidingWindow: Int
    let slidingWindowPattern: Int

    init(_ config: Gemma3Configuration) {
        self.numLayers = config.numHiddenLayers
        self.hiddenSize = config.hiddenSize
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<numLayers).map { idx in Gemma3DecoderLayer(config, layerIdx: idx) }
        self._norm.wrappedValue = Gemma3RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: inout [KVCache?]
    ) -> MLXArray {
        // Get embeddings and scale by sqrt(hiddenSize) - Gemma specific
        var hiddenStates = embedTokens(inputIds)
        let scale = MLXArray(sqrt(Float(hiddenSize)))
        hiddenStates = hiddenStates * scale.asType(hiddenStates.dtype)

        // Create masks for global and sliding window attention
        // Find a global layer to get its cache offset for the full mask
        let globalLayerIdx = slidingWindowPattern > 0 ? slidingWindowPattern - 1 : 0
        let globalCache = globalLayerIdx < cache.count ? cache[globalLayerIdx] : nil
        let fullMask = createAttentionMask(h: hiddenStates, cache: globalCache, windowSize: nil)

        let slidingCache = cache.first ?? nil
        let slidingMask = createAttentionMask(h: hiddenStates, cache: slidingCache, windowSize: slidingWindow)

        for i in 0..<layers.count {
            let isGlobal = layers[i].isGlobal
            let mask = isGlobal ? fullMask : slidingMask
            hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
        }

        return norm(hiddenStates)
    }
}

// MARK: - Top-Level Model

public class Gemma3Model: Module, LLMModel {
    public let vocabularySize: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int

    @ModuleInfo(key: "model") var model: Gemma3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private let config: Gemma3Configuration

    public var supportsCache: Bool { true }

    public init(_ config: Gemma3Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.numLayers = config.numHiddenLayers
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim

        self._model.wrappedValue = Gemma3ModelInner(config)
        // Gemma ties embed and lm_head weights
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Forward pass without cache
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var cache: [KVCache?] = Array(repeating: nil, count: numLayers)
        let h = model(inputIds, cache: &cache)
        return lmHead(h)
    }

    /// Forward pass with KV cache for efficient generation
    public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        var layerCaches: [KVCache?]
        if let existingCache = cache {
            layerCaches = existingCache.map { $0 as KVCache? }
        } else {
            layerCaches = Array(repeating: nil, count: numLayers)
        }

        let h = model(inputIds, cache: &layerCaches)

        cache = layerCaches.compactMap { $0 }

        return lmHead(h)
    }

    /// Create a new KV cache with appropriate cache types per layer
    public func newCache() -> [KVCache] {
        return (0..<numLayers).map { i in
            if config.isGlobalLayer(i) {
                // Global layers use standard cache
                return KVCacheSimple()
            } else {
                // Sliding window layers use rotating cache
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    /// Sanitize weight keys from HuggingFace format
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // model.language_model.X -> model.X (VLM format)
            if newKey.hasPrefix("model.language_model.") {
                newKey = "model." + String(newKey.dropFirst("model.language_model.".count))
            } else if newKey.hasPrefix("language_model.") {
                newKey = "model." + String(newKey.dropFirst("language_model.".count))
            }

            // Skip vision/audio tower weights
            if newKey.contains("vision_tower") || newKey.contains("audio_tower") ||
                newKey.contains("multi_modal_projector") {
                continue
            }

            result[newKey] = value
        }

        // Weight tying: copy embed_tokens to lm_head if lm_head is missing
        if result["lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let embedWeight = result["model.embed_tokens.\(suffix)"] {
                    result["lm_head.\(suffix)"] = embedWeight
                }
            }
        }

        return result
    }
}
