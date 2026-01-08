//
//  Gemma3n.swift
//  NodeMLXCore
//
//  Gemma 3n Language Model implementation.
//  This model has a unique architecture with AltUp and Laurel blocks.
//
//  The model-specific components are in Gemma3nSpecific.swift
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Gemma3nConfiguration: Decodable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int?
    public var intermediateSizes: [Int]  // Array for MoE/varying layer sizes
    public var vocabSize: Int
    public var headDim: Int
    public var rmsNormEps: Float?
    public var ropeTheta: Float?
    public var ropeLocalBaseFreq: Float?
    public var maxPositionEmbeddings: Int?
    public var queryPreAttnScalar: Int?
    public var layerTypes: [String]?
    public var slidingWindow: Int?

    // Gemma3n specific
    public var hiddenSizePerLayerInput: Int?
    public var vocabSizePerLayerInput: Int?
    public var altupNumInputs: Int?
    public var altupActiveIdx: Int?
    public var altupCorrectScale: Bool?
    public var laurelRank: Int?

    public var modelType: String?

    /// Get intermediate size for a specific layer
    public func intermediateSize(forLayer layer: Int) -> Int {
        if layer < intermediateSizes.count {
            return intermediateSizes[layer]
        }
        return intermediateSizes.first ?? 8192
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
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case maxPositionEmbeddings = "max_position_embeddings"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case altupNumInputs = "altup_num_inputs"
        case altupActiveIdx = "altup_active_idx"
        case altupCorrectScale = "altup_correct_scale"
        case laurelRank = "laurel_rank"
        case modelType = "model_type"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Try to decode from text_config first (VLM format), then from top level
        if let textContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
            hiddenSize = try textContainer.decode(Int.self, forKey: .hiddenSize)
            numHiddenLayers = try textContainer.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try textContainer.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try textContainer.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            // intermediate_size can be Int or [Int]
            if let sizes = try? textContainer.decode([Int].self, forKey: .intermediateSize) {
                intermediateSizes = sizes
            } else if let size = try? textContainer.decode(Int.self, forKey: .intermediateSize) {
                intermediateSizes = [size]
            } else {
                intermediateSizes = [8192]  // default
            }
            vocabSize = try textContainer.decode(Int.self, forKey: .vocabSize)
            headDim = try textContainer.decode(Int.self, forKey: .headDim)
            rmsNormEps = try textContainer.decodeIfPresent(Float.self, forKey: .rmsNormEps)
            ropeTheta = try textContainer.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ropeLocalBaseFreq = try textContainer.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq)
            maxPositionEmbeddings = try textContainer.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            queryPreAttnScalar = try textContainer.decodeIfPresent(Int.self, forKey: .queryPreAttnScalar)
            layerTypes = try textContainer.decodeIfPresent([String].self, forKey: .layerTypes)
            slidingWindow = try textContainer.decodeIfPresent(Int.self, forKey: .slidingWindow)
            hiddenSizePerLayerInput = try textContainer.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
            vocabSizePerLayerInput = try textContainer.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
            altupNumInputs = try textContainer.decodeIfPresent(Int.self, forKey: .altupNumInputs)
            altupActiveIdx = try textContainer.decodeIfPresent(Int.self, forKey: .altupActiveIdx)
            altupCorrectScale = try textContainer.decodeIfPresent(Bool.self, forKey: .altupCorrectScale)
            laurelRank = try textContainer.decodeIfPresent(Int.self, forKey: .laurelRank)
            modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        } else {
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            // intermediate_size can be Int or [Int]
            if let sizes = try? container.decode([Int].self, forKey: .intermediateSize) {
                intermediateSizes = sizes
            } else if let size = try? container.decode(Int.self, forKey: .intermediateSize) {
                intermediateSizes = [size]
            } else {
                intermediateSizes = [8192]  // default
            }
            vocabSize = try container.decode(Int.self, forKey: .vocabSize)
            headDim = try container.decode(Int.self, forKey: .headDim)
            rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
            ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq)
            maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            queryPreAttnScalar = try container.decodeIfPresent(Int.self, forKey: .queryPreAttnScalar)
            layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
            hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
            vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
            altupNumInputs = try container.decodeIfPresent(Int.self, forKey: .altupNumInputs)
            altupActiveIdx = try container.decodeIfPresent(Int.self, forKey: .altupActiveIdx)
            altupCorrectScale = try container.decodeIfPresent(Bool.self, forKey: .altupCorrectScale)
            laurelRank = try container.decodeIfPresent(Int.self, forKey: .laurelRank)
            modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        }
    }
}

// MARK: - Decoder Layer

/// Simplified Gemma3n Decoder Layer - only essential components
/// Full AltUp/Laurel/per-layer features are not implemented
class Gemma3nTextDecoderLayer: Module {
    let layerIdx: Int
    let attentionType: String

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma3nTextAttention
    @ModuleInfo(key: "mlp") var mlp: Gemma3nTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: Gemma3nRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: Gemma3nRMSNorm

    init(_ config: Gemma3nConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx

        // Determine attention type for this layer
        if let layerTypes = config.layerTypes, layerIdx < layerTypes.count {
            self.attentionType = layerTypes[layerIdx]
        } else {
            self.attentionType = "full_attention"
        }

        let eps = config.rmsNormEps ?? 1e-6
        let numKVHeads = config.numKeyValueHeads ?? config.numAttentionHeads
        let intermediateSize = config.intermediateSize(forLayer: layerIdx)

        // Initialize attention
        self._selfAttn.wrappedValue = Gemma3nTextAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKVHeads: numKVHeads,
            headDim: config.headDim,
            queryPreAttnScalar: config.queryPreAttnScalar,
            eps: eps
        )

        // Initialize MLP
        self._mlp.wrappedValue = Gemma3nTextMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: intermediateSize
        )

        // Initialize norms
        self._inputLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._postAttentionLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._preFeedforwardLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: eps)
        self._postFeedforwardLayernorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: eps)
    }

    /// Simplified forward pass (standard transformer without AltUp/Laurel)
    func callAsFunction(
        _ hiddenStates: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray)?,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // 1. Pre-norm + Self-attention
        let normed = inputLayernorm(hiddenStates)
        let attnOut = selfAttn(normed, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
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

// MARK: - Text Model Inner (Simplified)

/// Simplified Gemma3n inner model - standard transformer without AltUp/Laurel complexity
class Gemma3nTextModelInner: Module {
    let numLayers: Int
    let hiddenSize: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Gemma3nTextScaledWordEmbedding
    @ModuleInfo(key: "layers") var layers: [Gemma3nTextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm
    @ModuleInfo(key: "rotary_emb") var rotaryEmb: Gemma3nRotaryEmbedding

    init(_ config: Gemma3nConfiguration) {
        self.numLayers = config.numHiddenLayers
        self.hiddenSize = config.hiddenSize

        let eps = config.rmsNormEps ?? 1e-6

        // Main token embedding
        self._embedTokens.wrappedValue = Gemma3nTextScaledWordEmbedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
            embedScale: sqrt(Float(config.hiddenSize))
        )

        // Decoder layers
        self._layers.wrappedValue = (0..<numLayers).map { idx in
            Gemma3nTextDecoderLayer(config, layerIdx: idx)
        }

        // Final norm
        self._norm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize, eps: eps)

        // Rotary embedding
        self._rotaryEmb.wrappedValue = Gemma3nRotaryEmbedding(
            dim: config.headDim,
            maxPositions: config.maxPositionEmbeddings ?? 8192,
            ropeTheta: config.ropeTheta ?? 10000.0,
            ropeLocalBaseFreq: config.ropeLocalBaseFreq ?? 10000.0
        )
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [[KVCache]]? = nil) -> MLXArray {
        // 1. Embed tokens
        var hiddenStates = embedTokens(inputIds)

        // 2. Compute position embeddings
        let seqLen = inputIds.dim(1)
        let positions = MLXArray(Array(0..<seqLen).map { Int32($0) })

        // 3. Process through layers
        for (layerIdx, layer) in layers.enumerated() {
            let positionEmbeddings = rotaryEmb(positions, layerType: layer.attentionType)
            let layerCache = cache?[layerIdx].first

            hiddenStates = layer(hiddenStates, positionEmbeddings: positionEmbeddings, mask: nil, cache: layerCache)
        }

        // 4. Final norm
        hiddenStates = norm(hiddenStates)

        return hiddenStates
    }
}

// MARK: - Top-Level Model

public class Gemma3nModel: Module, LLMModel {
    public let vocabularySize: Int
    public let numLayers: Int

    @ModuleInfo(key: "model") var model: Gemma3nTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    private let config: Gemma3nConfiguration

    public init(_ config: Gemma3nConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.numLayers = config.numHiddenLayers

        self._model.wrappedValue = Gemma3nTextModelInner(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let h = model(inputIds, cache: nil)
        return lmHead(h)
    }

    /// Sanitize weight keys from HuggingFace format
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        // Keys to skip (complex Gemma3n-specific modules that we don't use in simplified forward)
        let skipPatterns = [
            "altup",
            "laurel",
            "per_layer_input_gate",
            "per_layer_projection",
            "post_per_layer_input_norm",
            "embed_tokens_per_layer",
            "per_layer_model_projection",
            "per_layer_projection_norm",
            "altup_projections",
            "altup_unembed_projections",
        ]

        for (key, value) in weights {
            var newKey = key

            // model.language_model.X -> model.X
            if newKey.hasPrefix("model.language_model.") {
                newKey = "model." + String(newKey.dropFirst("model.language_model.".count))
            } else if newKey.hasPrefix("language_model.") {
                newKey = "model." + String(newKey.dropFirst("language_model.".count))
            }

            // Skip complex modules we don't use
            let shouldSkip = skipPatterns.contains { newKey.contains($0) }
            if shouldSkip {
                continue
            }

            // Remap embed_tokens.X -> embed_tokens.inner.X (for our wrapper structure)
            if newKey.contains("embed_tokens.") && !newKey.contains("embed_tokens.inner.") {
                newKey = newKey.replacingOccurrences(of: "embed_tokens.", with: "embed_tokens.inner.")
            }

            result[newKey] = value
        }
        return result
    }
}
