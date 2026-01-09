//
//  Gemma3VLM.swift
//  NodeMLXCore
//
//  Gemma 3 Vision-Language Model
//  Combines SigLIP vision encoder with Gemma 3 text model for multimodal generation.
//
//  Supports Gemma 3 4B, 12B, and 27B vision variants.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Gemma3VLMConfiguration: Decodable, Sendable {
    /// Text model configuration
    public var textConfig: Gemma3Configuration

    /// Vision model configuration
    public var visionConfig: SiglipVisionConfiguration

    /// Number of image tokens per image (default: 256)
    public var mmTokensPerImage: Int

    /// Begin-of-image token index
    public var boiTokenIndex: Int

    /// End-of-image token index
    public var eoiTokenIndex: Int

    /// Image placeholder token index
    public var imageTokenIndex: Int

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case mmTokensPerImage = "mm_tokens_per_image"
        case boiTokenIndex = "boi_token_index"
        case eoiTokenIndex = "eoi_token_index"
        case imageTokenIndex = "image_token_index"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        textConfig = try container.decode(Gemma3Configuration.self, forKey: .textConfig)
        visionConfig = try container.decode(SiglipVisionConfiguration.self, forKey: .visionConfig)
        mmTokensPerImage = try container.decodeIfPresent(Int.self, forKey: .mmTokensPerImage) ?? 256
        boiTokenIndex = try container.decodeIfPresent(Int.self, forKey: .boiTokenIndex) ?? 255_999
        eoiTokenIndex = try container.decodeIfPresent(Int.self, forKey: .eoiTokenIndex) ?? 256_000
        imageTokenIndex = try container.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 262_144
    }

    public init(
        textConfig: Gemma3Configuration,
        visionConfig: SiglipVisionConfiguration,
        mmTokensPerImage: Int = 256,
        boiTokenIndex: Int = 255_999,
        eoiTokenIndex: Int = 256_000,
        imageTokenIndex: Int = 262_144
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.mmTokensPerImage = mmTokensPerImage
        self.boiTokenIndex = boiTokenIndex
        self.eoiTokenIndex = eoiTokenIndex
        self.imageTokenIndex = imageTokenIndex
    }
}

// MARK: - Vision Language Model

/// Gemma 3 Vision-Language Model
public class Gemma3VLMModel: Module, LLMModel {
    // LLMModel protocol
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var numLayers: Int { config.textConfig.numHiddenLayers }
    public let numKVHeads: Int
    public let headDim: Int
    public var supportsCache: Bool { true }

    /// Vision tower (SigLIP encoder)
    @ModuleInfo(key: "vision_tower") var visionTower: SiglipVisionModel

    /// Multi-modal projector
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Gemma3MultiModalProjector

    /// Language model (Gemma 3 text model, wrapped as inner)
    @ModuleInfo(key: "language_model") var languageModel: Gemma3Model

    private let config: Gemma3VLMConfiguration

    public init(_ config: Gemma3VLMConfiguration) {
        self.config = config
        self.numKVHeads = config.textConfig.numKeyValueHeads
        self.headDim = config.textConfig.headDim

        _visionTower.wrappedValue = SiglipVisionModel(config.visionConfig)
        _multiModalProjector.wrappedValue = Gemma3MultiModalProjector(
            visionConfig: config.visionConfig,
            textHiddenSize: config.textConfig.hiddenSize,
            mmTokensPerImage: config.mmTokensPerImage
        )
        _languageModel.wrappedValue = Gemma3Model(config.textConfig)
    }

    // MARK: - Vision Processing

    /// Get image features from pixel values
    /// - Parameter pixelValues: Image tensor [B, C, H, W]
    /// - Returns: Projected image features [B, mm_tokens_per_image, hidden_size]
    public func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        let visionOutputs = visionTower(pixelValues)
        let imageFeatures = multiModalProjector(visionOutputs)
        return imageFeatures
    }

    // MARK: - Forward Pass

    /// Forward pass with optional image input
    /// - Parameters:
    ///   - inputIds: Token IDs [B, L]
    ///   - pixelValues: Optional image tensor [B, C, H, W]
    ///   - cache: Optional KV cache
    /// - Returns: Logits [B, L, vocab_size]
    public func callAsFunction(
        _ inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        cache: inout [KVCache]?
    ) -> MLXArray {
        // Get text embeddings
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)

        // Scale embeddings (Gemma style)
        let scale = MLXArray(sqrt(Float(config.textConfig.hiddenSize)))
        inputsEmbeds = inputsEmbeds * scale.asType(inputsEmbeds.dtype)

        // Merge image features if provided
        if let pixelValues = pixelValues {
            let imageFeatures = getImageFeatures(pixelValues)
            inputsEmbeds = mergeImageFeatures(inputsEmbeds, imageFeatures: imageFeatures, inputIds: inputIds)
        }

        // Forward through language model with embeddings
        return languageModel.forward(inputsEmbeds: inputsEmbeds, cache: &cache)
    }

    /// Simple forward without images
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var cache: [KVCache]? = nil
        return callAsFunction(inputIds, pixelValues: nil, cache: &cache)
    }

    /// Forward with cache (LLMModel protocol)
    public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        return callAsFunction(inputIds, pixelValues: nil, cache: &cache)
    }

    // MARK: - Cache

    public func newCache() -> [KVCache] {
        return languageModel.newCache()
    }

    // MARK: - Weight Sanitization

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // Map HuggingFace VLM keys to our structure
            // HF: vision_tower.vision_model.* -> vision_tower.* (remove redundant vision_model)
            if newKey.hasPrefix("vision_tower.vision_model.") {
                newKey = "vision_tower." + String(newKey.dropFirst("vision_tower.vision_model.".count))
            }
            
            // MLX Conv2d expects weights in (out_channels, kH, kW, in_channels) format
            // HuggingFace may have (out_channels, in_channels, kH, kW) - need to transpose
            if newKey.contains("patch_embedding.weight") && newValue.ndim == 4 {
                // Check if format is (out, in, kH, kW) where in=3 for RGB
                if newValue.dim(1) == 3 && newValue.dim(2) == newValue.dim(3) {
                    // Transpose from (out, in, kH, kW) to (out, kH, kW, in)
                    newValue = newValue.transposed(0, 2, 3, 1)
                }
            }

            result[newKey] = newValue
        }

        // Weight tying fallback for language model
        if result["language_model.lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let embedWeight = result["language_model.model.embed_tokens.\(suffix)"] {
                    result["language_model.lm_head.\(suffix)"] = embedWeight
                }
            }
        }

        return result
    }

    // MARK: - Image Feature Merging

    /// Merge image features into text embeddings at placeholder positions
    private func mergeImageFeatures(
        _ inputsEmbeds: MLXArray,
        imageFeatures: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let imageTokenId = config.imageTokenIndex
        let vocabSize = vocabularySize

        // If image token is OOV, handle gracefully
        if imageTokenId >= vocabSize {
            // Find placeholder positions and replace with image features
            // For simplicity, assume single image and replace first occurrence
            return maskedScatter(inputsEmbeds, imageFeatures: imageFeatures, inputIds: inputIds, imageTokenId: imageTokenId)
        }

        return maskedScatter(inputsEmbeds, imageFeatures: imageFeatures, inputIds: inputIds, imageTokenId: imageTokenId)
    }

    /// Scatter image features at mask positions
    private func maskedScatter(
        _ inputsEmbeds: MLXArray,
        imageFeatures: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int
    ) -> MLXArray {
        // Create mask where input_ids == image_token_id
        let mask = inputIds .== imageTokenId

        // Expand mask to match embedding dimensions [B, L, 1]
        let expandedMask = mask.expandedDimensions(axis: -1)

        // Flatten image features to match sequence dimension
        let batchSize = inputsEmbeds.dim(0)
        let hiddenSize = inputsEmbeds.dim(2)
        _ = imageFeatures.dim(1)  // numImageTokens

        // For each batch, find image token positions and replace
        // This is a simplified version - full implementation would handle multiple images
        var result = inputsEmbeds

        // Get the mask sum to find number of image tokens
        let maskSum = mask.sum().item(Int.self)

        if maskSum > 0 {
            // Simple approach: broadcast image features across mask positions
            // This works for single image case
            let flatImageFeatures = imageFeatures.reshaped([batchSize, -1, hiddenSize])

            // Use where to conditionally select
            let broadcastedFeatures = MLX.broadcast(flatImageFeatures, to: inputsEmbeds.shape)
            result = MLX.where(expandedMask, broadcastedFeatures, inputsEmbeds)
        }

        return result
    }
}

// MARK: - Gemma3Model Extension for Embeddings Forward

extension Gemma3Model {
    /// Forward pass with pre-computed embeddings
    public func forward(inputsEmbeds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
        var layerCaches: [KVCache?]
        if let existingCache = cache {
            layerCaches = existingCache.map { $0 as KVCache? }
        } else {
            layerCaches = Array(repeating: nil, count: numLayers)
        }

        let h = model.forward(inputsEmbeds: inputsEmbeds, cache: &layerCaches)

        cache = layerCaches.compactMap { $0 }

        return lmHead(h)
    }
}

// MARK: - Gemma3ModelInner Extension for Embeddings Forward

extension Gemma3ModelInner {
    /// Forward pass with pre-computed embeddings (skips embed_tokens)
    public func forward(inputsEmbeds: MLXArray, cache: inout [KVCache?]) -> MLXArray {
        var hiddenStates = inputsEmbeds
        // Note: embedding scaling should be done before calling this

        // Create masks
        let globalLayerIdx = slidingWindowPattern - 1
        let globalCache = globalLayerIdx < cache.count ? cache[globalLayerIdx] : nil
        let globalMask = createAttentionMask(h: hiddenStates, cache: globalCache, windowSize: nil)

        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if slidingWindowPattern > 1 {
            let firstCache = cache.first ?? nil
            slidingMask = createAttentionMask(h: hiddenStates, cache: firstCache, windowSize: slidingWindow)
        } else {
            slidingMask = globalMask
        }

        for i in 0 ..< layers.count {
            let isGlobal = (i % slidingWindowPattern) == (slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingMask
            hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
        }

        return norm(hiddenStates)
    }
}
