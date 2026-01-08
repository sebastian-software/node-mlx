/**
 * Model and ModelInner component generators
 */

import type { ModelFeatures } from "../features.js"

export function generateModelInner(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Embedding call with optional sqrt(hiddenSize) scaling
  const embedCall = features.useEmbeddingScale
    ? `
        // Get embeddings and scale by sqrt(hiddenSize) - Gemma specific
        var hiddenStates = embedTokens(inputIds)
        let scale = MLXArray(sqrt(Float(hiddenSize)))
        hiddenStates = hiddenStates * scale.asType(hiddenStates.dtype)`
    : `
        var hiddenStates = embedTokens(inputIds)`

  // Sliding window mask handling
  let maskHandling: string
  let layerLoop: string

  if (features.useSlidingWindow) {
    maskHandling = `
        // Create masks following mlx-lm pattern:
        // - Global mask: uses cache from a global layer (pattern - 1)
        // - Sliding mask: uses cache from first layer with sliding window size
        let globalLayerIdx = slidingWindowPattern - 1
        let globalCache = globalLayerIdx < cache.count ? cache[globalLayerIdx] : nil
        let globalMask = createAttentionMask(h: hiddenStates, cache: globalCache, windowSize: nil)

        // Sliding window mask (only if pattern > 1)
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if slidingWindowPattern > 1 {
            let firstCache = cache.first ?? nil
            slidingMask = createAttentionMask(h: hiddenStates, cache: firstCache, windowSize: slidingWindow)
        } else {
            slidingMask = globalMask
        }`
    layerLoop = `
        for i in 0..<layers.count {
            // Layer is global if i % pattern == pattern - 1 (matching mlx-lm)
            let isGlobal = (i % slidingWindowPattern) == (slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingMask
            hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
        }`
  } else {
    maskHandling = `
        let mask = createAttentionMask(h: hiddenStates, cache: cache.first ?? nil, windowSize: nil)`
    layerLoop = `
        for i in 0..<layers.count {
            hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
        }`
  }

  const extraProps = features.useSlidingWindow
    ? `
    let slidingWindow: Int
    let slidingWindowPattern: Int`
    : ""

  const extraInit = features.useSlidingWindow
    ? `
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern`
    : ""

  return `// MARK: - Model Inner

class ${modelName}ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [${modelName}DecoderLayer]
    @ModuleInfo(key: "norm") var norm: ${normType}

    let numLayers: Int
    let hiddenSize: Int${extraProps}

    init(_ config: ${configClass}) {
        self.numLayers = config.numHiddenLayers
        self.hiddenSize = config.hiddenSize${extraInit}
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<numLayers).map { idx in ${modelName}DecoderLayer(config, layerIdx: idx) }
        self._norm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: inout [KVCache?]
    ) -> MLXArray {${embedCall}
${maskHandling}
${layerLoop}

        return norm(hiddenStates)
    }
}`
}

export function generateModel(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  // newCache implementation depends on sliding window
  let newCacheImpl: string
  if (features.useSlidingWindow) {
    newCacheImpl = `
    /// Create a new KV cache with appropriate cache types per layer
    /// Following mlx-lm pattern: global layers use KVCacheSimple, others use RotatingKVCache
    public func newCache() -> [KVCache] {
        return (0..<numLayers).map { i in
            // Layer is global if i % pattern == pattern - 1
            let isGlobal = (i % config.slidingWindowPattern) == (config.slidingWindowPattern - 1)
            if isGlobal {
                return KVCacheSimple()
            } else {
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }`
  } else {
    newCacheImpl = `
    /// Create a new KV cache
    public func newCache() -> [KVCache] {
        return (0..<numLayers).map { _ in KVCacheSimple() }
    }`
  }

  return `// MARK: - Top-Level Model

public class ${modelName}Model: Module, LLMModel {
    public let vocabularySize: Int
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int

    @ModuleInfo(key: "model") var model: ${modelName}ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private let config: ${configClass}

    public var supportsCache: Bool { true }

    public init(_ config: ${configClass}) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.numLayers = config.numHiddenLayers
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim

        self._model.wrappedValue = ${modelName}ModelInner(config)
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
${newCacheImpl}

    /// Sanitize weight keys from HuggingFace format
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // VLM format transformations:
            // - language_model.model.X -> model.X (transformer layers)
            // - language_model.lm_head.X -> lm_head.X (output projection)
            // - language_model.X -> X (other components)
            if newKey.hasPrefix("language_model.model.") {
                newKey = "model." + String(newKey.dropFirst("language_model.model.".count))
            } else if newKey.hasPrefix("language_model.lm_head.") {
                newKey = "lm_head." + String(newKey.dropFirst("language_model.lm_head.".count))
            } else if newKey.hasPrefix("language_model.") {
                newKey = String(newKey.dropFirst("language_model.".count))
            }

            // Skip vision/audio tower weights
            if newKey.contains("vision_tower") || newKey.contains("audio_tower") ||
                newKey.contains("multi_modal_projector") {
                continue
            }

            result[newKey] = value
        }

        // Weight tying fallback: copy embed_tokens to lm_head if missing
        if result["lm_head.weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let embedWeight = result["model.embed_tokens.\\(suffix)"] {
                    result["lm_head.\\(suffix)"] = embedWeight
                }
            }
        }

        return result
    }
}`
}
