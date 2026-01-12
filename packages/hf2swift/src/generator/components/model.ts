/**
 * Model and ModelInner component generators
 *
 * Supports:
 * - Standard transformer models
 * - AltUp-style models (Gemma3n) with:
 *   - Dual embeddings (token + per-layer)
 *   - AltUp projections
 *   - Cache index mapping for KV-shared layers
 *   - Weight tying
 *
 * Note: Output is not formatted - SwiftFormat handles that.
 */

import type { ModelFeatures } from "../features.js"

export function generateModelInner(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  if (features.hasAltUp) {
    return generateAltUpLanguageModel(modelName, configClass, features)
  }
  return generateStandardModelInner(modelName, configClass, features)
}

function generateStandardModelInner(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  const embedCall = features.useEmbeddingScale
    ? `var hiddenStates = embedTokens(inputIds)
let scale = MLXArray(sqrt(Float(hiddenSize)))
hiddenStates = hiddenStates * scale.asType(hiddenStates.dtype)`
    : `var hiddenStates = embedTokens(inputIds)`

  const { maskHandling, layerLoop, extraProps, extraInit } = buildModelInnerParts(features)

  return `
// MARK: - Model Inner

class ${modelName}ModelInner: Module {
@ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
@ModuleInfo(key: "layers") var layers: [${modelName}DecoderLayer]
@ModuleInfo(key: "norm") var norm: ${normType}

let numLayers: Int
let hiddenSize: Int
${extraProps}

init(_ config: ${configClass}) {
self.numLayers = config.numHiddenLayers
self.hiddenSize = config.hiddenSize
${extraInit}
self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
self._layers.wrappedValue = (0..<numLayers).map { idx in ${modelName}DecoderLayer(config, layerIdx: idx) }
self._norm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
}

func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache?]) -> MLXArray {
${embedCall}
${maskHandling}
${layerLoop}
return norm(hiddenStates)
}
}
`
}

function buildModelInnerParts(features: ModelFeatures): {
  maskHandling: string
  layerLoop: string
  extraProps: string
  extraInit: string
} {
  if (features.hasMoE) {
    return {
      maskHandling: `// Find first global layer for mask creation
var firstGlobalIdx = 0
for (i, layerType) in layerTypes.prefix(cache.count).enumerated() {
if layerType == "full_attention" { firstGlobalIdx = i; break }
}
let globalCache = firstGlobalIdx < cache.count ? cache[firstGlobalIdx] : nil
let globalOffset = globalCache?.offset ?? 0
let globalMask = createAttentionMask(n: hiddenStates.dim(1), offset: globalOffset, windowSize: nil)
let slidingOffset = cache.first??.offset ?? 0
let slidingMask = createAttentionMask(n: hiddenStates.dim(1), offset: slidingOffset, windowSize: slidingWindow)`,
      layerLoop: `for i in 0..<layers.count {
let layerType = i < layerTypes.count ? layerTypes[i] : "sliding_attention"
let isGlobal = layerType == "full_attention"
let mask = isGlobal ? globalMask : slidingMask
hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
}`,
      extraProps: `let slidingWindow: Int
let layerTypes: [String]`,
      extraInit: `self.slidingWindow = config.slidingWindow
self.layerTypes = config.layerTypes`
    }
  }

  if (features.useSlidingWindow) {
    return {
      maskHandling: `let globalLayerIdx = slidingWindowPattern - 1
let globalCache = globalLayerIdx < cache.count ? cache[globalLayerIdx] : nil
let globalOffset = globalCache?.offset ?? 0
let globalMask = createAttentionMask(n: hiddenStates.dim(1), offset: globalOffset, windowSize: nil)
let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
if slidingWindowPattern > 1 {
let slidingOffset = cache.first??.offset ?? 0
slidingMask = createAttentionMask(n: hiddenStates.dim(1), offset: slidingOffset, windowSize: slidingWindow)
} else {
slidingMask = globalMask
}`,
      layerLoop: `for i in 0..<layers.count {
let isGlobal = (i % slidingWindowPattern) == (slidingWindowPattern - 1)
let mask = isGlobal ? globalMask : slidingMask
hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
}`,
      extraProps: `let slidingWindow: Int
let slidingWindowPattern: Int`,
      extraInit: `self.slidingWindow = config.slidingWindow
self.slidingWindowPattern = config.slidingWindowPattern`
    }
  }

  return {
    maskHandling: `let offset = cache.first??.offset ?? 0
let mask = createAttentionMask(n: hiddenStates.dim(1), offset: offset, windowSize: nil)`,
    layerLoop: `for i in 0..<layers.count {
hiddenStates = layers[i](hiddenStates, mask: mask, cache: &cache[i])
}`,
    extraProps: "",
    extraInit: ""
  }
}

function generateAltUpLanguageModel(
  modelName: string,
  configClass: string,
  _features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  return `
// MARK: - Language Model (AltUp)

class ${modelName}LanguageModel: Module {
let config: ${configClass}
let hiddenSize: Int
let hiddenSizePerLayerInput: Int
let vocabSize: Int
let vocabSizePerLayerInput: Int
let numHiddenLayers: Int
let altupNumInputs: Int
let finalLogitSoftcapping: Float?

let firstKVSharedLayerIdx: Int
let layerIdxToCacheIdx: [Int]
let firstSlidingIdx: Int
let firstFullIdx: Int

@ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
@ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
@ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
@ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: ${normType}
@ModuleInfo(key: "altup_projections") var altupProjections: [Linear]
@ModuleInfo(key: "altup_unembed_projections") var altupUnembedProjections: [Linear]
@ModuleInfo(key: "layers") var layers: [${modelName}DecoderLayer]
@ModuleInfo(key: "norm") var norm: ${normType}

init(_ config: ${configClass}) {
self.config = config
self.hiddenSize = config.hiddenSize
self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
self.vocabSize = config.vocabSize
self.vocabSizePerLayerInput = config.vocabSizePerLayerInput
self.numHiddenLayers = config.numHiddenLayers
self.altupNumInputs = config.altupNumInputs
self.finalLogitSoftcapping = config.finalLogitSoftcapping

self.firstKVSharedLayerIdx = config.firstKVSharedLayerIdx

var firstSliding = 0
var firstFull = 0
for (i, layerType) in config.layerTypes.enumerated() {
if layerType == "sliding_attention" && firstSliding == 0 { firstSliding = i; break }
}
for (i, layerType) in config.layerTypes.enumerated() {
if layerType == "full_attention" { firstFull = i; break }
}
self.firstSlidingIdx = firstSliding
self.firstFullIdx = firstFull

let concreteLayers = Array(config.layerTypes.prefix(config.firstKVSharedLayerIdx))
var sharedFullIdx = 0
var sharedSlidingIdx = 0
for (i, layerType) in concreteLayers.enumerated().reversed() {
if layerType == "full_attention" && sharedFullIdx == 0 { sharedFullIdx = i }
if layerType == "sliding_attention" && sharedSlidingIdx == 0 { sharedSlidingIdx = i }
if sharedFullIdx > 0 && sharedSlidingIdx > 0 { break }
}

var mapping: [Int] = []
for i in 0..<config.numHiddenLayers {
if i < config.firstKVSharedLayerIdx {
mapping.append(i)
} else {
let layerType = i < config.layerTypes.count ? config.layerTypes[i] : "sliding_attention"
mapping.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
}
}
self.layerIdxToCacheIdx = mapping

_embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
_embedTokensPerLayer.wrappedValue = Embedding(embeddingCount: config.vocabSizePerLayerInput, dimensions: numHiddenLayers * config.hiddenSizePerLayerInput)
_perLayerModelProjection.wrappedValue = Linear(config.hiddenSize, numHiddenLayers * config.hiddenSizePerLayerInput, bias: false)
_perLayerProjectionNorm.wrappedValue = ${normType}(dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
_altupProjections.wrappedValue = (1..<config.altupNumInputs).map { _ in Linear(config.hiddenSize, config.hiddenSize, bias: false) }
_altupUnembedProjections.wrappedValue = (1..<config.altupNumInputs).map { _ in Linear(config.hiddenSize, config.hiddenSize, bias: false) }
_layers.wrappedValue = (0..<numHiddenLayers).map { idx in ${modelName}DecoderLayer(config, layerIdx: idx) }
_norm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
}

func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
let mask = inputIds .< Int32(vocabSizePerLayerInput)
let tokens = MLX.where(mask, inputIds, MLXArray.zeros(like: inputIds))
let embeds = embedTokensPerLayer(tokens)
let scaled = embeds * sqrt(Float(hiddenSizePerLayerInput))
let shape = inputIds.shape
return scaled.reshaped([shape[0], shape[1], numHiddenLayers, hiddenSizePerLayerInput])
}

func projectPerLayerInputs(_ inputsEmbeds: MLXArray, _ perLayerInputs: MLXArray) -> MLXArray {
var projection = perLayerModelProjection(inputsEmbeds)
projection = projection * pow(Float(hiddenSize), -0.5)
let shape = inputsEmbeds.shape
projection = projection.reshaped([shape[0], shape[1], numHiddenLayers, hiddenSizePerLayerInput])
projection = perLayerProjectionNorm(projection)
let sqrtTwoInv = Float(pow(2.0, -0.5))
return (projection + perLayerInputs) * sqrtTwoInv
}

func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache?]) -> MLXArray {
var h = embedTokens(inputIds)
h = h * sqrt(Float(hiddenSize))

let perLayerInputsRaw = getPerLayerInputs(inputIds)
let perLayerInputs = projectPerLayerInputs(h, perLayerInputsRaw)

let targetMagnitude = pow(mean(h.pow(2), axis: -1, keepDims: true), 0.5)
var hList: [MLXArray] = [h]
for proj in altupProjections { hList.append(proj(h)) }
var hiddenStates = stacked(hList, axis: 0)

let mags = pow(mean(hiddenStates[1...].pow(2), axis: -1, keepDims: true), 0.5)
let minVal = MLXArray(Float.leastNormalMagnitude)
let normalizedSlots = hiddenStates[1...] * (targetMagnitude / maximum(mags, minVal))
hiddenStates = concatenated([hiddenStates[0..<1], normalizedSlots], axis: 0)

let h0 = hiddenStates[0]
let globalCache = firstFullIdx < cache.count ? cache[firstFullIdx] : nil
let globalOffset = globalCache?.offset ?? 0
let globalMask = createAttentionMask(n: h0.dim(1), offset: globalOffset, windowSize: nil)
let slidingCache = firstSlidingIdx < cache.count ? cache[firstSlidingIdx] : nil
let slidingOffset = slidingCache?.offset ?? 0
let slidingMask = createAttentionMask(n: h0.dim(1), offset: slidingOffset, windowSize: config.slidingWindow)

for i in 0..<layers.count {
let isGlobal = config.isGlobalLayer(i)
let mask = isGlobal ? globalMask : slidingMask
let perLayerInput = perLayerInputs[0..., 0..., i, 0...]
let cacheIdx = layerIdxToCacheIdx[i]
if cacheIdx < cache.count {
hiddenStates = layers[i](hiddenStates, perLayerInput: perLayerInput, mask: mask, cache: &cache[cacheIdx])
} else {
var nilCache: KVCache? = nil
hiddenStates = layers[i](hiddenStates, perLayerInput: perLayerInput, mask: mask, cache: &nilCache)
}
}

let finalTargetMagnitude = pow(mean(hiddenStates[0].pow(2), axis: -1, keepDims: true), 0.5)
var unembedded: [MLXArray] = [hiddenStates[0]]
for i in 0..<altupUnembedProjections.count { unembedded.append(altupUnembedProjections[i](hiddenStates[i + 1])) }
var finalStates = stacked(unembedded, axis: 0)

let finalMags = pow(mean(finalStates[1...].pow(2), axis: -1, keepDims: true), 0.5)
let normalizedFinal = finalStates[1...] * (finalTargetMagnitude / maximum(finalMags, minVal))
finalStates = concatenated([finalStates[0..<1], normalizedFinal], axis: 0)

var output = mean(finalStates, axis: 0)
output = norm(output)

let logits = embedTokens.asLinear(output)
if let cap = finalLogitSoftcapping { return cap * tanh(logits / cap) }
return logits
}
}
`
}

export function generateModel(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  if (features.hasAltUp) {
    return generateAltUpModel(modelName, configClass, features)
  }
  return generateStandardModel(modelName, configClass, features)
}

function generateStandardModel(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const newCacheImpl = buildNewCacheImpl(features)

  // MoE models need a completely different sanitize implementation
  if (features.hasMoE) {
    return generateMoEModel(modelName, configClass, newCacheImpl)
  }

  return `
// MARK: - Top-Level Model

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

public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
var cache: [KVCache?] = Array(repeating: nil, count: numLayers)
let h = model(inputIds, cache: &cache)
return lmHead(h)
}

public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
var layerCaches: [KVCache?]
if let existingCache = cache { layerCaches = existingCache.map { $0 as KVCache? } }
else { layerCaches = Array(repeating: nil, count: numLayers) }
let h = model(inputIds, cache: &layerCaches)
cache = layerCaches.compactMap { $0 }
return lmHead(h)
}

${newCacheImpl}

public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
// Uses shared weight sanitization logic
return sanitizeWeights(weights)
}
}
`
}

function generateMoEModel(modelName: string, configClass: string, newCacheImpl: string): string {
  return `
// MARK: - Top-Level Model

public class ${modelName}Model: Module, LLMModel {
public let vocabularySize: Int
public let numLayers: Int
public let numKVHeads: Int
public let headDim: Int
public let kvHeads: [Int]

let model: ${modelName}ModelInner
private let configuration: ${configClass}
@ModuleInfo(key: "lm_head") var lmHead: Linear

public var supportsCache: Bool { true }

public init(_ config: ${configClass}) {
configuration = config
model = ${modelName}ModelInner(config)
vocabularySize = config.vocabSize
numLayers = config.numHiddenLayers
numKVHeads = config.numKeyValueHeads
headDim = config.headDim
kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
_lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
}

public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
var cache: [KVCache?] = Array(repeating: nil, count: numLayers)
let hidden = model(inputIds, cache: &cache)
return lmHead(hidden)
}

public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
var layerCaches: [KVCache?]
if let existingCache = cache { layerCaches = existingCache.map { $0 as KVCache? } }
else { layerCaches = Array(repeating: nil, count: numLayers) }
let hidden = model(inputIds, cache: &layerCaches)
cache = layerCaches.compactMap { $0 }
return lmHead(hidden)
}

${newCacheImpl}

${generateMoeSanitizeMethodInline()}
}
`
}

/**
 * Generate MoE sanitize method that delegates to shared MoESanitizer.
 * Reduces generated code from 80+ lines to 3 lines.
 */
function generateMoeSanitizeMethodInline(): string {
  return `// MARK: - Weight Sanitization

/// Sanitize MoE weights - delegates to shared implementation
public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
MoESanitizer.sanitize(weights: weights)
}`
}

function buildNewCacheImpl(features: ModelFeatures): string {
  if (features.hasMoE) {
    return `public func newCache() -> [KVCache] {
return (0..<numLayers).map { i in
let layerType = i < configuration.layerTypes.count ? configuration.layerTypes[i] : "sliding_attention"
if layerType == "full_attention" { return KVCacheSimple() }
else { return RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0) }
}
}`
  }

  if (features.useSlidingWindow) {
    return `public func newCache() -> [KVCache] {
return (0..<numLayers).map { i in
let isGlobal = (i % config.slidingWindowPattern) == (config.slidingWindowPattern - 1)
if isGlobal { return KVCacheSimple() }
else { return RotatingKVCache(maxSize: config.slidingWindow, keep: 0) }
}
}`
  }

  return `public func newCache() -> [KVCache] {
return (0..<numLayers).map { _ in KVCacheSimple() }
}`
}

function generateAltUpModel(
  modelName: string,
  configClass: string,
  _features: ModelFeatures
): string {
  return `
// MARK: - Inner Wrapper

class ${modelName}Inner: Module {
@ModuleInfo(key: "language_model") var languageModel: ${modelName}LanguageModel

init(_ config: ${configClass}) {
_languageModel.wrappedValue = ${modelName}LanguageModel(config)
}

func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache?]) -> MLXArray {
return languageModel(inputIds, cache: &cache)
}
}

// MARK: - Top-Level Model

public class ${modelName}Model: Module, LLMModel {
public let vocabularySize: Int
public let numLayers: Int
public let numKVHeads: Int
public let headDim: Int

@ModuleInfo(key: "model") var model: ${modelName}Inner
private let config: ${configClass}

public var supportsCache: Bool { true }

public init(_ config: ${configClass}) {
self.config = config
self.vocabularySize = config.vocabSize
self.numLayers = config.numHiddenLayers
self.numKVHeads = config.numKeyValueHeads
self.headDim = config.headDim
_model.wrappedValue = ${modelName}Inner(config)
}

public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
var cache: [KVCache?] = Array(repeating: nil, count: numLayers)
return model(inputIds, cache: &cache)
}

public func callAsFunction(_ inputIds: MLXArray, cache: inout [KVCache]?) -> MLXArray {
var layerCaches: [KVCache?]
if let existingCache = cache { layerCaches = existingCache.map { $0 as KVCache? } }
else { layerCaches = Array(repeating: nil, count: numLayers) }
let output = model(inputIds, cache: &layerCaches)
cache = layerCaches.compactMap { $0 }
return output
}

public func newCache() -> [KVCache] {
let numCaches = config.firstKVSharedLayerIdx
return (0..<numCaches).map { i in
let layerType = i < config.layerTypes.count ? config.layerTypes[i] : "sliding_attention"
if layerType == "full_attention" { return KVCacheSimple() }
else { return RotatingKVCache(maxSize: config.slidingWindow, keep: 0) }
}
}

public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
var result: [String: MLXArray] = [:]
let skipPatterns = ["rotary_emb", "vision_tower", "audio_tower", "embed_audio", "embed_vision"]
for (key, value) in weights {
let shouldSkip = skipPatterns.contains { key.contains($0) }
if shouldSkip { continue }
var newKey = key
// Transform language_model.model.X -> model.language_model.X
if key.hasPrefix("language_model.model.") {
let suffix = String(key.dropFirst("language_model.model.".count))
newKey = "model.language_model." + suffix
}
// Transform language_model.X (without .model.) -> model.language_model.X
else if key.hasPrefix("language_model.") {
let suffix = String(key.dropFirst("language_model.".count))
newKey = "model.language_model." + suffix
}
result[newKey] = value
}
return result
}
}
`
}
