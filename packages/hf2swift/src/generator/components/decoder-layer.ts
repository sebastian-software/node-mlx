/**
 * DecoderLayer component generator
 *
 * Supports:
 * - Standard decoder layer (2 norms)
 * - Gemma-style (4 norms, clipResidual)
 * - AltUp predict/correct (Gemma3n)
 * - Laurel blocks (Gemma3n)
 * - Per-layer inputs (Gemma3n)
 * - KV-sharing (Gemma3n)
 *
 * Note: Output is not formatted - SwiftFormat handles that.
 */

import type { ModelFeatures } from "../features.js"

export function generateDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  if (features.hasAltUp) {
    return generateAltUpDecoderLayer(modelName, configClass, features)
  }

  // Check if model can use shared StandardDecoderLayer
  if (canUseSharedStandardDecoderLayer(features)) {
    return generateSharedStandardDecoderLayer(modelName, configClass)
  }

  return generateStandardDecoderLayer(modelName, configClass, features)
}

/**
 * Check if a model can use the shared StandardDecoderLayer.
 * Requires both StandardAttention and StandardMLP to be usable.
 */
function canUseSharedStandardDecoderLayer(features: ModelFeatures): boolean {
  // Must be able to use both shared components
  const canUseSharedAttention =
    !features.useSlidingWindow &&
    !features.hasKVSharing &&
    !features.hasMoE &&
    !features.hasNoRopeLayers &&
    !features.hasQKNorms &&
    !features.hasVNorm &&
    !features.hasAttentionSinks &&
    features.attentionScale === undefined

  const canUseSharedMLP =
    features.activation === "silu" &&
    !features.hasPerLayerIntermediateSize &&
    !features.hasSparseActivation &&
    !features.hasMoE

  // Also requires standard RMSNorm (not Gemma-style)
  const usesStandardNorm = features.rmsNormStyle === "standard" && features.normsPerLayer === 2

  return canUseSharedAttention && canUseSharedMLP && usesStandardNorm
}

/**
 * Generate decoder layer using shared StandardDecoderLayer<C>.
 */
function generateSharedStandardDecoderLayer(modelName: string, configClass: string): string {
  return `
// MARK: - Decoder Layer

/// Standard decoder layer - uses shared implementation
typealias ${modelName}DecoderLayer = StandardDecoderLayer<${configClass}>
`
}

function generateStandardDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Layer index for sliding window, MoE, or no-rope layers
  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing -- logical OR for booleans
  const needsLayerIdx = features.useSlidingWindow || features.hasMoE || features.hasNoRopeLayers
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int" : ", layerIdx: Int = 0"
  const attnInit = needsLayerIdx
    ? `${modelName}Attention(config, layerIdx: layerIdx)`
    : `${modelName}Attention(config)`

  // Build declarations and forward body based on norm count
  const declarations = buildStandardDeclarations(modelName, normType, features)
  const initializations = buildStandardInitializations(modelName, normType, attnInit, features)
  const forwardBody = buildStandardForwardBody(features)

  return `
// MARK: - Decoder Layer

class ${modelName}DecoderLayer: Module {
${declarations}

init(_ config: ${configClass}${layerIdxParam}) {
${initializations}
}

func callAsFunction(
_ hiddenStates: MLXArray,
mask: MLXFast.ScaledDotProductAttentionMaskMode,
cache: inout KVCache?
) -> MLXArray {
${forwardBody}
}
}
`
}

function buildStandardDeclarations(
  modelName: string,
  normType: string,
  features: ModelFeatures
): string {
  const lines: string[] = []

  lines.push(`@ModuleInfo(key: "self_attn") var selfAttn: ${modelName}Attention`)
  lines.push(`@ModuleInfo(key: "mlp") var mlp: ${modelName}MLP`)
  lines.push(`@ModuleInfo(key: "input_layernorm") var inputLayernorm: ${normType}`)
  lines.push(`@ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: ${normType}`)

  if (features.normsPerLayer === 4) {
    lines.push(
      `@ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: ${normType}`
    )
    lines.push(
      `@ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: ${normType}`
    )
  }

  return lines.join("\n")
}

function buildStandardInitializations(
  modelName: string,
  normType: string,
  attnInit: string,
  features: ModelFeatures
): string {
  const lines: string[] = []

  lines.push(`self._selfAttn.wrappedValue = ${attnInit}`)
  lines.push(`self._mlp.wrappedValue = ${modelName}MLP(config)`)
  lines.push(
    `self._inputLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )
  lines.push(
    `self._postAttentionLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )

  if (features.normsPerLayer === 4) {
    lines.push(
      `self._preFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
    )
    lines.push(
      `self._postFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
    )
  }

  return lines.join("\n")
}

function buildStandardForwardBody(features: ModelFeatures): string {
  const lines: string[] = []

  if (features.normsPerLayer === 4) {
    // Gemma-style with 4 norms
    lines.push(`// 1. Pre-norm + Self-attention`)
    lines.push(`let normed = inputLayernorm(hiddenStates)`)
    lines.push(`let attnOut = selfAttn(normed, mask: mask, cache: &cache)`)
    lines.push(`let attnNormed = postAttentionLayernorm(attnOut)`)

    const residual1 = features.useClipResidual
      ? "clipResidual(hiddenStates, attnNormed)"
      : "hiddenStates + attnNormed"
    lines.push(`var h = ${residual1}`)

    lines.push(``)
    lines.push(`// 2. Pre-norm + MLP`)
    lines.push(`let mlpIn = preFeedforwardLayernorm(h)`)
    lines.push(`let mlpOut = mlp(mlpIn)`)
    lines.push(`let mlpNormed = postFeedforwardLayernorm(mlpOut)`)

    const residual2 = features.useClipResidual ? "clipResidual(h, mlpNormed)" : "h + mlpNormed"
    lines.push(`h = ${residual2}`)
    lines.push(`return h`)
  } else {
    // Standard 2-norm style
    lines.push(`// 1. Pre-norm + Self-attention`)
    lines.push(`let normed = inputLayernorm(hiddenStates)`)
    lines.push(`let attnOut = selfAttn(normed, mask: mask, cache: &cache)`)
    lines.push(`var h = hiddenStates + attnOut`)
    lines.push(``)
    lines.push(`// 2. Pre-norm + MLP`)
    lines.push(`let mlpNormed = postAttentionLayernorm(h)`)
    lines.push(`let mlpOut = mlp(mlpNormed)`)
    lines.push(`h = h + mlpOut`)
    lines.push(`return h`)
  }

  return lines.join("\n")
}

function generateAltUpDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`
  const perLayerParam = features.hasPerLayerInputs ? ", perLayerInput: MLXArray" : ""

  const declarations = buildAltUpDeclarations(modelName, normType, features)
  const initializations = buildAltUpInitializations(modelName, normType, features)
  const forwardBody = buildAltUpForwardBody(modelName, features)

  return `
// MARK: - Decoder Layer (with AltUp)

class ${modelName}DecoderLayer: Module {
let layerIdx: Int
let activeIdx: Int
let altupCorrectScale: Bool

${declarations}

init(_ config: ${configClass}, layerIdx: Int) {
self.layerIdx = layerIdx
self.activeIdx = config.altupActiveIdx
self.altupCorrectScale = config.altupCorrectScale

${initializations}
}

/// Forward pass with AltUp predict/correct
/// - Parameters:
///   - hiddenStates: [numInputs, batch, seq, hidden]
///   - perLayerInput: [batch, seq, hiddenPerLayerInput] (if hasPerLayerInputs)
///   - mask: attention mask
///   - cache: KV cache (for KV-shared layers, the cache contains pre-computed KV)
/// - Returns: [numInputs, batch, seq, hidden]
func callAsFunction(
_ hiddenStates: MLXArray${perLayerParam},
mask: MLXFast.ScaledDotProductAttentionMaskMode,
cache: inout KVCache?
) -> MLXArray {
${forwardBody}
}
}
`
}

function buildAltUpDeclarations(
  modelName: string,
  normType: string,
  features: ModelFeatures
): string {
  const lines: string[] = []

  lines.push(`@ModuleInfo(key: "self_attn") var selfAttn: ${modelName}Attention`)
  lines.push(`@ModuleInfo(key: "mlp") var mlp: ${modelName}MLP`)
  lines.push(`@ModuleInfo(key: "input_layernorm") var inputLayernorm: ${normType}`)
  lines.push(`@ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: ${normType}`)
  lines.push(
    `@ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: ${normType}`
  )
  lines.push(
    `@ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: ${normType}`
  )
  lines.push(`@ModuleInfo(key: "altup") var altup: ${modelName}AltUp`)

  if (features.hasLaurel) {
    lines.push(`@ModuleInfo(key: "laurel") var laurel: ${modelName}LaurelBlock`)
  }

  if (features.hasPerLayerInputs) {
    lines.push(`@ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear`)
    lines.push(`@ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear`)
    lines.push(
      `@ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: ${normType}`
    )
  }

  return lines.join("\n")
}

function buildAltUpInitializations(
  modelName: string,
  normType: string,
  features: ModelFeatures
): string {
  const lines: string[] = []

  lines.push(`_selfAttn.wrappedValue = ${modelName}Attention(config, layerIdx: layerIdx)`)
  lines.push(`_mlp.wrappedValue = ${modelName}MLP(config, layerIdx: layerIdx)`)
  lines.push(
    `_inputLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )
  lines.push(
    `_postAttentionLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )
  lines.push(
    `_preFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )
  lines.push(
    `_postFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
  )
  lines.push(`_altup.wrappedValue = ${modelName}AltUp(config)`)

  if (features.hasLaurel) {
    lines.push(`_laurel.wrappedValue = ${modelName}LaurelBlock(config)`)
  }

  if (features.hasPerLayerInputs) {
    lines.push(
      `_perLayerInputGate.wrappedValue = Linear(config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)`
    )
    lines.push(
      `_perLayerProjection.wrappedValue = Linear(config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)`
    )
    lines.push(
      `_postPerLayerInputNorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
    )
  }

  return lines.join("\n")
}

function buildAltUpForwardBody(modelName: string, features: ModelFeatures): string {
  const lines: string[] = []

  lines.push(`// 1. AltUp predict`)
  lines.push(`let predictions = altup.predict(hiddenStates)`)
  lines.push(`let activePrediction = predictions[activeIdx]`)
  lines.push(``)
  lines.push(`// 2. Input layernorm`)
  lines.push(`let activePredictionNormed = inputLayernorm(activePrediction)`)
  lines.push(``)

  // 3. Laurel
  if (features.hasLaurel) {
    lines.push(`// 3. Laurel (adds residual internally)`)
    lines.push(`let laurelOutput = laurel(activePredictionNormed)`)
  } else {
    lines.push(`// 3. Skip Laurel`)
    lines.push(`let laurelOutput = activePredictionNormed`)
  }
  lines.push(``)

  lines.push(`// 4. Self attention`)
  lines.push(`var attn = selfAttn(activePredictionNormed, mask: mask, cache: &cache)`)
  lines.push(`attn = postAttentionLayernorm(attn)`)
  lines.push(``)
  lines.push(`// 5. Residual + scale with sqrt(2)`)
  lines.push(`let attnGated = activePrediction + attn`)
  lines.push(`let sqrtTwoInv = Float(pow(2.0, -0.5))`)
  lines.push(`let attnLaurel = (attnGated + laurelOutput) * sqrtTwoInv`)
  lines.push(``)
  lines.push(`// 6. MLP`)
  lines.push(`let attnNorm = preFeedforwardLayernorm(attnLaurel)`)
  lines.push(`let attnFfw = mlp(attnNorm)`)
  lines.push(`let attnFfwNorm = postFeedforwardLayernorm(attnFfw)`)
  lines.push(`let attnFfwLaurelGated = attnLaurel + attnFfwNorm`)
  lines.push(``)
  lines.push(`// 7. AltUp correct`)
  lines.push(`let correctedPredictions = altup.correct(predictions, activated: attnFfwLaurelGated)`)
  lines.push(``)
  lines.push(`// 8. Scale corrected output if configured`)
  lines.push(`var firstPrediction = correctedPredictions[activeIdx]`)
  lines.push(`if altupCorrectScale {`)
  lines.push(`firstPrediction = altup.scaleCorrectOutput(firstPrediction)`)
  lines.push(`}`)
  lines.push(``)

  if (features.hasPerLayerInputs) {
    lines.push(`// 9. Per-layer input gate and projection`)
    lines.push(`var perLayerOut = perLayerInputGate(firstPrediction)`)
    lines.push(`perLayerOut = geluApproximate(perLayerOut)`)
    lines.push(`perLayerOut = perLayerOut * perLayerInput`)
    lines.push(`perLayerOut = perLayerProjection(perLayerOut)`)
    lines.push(`perLayerOut = postPerLayerInputNorm(perLayerOut)`)
    lines.push(``)
  }

  lines.push(`// Update all slots`)
  lines.push(`var updatedSlots: [MLXArray] = [correctedPredictions[0]]`)
  lines.push(`for i in 1..<correctedPredictions.dim(0) {`)

  if (features.hasPerLayerInputs) {
    lines.push(`updatedSlots.append(correctedPredictions[i] + perLayerOut)`)
  } else {
    lines.push(`updatedSlots.append(correctedPredictions[i] + firstPrediction)`)
  }

  lines.push(`}`)
  lines.push(`return stacked(updatedSlots, axis: 0)`)

  return lines.join("\n")
}
