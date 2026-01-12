/**
 * Attention component generator
 *
 * Supports:
 * - Standard attention with separate q/k/v projections
 * - Fused QKV projection (Phi3, Phi4)
 * - Q/K norms (Gemma3, Gemma3n)
 * - V norm / RMSNoScale (Gemma3n)
 * - KV-cache sharing (Gemma3n)
 * - Sliding window attention
 * - Custom attention scale
 *
 * Note: Output is not formatted - SwiftFormat handles that.
 */

import type { ModelFeatures } from "../features.js"

export function generateAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  if (features.hasFusedQKV) {
    return generateFusedQKVAttention(modelName, configClass, features)
  }
  return generateStandardAttention(modelName, configClass, features)
}

/**
 * Generate attention with fused QKV projection (Phi3, Phi4 style)
 *
 * Uses the shared FusedQKVAttention<C> generic class.
 * Only generates a typealias and protocol conformance.
 */
function generateFusedQKVAttention(
  modelName: string,
  configClass: string,
  _features: ModelFeatures
): string {
  return `
// MARK: - Attention

/// Protocol conformance for shared FusedQKVAttention
extension ${configClass}: AttentionConfiguration {}

/// Fused QKV attention - uses shared implementation
typealias ${modelName}Attention = FusedQKVAttention<${configClass}>
`
}

/**
 * Generate standard attention with separate q/k/v projections
 */
function generateStandardAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Attention scale
  const scaleExpr =
    features.attentionScale !== undefined
      ? String(features.attentionScale)
      : "1.0 / sqrt(Float(headDim))"

  // Layer index parameter - need || for boolean OR (not nullish coalescing)
  /* eslint-disable @typescript-eslint/prefer-nullish-coalescing */
  const needsLayerIdx =
    features.useSlidingWindow ||
    features.hasKVSharing ||
    features.hasMoE ||
    features.hasNoRopeLayers
  /* eslint-enable @typescript-eslint/prefer-nullish-coalescing */
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int" : ""

  // Build declarations
  const declarations = buildDeclarations(features, normType)
  const initializations = buildInitializations(features, normType, configClass, scaleExpr)
  const forwardBody = buildForwardBody(features)

  return `
// MARK: - Attention

class ${modelName}Attention: Module {
@ModuleInfo(key: "q_proj") var qProj: Linear
@ModuleInfo(key: "k_proj") var kProj: Linear
@ModuleInfo(key: "v_proj") var vProj: Linear
@ModuleInfo(key: "o_proj") var oProj: Linear
${declarations}

let numHeads: Int
let numKVHeads: Int
let headDim: Int
let scale: Float
${buildRopeDecl(features)}
${features.hasKVSharing ? "let isKVSharedLayer: Bool" : ""}

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

function buildDeclarations(features: ModelFeatures, normType: string): string {
  const decls: string[] = []

  if (features.hasQKNorms) {
    decls.push(`@ModuleInfo(key: "q_norm") var qNorm: ${normType}`)
    decls.push(`@ModuleInfo(key: "k_norm") var kNorm: ${normType}`)
  }
  if (features.hasVNorm) {
    decls.push(`@ModuleInfo(key: "v_norm") var vNorm: RMSNoScale`)
  }
  if (features.hasAttentionSinks) {
    decls.push(`@ModuleInfo(key: "sinks") var sinks: MLXArray`)
  }

  return decls.join("\n")
}

function buildRopeDecl(features: ModelFeatures): string {
  if (features.useSlidingWindow) {
    return `let rope: RoPE
let isSliding: Bool`
  }
  if (features.hasNoRopeLayers) {
    return `let rope: RoPE
let skipRope: Bool`
  }
  return "let rope: RoPE"
}

function buildInitializations(
  features: ModelFeatures,
  normType: string,
  configClass: string,
  scaleExpr: string
): string {
  const lines: string[] = []

  // Basic properties
  lines.push(`self.numHeads = config.numAttentionHeads`)
  lines.push(`self.numKVHeads = config.numKeyValueHeads`)
  lines.push(`self.headDim = config.headDim`)
  lines.push(`self.scale = ${scaleExpr}`)
  lines.push(``)
  lines.push(`let qDim = numHeads * headDim`)
  lines.push(`let kvDim = numKVHeads * headDim`)
  lines.push(`let attnBias = config.attentionBias`)
  lines.push(``)
  lines.push(`self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: attnBias)`)
  lines.push(`self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)`)
  lines.push(`self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)`)
  lines.push(`self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: attnBias)`)

  // Q/K/V norms
  if (features.hasQKNorms) {
    lines.push(
      `self._qNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)`
    )
    lines.push(
      `self._kNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)`
    )
  }
  if (features.hasVNorm) {
    lines.push(`self._vNorm.wrappedValue = RMSNoScale(eps: config.rmsNormEps)`)
  }
  if (features.hasAttentionSinks) {
    lines.push(`self._sinks.wrappedValue = MLXArray.zeros([numHeads])`)
  }

  // RoPE initialization
  const traditionalRope = features.useTraditionalRope ? "true" : "false"
  if (features.useSlidingWindow) {
    lines.push(`self.isSliding = !config.isGlobalLayer(layerIdx)`)
    const ropeBase = features.hasLocalRopeTheta
      ? "isSliding ? config.ropeLocalBaseFreq : config.ropeTheta"
      : "config.ropeTheta"
    lines.push(`let ropeBase = ${ropeBase}`)
    lines.push(
      `self.rope = RoPE(dimensions: headDim, traditional: ${traditionalRope}, base: ropeBase)`
    )
  } else if (features.hasNoRopeLayers) {
    lines.push(`self.skipRope = config.shouldSkipRope(layerIdx)`)
    lines.push(
      `self.rope = RoPE(dimensions: headDim, traditional: ${traditionalRope}, base: config.ropeTheta)`
    )
  } else {
    lines.push(
      `self.rope = RoPE(dimensions: headDim, traditional: ${traditionalRope}, base: config.ropeTheta)`
    )
  }

  // KV sharing
  if (features.hasKVSharing) {
    lines.push(`self.isKVSharedLayer = config.isKVSharedLayer(layerIdx)`)
  }

  return lines.join("\n")
}

function buildForwardBody(features: ModelFeatures): string {
  const lines: string[] = []

  lines.push(`let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))`)
  lines.push(``)
  lines.push(`var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])`)

  if (features.hasQKNorms) {
    lines.push(`queries = qNorm(queries)`)
  }

  if (features.hasKVSharing) {
    // KV-sharing path
    lines.push(`queries = queries.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`var keys: MLXArray`)
    lines.push(`var values: MLXArray`)
    lines.push(`var offset: Int`)
    lines.push(``)
    lines.push(`if isKVSharedLayer, let c = cache, let state = c.state {`)
    lines.push(`// For KV-shared layers, retrieve KV from the designated cache`)
    lines.push(`keys = state.keys`)
    lines.push(`values = state.values`)
    lines.push(`offset = c.offset`)
    lines.push(`} else {`)
    lines.push(`// Compute KV for this layer`)
    lines.push(`offset = cache?.offset ?? 0`)
    lines.push(`keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasQKNorms) {
      lines.push(`keys = kNorm(keys)`)
    }
    lines.push(`keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`keys = rope(keys, offset: offset)`)
    lines.push(`values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasVNorm) {
      lines.push(`values = vNorm(values)`)
    }
    lines.push(`values = values.transposed(0, 2, 1, 3)`)
    lines.push(`if let c = cache {`)
    lines.push(`(keys, values) = c.update(keys: keys, values: values)`)
    lines.push(`}`)
    lines.push(`}`)
    lines.push(`queries = rope(queries, offset: offset)`)
  } else {
    // Standard path
    lines.push(`var keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    lines.push(`var values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)

    if (features.hasQKNorms) {
      lines.push(`keys = kNorm(keys)`)
    }
    if (features.hasVNorm) {
      lines.push(`values = vNorm(values)`)
    }

    lines.push(``)
    lines.push(`// Transpose for attention: [B, heads, L, headDim]`)
    lines.push(`queries = queries.transposed(0, 2, 1, 3)`)
    lines.push(`keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`values = values.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`// Apply RoPE with cache offset`)
    lines.push(`let offset = cache?.offset ?? 0`)

    if (features.hasNoRopeLayers) {
      lines.push(`if !skipRope {`)
      lines.push(`queries = rope(queries, offset: offset)`)
      lines.push(`keys = rope(keys, offset: offset)`)
      lines.push(`}`)
    } else {
      lines.push(`queries = rope(queries, offset: offset)`)
      lines.push(`keys = rope(keys, offset: offset)`)
    }

    lines.push(``)
    lines.push(`// Update cache`)
    lines.push(`if let c = cache {`)
    lines.push(`(keys, values) = c.update(keys: keys, values: values)`)
    lines.push(`}`)
  }

  // Attention computation (common for both paths)
  lines.push(``)
  lines.push(`// Attention using MLXFast (handles GQA automatically)`)
  lines.push(`let output = MLXFast.scaledDotProductAttention(`)
  lines.push(`queries: queries,`)
  lines.push(`keys: keys,`)
  lines.push(`values: values,`)
  lines.push(`scale: scale,`)
  lines.push(`mask: mask`)
  lines.push(`)`)
  lines.push(``)
  lines.push(`// Reshape back: [B, heads, L, headDim] -> [B, L, hidden]`)
  lines.push(`let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])`)
  lines.push(`return oProj(outputReshaped)`)

  return lines.join("\n")
}
