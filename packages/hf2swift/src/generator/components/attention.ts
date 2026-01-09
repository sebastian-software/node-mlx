/**
 * Attention component generator
 *
 * Supports:
 * - Standard attention
 * - Q/K norms (Gemma3, Gemma3n)
 * - V norm / RMSNoScale (Gemma3n)
 * - KV-cache sharing (Gemma3n)
 * - Sliding window attention
 * - Custom attention scale
 */

import type { ModelFeatures } from "../features.js"

export function generateAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Build Q/K norm declarations if needed
  let qkNormDecl = ""
  if (features.hasQKNorms) {
    qkNormDecl = `
    @ModuleInfo(key: "q_norm") var qNorm: ${normType}
    @ModuleInfo(key: "k_norm") var kNorm: ${normType}`
  }

  // Add V norm for models that need it
  if (features.hasVNorm) {
    qkNormDecl += `
    @ModuleInfo(key: "v_norm") var vNorm: RMSNoScale`
  }

  // Q/K/V norm initialization
  let normInit = ""
  if (features.hasQKNorms) {
    normInit = `
        self._qNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)`
  }
  if (features.hasVNorm) {
    normInit += `
        self._vNorm.wrappedValue = RMSNoScale(eps: config.rmsNormEps)`
  }

  // KV-sharing properties for Gemma3n-style models
  const kvSharingDecl = features.hasKVSharing
    ? `
    let isKVSharedLayer: Bool`
    : ""

  const kvSharingInit = features.hasKVSharing
    ? `
        self.isKVSharedLayer = config.isKVSharedLayer(layerIdx)`
    : ""

  // RoPE initialization - use provider for sliding window support
  let ropeDecl: string
  let ropeInit: string

  if (features.useSlidingWindow) {
    ropeDecl = `let rope: RoPE
    let isSliding: Bool`

    const ropeBaseExpr = features.hasLocalRopeTheta
      ? "isSliding ? config.ropeLocalBaseFreq : config.ropeTheta"
      : "config.ropeTheta"

    ropeInit = `
        self.isSliding = !config.isGlobalLayer(layerIdx)
        let ropeBase = ${ropeBaseExpr}
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)`
  } else {
    ropeDecl = `let rope: RoPE`
    ropeInit = `
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)`
  }

  // Attention scale - some models use 1.0 instead of 1/sqrt(headDim)
  const scaleExpr =
    features.attentionScale !== undefined
      ? `${features.attentionScale}`
      : "1.0 / sqrt(Float(headDim))"

  // Layer index parameter needed for sliding window or KV sharing
  const needsLayerIdx = features.useSlidingWindow || features.hasKVSharing
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int" : ""

  // Shared KV parameter for KV-sharing models
  const sharedKVParam = features.hasKVSharing
    ? `,
        sharedKV: (keys: MLXArray, values: MLXArray, offset: Int)? = nil`
    : ""

  // Generate forward function body
  const forwardBody = generateForwardBody(features, normType)

  return `// MARK: - Attention

class ${modelName}Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear${qkNormDecl}

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    ${ropeDecl}${kvSharingDecl}

    init(_ config: ${configClass}${layerIdxParam}) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = ${scaleExpr}

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
${normInit}${ropeInit}${kvSharingInit}
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?${sharedKVParam}
    ) -> MLXArray {
${forwardBody}
    }
}`
}

function generateForwardBody(features: ModelFeatures, _normType: string): string {
  const lines: string[] = []

  lines.push(
    `        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))`
  )
  lines.push(``)

  // Q projection
  lines.push(`        var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])`)
  if (features.hasQKNorms) {
    lines.push(`        queries = qNorm(queries)`)
  }
  lines.push(`        queries = queries.transposed(0, 2, 1, 3)`)
  lines.push(``)

  // K/V handling depends on KV-sharing
  if (features.hasKVSharing) {
    lines.push(`        var keys: MLXArray`)
    lines.push(`        var values: MLXArray`)
    lines.push(`        var offset: Int`)
    lines.push(``)
    lines.push(`        if isKVSharedLayer, let shared = sharedKV {`)
    lines.push(`            // For KV-shared layers, use pre-computed KV from designated cache`)
    lines.push(`            keys = shared.keys`)
    lines.push(`            values = shared.values`)
    lines.push(`            offset = shared.offset`)
    lines.push(`        } else {`)
    lines.push(`            // Compute KV for this layer`)
    lines.push(`            offset = cache?.offset ?? 0`)
    lines.push(``)
    lines.push(`            keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasQKNorms) {
      lines.push(`            keys = kNorm(keys)`)
    }
    lines.push(`            keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`            keys = rope(keys, offset: offset)`)
    lines.push(``)
    lines.push(`            values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasVNorm) {
      lines.push(`            values = vNorm(values)`)
    }
    lines.push(`            values = values.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`            if let c = cache {`)
    lines.push(`                (keys, values) = c.update(keys: keys, values: values)`)
    lines.push(`            }`)
    lines.push(`        }`)
  } else {
    // Standard K/V computation
    lines.push(`        var keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    lines.push(`        var values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasQKNorms) {
      lines.push(`        keys = kNorm(keys)`)
    }
    if (features.hasVNorm) {
      lines.push(`        values = vNorm(values)`)
    }
    lines.push(``)
    lines.push(`        // Transpose for attention: [B, heads, L, headDim]`)
    lines.push(`        keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`        values = values.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`        // Apply RoPE with cache offset`)
    lines.push(`        let offset = cache?.offset ?? 0`)
    lines.push(`        keys = rope(keys, offset: offset)`)
    lines.push(``)
    lines.push(`        // Update cache`)
    lines.push(`        if let c = cache {`)
    lines.push(`            (keys, values) = c.update(keys: keys, values: values)`)
    lines.push(`        }`)
  }

  lines.push(``)
  lines.push(`        queries = rope(queries, offset: offset)`)
  lines.push(``)
  lines.push(`        // Attention using MLXFast (handles GQA automatically)`)
  lines.push(`        let output = MLXFast.scaledDotProductAttention(`)
  lines.push(`            queries: queries,`)
  lines.push(`            keys: keys,`)
  lines.push(`            values: values,`)
  lines.push(`            scale: scale,`)
  lines.push(`            mask: mask`)
  lines.push(`        )`)
  lines.push(``)
  lines.push(`        // Reshape back: [B, heads, L, headDim] -> [B, L, hidden]`)
  lines.push(`        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])`)
  lines.push(``)
  lines.push(`        return oProj(outputReshaped)`)

  return lines.join("\n")
}
