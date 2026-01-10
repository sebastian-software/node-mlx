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
 */

import type { ModelFeatures } from "../features.js"

export function generateAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  // Use fused QKV generator for models that need it
  if (features.hasFusedQKV) {
    return generateFusedQKVAttention(modelName, configClass, features)
  }

  return generateStandardAttention(modelName, configClass, features)
}

/**
 * Generate attention with fused QKV projection (Phi3, Phi4 style)
 */
function generateFusedQKVAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  // Attention scale - some models use 1.0 instead of 1/sqrt(headDim)
  const scaleExpr =
    features.attentionScale !== undefined
      ? String(features.attentionScale)
      : "1.0 / sqrt(Float(headDim))"

  return `// MARK: - Attention

class ${modelName}Attention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ config: ${configClass}) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = ${scaleExpr}

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        let opSize = qDim + 2 * kvDim

        self._qkvProj.wrappedValue = Linear(config.hiddenSize, opSize, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        let qkv = qkvProj(hiddenStates)
        let queryPos = numHeads * headDim
        let kvPos = queryPos + numKVHeads * headDim

        var queries = qkv[0..., 0..., ..<queryPos].reshaped([B, L, numHeads, headDim])
        var keys = qkv[0..., 0..., queryPos..<kvPos].reshaped([B, L, numKVHeads, headDim])
        var values = qkv[0..., 0..., kvPos...].reshaped([B, L, numKVHeads, headDim])

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
}`
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

  // Add attention sinks for models that need it
  if (features.hasAttentionSinks) {
    qkNormDecl += `
    @ModuleInfo(key: "sinks") var sinks: MLXArray`
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

  if (features.hasAttentionSinks) {
    normInit += `
        self._sinks.wrappedValue = MLXArray.zeros([numHeads])`
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
  } else if (features.hasNoRopeLayers) {
    // SmolLM3: Some layers skip RoPE entirely
    ropeDecl = `let rope: RoPE
    let skipRope: Bool`

    ropeInit = `
        self.skipRope = config.shouldSkipRope(layerIdx)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)`
  } else {
    ropeDecl = `let rope: RoPE`
    ropeInit = `
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)`
  }

  // Attention scale - some models use 1.0 instead of 1/sqrt(headDim)
  const scaleExpr =
    features.attentionScale !== undefined
      ? String(features.attentionScale)
      : "1.0 / sqrt(Float(headDim))"

  // Layer index parameter needed for sliding window, KV sharing, MoE, or no-rope layers
  const needsLayerIdx =
    features.useSlidingWindow ||
    features.hasKVSharing ||
    features.hasMoE ||
    features.hasNoRopeLayers
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int" : ""

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
        let attnBias = config.attentionBias

        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: attnBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: attnBias)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: attnBias)
${normInit}${ropeInit}${kvSharingInit}
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
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

  // K/V handling depends on KV-sharing
  if (features.hasKVSharing) {
    // For KV-sharing, transpose queries early (before entering the branch)
    lines.push(`        queries = queries.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`        var keys: MLXArray`)
    lines.push(`        var values: MLXArray`)
    lines.push(`        var offset: Int`)
    lines.push(``)
    lines.push(`        if isKVSharedLayer, let c = cache, let state = c.state {`)
    lines.push(
      `            // For KV-shared layers, retrieve KV from the designated cache (via cache mapping)`
    )
    lines.push(`            keys = state.keys`)
    lines.push(`            values = state.values`)
    lines.push(`            offset = c.offset`)
    lines.push(`        } else {`)
    lines.push(`            // Compute KV for this layer`)
    lines.push(`            offset = cache?.offset ?? 0`)
    lines.push(``)
    lines.push(`            keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])`)
    if (features.hasQKNorms) {
      lines.push(`            keys = kNorm(keys)`)
    }
    lines.push(`            keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`            keys = rope.apply(keys, offset: offset)`)
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
    lines.push(``)
    lines.push(`        queries = rope.apply(queries, offset: offset)`)
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
    lines.push(`        queries = queries.transposed(0, 2, 1, 3)`)
    lines.push(`        keys = keys.transposed(0, 2, 1, 3)`)
    lines.push(`        values = values.transposed(0, 2, 1, 3)`)
    lines.push(``)
    lines.push(`        // Apply RoPE with cache offset`)
    lines.push(`        let offset = cache?.offset ?? 0`)
    if (features.hasNoRopeLayers) {
      lines.push(`        if !skipRope {`)
      lines.push(`            queries = rope.apply(queries, offset: offset)`)
      lines.push(`            keys = rope.apply(keys, offset: offset)`)
      lines.push(`        }`)
    } else {
      lines.push(`        queries = rope.apply(queries, offset: offset)`)
      lines.push(`        keys = rope.apply(keys, offset: offset)`)
    }
    lines.push(``)
    lines.push(`        // Update cache`)
    lines.push(`        if let c = cache {`)
    lines.push(`            (keys, values) = c.update(keys: keys, values: values)`)
    lines.push(`        }`)
  }

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
