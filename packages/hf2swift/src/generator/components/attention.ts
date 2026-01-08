/**
 * Attention component generator
 */

import type { ModelFeatures } from "../features.js"

export function generateAttention(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Build Q/K norm declarations if needed
  const qkNormDecl = features.hasQKNorms
    ? `
    @ModuleInfo(key: "q_norm") var qNorm: ${normType}
    @ModuleInfo(key: "k_norm") var kNorm: ${normType}`
    : ""

  const qkNormInit = features.hasQKNorms
    ? `
        self._qNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = ${normType}(dimensions: headDim, eps: config.rmsNormEps)`
    : ""

  const qkNormApply = features.hasQKNorms
    ? `
        // Apply RMSNorm to Q and K
        queries = qNorm(queries)
        keys = kNorm(keys)`
    : ""

  // RoPE initialization - use provider for sliding window support
  const ropeDecl = features.useSlidingWindow
    ? `let rope: any RoPEProvider
    let isGlobal: Bool
    let slidingWindow: Int?`
    : `let rope: RoPE`

  // Build RoPE initialization based on features
  let ropeInit: string
  if (features.useSlidingWindow) {
    const ropeBaseExpr = features.hasLocalRopeTheta
      ? "isGlobal ? config.ropeTheta : config.ropeLocalTheta"
      : "config.ropeTheta"
    ropeInit = `
        self.isGlobal = config.isGlobalLayer(layerIdx)
        self.slidingWindow = isGlobal ? nil : config.slidingWindow

        // Different RoPE theta for sliding vs global layers
        let ropeBase = ${ropeBaseExpr}

        // Use initializeRope to handle linear scaling for larger models
        self.rope = initializeRope(
            dims: headDim,
            base: ropeBase,
            traditional: false,
            scalingConfig: isGlobal ? config.ropeScaling : nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )`
  } else {
    ropeInit = `
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)`
  }

  // Layer index parameter needed for sliding window
  const layerIdxParam = features.useSlidingWindow ? ", layerIdx: Int" : ""

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
    ${ropeDecl}

    init(_ config: ${configClass}${layerIdxParam}) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
${qkNormInit}
${ropeInit}
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
${qkNormApply}

        // Transpose for attention: [B, heads, L, headDim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Apply RoPE with cache offset
        let offset = cache?.offset ?? 0
        queries = rope.apply(queries, offset: offset)
        keys = rope.apply(keys, offset: offset)

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
