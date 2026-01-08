/**
 * DecoderLayer component generator
 */

import type { ModelFeatures } from "../features.js"

export function generateDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Extra norms for Gemma-style (4 norms per layer)
  const extraNormDecl =
    features.normsPerLayer === 4
      ? `
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: ${normType}
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: ${normType}`
      : ""

  const extraNormInit =
    features.normsPerLayer === 4
      ? `
        self._preFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
      : ""

  // Layer index for sliding window
  const layerIdxParam = features.useSlidingWindow ? ", layerIdx: Int" : ", layerIdx: Int = 0"
  const attnInit = features.useSlidingWindow
    ? `${modelName}Attention(config, layerIdx: layerIdx)`
    : `${modelName}Attention(config)`

  // Forward pass body
  let forwardBody: string
  if (features.normsPerLayer === 4) {
    // Gemma-style with 4 norms and post-norms before residual
    forwardBody = `
        // 1. Pre-norm + Self-attention
        let normed = inputLayernorm(hiddenStates)
        let attnOut = selfAttn(normed, mask: mask, cache: &cache)
        let attnNormed = postAttentionLayernorm(attnOut)
        var h = ${features.useClipResidual ? "clipResidual(hiddenStates, attnNormed)" : "hiddenStates + attnNormed"}

        // 2. Pre-norm + MLP
        let mlpIn = preFeedforwardLayernorm(h)
        let mlpOut = mlp(mlpIn)
        let mlpNormed = postFeedforwardLayernorm(mlpOut)
        h = ${features.useClipResidual ? "clipResidual(h, mlpNormed)" : "h + mlpNormed"}

        return h`
  } else {
    // Standard 2-norm style
    forwardBody = `
        // 1. Pre-norm + Self-attention
        let normed = inputLayernorm(hiddenStates)
        let attnOut = selfAttn(normed, mask: mask, cache: &cache)
        var h = hiddenStates + attnOut

        // 2. Pre-norm + MLP
        let mlpNormed = postAttentionLayernorm(h)
        let mlpOut = mlp(mlpNormed)
        h = h + mlpOut

        return h`
  }

  return `// MARK: - Decoder Layer

class ${modelName}DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ${modelName}Attention
    @ModuleInfo(key: "mlp") var mlp: ${modelName}MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: ${normType}
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: ${normType}${extraNormDecl}

    init(_ config: ${configClass}${layerIdxParam}) {
        self._selfAttn.wrappedValue = ${attnInit}
        self._mlp.wrappedValue = ${modelName}MLP(config)
        self._inputLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)${extraNormInit}
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?
    ) -> MLXArray {${forwardBody}
    }
}`
}
