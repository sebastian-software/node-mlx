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
 */

import type { ModelFeatures } from "../features.js"

export function generateDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  // Models with AltUp have a completely different decoder layer structure
  if (features.hasAltUp) {
    return generateAltUpDecoderLayer(modelName, configClass, features)
  }

  return generateStandardDecoderLayer(modelName, configClass, features)
}

function generateStandardDecoderLayer(
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

function generateAltUpDecoderLayer(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const normType = `${modelName}RMSNorm`

  // Laurel block declaration
  const laurelDecl = features.hasLaurel
    ? `
    @ModuleInfo(key: "laurel") var laurel: ${modelName}LaurelBlock`
    : ""

  const laurelInit = features.hasLaurel
    ? `
        _laurel.wrappedValue = ${modelName}LaurelBlock(config)`
    : ""

  // Per-layer input declarations
  const perLayerDecl = features.hasPerLayerInputs
    ? `
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: ${normType}`
    : ""

  const perLayerInit = features.hasPerLayerInputs
    ? `
        _perLayerInputGate.wrappedValue = Linear(config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
        _perLayerProjection.wrappedValue = Linear(config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
        _postPerLayerInputNorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)`
    : ""

  // Function signature with per-layer input and shared KV
  const perLayerParam = features.hasPerLayerInputs
    ? `,
        perLayerInput: MLXArray`
    : ""
  const sharedKVParam = features.hasKVSharing
    ? `,
        sharedKV: (keys: MLXArray, values: MLXArray, offset: Int)? = nil`
    : ""

  // Attention call with shared KV
  const attnCall = features.hasKVSharing
    ? `selfAttn(activePredictionNormed, mask: mask, cache: &cache, sharedKV: sharedKV)`
    : `selfAttn(activePredictionNormed, mask: mask, cache: &cache)`

  return `// MARK: - Decoder Layer (with AltUp)

class ${modelName}DecoderLayer: Module {
    let layerIdx: Int
    let activeIdx: Int
    let altupCorrectScale: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: ${modelName}Attention
    @ModuleInfo(key: "mlp") var mlp: ${modelName}MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: ${normType}
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: ${normType}
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: ${normType}
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: ${normType}
    @ModuleInfo(key: "altup") var altup: ${modelName}AltUp${laurelDecl}${perLayerDecl}

    init(_ config: ${configClass}, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.activeIdx = config.altupActiveIdx
        self.altupCorrectScale = config.altupCorrectScale

        _selfAttn.wrappedValue = ${modelName}Attention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = ${modelName}MLP(config, layerIdx: layerIdx)
        _inputLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = ${normType}(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _altup.wrappedValue = ${modelName}AltUp(config)${laurelInit}${perLayerInit}
    }

    /// Forward pass with AltUp predict/correct
    /// - Parameters:
    ///   - hiddenStates: [numInputs, batch, seq, hidden]
    ///   - perLayerInput: [batch, seq, hiddenPerLayerInput] (if hasPerLayerInputs)
    ///   - mask: attention mask
    ///   - cache: KV cache
    ///   - sharedKV: Optional shared KV for KV-shared layers (if hasKVSharing)
    /// - Returns: [numInputs, batch, seq, hidden]
    func callAsFunction(
        _ hiddenStates: MLXArray${perLayerParam},
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout KVCache?${sharedKVParam}
    ) -> MLXArray {
        // 1. AltUp predict
        let predictions = altup.predict(hiddenStates)
        let activePrediction = predictions[activeIdx]

        // 2. Input layernorm
        let activePredictionNormed = inputLayernorm(activePrediction)

        // 3. Laurel (adds residual internally: x + laurel_output)
        ${features.hasLaurel ? "let laurelOutput = laurel(activePredictionNormed)" : "let laurelOutput = activePredictionNormed"}

        // 4. Self attention
        var attn = ${attnCall}
        attn = postAttentionLayernorm(attn)

        // 5. Residual + scale with sqrt(2)
        let attnGated = activePrediction + attn
        let sqrtTwoInv = Float(pow(2.0, -0.5))
        let attnLaurel = (attnGated + laurelOutput) * sqrtTwoInv

        // 6. MLP
        let attnNorm = preFeedforwardLayernorm(attnLaurel)
        let attnFfw = mlp(attnNorm)
        let attnFfwNorm = postFeedforwardLayernorm(attnFfw)
        let attnFfwLaurelGated = attnLaurel + attnFfwNorm

        // 7. AltUp correct
        var correctedPredictions = altup.correct(predictions, activated: attnFfwLaurelGated)

        // 8. Scale corrected output if configured
        var firstPrediction = correctedPredictions[activeIdx]
        if altupCorrectScale {
            firstPrediction = altup.scaleCorrectOutput(firstPrediction)
        }

        ${features.hasPerLayerInputs ? generatePerLayerInputHandling() : ""}

        // Update all slots except the active one
        var updatedSlots: [MLXArray] = [correctedPredictions[0]]
        for i in 1..<correctedPredictions.dim(0) {
            ${features.hasPerLayerInputs ? "updatedSlots.append(correctedPredictions[i] + perLayerOut)" : "updatedSlots.append(correctedPredictions[i] + firstPrediction)"}
        }

        return stacked(updatedSlots, axis: 0)
    }
}`
}

function generatePerLayerInputHandling(): string {
  return `// 9. Per-layer input gate and projection
        var perLayerOut = perLayerInputGate(firstPrediction)
        perLayerOut = geluApproximate(perLayerOut)
        perLayerOut = perLayerOut * perLayerInput
        perLayerOut = perLayerProjection(perLayerOut)
        perLayerOut = postPerLayerInputNorm(perLayerOut)

`
}
