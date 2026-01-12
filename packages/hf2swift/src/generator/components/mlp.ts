/**
 * MLP component generator
 *
 * Supports:
 * - Standard MLP with separate gate/up projections
 * - Fused gate_up_proj (Phi3, Phi4)
 * - Per-layer intermediate sizes (Gemma3n)
 * - Sparse activation with gelu_topk (Gemma3n)
 * - Mixture of Experts MLP (GPT-OSS)
 *
 * Note: Output is not formatted - SwiftFormat handles that.
 */

import type { ModelFeatures } from "../features.js"

export function generateMlp(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const activation =
    features.activation === "geluApproximate"
      ? "geluApproximate"
      : features.activation === "silu"
        ? "silu"
        : "gelu"

  if (features.hasMoE) {
    return generateMoEMlp(modelName, configClass, features)
  }

  if (features.hasFusedGateUp) {
    return generateFusedGateUpMlp(modelName, configClass, activation)
  }

  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing -- logical OR for booleans
  const needsLayerIdx = features.hasPerLayerIntermediateSize || features.hasSparseActivation
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int = 0" : ""
  const intermediateSizeExpr = features.hasPerLayerIntermediateSize
    ? "config.intermediateSize(forLayer: layerIdx)"
    : "config.intermediateSize"

  if (features.hasSparseActivation) {
    return generateMlpWithSparseActivation(modelName, configClass, activation, intermediateSizeExpr)
  }

  return `
// MARK: - MLP

class ${modelName}MLP: Module {
@ModuleInfo(key: "gate_proj") var gateProj: Linear
@ModuleInfo(key: "up_proj") var upProj: Linear
@ModuleInfo(key: "down_proj") var downProj: Linear

init(_ config: ${configClass}${layerIdxParam}) {
let intermediateSize = ${intermediateSizeExpr}
let mlpBias = config.mlpBias
self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: mlpBias)
self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: mlpBias)
self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: mlpBias)
}

func callAsFunction(_ x: MLXArray) -> MLXArray {
return downProj(${activation}(gateProj(x)) * upProj(x))
}
}
`
}

function generateFusedGateUpMlp(
  modelName: string,
  configClass: string,
  activation: string
): string {
  return `
// MARK: - MLP

class ${modelName}MLP: Module {
@ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
@ModuleInfo(key: "down_proj") var downProj: Linear

init(_ config: ${configClass}) {
let intermediateSize = config.intermediateSize
self._gateUpProj.wrappedValue = Linear(config.hiddenSize, 2 * intermediateSize, bias: false)
self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
}

func callAsFunction(_ x: MLXArray) -> MLXArray {
let gateUp = gateUpProj(x)
let chunks = split(gateUp, parts: 2, axis: -1)
let gate = chunks[0]
let up = chunks[1]
return downProj(${activation}(gate) * up)
}
}
`
}

function generateMlpWithSparseActivation(
  modelName: string,
  configClass: string,
  activation: string,
  intermediateSizeExpr: string
): string {
  return `
// MARK: - MLP

class ${modelName}MLP: Module {
@ModuleInfo(key: "gate_proj") var gateProj: Linear
@ModuleInfo(key: "up_proj") var upProj: Linear
@ModuleInfo(key: "down_proj") var downProj: Linear

let activationSparsity: Float
let stdMultiplier: Float?

init(_ config: ${configClass}, layerIdx: Int = 0) {
let intermediateSize = ${intermediateSizeExpr}
_gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
_upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
_downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)

// Get activation sparsity for this layer
if layerIdx < config.activationSparsityPattern.count {
self.activationSparsity = config.activationSparsityPattern[layerIdx]
} else {
self.activationSparsity = 0.0
}

// Precompute std multiplier for gelu_topk if sparsity > 0
if activationSparsity > 0 {
// sqrt(2) * erfinv(2 * sparsity - 1)
self.stdMultiplier = Float(sqrt(2.0)) * Self.erfinv(2.0 * activationSparsity - 1.0)
} else {
self.stdMultiplier = nil
}
}

/// Approximate inverse error function
private static func erfinv(_ x: Float) -> Float {
let a: Float = 0.147
let sign: Float = x < 0 ? -1 : 1
let x2 = x * x
let lnTerm = log(1 - x2)
let term1 = 2 / (Float.pi * a) + lnTerm / 2
let term2 = lnTerm / a
return sign * sqrt(sqrt(term1 * term1 - term2) - term1)
}

func callAsFunction(_ x: MLXArray) -> MLXArray {
let gateOutput = gateProj(x)
let activations: MLXArray

if let stdMult = stdMultiplier, activationSparsity > 0 {
// gelu_topk: sparse activation
let inputMean = mean(gateOutput, axis: -1, keepDims: true)
let inputStd = sqrt(mean((gateOutput - inputMean).pow(2), axis: -1, keepDims: true))
let cutoffX = inputMean + inputStd * stdMult
activations = ${activation}(maximum(MLXArray(Float(0)), gateOutput - cutoffX))
} else {
activations = ${activation}(gateOutput)
}

return downProj(activations * upProj(x))
}
}
`
}

function generateMoEMlp(modelName: string, configClass: string, features: ModelFeatures): string {
  // Determine which SwitchGLU variant to use
  // GPT-OSS uses SwiGLU activation, others might use standard GELU
  const expertsClass = features.useCustomSwiGLU ? "SwiGLUSwitchGLU" : "SwitchGLU"

  return `
// MARK: - MoE MLP

/// Mixture of Experts MLP with router and experts
/// Uses vendored SwitchLayers from mlx-swift-lm
class ${modelName}MLP: Module {
@ModuleInfo(key: "experts") var experts: ${expertsClass}
@ModuleInfo(key: "router") var router: Linear

let hiddenSize: Int
let numLocalExperts: Int
let numExpertsPerTok: Int

init(_ config: ${configClass}) {
hiddenSize = config.hiddenSize
numLocalExperts = config.numLocalExperts
numExpertsPerTok = config.numExpertsPerTok

_experts.wrappedValue = ${expertsClass}(
inputDims: config.hiddenSize,
hiddenDims: config.intermediateSize,
numExperts: config.numLocalExperts,
bias: config.mlpBias
)
_router.wrappedValue = Linear(config.hiddenSize, config.numLocalExperts, bias: config.mlpBias)
}

func callAsFunction(_ x: MLXArray) -> MLXArray {
let g = router(x)
let (expertScores, indices) = mlxTopK(g, k: numExpertsPerTok, axis: -1)
let expertWeights = softmax(expertScores, axis: -1, precise: true)

var output = self.experts(x, indices: indices)

output = output * expandedDimensions(expertWeights, axis: -1)
return output.sum(axis: -2)
}
}
`
}
