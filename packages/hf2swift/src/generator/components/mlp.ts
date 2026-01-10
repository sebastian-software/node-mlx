/**
 * MLP component generator
 *
 * Supports:
 * - Standard MLP with separate gate/up projections
 * - Fused gate_up_proj (Phi3, Phi4)
 * - Per-layer intermediate sizes (Gemma3n)
 * - Sparse activation with gelu_topk (Gemma3n)
 * - Mixture of Experts MLP (GPT-OSS)
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

  // Use MoE MLP for models that need it
  if (features.hasMoE) {
    return generateMoEMlp(modelName, configClass, features)
  }

  // Use fused gate_up_proj for models that need it
  if (features.hasFusedGateUp) {
    return generateFusedGateUpMlp(modelName, configClass, activation)
  }

  // Per-layer intermediate size support
  // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing -- logical OR for booleans
  const needsLayerIdx = features.hasPerLayerIntermediateSize || features.hasSparseActivation
  const layerIdxParam = needsLayerIdx ? ", layerIdx: Int = 0" : ""
  const intermediateSizeExpr = features.hasPerLayerIntermediateSize
    ? "config.intermediateSize(forLayer: layerIdx)"
    : "config.intermediateSize"

  // Sparse activation support
  if (features.hasSparseActivation) {
    return generateMlpWithSparseActivation(modelName, configClass, activation, intermediateSizeExpr)
  }

  return `// MARK: - MLP

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
}`
}

/**
 * Generate MLP with fused gate_up_proj (Phi3/Phi4 style)
 */
function generateFusedGateUpMlp(
  modelName: string,
  configClass: string,
  activation: string
): string {
  return `// MARK: - MLP

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
}`
}

function generateMlpWithSparseActivation(
  modelName: string,
  configClass: string,
  activation: string,
  intermediateSizeExpr: string
): string {
  return `// MARK: - MLP

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
}`
}

/**
 * Generate MoE MLP (GPT-OSS style with SwitchGLU experts)
 */
function generateMoEMlp(modelName: string, configClass: string, features: ModelFeatures): string {
  const useCustomSwiGLU = features.useCustomSwiGLU ?? false

  return `// MARK: - MoE MLP

/// Mixture of Experts MLP using shared MoEMLP infrastructure
class ${modelName}MLP: Module {
    @ModuleInfo(key: "router") var router: MoERouter
    @ModuleInfo(key: "experts") var experts: SwitchGLU

    let numExperts: Int
    let topK: Int

    init(_ config: ${configClass}) {
        self.numExperts = config.numLocalExperts
        self.topK = config.numExpertsPerTok

        _router.wrappedValue = MoERouter(
            hiddenSize: config.hiddenSize,
            numExperts: config.numLocalExperts,
            topK: config.numExpertsPerTok,
            bias: config.mlpBias
        )
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numLocalExperts,
            bias: config.mlpBias,
            useCustomSwiGLU: ${String(useCustomSwiGLU)}
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let batchSeq = shape.dropLast().reduce(1, *)
        let hidden = shape.last!

        // Flatten to [batch * seq, hidden]
        let xFlat = x.reshaped([batchSeq, hidden])

        // Get routing weights and expert indices
        let (weights, indices) = router(xFlat)

        // Get expert outputs [batch * seq, topK, hidden]
        let expertOutput = experts(xFlat, indices: indices)

        // Weighted sum of expert outputs
        let weightsExpanded = weights[.ellipsis, .newAxis]
        let weightedOutput = sum(expertOutput * weightsExpanded, axis: 1)

        // Reshape back to original shape
        return weightedOutput.reshaped(shape)
    }
}`
}
