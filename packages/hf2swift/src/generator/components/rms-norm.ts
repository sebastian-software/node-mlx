/**
 * RMSNorm component generator
 */

import type { ModelFeatures } from "../features.js"

export function generateRmsNorm(modelName: string, features: ModelFeatures): string {
  if (features.rmsNormStyle === "gemma") {
    return `// MARK: - RMS Norm (Gemma style - uses 1+weight scaling)

class ${modelName}RMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        // Initialize to zeros - will be (1 + weight) in forward
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma uses (1 + weight) scaling
        return MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}`
  }

  // Standard RMSNorm
  return `// MARK: - RMS Norm

class ${modelName}RMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}`
}
