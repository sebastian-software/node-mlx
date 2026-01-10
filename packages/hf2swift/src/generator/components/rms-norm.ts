/**
 * RMSNorm component generator
 *
 * Note: Output is not formatted - SwiftFormat handles that.
 */

import type { ModelFeatures } from "../features.js"

export function generateRmsNorm(modelName: string, features: ModelFeatures): string {
  const parts: string[] = []

  // Add RMSNoScale if needed for v_norm
  if (features.hasVNorm) {
    parts.push(`
// MARK: - RMS Norm Variants

/// RMSNorm without learnable scale weights - used for value normalization
class RMSNoScale: Module {
let eps: Float

init(eps: Float = 1e-5) {
self.eps = eps
}

func callAsFunction(_ x: MLXArray) -> MLXArray {
let variance = mean(x.pow(2), axis: -1, keepDims: true)
return x * rsqrt(variance + eps)
}
}
`)
  } else {
    parts.push("\n// MARK: - RMS Norm\n")
  }

  if (features.rmsNormStyle === "gemma") {
    parts.push(`
/// RMSNorm with Gemma-style (1 + weight) scaling
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
}
`)
  } else {
    parts.push(`
/// Standard RMSNorm
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
}
`)
  }

  return parts.join("")
}
