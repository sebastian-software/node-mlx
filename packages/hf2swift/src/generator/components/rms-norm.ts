/**
 * RMSNorm component generator
 *
 * For standard models, uses the shared RMSNorm class via typealias.
 * For Gemma models, generates a custom RMSNorm with (1 + weight) scaling.
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
    // Gemma models use the ported GemmaRMSNorm with (1 + weight) scaling
    parts.push(`
/// Uses ported GemmaRMSNorm (1 + weight scaling)
typealias ${modelName}RMSNorm = GemmaRMSNorm
`)
  } else {
    // Standard models use the shared RMSNorm class
    parts.push(`
/// Uses shared RMSNorm implementation
typealias ${modelName}RMSNorm = RMSNorm
`)
  }

  return parts.join("")
}
