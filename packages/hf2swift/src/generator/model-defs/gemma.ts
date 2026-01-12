/**
 * Gemma model family definition
 *
 * Includes: Gemma 3, Gemma 3n
 * Features: Gemma-style RMSNorm (1+weight), GELU approximate activation,
 * Q/K norms, 4 norms per layer, embedding scaling.
 *
 * Gemma 3n adds: AltUp, Laurel, KV-sharing, sparse activation, VLM support.
 */

import { DEFAULT_CONFIG, type ModelDefinition, type ArchitecturalFeatures } from "./types.js"

/**
 * Check if model is Gemma 3n (needs special handling)
 */
export function isGemma3n(modelType: string): boolean {
  const lower = modelType.toLowerCase()
  return lower.includes("gemma3n") || lower.includes("gemma-3n") || lower.includes("gemma_3n")
}

const gemmaBaseArchitectural: ArchitecturalFeatures = {
  rmsNormStyle: "gemma",
  activation: "geluApproximate",
  useClipResidual: true,
  useEmbeddingScale: true,
  hasQKNorms: true,
  normsPerLayer: 4
}

export const gemma3n: ModelDefinition = {
  name: "Gemma3n",

  matches: isGemma3n,

  architectural: {
    ...gemmaBaseArchitectural,
    rmsNormStyle: "standard", // Gemma3n uses standard RMSNorm
    useClipResidual: false,
    hasAltUp: true,
    hasLaurel: true,
    hasPerLayerInputs: true,
    hasKVSharing: true,
    hasPerLayerIntermediateSize: true,
    hasSparseActivation: true,
    hasVNorm: true,
    hasLogitSoftcapping: true,
    attentionScale: 1.0
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    useSlidingWindow: true,
    ropeTheta: 1000000,
    hasLocalRopeTheta: true,
    rmsNormEps: 1e-6,
    hasWeightTying: true
  }
}

export const gemma3: ModelDefinition = {
  name: "Gemma3",

  // Matches "gemma3" but not "gemma3n"
  matches: (modelType) => {
    const lower = modelType.toLowerCase()
    return (lower.includes("gemma3") || lower.includes("gemma-3")) && !isGemma3n(modelType)
  },

  architectural: gemmaBaseArchitectural,

  configDefaults: {
    ...DEFAULT_CONFIG,
    useSlidingWindow: true,
    ropeTheta: 1000000,
    hasLocalRopeTheta: true,
    rmsNormEps: 1e-6
  }
}
