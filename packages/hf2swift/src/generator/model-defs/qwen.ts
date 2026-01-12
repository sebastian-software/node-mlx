/**
 * Qwen model family definition
 *
 * Includes: Qwen2, Qwen2.5, Qwen3
 * Qwen3 adds Q/K norms and uses weight tying.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const qwen3: ModelDefinition = {
  name: "Qwen3",

  matches: (modelType) => modelType.toLowerCase().includes("qwen3"),

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu",
    hasQKNorms: true
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    ropeTheta: 1000000,
    rmsNormEps: 1e-6,
    hasWeightTying: true
  }
}

export const qwen2: ModelDefinition = {
  name: "Qwen2",

  // Matches "qwen" but not "qwen3"
  matches: (modelType) => {
    const lower = modelType.toLowerCase()
    return lower.includes("qwen") && !lower.includes("qwen3")
  },

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu"
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    hasAttentionBias: true,
    rmsNormEps: 1e-6
  }
}
