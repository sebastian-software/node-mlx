/**
 * SmolLM model family definition
 *
 * SmolLM3 has no-RoPE layers and weight tying.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const smolLm3: ModelDefinition = {
  name: "SmolLM3",

  matches: (modelType) => {
    const lower = modelType.toLowerCase()
    return lower.includes("smollm3") || lower.includes("smollm-3") || lower.includes("smollm_3")
  },

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu",
    hasNoRopeLayers: true
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    ropeTheta: 5000000,
    hasWeightTying: true
  }
}
