/**
 * Mistral model family definition
 *
 * Includes: Mistral 7B, Mixtral, Mistral 3 (Ministral)
 * Mistral 3/Ministral adds YaRN RoPE and removes sliding window.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

/**
 * Check if model is Mistral 3 / Ministral 3
 */
function isMistral3(modelType: string): boolean {
  const lower = modelType.toLowerCase()
  return (
    lower.includes("mistral3") ||
    lower.includes("mistral-3") ||
    lower.includes("ministral3") ||
    lower.includes("ministral-3")
  )
}

export const mistral3: ModelDefinition = {
  name: "Mistral3",

  matches: isMistral3,

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu",
    hasYarnRope: true
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    ropeTheta: 1000000
  }
}

export const mistral: ModelDefinition = {
  name: "Mistral",

  // Matches "mistral" or "ministral" but not "mistral3/ministral3"
  matches: (modelType) => {
    const lower = modelType.toLowerCase()
    return (lower.includes("mistral") || lower.includes("ministral")) && !isMistral3(modelType)
  },

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu"
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    useSlidingWindow: true
  }
}
