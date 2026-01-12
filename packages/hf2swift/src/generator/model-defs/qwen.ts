/**
 * Qwen model family definition
 *
 * Qwen3 adds Q/K norms and uses weight tying.
 * Note: Qwen2/2.5 support was removed - use Qwen3 instead.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const qwen3: ModelDefinition = {
  name: "Qwen3",

  // Matches any "qwen" model (Qwen3 is the only supported version now)
  matches: (modelType) => modelType.toLowerCase().includes("qwen"),

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
