/**
 * Phi model family definition
 *
 * Includes: Phi-3, Phi-3.5, Phi-4
 * Features: Fused QKV projection, fused gate_up_proj.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const phi: ModelDefinition = {
  name: "Phi",

  matches: (modelType) => modelType.toLowerCase().includes("phi"),

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu",
    hasFusedQKV: true,
    hasFusedGateUp: true
  },

  configDefaults: {
    ...DEFAULT_CONFIG
  }
}
