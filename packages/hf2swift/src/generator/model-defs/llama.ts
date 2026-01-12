/**
 * Llama model family definition
 *
 * Includes: Llama 2, Llama 3, Llama 3.1, Llama 3.2, etc.
 * Standard transformer architecture with SiLU activation.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const llama: ModelDefinition = {
  name: "Llama",

  matches: (modelType) => modelType.toLowerCase().includes("llama"),

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu"
  },

  configDefaults: {
    ...DEFAULT_CONFIG
  }
}
