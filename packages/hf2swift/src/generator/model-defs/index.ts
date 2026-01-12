/**
 * Model definitions registry
 *
 * Central registry of all supported model families.
 * Order matters - more specific matchers should come first.
 */

// Re-export types
export type {
  ArchitecturalFeatures,
  ConfigValues,
  ModelFeatures,
  ModelDefinition
} from "./types.js"

export { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG } from "./types.js"

// Import model definitions
import { gemma3n, gemma3, isGemma3n } from "./gemma.js"
import { qwen3 } from "./qwen.js"
import { mistral3, mistral } from "./mistral.js"
import { phi } from "./phi.js"
import { llama } from "./llama.js"
import { gptOss } from "./gpt-oss.js"
import { smolLm3 } from "./smollm.js"

import type { ModelDefinition, ArchitecturalFeatures, ConfigValues } from "./types.js"
import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG } from "./types.js"

// Re-export isGemma3n for external use
export { isGemma3n }

/**
 * Model registry - order matters!
 * More specific matchers should come first.
 */
const MODEL_REGISTRY: ModelDefinition[] = [
  // Gemma (3n before 3)
  gemma3n,
  gemma3,

  // Qwen (only Qwen3 supported)
  qwen3,

  // Mistral (3 before base)
  mistral3,
  mistral,

  // Others (no ordering needed)
  phi,
  gptOss,
  smolLm3,
  llama // Llama is generic, keep last among specific models
]

/**
 * Find matching model definition
 */
export function findModelDefinition(modelType: string): ModelDefinition | undefined {
  return MODEL_REGISTRY.find((def) => def.matches(modelType))
}

/**
 * Get architectural features for a model type
 */
export function getArchitecturalFeatures(modelType: string): ArchitecturalFeatures {
  const def = findModelDefinition(modelType)
  return def?.architectural ?? DEFAULT_ARCHITECTURAL
}

/**
 * Get default config values for a model type
 */
export function getDefaultConfigValues(modelType: string): ConfigValues {
  const def = findModelDefinition(modelType)
  return def?.configDefaults ?? DEFAULT_CONFIG
}

/**
 * Get list of all supported model names
 */
export function getSupportedModels(): string[] {
  return MODEL_REGISTRY.map((def) => def.name)
}
