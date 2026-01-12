/**
 * Model-specific feature flags for code generation
 *
 * Two-tier system:
 * 1. Architectural features - determined by model family (from model-defs/)
 * 2. Config values - read from config.json with model-specific defaults
 *
 * This separation ensures:
 * - Model-specific code paths are feature-driven, not name-driven
 * - Config values come from the source of truth (config.json)
 * - Reasonable defaults when config values are missing
 */

// Re-export types from model-defs
export type { ArchitecturalFeatures, ConfigValues, ModelFeatures } from "./model-defs/index.js"

// Re-export isGemma3n for backward compatibility
export { isGemma3n } from "./model-defs/index.js"

import {
  getArchitecturalFeatures,
  getDefaultConfigValues,
  type ConfigValues,
  type ModelFeatures
} from "./model-defs/index.js"

/**
 * Raw config.json structure (partial)
 */
interface ConfigJson {
  rope_theta?: number
  attention_bias?: boolean
  mlp_bias?: boolean
  rms_norm_eps?: number
  sliding_window?: number | null
  num_local_experts?: number
  num_experts_per_tok?: number
  tie_word_embeddings?: boolean
  rope_local_base_freq?: number
}

/**
 * Extract config values from config.json
 * Returns only values that are explicitly set
 */
function extractConfigValues(configJson: ConfigJson): Partial<ConfigValues> {
  const values: Partial<ConfigValues> = {}

  if (configJson.rope_theta !== undefined) {
    values.ropeTheta = configJson.rope_theta
  }

  if (configJson.attention_bias !== undefined) {
    values.hasAttentionBias = configJson.attention_bias
  }

  if (configJson.mlp_bias !== undefined) {
    values.hasMlpBias = configJson.mlp_bias
  }

  if (configJson.rms_norm_eps !== undefined) {
    values.rmsNormEps = configJson.rms_norm_eps
  }

  if (configJson.sliding_window !== undefined && configJson.sliding_window !== null) {
    values.slidingWindow = configJson.sliding_window
    values.useSlidingWindow = true
  }

  if (configJson.num_local_experts !== undefined) {
    values.numExperts = configJson.num_local_experts
  }

  if (configJson.num_experts_per_tok !== undefined) {
    values.numExpertsPerTok = configJson.num_experts_per_tok
  }

  if (configJson.tie_word_embeddings !== undefined) {
    values.hasWeightTying = configJson.tie_word_embeddings
  }

  if (configJson.rope_local_base_freq !== undefined) {
    values.hasLocalRopeTheta = true
  }

  return values
}

/**
 * Get complete model features
 *
 * @param modelType - Model type name (e.g., "gemma3", "qwen2", "llama")
 * @param configJson - Optional config.json contents to extract values from
 * @returns Combined architectural features and config values
 */
export function getModelFeatures(
  modelType: string,
  configJson?: Record<string, unknown>
): ModelFeatures {
  const architectural = getArchitecturalFeatures(modelType)
  const defaults = getDefaultConfigValues(modelType)
  const fromConfig = configJson ? extractConfigValues(configJson as ConfigJson) : {}

  // Merge: architectural + defaults + config (config wins)
  return {
    ...architectural,
    ...defaults,
    ...fromConfig
  }
}
