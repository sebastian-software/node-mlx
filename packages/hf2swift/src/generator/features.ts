/**
 * Model-specific feature flags for code generation
 *
 * Two-tier system:
 * 1. Architectural features - determined by model family (immutable)
 * 2. Config values - read from config.json with model-specific defaults
 *
 * This separation ensures:
 * - Model-specific code paths are feature-driven, not name-driven
 * - Config values come from the source of truth (config.json)
 * - Reasonable defaults when config values are missing
 */

/**
 * Architectural features - determined by model family
 * These control which Swift code patterns are generated
 */
export interface ArchitecturalFeatures {
  /** RMSNorm style: "gemma" uses (1+weight), "standard" uses weight directly */
  rmsNormStyle: "gemma" | "standard"

  /** Activation function: "gelu", "geluApproximate" (Gemma), or "silu" */
  activation: "gelu" | "geluApproximate" | "silu"

  /** Use clipResidual for float16 overflow protection */
  useClipResidual: boolean

  /** Gemma-style embedding scaling (multiply by sqrt(hiddenSize)) */
  useEmbeddingScale: boolean

  /** Has Q/K norms before attention */
  hasQKNorms: boolean

  /** Number of norms per decoder layer (2 for most, 4 for Gemma3) */
  normsPerLayer: 2 | 4

  /** Use fused QKV projection instead of separate q_proj, k_proj, v_proj */
  hasFusedQKV?: boolean

  /** Use fused gate_up_proj instead of separate gate_proj, up_proj */
  hasFusedGateUp?: boolean

  /** Uses Mixture of Experts architecture */
  hasMoE?: boolean

  /** Has learnable attention sinks (GPT-OSS) */
  hasAttentionSinks?: boolean

  /** Uses custom SwiGLU activation (alpha=1.702, limit=7.0) */
  useCustomSwiGLU?: boolean

  /** Use traditional RoPE instead of modern */
  useTraditionalRope?: boolean

  // === Advanced Features (Gemma3n) ===
  hasAltUp?: boolean
  hasLaurel?: boolean
  hasPerLayerInputs?: boolean
  hasKVSharing?: boolean
  hasPerLayerIntermediateSize?: boolean
  hasSparseActivation?: boolean
  hasVNorm?: boolean
  hasLogitSoftcapping?: boolean
  attentionScale?: number

  // === SmolLM3 / Ministral specific ===
  hasNoRopeLayers?: boolean
  hasYarnRope?: boolean
}

/**
 * Config values - read from config.json with defaults
 */
export interface ConfigValues {
  /** Sliding window attention support */
  useSlidingWindow: boolean

  /** RoPE theta */
  ropeTheta: number

  /** Has separate local RoPE theta for sliding window layers */
  hasLocalRopeTheta: boolean

  /** Has attention bias */
  hasAttentionBias: boolean

  /** Has MLP bias */
  hasMlpBias: boolean

  /** RMS norm epsilon */
  rmsNormEps: number

  /** Sliding window size */
  slidingWindow?: number

  /** Number of experts (MoE) */
  numExperts?: number

  /** Experts per token (MoE) */
  numExpertsPerTok?: number

  /** Weight tying (use embed_tokens.weight for lm_head) */
  hasWeightTying?: boolean
}

/**
 * Combined model features = Architectural + Config
 */
export type ModelFeatures = ArchitecturalFeatures & ConfigValues

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
 * Check if model is Gemma 3n
 */
export function isGemma3n(modelType: string): boolean {
  const lower = modelType.toLowerCase()
  return lower.includes("gemma3n") || lower.includes("gemma-3n") || lower.includes("gemma_3n")
}

/**
 * Get architectural features for a model type
 * These are immutable per model family
 */
function getArchitecturalFeatures(modelType: string): ArchitecturalFeatures {
  const lower = modelType.toLowerCase()

  // Gemma 3n - Very specialized architecture
  if (isGemma3n(modelType)) {
    return {
      rmsNormStyle: "standard",
      activation: "geluApproximate",
      useClipResidual: false,
      useEmbeddingScale: true,
      hasQKNorms: true,
      normsPerLayer: 4,
      hasAltUp: true,
      hasLaurel: true,
      hasPerLayerInputs: true,
      hasKVSharing: true,
      hasPerLayerIntermediateSize: true,
      hasSparseActivation: true,
      hasVNorm: true,
      hasLogitSoftcapping: true,
      attentionScale: 1.0
    }
  }

  // Gemma 3
  if (lower.includes("gemma3") || lower.includes("gemma-3")) {
    return {
      rmsNormStyle: "gemma",
      activation: "geluApproximate",
      useClipResidual: true,
      useEmbeddingScale: true,
      hasQKNorms: true,
      normsPerLayer: 4
    }
  }

  // Qwen3
  if (lower.includes("qwen3")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: true,
      normsPerLayer: 2
    }
  }

  // Qwen2
  if (lower.includes("qwen")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // Llama
  if (lower.includes("llama")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // Phi3/Phi4
  if (lower.includes("phi")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2,
      hasFusedQKV: true,
      hasFusedGateUp: true
    }
  }

  // Mistral
  if (lower.includes("mistral") || lower.includes("ministral")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // GPT-OSS - MoE architecture
  if (lower.includes("gpt_oss") || lower.includes("gptoss") || lower.includes("gpt-oss")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2,
      hasMoE: true,
      hasAttentionSinks: true,
      useCustomSwiGLU: true,
      useTraditionalRope: true
    }
  }

  // SmolLM3
  if (lower.includes("smollm3") || lower.includes("smollm-3") || lower.includes("smollm_3")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2,
      hasNoRopeLayers: true
    }
  }

  // Mistral 3 / Ministral 3
  if (
    lower.includes("mistral3") ||
    lower.includes("mistral-3") ||
    lower.includes("ministral3") ||
    lower.includes("ministral-3")
  ) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2,
      hasYarnRope: true
    }
  }

  // Default
  return {
    rmsNormStyle: "standard",
    activation: "gelu",
    useClipResidual: false,
    useEmbeddingScale: false,
    hasQKNorms: false,
    normsPerLayer: 2
  }
}

/**
 * Get default config values for a model type
 * These serve as fallbacks when config.json doesn't have the value
 */
function getDefaultConfigValues(modelType: string): ConfigValues {
  const lower = modelType.toLowerCase()

  // Gemma family defaults
  if (lower.includes("gemma")) {
    return {
      useSlidingWindow: true,
      ropeTheta: 1000000,
      hasLocalRopeTheta: true,
      hasAttentionBias: false,
      hasMlpBias: false,
      rmsNormEps: 1e-6,
      hasWeightTying: isGemma3n(modelType)
    }
  }

  // Qwen3
  if (lower.includes("qwen3")) {
    return {
      useSlidingWindow: false,
      ropeTheta: 1000000,
      hasLocalRopeTheta: false,
      hasAttentionBias: false,
      hasMlpBias: false,
      rmsNormEps: 1e-6,
      hasWeightTying: true
    }
  }

  // Qwen2
  if (lower.includes("qwen")) {
    return {
      useSlidingWindow: false,
      ropeTheta: 10000,
      hasLocalRopeTheta: false,
      hasAttentionBias: true,
      hasMlpBias: false,
      rmsNormEps: 1e-6
    }
  }

  // Mistral family
  if (lower.includes("mistral") || lower.includes("ministral")) {
    const isMistral3 =
      lower.includes("mistral3") ||
      lower.includes("mistral-3") ||
      lower.includes("ministral3") ||
      lower.includes("ministral-3")

    return {
      useSlidingWindow: !isMistral3,
      ropeTheta: isMistral3 ? 1000000 : 10000,
      hasLocalRopeTheta: false,
      hasAttentionBias: false,
      hasMlpBias: false,
      rmsNormEps: 1e-5
    }
  }

  // GPT-OSS
  if (lower.includes("gpt_oss") || lower.includes("gptoss") || lower.includes("gpt-oss")) {
    return {
      useSlidingWindow: true,
      ropeTheta: 150000,
      hasLocalRopeTheta: false,
      hasAttentionBias: true,
      hasMlpBias: true,
      rmsNormEps: 1e-5,
      slidingWindow: 128,
      numExperts: 128,
      numExpertsPerTok: 4
    }
  }

  // SmolLM3
  if (lower.includes("smollm3") || lower.includes("smollm-3") || lower.includes("smollm_3")) {
    return {
      useSlidingWindow: false,
      ropeTheta: 5000000,
      hasLocalRopeTheta: false,
      hasAttentionBias: false,
      hasMlpBias: false,
      rmsNormEps: 1e-5,
      hasWeightTying: true
    }
  }

  // Default (Llama, Phi, etc.)
  return {
    useSlidingWindow: false,
    ropeTheta: 10000,
    hasLocalRopeTheta: false,
    hasAttentionBias: false,
    hasMlpBias: false,
    rmsNormEps: 1e-5
  }
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

// Note: ModelFeatures is already exported above as a type alias
