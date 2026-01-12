/**
 * Model definition types
 *
 * Each model family defines its architectural features and config defaults.
 * This enables clean separation and easy addition of new models.
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
 * Model definition - architectural features + config defaults + matcher
 */
export interface ModelDefinition {
  /** Human-readable name */
  name: string

  /** Check if a model type string matches this definition */
  matches: (modelType: string) => boolean

  /** Architectural features (immutable per model family) */
  architectural: ArchitecturalFeatures

  /** Default config values (fallback when config.json missing) */
  configDefaults: ConfigValues
}

/**
 * Default architectural features (used as base)
 */
export const DEFAULT_ARCHITECTURAL: ArchitecturalFeatures = {
  rmsNormStyle: "standard",
  activation: "gelu",
  useClipResidual: false,
  useEmbeddingScale: false,
  hasQKNorms: false,
  normsPerLayer: 2
}

/**
 * Default config values (used as base)
 */
export const DEFAULT_CONFIG: ConfigValues = {
  useSlidingWindow: false,
  ropeTheta: 10000,
  hasLocalRopeTheta: false,
  hasAttentionBias: false,
  hasMlpBias: false,
  rmsNormEps: 1e-5
}
