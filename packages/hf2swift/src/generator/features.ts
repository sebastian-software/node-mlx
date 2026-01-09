/**
 * Model-specific feature flags for code generation
 *
 * These flags control which Swift code patterns are generated
 * for different model architectures.
 */

/**
 * Model-specific feature configuration
 */
export interface ModelFeatures {
  // === Core Architecture ===

  /** RMSNorm style: "gemma" uses (1+weight), "standard" uses weight directly */
  rmsNormStyle: "gemma" | "standard"

  /** Activation function: "gelu", "geluApproximate" (Gemma), or "silu" */
  activation: "gelu" | "geluApproximate" | "silu"

  /** Use clipResidual for float16 overflow protection */
  useClipResidual: boolean

  /** Sliding window attention support */
  useSlidingWindow: boolean

  /** Default RoPE theta (10000 for most, 1000000 for Gemma3) */
  defaultRopeTheta: number

  /** Has separate local RoPE theta for sliding window layers */
  hasLocalRopeTheta: boolean

  /** Gemma-style embedding scaling (multiply by sqrt(hiddenSize)) */
  useEmbeddingScale: boolean

  /** Has Q/K norms before attention */
  hasQKNorms: boolean

  /** Number of norms per decoder layer (2 for most, 4 for Gemma3) */
  normsPerLayer: 2 | 4

  /** Has attention bias (read from config.attention_bias, default varies by model) */
  hasAttentionBias?: boolean

  /** Has MLP bias (read from config.mlp_bias, default false) */
  hasMlpBias?: boolean

  // === Advanced Features (Gemma3n and future models) ===

  /** AltUp (Alternating Updates) for efficient sparse computation */
  hasAltUp?: boolean

  /** Laurel (Learned Augmented Residual) blocks */
  hasLaurel?: boolean

  /** Per-layer input embeddings */
  hasPerLayerInputs?: boolean

  /** KV-cache sharing for later layers */
  hasKVSharing?: boolean

  /** Per-layer intermediate MLP sizes (array instead of single value) */
  hasPerLayerIntermediateSize?: boolean

  /** Sparse activation with gelu_topk */
  hasSparseActivation?: boolean

  /** Value normalization (RMSNoScale) in attention */
  hasVNorm?: boolean

  /** Weight tying (use embed_tokens.weight for lm_head) */
  hasWeightTying?: boolean

  /** Logit softcapping */
  hasLogitSoftcapping?: boolean

  /** Attention scale override (e.g., 1.0 for Gemma3n instead of 1/sqrt(headDim)) */
  attentionScale?: number
}

/**
 * Check if model is Gemma 3n
 */
export function isGemma3n(modelType: string): boolean {
  const lower = modelType.toLowerCase()
  return lower.includes("gemma3n") || lower.includes("gemma-3n") || lower.includes("gemma_3n")
}

/**
 * Get default features for a model type
 */
export function getModelFeatures(modelType: string): ModelFeatures {
  const lower = modelType.toLowerCase()

  // Gemma 3n - Very specialized architecture (check first!)
  if (isGemma3n(modelType)) {
    return {
      // Base features (similar to Gemma 3)
      rmsNormStyle: "standard", // Gemma3n uses standard RMSNorm (not 1+weight)
      activation: "geluApproximate",
      useClipResidual: false,
      useSlidingWindow: true,
      defaultRopeTheta: 1000000,
      hasLocalRopeTheta: true,
      useEmbeddingScale: true,
      hasQKNorms: true,
      normsPerLayer: 4,

      // Gemma 3n specific advanced features
      hasAltUp: true,
      hasLaurel: true,
      hasPerLayerInputs: true,
      hasKVSharing: true,
      hasPerLayerIntermediateSize: true,
      hasSparseActivation: true,
      hasVNorm: true,
      hasWeightTying: true,
      hasLogitSoftcapping: true,
      attentionScale: 1.0
    }
  }

  // Gemma 3 - Advanced features
  if (lower.includes("gemma3") || lower.includes("gemma-3")) {
    return {
      rmsNormStyle: "gemma",
      activation: "geluApproximate",
      useClipResidual: true,
      useSlidingWindow: true,
      defaultRopeTheta: 1000000,
      hasLocalRopeTheta: true,
      useEmbeddingScale: true,
      hasQKNorms: true,
      normsPerLayer: 4
    }
  }

  // Qwen2 - Standard with SiLU, has attention bias by default
  if (lower.includes("qwen")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useSlidingWindow: false,
      defaultRopeTheta: 10000,
      hasLocalRopeTheta: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2,
      hasAttentionBias: true, // Qwen2/2.5 has attention_bias: true by default
      hasMlpBias: false
    }
  }

  // Llama - Standard with SiLU
  if (lower.includes("llama")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useSlidingWindow: false,
      defaultRopeTheta: 10000,
      hasLocalRopeTheta: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // Phi - Standard with SiLU
  if (lower.includes("phi")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useSlidingWindow: false,
      defaultRopeTheta: 10000,
      hasLocalRopeTheta: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // Mistral - with sliding window
  if (lower.includes("mistral") || lower.includes("ministral")) {
    return {
      rmsNormStyle: "standard",
      activation: "silu",
      useClipResidual: false,
      useSlidingWindow: true,
      defaultRopeTheta: 10000,
      hasLocalRopeTheta: false,
      useEmbeddingScale: false,
      hasQKNorms: false,
      normsPerLayer: 2
    }
  }

  // Default features
  return {
    rmsNormStyle: "standard",
    activation: "gelu",
    useClipResidual: false,
    useSlidingWindow: false,
    defaultRopeTheta: 10000,
    hasLocalRopeTheta: false,
    useEmbeddingScale: false,
    hasQKNorms: false,
    normsPerLayer: 2
  }
}
