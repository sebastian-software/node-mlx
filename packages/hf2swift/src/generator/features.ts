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
}

/**
 * Get default features for a model type
 */
export function getModelFeatures(modelType: string): ModelFeatures {
  const lower = modelType.toLowerCase()

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

  // Qwen2 - Standard with SiLU
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
      normsPerLayer: 2
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
