/**
 * GPT-OSS model family definition
 *
 * Mixture of Experts architecture with attention sinks and sliding window.
 */

import { DEFAULT_ARCHITECTURAL, DEFAULT_CONFIG, type ModelDefinition } from "./types.js"

export const gptOss: ModelDefinition = {
  name: "GPT-OSS",

  matches: (modelType) => {
    const lower = modelType.toLowerCase()
    return lower.includes("gpt_oss") || lower.includes("gptoss") || lower.includes("gpt-oss")
  },

  architectural: {
    ...DEFAULT_ARCHITECTURAL,
    activation: "silu",
    hasMoE: true,
    hasAttentionSinks: true,
    useCustomSwiGLU: true,
    useTraditionalRope: true
  },

  configDefaults: {
    ...DEFAULT_CONFIG,
    useSlidingWindow: true,
    ropeTheta: 150000,
    hasAttentionBias: true,
    hasMlpBias: true,
    slidingWindow: 128,
    numExperts: 128,
    numExpertsPerTok: 4
  }
}
