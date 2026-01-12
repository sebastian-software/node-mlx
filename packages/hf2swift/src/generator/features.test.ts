import { describe, it, expect } from "vitest"
import { getModelFeatures } from "./features.js"

describe("getModelFeatures", () => {
  describe("architectural features (model-specific)", () => {
    it("returns Gemma3 architectural features", () => {
      const features = getModelFeatures("gemma3")

      expect(features.rmsNormStyle).toBe("gemma")
      expect(features.activation).toBe("geluApproximate")
      expect(features.useClipResidual).toBe(true)
      expect(features.useEmbeddingScale).toBe(true)
      expect(features.hasQKNorms).toBe(true)
      expect(features.normsPerLayer).toBe(4)
    })

    it("returns Gemma3 features for gemma-3 variant", () => {
      const features = getModelFeatures("gemma-3")

      expect(features.rmsNormStyle).toBe("gemma")
      expect(features.activation).toBe("geluApproximate")
    })

    it("returns Qwen2 architectural features", () => {
      const features = getModelFeatures("qwen2")

      expect(features.rmsNormStyle).toBe("standard")
      expect(features.activation).toBe("silu")
      expect(features.useClipResidual).toBe(false)
      expect(features.normsPerLayer).toBe(2)
    })

    it("returns Llama architectural features", () => {
      const features = getModelFeatures("llama")

      expect(features.rmsNormStyle).toBe("standard")
      expect(features.activation).toBe("silu")
    })

    it("returns Phi architectural features with fused projections", () => {
      const features = getModelFeatures("phi3")

      expect(features.rmsNormStyle).toBe("standard")
      expect(features.activation).toBe("silu")
      expect(features.hasFusedQKV).toBe(true)
      expect(features.hasFusedGateUp).toBe(true)
    })

    it("returns GPT-OSS architectural features with MoE", () => {
      const features = getModelFeatures("gpt_oss")

      expect(features.hasMoE).toBe(true)
      expect(features.hasAttentionSinks).toBe(true)
      expect(features.useCustomSwiGLU).toBe(true)
    })

    it("returns default features for unknown models", () => {
      const features = getModelFeatures("unknown_model")

      expect(features.rmsNormStyle).toBe("standard")
      expect(features.activation).toBe("gelu")
      expect(features.useClipResidual).toBe(false)
    })
  })

  describe("config values (from defaults)", () => {
    it("returns Gemma3 default config values", () => {
      const features = getModelFeatures("gemma3")

      expect(features.useSlidingWindow).toBe(true)
      expect(features.ropeTheta).toBe(1000000)
      expect(features.hasLocalRopeTheta).toBe(true)
    })

    it("returns Mistral default config values with sliding window", () => {
      const features = getModelFeatures("mistral")

      expect(features.useSlidingWindow).toBe(true)
    })

    it("returns GPT-OSS default MoE config values", () => {
      const features = getModelFeatures("gpt_oss")

      expect(features.numExperts).toBe(128)
      expect(features.numExpertsPerTok).toBe(4)
      expect(features.slidingWindow).toBe(128)
      expect(features.ropeTheta).toBe(150000)
    })
  })

  describe("config override (from config.json)", () => {
    it("overrides ropeTheta from config.json", () => {
      const features = getModelFeatures("llama", { rope_theta: 500000 })

      expect(features.ropeTheta).toBe(500000)
    })

    it("overrides attentionBias from config.json", () => {
      const features = getModelFeatures("llama", { attention_bias: true })

      expect(features.hasAttentionBias).toBe(true)
    })

    it("overrides mlpBias from config.json", () => {
      const features = getModelFeatures("llama", { mlp_bias: true })

      expect(features.hasMlpBias).toBe(true)
    })

    it("overrides slidingWindow from config.json", () => {
      const features = getModelFeatures("llama", { sliding_window: 4096 })

      expect(features.slidingWindow).toBe(4096)
      expect(features.useSlidingWindow).toBe(true)
    })

    it("overrides numExperts from config.json", () => {
      const features = getModelFeatures("gpt_oss", { num_local_experts: 64 })

      expect(features.numExperts).toBe(64)
    })

    it("preserves architectural features when config provided", () => {
      const features = getModelFeatures("gemma3", { rope_theta: 999999 })

      // Config override
      expect(features.ropeTheta).toBe(999999)
      // Architectural preserved
      expect(features.rmsNormStyle).toBe("gemma")
      expect(features.activation).toBe("geluApproximate")
    })
  })
})
