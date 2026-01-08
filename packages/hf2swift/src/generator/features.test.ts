import { describe, it, expect } from "vitest"
import { getModelFeatures } from "./features.js"

describe("getModelFeatures", () => {
  it("returns Gemma3 features for gemma3 model", () => {
    const features = getModelFeatures("gemma3")

    expect(features.rmsNormStyle).toBe("gemma")
    expect(features.activation).toBe("geluApproximate")
    expect(features.useClipResidual).toBe(true)
    expect(features.useSlidingWindow).toBe(true)
    expect(features.defaultRopeTheta).toBe(1000000)
    expect(features.hasLocalRopeTheta).toBe(true)
    expect(features.useEmbeddingScale).toBe(true)
    expect(features.hasQKNorms).toBe(true)
    expect(features.normsPerLayer).toBe(4)
  })

  it("returns Gemma3 features for gemma-3 model", () => {
    const features = getModelFeatures("gemma-3")

    expect(features.rmsNormStyle).toBe("gemma")
    expect(features.activation).toBe("geluApproximate")
  })

  it("returns Qwen features for qwen2 model", () => {
    const features = getModelFeatures("qwen2")

    expect(features.rmsNormStyle).toBe("standard")
    expect(features.activation).toBe("silu")
    expect(features.useClipResidual).toBe(false)
    expect(features.useSlidingWindow).toBe(false)
    expect(features.normsPerLayer).toBe(2)
  })

  it("returns Llama features for llama model", () => {
    const features = getModelFeatures("llama")

    expect(features.rmsNormStyle).toBe("standard")
    expect(features.activation).toBe("silu")
    expect(features.useSlidingWindow).toBe(false)
  })

  it("returns Phi features for phi model", () => {
    const features = getModelFeatures("phi3")

    expect(features.rmsNormStyle).toBe("standard")
    expect(features.activation).toBe("silu")
  })

  it("returns Mistral features with sliding window", () => {
    const features = getModelFeatures("mistral")

    expect(features.rmsNormStyle).toBe("standard")
    expect(features.activation).toBe("silu")
    expect(features.useSlidingWindow).toBe(true)
  })

  it("returns default features for unknown models", () => {
    const features = getModelFeatures("unknown_model")

    expect(features.rmsNormStyle).toBe("standard")
    expect(features.activation).toBe("gelu")
    expect(features.useClipResidual).toBe(false)
    expect(features.useSlidingWindow).toBe(false)
  })
})
