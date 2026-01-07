import { describe, it, expect } from "vitest"
import { isSupported, RECOMMENDED_MODELS } from "../src/index.js"

describe("node-mlx", () => {
  describe("isSupported", () => {
    it("returns boolean", () => {
      const result = isSupported()
      expect(typeof result).toBe("boolean")
    })

    it("returns true on macOS ARM64", () => {
      // This test will only pass on Apple Silicon Macs
      if (process.platform === "darwin" && process.arch === "arm64") {
        expect(isSupported()).toBe(true)
      }
    })

    it("returns false on non-macOS platforms", () => {
      if (process.platform !== "darwin") {
        expect(isSupported()).toBe(false)
      }
    })
  })

  describe("RECOMMENDED_MODELS", () => {
    it("contains Gemma 3n models", () => {
      expect(RECOMMENDED_MODELS["gemma-3n-2b"]).toBeDefined()
      expect(RECOMMENDED_MODELS["gemma-3n-4b"]).toBeDefined()
    })

    it("contains Qwen models", () => {
      expect(RECOMMENDED_MODELS["qwen-3-1.7b"]).toBeDefined()
      expect(RECOMMENDED_MODELS["qwen-3-4b"]).toBeDefined()
    })

    it("contains Phi model", () => {
      expect(RECOMMENDED_MODELS["phi-4"]).toBeDefined()
    })

    it("contains Llama model", () => {
      expect(RECOMMENDED_MODELS["llama-4-scout"]).toBeDefined()
    })

    it("all models are valid HuggingFace paths", () => {
      for (const model of Object.values(RECOMMENDED_MODELS)) {
        expect(model).toMatch(/^mlx-community\//)
      }
    })
  })
})
