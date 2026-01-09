import { describe, it, expect } from "vitest"
import { SwiftGenerator } from "./index.js"

describe("SwiftGenerator", () => {
  describe("generate", () => {
    it("generates non-empty Swift code", () => {
      const generator = new SwiftGenerator("qwen2")
      const result = generator.generate([])

      // Basic sanity checks - the Swift compiler is the real test
      expect(result.length).toBeGreaterThan(1000)
      expect(result).toContain("import MLX")
      expect(result).toContain("class")
      expect(result).toContain("Module")
    })

    it("uses model name in generated classes", () => {
      const generator = new SwiftGenerator("qwen2")
      const result = generator.generate([])

      expect(result).toContain("Qwen2")
    })

    it("generates different code for different models", () => {
      const qwen = new SwiftGenerator("qwen2").generate([])
      const gemma = new SwiftGenerator("gemma3").generate([])

      // Different models should produce different output
      expect(qwen).not.toBe(gemma)
      expect(qwen).toContain("Qwen2")
      expect(gemma).toContain("Gemma3")
    })

    it("includes LLMModel protocol conformance", () => {
      const generator = new SwiftGenerator("test")
      const result = generator.generate([])

      expect(result).toContain("LLMModel")
      expect(result).toContain("supportsCache")
      expect(result).toContain("newCache")
    })

    it("includes weight sanitization", () => {
      const generator = new SwiftGenerator("test")
      const result = generator.generate([])

      expect(result).toContain("sanitize")
    })
  })

  describe("model-specific features", () => {
    it("gemma3 has sliding window support", () => {
      const result = new SwiftGenerator("gemma3").generate([])

      // Gemma3 uses sliding window attention
      expect(result).toContain("slidingWindow")
    })

    it("gemma3 has clip residual for fp16 stability", () => {
      const result = new SwiftGenerator("gemma3").generate([])

      expect(result).toContain("clipResidual")
    })

    it("standard models do not have clip residual", () => {
      const result = new SwiftGenerator("qwen2").generate([])

      expect(result).not.toContain("clipResidual")
    })
  })
})
