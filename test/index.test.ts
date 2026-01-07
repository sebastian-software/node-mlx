import { describe, it, expect, beforeAll, afterAll } from "vitest"
import { platform, arch } from "node:os"

// Only import if on supported platform
const isAppleSilicon = platform() === "darwin" && arch() === "arm64"

describe("node-mlx", () => {
  describe("platform detection", () => {
    it("should correctly detect Apple Silicon", async () => {
      const { isSupported } = await import("../src/index.js")
      expect(isSupported()).toBe(isAppleSilicon)
    })
  })

  describe("exports", () => {
    it("should export all expected functions", async () => {
      const exports = await import("../src/index.js")

      expect(exports.isSupported).toBeDefined()
      expect(exports.getVersion).toBeDefined()
      expect(exports.loadModel).toBeDefined()
      expect(exports.generate).toBeDefined()
      expect(exports.RECOMMENDED_MODELS).toBeDefined()
    })

    it("should have recommended models defined", async () => {
      const { RECOMMENDED_MODELS } = await import("../src/index.js")

      expect(Object.keys(RECOMMENDED_MODELS).length).toBeGreaterThan(0)
      expect(RECOMMENDED_MODELS["llama-3.2-1b"]).toBe("mlx-community/Llama-3.2-1B-Instruct-4bit")
      expect(RECOMMENDED_MODELS["qwen-2.5-0.5b"]).toBe("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    })
  })

  // Integration tests - only run on Apple Silicon with built native addon
  describe.skipIf(!isAppleSilicon)("native binding", () => {
    it("should get version", async () => {
      const { getVersion } = await import("../src/index.js")
      const version = getVersion()
      expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    })
  })

  // Full model tests - skip by default (require model download)
  describe.skip("model inference", () => {
    let model: Awaited<ReturnType<(typeof import("../src/index.js"))["loadModel"]>>

    beforeAll(async () => {
      const { loadModel, RECOMMENDED_MODELS } = await import("../src/index.js")
      model = loadModel(RECOMMENDED_MODELS["gemma-3n-2b"])
    })

    afterAll(() => {
      model?.unload()
    })

    it("should generate text", () => {
      const result = model.generate("Say hello in one word:", { maxTokens: 10 })

      expect(result.text).toBeDefined()
      expect(result.text.length).toBeGreaterThan(0)
      expect(result.tokenCount).toBeGreaterThan(0)
      expect(result.tokensPerSecond).toBeGreaterThan(0)
    })

    it("should respect maxTokens", () => {
      const result = model.generate("Count from 1 to 100:", { maxTokens: 5 })
      expect(result.tokenCount).toBeLessThanOrEqual(5)
    })
  })
})
