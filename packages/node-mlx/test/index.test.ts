import { describe, it, expect, beforeAll, afterAll } from "vitest"
import { platform, arch } from "node:os"

const isAppleSilicon = platform() === "darwin" && arch() === "arm64"

describe("node-mlx", () => {
  describe("platform detection", () => {
    it("correctly detects platform support", async () => {
      const { isSupported, isPlatformSupported } = await import("../src/index.js")

      // isPlatformSupported only checks OS/arch
      expect(isPlatformSupported()).toBe(isAppleSilicon)

      // isSupported also checks for native bindings
      // On non-Apple Silicon, both should be false
      if (!isAppleSilicon) {
        expect(isSupported()).toBe(false)
      }
    })
  })

  describe("API exports", () => {
    it("exports required functions", async () => {
      const exports = await import("../src/index.js")

      // Core API
      expect(typeof exports.loadModel).toBe("function")
      expect(typeof exports.generate).toBe("function")
      expect(typeof exports.isSupported).toBe("function")
      expect(typeof exports.getVersion).toBe("function")

      // Constants
      expect(typeof exports.RECOMMENDED_MODELS).toBe("object")
      expect(typeof exports.VERSION).toBe("string")
    })

    it("has model shortcuts for major families", async () => {
      const { RECOMMENDED_MODELS } = await import("../src/index.js")

      // Check model families exist (not specific IDs which may change)
      const keys = Object.keys(RECOMMENDED_MODELS)

      expect(keys.some((k) => k.includes("qwen"))).toBe(true)
      expect(keys.some((k) => k.includes("llama"))).toBe(true)
      expect(keys.some((k) => k.includes("phi"))).toBe(true)
      expect(keys.some((k) => k.includes("gemma"))).toBe(true)
    })
  })

  // Native binding tests - only on Apple Silicon with built binaries
  describe.skipIf(!isAppleSilicon)("native binding", () => {
    it("returns valid version string", async () => {
      const { getVersion, isSupported } = await import("../src/index.js")

      if (isSupported()) {
        const version = getVersion()
        expect(version).toMatch(/^\d+\.\d+\.\d+$/)
      }
    })
  })

  // Full integration tests - require model downloads, run manually
  describe.skip("model inference (manual)", () => {
    let model: Awaited<ReturnType<(typeof import("../src/index.js"))["loadModel"]>>

    beforeAll(async () => {
      const { loadModel, RECOMMENDED_MODELS } = await import("../src/index.js")
      model = loadModel(RECOMMENDED_MODELS["qwen-2.5-0.5b"])
    })

    afterAll(() => {
      model?.unload()
    })

    it("generates text", () => {
      const result = model.generate("Say hello:", { maxTokens: 10 })

      expect(result.text.length).toBeGreaterThan(0)
      expect(result.tokenCount).toBeGreaterThan(0)
      expect(result.tokensPerSecond).toBeGreaterThan(0)
    })
  })
})
