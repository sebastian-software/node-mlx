import { describe, it, expect } from "vitest"
import { inferSwiftType, generateConfigFromJson } from "../src/config.js"

describe("config utilities", () => {
  describe("inferSwiftType", () => {
    it("infers boolean type", () => {
      expect(inferSwiftType(true)).toEqual(["Bool", true])
      expect(inferSwiftType(false)).toEqual(["Bool", true])
    })

    it("infers integer type", () => {
      expect(inferSwiftType(768)).toEqual(["Int", true])
      expect(inferSwiftType(0)).toEqual(["Int", true])
      expect(inferSwiftType(-1)).toEqual(["Int", true])
    })

    it("infers float type", () => {
      expect(inferSwiftType(1e-6)).toEqual(["Float", true])
      expect(inferSwiftType(3.14)).toEqual(["Float", true])
    })

    it("infers string type", () => {
      expect(inferSwiftType("hello")).toEqual(["String", true])
      expect(inferSwiftType("")).toEqual(["String", true])
    })

    it("infers array types", () => {
      expect(inferSwiftType([1, 2, 3])).toEqual(["[Int]", true])
      // In JavaScript, 1.0 and 2.0 are integers, so we use actual floats
      expect(inferSwiftType([1.5, 2.5])).toEqual(["[Float]", true])
      expect(inferSwiftType(["a", "b"])).toEqual(["[String]", true])
      expect(inferSwiftType([])).toEqual(["[Any]", false])
    })

    it("handles null and undefined", () => {
      expect(inferSwiftType(null)).toEqual(["Any?", true])
      expect(inferSwiftType(undefined)).toEqual(["Any?", true])
    })

    it("handles unknown types", () => {
      expect(inferSwiftType({})).toEqual(["Any", false])
    })
  })

  describe("generateConfigFromJson", () => {
    it("generates basic config struct", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000,
        model_type: "qwen2"
      }

      const result = generateConfigFromJson(config, "qwen2")

      expect(result).toContain("public struct Qwen2Configuration: Codable, Sendable")
      expect(result).toContain("public var hiddenSize: Int")
      expect(result).toContain("public var numHiddenLayers: Int")
      expect(result).toContain("public var numAttentionHeads: Int")
      expect(result).toContain("enum CodingKeys: String, CodingKey")
      expect(result).toContain('case hiddenSize = "hidden_size"')
    })

    it("includes computed headDim property", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000
      }

      const result = generateConfigFromJson(config, "test")

      expect(result).toContain("public var headDim: Int {")
      expect(result).toContain("hiddenSize / numAttentionHeads")
    })

    it("skips headDim if already present", () => {
      const config = {
        hidden_size: 768,
        head_dim: 64,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000
      }

      const result = generateConfigFromJson(config, "test")

      expect(result).toContain("public var headDim: Int")
      expect(result).not.toContain("hiddenSize / numAttentionHeads")
    })

    it("makes optional fields nullable", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000,
        rms_norm_eps: 1e-6
      }

      const result = generateConfigFromJson(config, "test")

      // rms_norm_eps is not in importantFields, so should be optional
      expect(result).toContain("public var rmsNormEps: Float?")
    })

    it("skips internal and meta fields", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000,
        _name_or_path: "test",
        architectures: ["TestModel"],
        torch_dtype: "float16",
        transformers_version: "4.0.0"
      }

      const result = generateConfigFromJson(config, "test")

      expect(result).not.toContain("nameOrPath")
      expect(result).not.toContain("architectures")
      expect(result).not.toContain("torchDtype")
      expect(result).not.toContain("transformersVersion")
    })

    it("skips complex nested fields", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000,
        rope_scaling: { type: "linear", factor: 2.0 },
        quantization_config: { bits: 4 }
      }

      const result = generateConfigFromJson(config, "test")

      expect(result).not.toContain("ropeScaling")
      expect(result).not.toContain("quantizationConfig")
    })

    it("adds default fields if not present", () => {
      const config = {
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        vocab_size: 32000
      }

      const result = generateConfigFromJson(config, "test")

      // Should have added attention_bias and rms_norm_eps defaults
      expect(result).toContain("attentionBias")
      expect(result).toContain("rmsNormEps")
    })

    it("handles VLM configs with text_config", () => {
      const config = {
        text_config: {
          hidden_size: 768,
          num_hidden_layers: 12,
          num_attention_heads: 12,
          intermediate_size: 3072,
          vocab_size: 32000
        },
        vision_config: {
          hidden_size: 1024
        },
        model_type: "vlm"
      }

      const result = generateConfigFromJson(config, "test")

      expect(result).toContain("Decodable, Sendable")
      expect(result).toContain('textConfig = "text_config"')
      expect(result).toContain("TextConfigCodingKeys")
      expect(result).toContain("public init(from decoder: Swift.Decoder) throws")
      expect(result).toContain("nestedContainer")
    })
  })
})
