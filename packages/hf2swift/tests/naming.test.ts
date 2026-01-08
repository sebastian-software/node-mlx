import { describe, it, expect } from "vitest"
import { toCamel, toPascal, capitalize, convertExpr } from "../src/naming.js"
import { EXPR_CONVERSIONS } from "../src/types.js"

describe("naming utilities", () => {
  describe("toCamel", () => {
    it("converts snake_case to camelCase", () => {
      expect(toCamel("hidden_size")).toBe("hiddenSize")
      expect(toCamel("num_attention_heads")).toBe("numAttentionHeads")
      expect(toCamel("q_proj")).toBe("qProj")
    })

    it("handles single words", () => {
      expect(toCamel("hidden")).toBe("hidden")
      expect(toCamel("size")).toBe("size")
    })

    it("handles multiple underscores", () => {
      expect(toCamel("num_key_value_heads")).toBe("numKeyValueHeads")
    })
  })

  describe("toPascal", () => {
    it("converts snake_case to PascalCase", () => {
      expect(toPascal("qwen2")).toBe("Qwen2")
      expect(toPascal("gemma_3n")).toBe("Gemma3n")
      expect(toPascal("llama")).toBe("Llama")
    })

    it("handles special case gpt_oss", () => {
      expect(toPascal("gpt_oss")).toBe("GptOSS")
    })

    it("handles hyphens", () => {
      expect(toPascal("gemma-3n")).toBe("Gemma3n")
    })
  })

  describe("capitalize", () => {
    it("capitalizes first letter", () => {
      expect(capitalize("hello")).toBe("Hello")
      expect(capitalize("world")).toBe("World")
    })

    it("handles empty string", () => {
      expect(capitalize("")).toBe("")
    })

    it("handles single character", () => {
      expect(capitalize("a")).toBe("A")
    })
  })

  describe("convertExpr", () => {
    it("converts Python booleans to Swift", () => {
      expect(convertExpr("True", EXPR_CONVERSIONS)).toBe("true")
      expect(convertExpr("False", EXPR_CONVERSIONS)).toBe("false")
    })

    it("converts None to nil", () => {
      expect(convertExpr("None", EXPR_CONVERSIONS)).toBe("nil")
    })

    it("converts logical operators", () => {
      expect(convertExpr("a and b", EXPR_CONVERSIONS)).toBe("a && b")
      expect(convertExpr("a or b", EXPR_CONVERSIONS)).toBe("a || b")
      expect(convertExpr("not x", EXPR_CONVERSIONS)).toBe("! x")
    })

    it("converts type casts", () => {
      expect(convertExpr("int(x)", EXPR_CONVERSIONS)).toBe("Int(x)")
      expect(convertExpr("float(y)", EXPR_CONVERSIONS)).toBe("Float(y)")
    })

    it("removes self. prefix", () => {
      expect(convertExpr("self.hidden_size", EXPR_CONVERSIONS)).toBe("hidden_size")
    })

    it("converts string quotes", () => {
      expect(convertExpr("'hello'", EXPR_CONVERSIONS)).toBe('"hello"')
    })

    it("converts tensor operations", () => {
      expect(convertExpr("x.view(1, 2)", EXPR_CONVERSIONS)).toBe("x.reshaped([1, 2)")
      expect(convertExpr("x.transpose(0, 1)", EXPR_CONVERSIONS)).toBe("x.transposed(0, 1)")
      expect(convertExpr("x.unsqueeze(0)", EXPR_CONVERSIONS)).toBe("x.expandedDimensions(axis: 0)")
    })

    it("converts is None checks", () => {
      expect(convertExpr("x is nil", EXPR_CONVERSIONS)).toBe("x == nil")
      expect(convertExpr("x is not nil", EXPR_CONVERSIONS)).toBe("x != nil")
    })

    it("converts getattr", () => {
      expect(convertExpr("getattr(config, 'hidden_size', 768)", EXPR_CONVERSIONS)).toBe(
        "config.hidden_size ?? 768"
      )
      expect(convertExpr("getattr(config, 'eps')", EXPR_CONVERSIONS)).toBe("config.eps")
    })
  })
})
