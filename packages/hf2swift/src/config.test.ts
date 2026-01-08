import { describe, it, expect } from "vitest"
import { inferSwiftType, generateConfigFromJson } from "./config.js"

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
      expect(inferSwiftType([1.5, 2.5])).toEqual(["[Float]", true])
      expect(inferSwiftType(["a", "b"])).toEqual(["[String]", true])
      expect(inferSwiftType([])).toEqual(["[Any]", false])
    })

    it("handles null and undefined", () => {
      expect(inferSwiftType(null)).toEqual(["Any?", true])
      expect(inferSwiftType(undefined)).toEqual(["Any?", true])
    })

    it("handles object types as non-codable", () => {
      // Objects are skipped in config generation (isCodable = false)
      expect(inferSwiftType({ type: "linear" })).toEqual(["Any", false])
    })
  })
})
