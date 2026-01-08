/**
 * Swift Configuration struct generator
 *
 * Generates Swift Codable structs from HuggingFace config.json
 * with model-specific features and defaults.
 */

import { ConfigField } from "./types.js"
import { toCamel, toPascal } from "./naming.js"
import { ModelFeatures } from "./generator.js"

/**
 * Infer Swift type from a JSON value
 */
export function inferSwiftType(value: unknown): [string, boolean] {
  if (value === null || value === undefined) {
    return ["Any?", true]
  }
  if (typeof value === "boolean") {
    return ["Bool", true]
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? ["Int", true] : ["Float", true]
  }
  if (typeof value === "string") {
    return ["String", true]
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return ["[Any]", false]
    }
    const first = value[0]
    if (typeof first === "number") {
      return Number.isInteger(first) ? ["[Int]", true] : ["[Float]", true]
    }
    if (typeof first === "string") {
      return ["[String]", true]
    }
    return ["[Any]", false]
  }
  // Check for rope_scaling object
  if (typeof value === "object") {
    return ["[String: StringOrNumber]", true]
  }
  return ["Any", false]
}

/**
 * Generate Swift Configuration struct from config.json
 */
export function generateConfigFromJson(
  configJson: Record<string, unknown>,
  modelName: string,
  features?: ModelFeatures
): string {
  const importantFields = new Set([
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "model_type"
  ])

  const skipFields = new Set([
    "auto_map",
    "quantization",
    "quantization_config",
    "task_specific_params",
    "id2label",
    "label2id",
    "vision_config",
    "audio_config"
  ])

  // Fields we want to keep as complex types
  const complexFields = new Set(["rope_scaling"])

  const defaultFields: Record<string, unknown> = {
    attention_bias: false,
    rms_norm_eps: 1e-6,
    num_key_value_heads: null
  }

  // Add model-specific defaults
  if (features?.useSlidingWindow) {
    defaultFields["sliding_window"] = 512
    defaultFields["sliding_window_pattern"] = 6
  }
  if (features?.hasLocalRopeTheta) {
    defaultFields["rope_local_base_freq"] = 10000.0
  }

  // Add defaults if not present
  for (const [key, defaultValue] of Object.entries(defaultFields)) {
    if (!(key in configJson)) {
      configJson[key] = defaultValue
    }
  }

  // Check for VLM (Vision-Language Model) with nested text_config
  const isVlm =
    "text_config" in configJson &&
    typeof configJson.text_config === "object" &&
    configJson.text_config !== null
  const textConfig = isVlm ? (configJson.text_config as Record<string, unknown>) : {}

  const fields: ConfigField[] = []

  // Process text_config fields first for VLMs
  if (isVlm) {
    for (const [key, value] of Object.entries(textConfig)) {
      if (key.startsWith("_") || (skipFields.has(key) && !complexFields.has(key))) {
        continue
      }
      const swiftName = toCamel(key)
      const [swiftType, isCodable] = inferSwiftType(value)
      if (!isCodable) {
        continue
      }
      const optional = !importantFields.has(key)
      fields.push({
        name: key,
        swiftName,
        swiftType: optional && !swiftType.endsWith("?") ? `${swiftType}?` : swiftType,
        default: null,
        optional,
        codingKey: key
      })
    }
  }

  // Process main config fields
  for (const [key, value] of Object.entries(configJson)) {
    if (
      key.startsWith("_") ||
      ["architectures", "transformers_version", "torch_dtype"].includes(key)
    ) {
      continue
    }
    if ((skipFields.has(key) && !complexFields.has(key)) || (isVlm && key === "text_config")) {
      continue
    }
    if (fields.some((f) => f.name === key)) {
      continue
    }

    const swiftName = toCamel(key)
    const [swiftType, isCodable] = inferSwiftType(value)
    if (!isCodable) {
      continue
    }

    const optional = value === null || !importantFields.has(key)
    fields.push({
      name: key,
      swiftName,
      swiftType: optional && !swiftType.endsWith("?") ? `${swiftType}?` : swiftType,
      default: null,
      optional,
      codingKey: key
    })
  }

  // Ensure essential fields exist with proper types (non-optional for core fields)
  const essentialFields: Array<{ name: string; swiftName: string; swiftType: string }> = [
    { name: "hidden_size", swiftName: "hiddenSize", swiftType: "Int" },
    { name: "num_hidden_layers", swiftName: "numHiddenLayers", swiftType: "Int" },
    { name: "num_attention_heads", swiftName: "numAttentionHeads", swiftType: "Int" },
    { name: "num_key_value_heads", swiftName: "numKeyValueHeads", swiftType: "Int" },
    { name: "intermediate_size", swiftName: "intermediateSize", swiftType: "Int" },
    { name: "vocab_size", swiftName: "vocabSize", swiftType: "Int" },
    { name: "rms_norm_eps", swiftName: "rmsNormEps", swiftType: "Float" },
    { name: "rope_theta", swiftName: "ropeTheta", swiftType: "Float" },
    { name: "max_position_embeddings", swiftName: "maxPositionEmbeddings", swiftType: "Int" }
  ]

  // Add sliding window fields for models that use them
  if (features?.useSlidingWindow) {
    essentialFields.push(
      { name: "sliding_window", swiftName: "slidingWindow", swiftType: "Int" },
      { name: "sliding_window_pattern", swiftName: "slidingWindowPattern", swiftType: "Int" }
    )
  }
  if (features?.hasLocalRopeTheta) {
    essentialFields.push({
      name: "rope_local_base_freq",
      swiftName: "ropeLocalTheta",
      swiftType: "Float"
    })
  }

  // Generate Swift code
  const className = `${toPascal(modelName)}Configuration`
  const lines: string[] = [
    "// MARK: - Configuration",
    "",
    `public struct ${className}: Decodable, Sendable {`
  ]

  // Properties - first add essential fields
  for (const ef of essentialFields) {
    const existing = fields.find((f) => f.swiftName === ef.swiftName)
    if (!existing) {
      fields.unshift({
        name: ef.name,
        swiftName: ef.swiftName,
        swiftType: ef.swiftType,
        default: null,
        optional: false,
        codingKey: ef.name
      })
    } else {
      // Ensure non-optional
      existing.swiftType = ef.swiftType
      existing.optional = false
    }
  }

  // Add head_dim if needed
  const hasHeadDim = fields.some((f) => f.name === "head_dim")
  if (!hasHeadDim) {
    fields.push({
      name: "head_dim",
      swiftName: "headDim",
      swiftType: "Int",
      default: null,
      optional: false,
      codingKey: "head_dim"
    })
  }

  // Add rope_scaling for VLM support
  const hasRopeScaling = fields.some((f) => f.name === "rope_scaling")
  if (!hasRopeScaling) {
    fields.push({
      name: "rope_scaling",
      swiftName: "ropeScaling",
      swiftType: "[String: StringOrNumber]?",
      default: null,
      optional: true,
      codingKey: "rope_scaling"
    })
  }

  // Add model_type
  const hasModelType = fields.some((f) => f.name === "model_type")
  if (!hasModelType) {
    fields.push({
      name: "model_type",
      swiftName: "modelType",
      swiftType: "String?",
      default: null,
      optional: true,
      codingKey: "model_type"
    })
  }

  // Sort fields: non-optional first, then alphabetical
  fields.sort((a, b) => {
    if (a.optional !== b.optional) return a.optional ? 1 : -1
    return a.swiftName.localeCompare(b.swiftName)
  })

  // Remove duplicates
  const uniqueFields = fields.filter(
    (f, i, arr) => arr.findIndex((x) => x.swiftName === f.swiftName) === i
  )

  // Write properties
  for (const f of uniqueFields) {
    lines.push(`    public var ${f.swiftName}: ${f.swiftType}`)
  }

  lines.push("")

  // Add helper method for sliding window pattern
  if (features?.useSlidingWindow) {
    lines.push("    /// Check if a layer is a global attention layer")
    lines.push("    /// Following mlx-lm pattern: layer is global if i % pattern == pattern - 1")
    lines.push("    public func isGlobalLayer(_ layerIdx: Int) -> Bool {")
    lines.push("        return (layerIdx % slidingWindowPattern) == (slidingWindowPattern - 1)")
    lines.push("    }")
    lines.push("")
  }

  // CodingKeys
  lines.push("    enum CodingKeys: String, CodingKey {")
  if (isVlm) {
    lines.push('        case textConfig = "text_config"')
  }
  for (const f of uniqueFields) {
    lines.push(`        case ${f.swiftName} = "${f.name}"`)
  }
  lines.push("    }")
  lines.push("")

  // Custom init(from decoder:) for VLM or models needing defaults
  lines.push("    public init(from decoder: Swift.Decoder) throws {")
  lines.push("        let container = try decoder.container(keyedBy: CodingKeys.self)")
  lines.push("")
  lines.push("        // Helper to get value from text_config or top level")
  lines.push(
    "        func getValue<T: Decodable>(_ key: CodingKeys, type: T.Type, default defaultValue: T? = nil) throws -> T {"
  )
  lines.push(
    "            if let textContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {"
  )
  lines.push("                if let value = try? textContainer.decode(T.self, forKey: key) {")
  lines.push("                    return value")
  lines.push("                }")
  lines.push("            }")
  lines.push("            if let value = try? container.decode(T.self, forKey: key) {")
  lines.push("                return value")
  lines.push("            }")
  lines.push("            if let defaultValue = defaultValue {")
  lines.push("                return defaultValue")
  lines.push("            }")
  lines.push(
    '            throw DecodingError.keyNotFound(key, DecodingError.Context(codingPath: container.codingPath, debugDescription: "Key not found"))'
  )
  lines.push("        }")
  lines.push("")
  lines.push("        func getOptionalValue<T: Decodable>(_ key: CodingKeys, type: T.Type) -> T? {")
  lines.push(
    "            if let textContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {"
  )
  lines.push("                if let value = try? textContainer.decode(T.self, forKey: key) {")
  lines.push("                    return value")
  lines.push("                }")
  lines.push("            }")
  lines.push("            return try? container.decode(T.self, forKey: key)")
  lines.push("        }")
  lines.push("")

  // Assign fields with defaults
  for (const f of uniqueFields) {
    const baseType = f.swiftType.replace("?", "")
    const isOptional = f.swiftType.endsWith("?")

    // Get default value based on field name
    let defaultValue: string | null = null
    switch (f.name) {
      case "rms_norm_eps":
        defaultValue = "1e-6"
        break
      case "rope_theta":
        defaultValue = features?.defaultRopeTheta?.toString() ?? "10000.0"
        break
      case "rope_local_base_freq":
        defaultValue = "10000.0"
        break
      case "max_position_embeddings":
        defaultValue = "32768"
        break
      case "sliding_window":
        defaultValue = "512"
        break
      case "sliding_window_pattern":
        defaultValue = "6"
        break
      case "head_dim":
        defaultValue = "hiddenSize / numAttentionHeads"
        break
    }

    if (isOptional) {
      lines.push(
        `        ${f.swiftName} = getOptionalValue(.${f.swiftName}, type: ${baseType}.self)`
      )
    } else if (defaultValue) {
      if (defaultValue.includes("/") || defaultValue.includes("numAttentionHeads")) {
        // Computed default
        lines.push(
          `        ${f.swiftName} = getOptionalValue(.${f.swiftName}, type: ${baseType}.self) ?? ${defaultValue}`
        )
      } else {
        lines.push(
          `        ${f.swiftName} = getOptionalValue(.${f.swiftName}, type: ${baseType}.self) ?? ${defaultValue}`
        )
      }
    } else {
      lines.push(`        ${f.swiftName} = try getValue(.${f.swiftName}, type: ${baseType}.self)`)
    }
  }

  lines.push("    }")

  lines.push("}")
  return lines.join("\n")
}
