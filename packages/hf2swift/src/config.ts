/**
 * Swift Configuration struct generator
 *
 * Generates Swift Codable structs from HuggingFace config.json
 */

import { ConfigField } from "./types.js"
import { toCamel, toPascal } from "./naming.js"

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
  return ["Any", false]
}

/**
 * Generate Swift Configuration struct from config.json
 */
export function generateConfigFromJson(
  configJson: Record<string, unknown>,
  modelName: string
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
    "rope_scaling",
    "quantization_config",
    "task_specific_params",
    "id2label",
    "label2id",
    "vision_config",
    "audio_config"
  ])

  const defaultFields: Record<string, unknown> = {
    attention_bias: false,
    rms_norm_eps: 1e-6,
    num_key_value_heads: null
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
      if (key.startsWith("_") || skipFields.has(key)) {
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
    if (skipFields.has(key) || (isVlm && key === "text_config")) {
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

  // Generate Swift code
  const className = `${toPascal(modelName)}Configuration`
  const protocol = isVlm ? "Decodable, Sendable" : "Codable, Sendable"
  const lines: string[] = [`public struct ${className}: ${protocol} {`]

  // Properties
  for (const f of fields) {
    lines.push(`    public var ${f.swiftName}: ${f.swiftType}`)
  }

  lines.push("")

  // CodingKeys
  if (isVlm) {
    const textConfigFieldNames = new Set(
      Object.keys(textConfig)
        .filter((k) => !k.startsWith("_"))
        .map((k) => toCamel(k))
    )

    lines.push("    enum CodingKeys: String, CodingKey {")
    lines.push('        case textConfig = "text_config"')
    for (const f of fields) {
      if (!textConfigFieldNames.has(f.swiftName)) {
        lines.push(`        case ${f.swiftName} = "${f.name}"`)
      }
    }
    lines.push("    }")
    lines.push("")

    lines.push("    enum TextConfigCodingKeys: String, CodingKey {")
    for (const f of fields) {
      if (textConfigFieldNames.has(f.swiftName)) {
        lines.push(`        case ${f.swiftName} = "${f.name}"`)
      }
    }
    lines.push("    }")
    lines.push("")

    // Custom init(from decoder:)
    lines.push("    public init(from decoder: Swift.Decoder) throws {")
    lines.push("        let container = try decoder.container(keyedBy: CodingKeys.self)")
    lines.push(
      "        let textContainer = try container.nestedContainer(keyedBy: TextConfigCodingKeys.self, forKey: .textConfig)"
    )
    lines.push("")

    for (const f of fields) {
      const baseType = f.swiftType.replace("?", "")
      if (textConfigFieldNames.has(f.swiftName)) {
        if (f.optional || f.swiftType.endsWith("?")) {
          lines.push(
            `        self.${f.swiftName} = try textContainer.decodeIfPresent(${baseType}.self, forKey: .${f.swiftName})`
          )
        } else {
          lines.push(
            `        self.${f.swiftName} = try textContainer.decode(${baseType}.self, forKey: .${f.swiftName})`
          )
        }
      } else {
        if (f.optional || f.swiftType.endsWith("?")) {
          lines.push(
            `        self.${f.swiftName} = try container.decodeIfPresent(${baseType}.self, forKey: .${f.swiftName})`
          )
        } else {
          lines.push(
            `        self.${f.swiftName} = try container.decode(${baseType}.self, forKey: .${f.swiftName})`
          )
        }
      }
    }
    lines.push("    }")
  } else {
    lines.push("    enum CodingKeys: String, CodingKey {")
    for (const f of fields) {
      lines.push(`        case ${f.swiftName} = "${f.name}"`)
    }
    lines.push("    }")
  }

  lines.push("")

  // Computed property for headDim if not present
  const hasHeadDim = fields.some((f) => f.name === "head_dim")
  if (!hasHeadDim) {
    lines.push("    public var headDim: Int {")
    lines.push("        hiddenSize / numAttentionHeads")
    lines.push("    }")
  }

  lines.push("}")
  return lines.join("\n")
}
