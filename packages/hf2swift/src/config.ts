/**
 * Swift Configuration struct generator
 *
 * Generates minimal, essential Swift config structs.
 * Focus on fields that are actually needed for model forward pass.
 */

import { toPascal } from "./naming.js"
import type { ModelFeatures } from "./generator.js"

/**
 * Generate Swift Configuration struct - minimal essential fields only
 */
export function generateConfigFromJson(
  _configJson: Record<string, unknown>,
  modelName: string,
  features?: ModelFeatures
): string {
  const className = `${toPascal(modelName)}Configuration`

  const lines: string[] = [
    "// MARK: - Configuration",
    "",
    `public struct ${className}: Decodable, Sendable {`
  ]

  // Essential fields only - these are what the model actually needs
  lines.push("    public var hiddenSize: Int")
  lines.push("    public var numHiddenLayers: Int")
  lines.push("    public var numAttentionHeads: Int")
  lines.push("    public var numKeyValueHeads: Int")
  lines.push("    public var intermediateSize: Int")
  lines.push("    public var vocabSize: Int")
  lines.push("    public var headDim: Int")
  lines.push("    public var rmsNormEps: Float")
  lines.push("    public var ropeTheta: Float")
  lines.push("    public var maxPositionEmbeddings: Int")

  // Sliding window fields
  if (features?.useSlidingWindow) {
    lines.push("    public var slidingWindow: Int")
    lines.push("    public var slidingWindowPattern: Int")
  }

  // Local RoPE theta for sliding window layers
  if (features?.hasLocalRopeTheta) {
    lines.push("    public var ropeLocalTheta: Float")
  }

  // Optional fields
  lines.push("    public var ropeScaling: [String: StringOrNumber]?")

  lines.push("")

  // Helper method for sliding window
  if (features?.useSlidingWindow) {
    lines.push("    /// Check if a layer is a global attention layer")
    lines.push("    public func isGlobalLayer(_ layerIdx: Int) -> Bool {")
    lines.push("        return (layerIdx % slidingWindowPattern) == (slidingWindowPattern - 1)")
    lines.push("    }")
    lines.push("")
  }

  // CodingKeys
  lines.push("    enum CodingKeys: String, CodingKey {")
  lines.push('        case textConfig = "text_config"')
  lines.push('        case hiddenSize = "hidden_size"')
  lines.push('        case numHiddenLayers = "num_hidden_layers"')
  lines.push('        case numAttentionHeads = "num_attention_heads"')
  lines.push('        case numKeyValueHeads = "num_key_value_heads"')
  lines.push('        case intermediateSize = "intermediate_size"')
  lines.push('        case vocabSize = "vocab_size"')
  lines.push('        case headDim = "head_dim"')
  lines.push('        case rmsNormEps = "rms_norm_eps"')
  lines.push('        case ropeTheta = "rope_theta"')
  lines.push('        case maxPositionEmbeddings = "max_position_embeddings"')
  if (features?.useSlidingWindow) {
    lines.push('        case slidingWindow = "sliding_window"')
    lines.push('        case slidingWindowPattern = "sliding_window_pattern"')
  }
  if (features?.hasLocalRopeTheta) {
    lines.push('        case ropeLocalTheta = "rope_local_base_freq"')
  }
  lines.push('        case ropeScaling = "rope_scaling"')
  lines.push("    }")
  lines.push("")

  // Custom decoder with VLM support and defaults
  lines.push("    public init(from decoder: Swift.Decoder) throws {")
  lines.push("        let container = try decoder.container(keyedBy: CodingKeys.self)")
  lines.push("")
  lines.push("        // Helper to decode from text_config or top level")
  lines.push(
    "        func decode<T: Decodable>(_ key: CodingKeys, default defaultValue: T? = nil) throws -> T {"
  )
  lines.push(
    "            if let nested = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig),"
  )
  lines.push("               let value = try? nested.decode(T.self, forKey: key) {")
  lines.push("                return value")
  lines.push("            }")
  lines.push("            if let value = try? container.decode(T.self, forKey: key) {")
  lines.push("                return value")
  lines.push("            }")
  lines.push("            if let defaultValue = defaultValue {")
  lines.push("                return defaultValue")
  lines.push("            }")
  lines.push(
    '            throw DecodingError.keyNotFound(key, DecodingError.Context(codingPath: [], debugDescription: "Missing \\(key)"))'
  )
  lines.push("        }")
  lines.push("")

  // Decode fields
  lines.push("        hiddenSize = try decode(.hiddenSize)")
  lines.push("        numHiddenLayers = try decode(.numHiddenLayers)")
  lines.push("        numAttentionHeads = try decode(.numAttentionHeads)")
  lines.push("        numKeyValueHeads = try decode(.numKeyValueHeads, default: numAttentionHeads)")
  lines.push("        intermediateSize = try decode(.intermediateSize)")
  lines.push("        vocabSize = try decode(.vocabSize)")
  lines.push("        headDim = try decode(.headDim, default: hiddenSize / numAttentionHeads)")
  lines.push("        rmsNormEps = try decode(.rmsNormEps, default: 1e-6)")

  const defaultTheta = features?.defaultRopeTheta ?? 10000
  lines.push(`        ropeTheta = try decode(.ropeTheta, default: ${defaultTheta}.0)`)
  lines.push("        maxPositionEmbeddings = try decode(.maxPositionEmbeddings, default: 32768)")

  if (features?.useSlidingWindow) {
    lines.push("        slidingWindow = try decode(.slidingWindow, default: 512)")
    lines.push("        slidingWindowPattern = try decode(.slidingWindowPattern, default: 6)")
  }

  if (features?.hasLocalRopeTheta) {
    lines.push("        ropeLocalTheta = try decode(.ropeLocalTheta, default: 10000.0)")
  }

  lines.push(
    "        ropeScaling = try? container.decode([String: StringOrNumber].self, forKey: .ropeScaling)"
  )

  lines.push("    }")
  lines.push("}")

  return lines.join("\n")
}

/**
 * Infer Swift type from a JSON value (for reference)
 */
export function inferSwiftType(value: unknown): [string, boolean] {
  if (value === null || value === undefined) return ["Any?", true]
  if (typeof value === "boolean") return ["Bool", true]
  if (typeof value === "number") return Number.isInteger(value) ? ["Int", true] : ["Float", true]
  if (typeof value === "string") return ["String", true]
  if (Array.isArray(value)) {
    if (value.length === 0) return ["[Any]", false]
    const first = value[0]
    if (typeof first === "number")
      return Number.isInteger(first) ? ["[Int]", true] : ["[Float]", true]
    if (typeof first === "string") return ["[String]", true]
    return ["[Any]", false]
  }
  return ["Any", false]
}
