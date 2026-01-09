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

  // Per-layer intermediate sizes for AltUp models
  if (features?.hasPerLayerIntermediateSize) {
    lines.push("    public var intermediateSizes: [Int]  // Per-layer intermediate sizes")
  } else {
    lines.push("    public var intermediateSize: Int")
  }

  lines.push("    public var vocabSize: Int")
  lines.push("    public var headDim: Int")
  lines.push("    public var rmsNormEps: Float")
  lines.push("    public var ropeTheta: Float")
  lines.push("    public var maxPositionEmbeddings: Int")

  // Sliding window fields
  if (features?.useSlidingWindow) {
    lines.push("    public var slidingWindow: Int")
  }

  // Sliding window pattern (only for non-AltUp models)
  if (features?.useSlidingWindow && !features?.hasAltUp) {
    lines.push("    public var slidingWindowPattern: Int")
  }

  // Layer types for models with mixed attention (Gemma3n)
  if (features?.hasAltUp) {
    lines.push("    public var layerTypes: [String]")
  }

  // Local RoPE theta for sliding window layers
  if (features?.hasLocalRopeTheta) {
    lines.push("    public var ropeLocalBaseFreq: Float")
  }

  // KV-cache sharing
  if (features?.hasKVSharing) {
    lines.push("    public var numKVSharedLayers: Int")
  }

  // AltUp specific fields
  if (features?.hasAltUp) {
    lines.push("")
    lines.push("    // AltUp configuration")
    lines.push("    public var altupNumInputs: Int")
    lines.push("    public var altupActiveIdx: Int")
    lines.push("    public var altupCorrectScale: Bool")
    lines.push("    public var altupCoefClip: Float?")
  }

  // Laurel block
  if (features?.hasLaurel) {
    lines.push("    public var laurelRank: Int")
  }

  // Per-layer inputs
  if (features?.hasPerLayerInputs) {
    lines.push("    public var hiddenSizePerLayerInput: Int")
    lines.push("    public var vocabSizePerLayerInput: Int")
  }

  // Sparse activation
  if (features?.hasSparseActivation) {
    lines.push("    public var activationSparsityPattern: [Float]")
  }

  // Logit softcapping
  if (features?.hasLogitSoftcapping) {
    lines.push("    public var finalLogitSoftcapping: Float?")
  }

  // Optional fields
  lines.push("    public var ropeScaling: [String: StringOrNumber]?")
  lines.push("    public var modelType: String?")

  lines.push("")

  // Helper methods
  if (features?.hasPerLayerIntermediateSize) {
    lines.push("    /// Get intermediate size for a specific layer")
    lines.push("    public func intermediateSize(forLayer idx: Int) -> Int {")
    lines.push("        if idx < intermediateSizes.count {")
    lines.push("            return intermediateSizes[idx]")
    lines.push("        }")
    lines.push("        return intermediateSizes.first ?? 16384")
    lines.push("    }")
    lines.push("")
  }

  if (features?.hasKVSharing) {
    lines.push("    /// First KV shared layer index")
    lines.push("    public var firstKVSharedLayerIdx: Int {")
    lines.push("        return numHiddenLayers - numKVSharedLayers")
    lines.push("    }")
    lines.push("")
    lines.push("    /// Check if a layer uses shared KV cache")
    lines.push("    public func isKVSharedLayer(_ layerIdx: Int) -> Bool {")
    lines.push("        return layerIdx >= firstKVSharedLayerIdx")
    lines.push("    }")
    lines.push("")
  }

  // isGlobalLayer implementation
  if (features?.useSlidingWindow) {
    lines.push("    /// Check if a layer is a global attention layer")
    lines.push("    public func isGlobalLayer(_ layerIdx: Int) -> Bool {")
    if (features?.hasAltUp) {
      lines.push("        if layerIdx < layerTypes.count {")
      lines.push("            let layerType = layerTypes[layerIdx].lowercased()")
      lines.push('            return layerType == "full_attention" || layerType == "global"')
      lines.push("        }")
      lines.push("        return false")
    } else {
      lines.push("        return (layerIdx % slidingWindowPattern) == (slidingWindowPattern - 1)")
    }
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
    if (!features?.hasAltUp) {
      lines.push('        case slidingWindowPattern = "sliding_window_pattern"')
    }
  }
  if (features?.hasAltUp) {
    lines.push('        case layerTypes = "layer_types"')
  }
  if (features?.hasLocalRopeTheta) {
    lines.push('        case ropeLocalBaseFreq = "rope_local_base_freq"')
  }
  if (features?.hasKVSharing) {
    lines.push('        case numKVSharedLayers = "num_kv_shared_layers"')
  }
  if (features?.hasAltUp) {
    lines.push('        case altupNumInputs = "altup_num_inputs"')
    lines.push('        case altupActiveIdx = "altup_active_idx"')
    lines.push('        case altupCorrectScale = "altup_correct_scale"')
    lines.push('        case altupCoefClip = "altup_coef_clip"')
  }
  if (features?.hasLaurel) {
    lines.push('        case laurelRank = "laurel_rank"')
  }
  if (features?.hasPerLayerInputs) {
    lines.push('        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"')
    lines.push('        case vocabSizePerLayerInput = "vocab_size_per_layer_input"')
  }
  if (features?.hasSparseActivation) {
    lines.push('        case activationSparsityPattern = "activation_sparsity_pattern"')
  }
  if (features?.hasLogitSoftcapping) {
    lines.push('        case finalLogitSoftcapping = "final_logit_softcapping"')
  }
  lines.push('        case ropeScaling = "rope_scaling"')
  lines.push('        case modelType = "model_type"')
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

  // Decode essential fields
  lines.push("        hiddenSize = try decode(.hiddenSize)")
  lines.push("        numHiddenLayers = try decode(.numHiddenLayers)")
  lines.push("        numAttentionHeads = try decode(.numAttentionHeads)")
  lines.push("        numKeyValueHeads = try decode(.numKeyValueHeads, default: numAttentionHeads)")

  // Per-layer intermediate sizes
  if (features?.hasPerLayerIntermediateSize) {
    lines.push("")
    lines.push("        // intermediate_size can be a single Int or [Int] array")
    lines.push("        if let sizes: [Int] = try? decode(.intermediateSize) {")
    lines.push("            intermediateSizes = sizes")
    lines.push("        } else if let size: Int = try? decode(.intermediateSize) {")
    lines.push("            intermediateSizes = Array(repeating: size, count: numHiddenLayers)")
    lines.push("        } else {")
    lines.push("            intermediateSizes = Array(repeating: 16384, count: numHiddenLayers)")
    lines.push("        }")
  } else {
    lines.push("        intermediateSize = try decode(.intermediateSize)")
  }

  lines.push("        vocabSize = try decode(.vocabSize)")
  lines.push("        headDim = try decode(.headDim, default: hiddenSize / numAttentionHeads)")
  lines.push("        rmsNormEps = try decode(.rmsNormEps, default: 1e-6)")

  const defaultTheta = features?.defaultRopeTheta ?? 10000
  lines.push(`        ropeTheta = try decode(.ropeTheta, default: ${defaultTheta}.0)`)
  lines.push("        maxPositionEmbeddings = try decode(.maxPositionEmbeddings, default: 32768)")

  if (features?.useSlidingWindow) {
    lines.push("        slidingWindow = try decode(.slidingWindow, default: 512)")
    if (!features?.hasAltUp) {
      lines.push("        slidingWindowPattern = try decode(.slidingWindowPattern, default: 6)")
    }
  }

  if (features?.hasAltUp) {
    lines.push("")
    lines.push("        if let types: [String] = try? decode(.layerTypes) {")
    lines.push("            layerTypes = types")
    lines.push("        } else {")
    lines.push("            layerTypes = []")
    lines.push("        }")
  }

  if (features?.hasLocalRopeTheta) {
    lines.push("        ropeLocalBaseFreq = try decode(.ropeLocalBaseFreq, default: 10000.0)")
  }

  if (features?.hasKVSharing) {
    lines.push("        numKVSharedLayers = try decode(.numKVSharedLayers, default: 0)")
  }

  if (features?.hasAltUp) {
    lines.push("")
    lines.push("        // AltUp configuration with defaults")
    lines.push("        altupNumInputs = try decode(.altupNumInputs, default: 4)")
    lines.push("        altupActiveIdx = try decode(.altupActiveIdx, default: 0)")
    lines.push("        altupCorrectScale = try decode(.altupCorrectScale, default: true)")
    lines.push("        altupCoefClip = try? decode(.altupCoefClip) as Float")
  }

  if (features?.hasLaurel) {
    lines.push("        laurelRank = try decode(.laurelRank, default: 64)")
  }

  if (features?.hasPerLayerInputs) {
    lines.push(
      "        hiddenSizePerLayerInput = try decode(.hiddenSizePerLayerInput, default: 256)"
    )
    lines.push(
      "        vocabSizePerLayerInput = try decode(.vocabSizePerLayerInput, default: 262144)"
    )
  }

  if (features?.hasSparseActivation) {
    lines.push("")
    lines.push("        if let pattern: [Float] = try? decode(.activationSparsityPattern) {")
    lines.push("            activationSparsityPattern = pattern")
    lines.push("        } else {")
    lines.push("            activationSparsityPattern = []")
    lines.push("        }")
  }

  if (features?.hasLogitSoftcapping) {
    lines.push("        finalLogitSoftcapping = try? decode(.finalLogitSoftcapping)")
  }

  lines.push(
    "        ropeScaling = try? container.decode([String: StringOrNumber].self, forKey: .ropeScaling)"
  )
  lines.push("        modelType = try? container.decode(String.self, forKey: .modelType)")

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
