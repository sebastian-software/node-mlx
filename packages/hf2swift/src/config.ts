/**
 * Swift Configuration struct generator
 *
 * Generates minimal, essential Swift config structs.
 * Focus on fields that are actually needed for model forward pass.
 *
 * Note: Output is not formatted - SwiftFormat handles that.
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

  const parts: string[] = []
  parts.push("// MARK: - Configuration\n")

  // Add RoPEParameters struct for YaRN models
  if (features?.hasYarnRope) {
    parts.push(generateRoPEParametersStruct())
  }

  // Main configuration struct
  parts.push(`public struct ${className}: Decodable, Sendable {`)
  parts.push(generatePropertyDeclarations(features))
  parts.push(generateHelperMethods(features))
  parts.push(generateCodingKeys(features))
  parts.push(generateDecoder(features))
  parts.push("}")

  return parts.join("\n")
}

function generateRoPEParametersStruct(): string {
  return `
/// YaRN RoPE parameters for long context support
public struct RoPEParameters: Decodable, Sendable {
public var ropeTheta: Float
public var ropeType: String
public var factor: Float
public var mscale: Float
public var mscaleAllDim: Float
public var originalMaxPositionEmbeddings: Int
public var betaFast: Float
public var betaSlow: Float

enum CodingKeys: String, CodingKey {
case ropeTheta = "rope_theta"
case ropeType = "rope_type"
case factor
case mscale
case mscaleAllDim = "mscale_all_dim"
case originalMaxPositionEmbeddings = "original_max_position_embeddings"
case betaFast = "beta_fast"
case betaSlow = "beta_slow"
}

public init(from decoder: Swift.Decoder) throws {
let container = try decoder.container(keyedBy: CodingKeys.self)
ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "yarn"
factor = try container.decodeIfPresent(Float.self, forKey: .factor) ?? 1.0
mscale = try container.decodeIfPresent(Float.self, forKey: .mscale) ?? 1.0
mscaleAllDim = try container.decodeIfPresent(Float.self, forKey: .mscaleAllDim) ?? 1.0
originalMaxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings) ?? 16384
betaFast = try container.decodeIfPresent(Float.self, forKey: .betaFast) ?? 32.0
betaSlow = try container.decodeIfPresent(Float.self, forKey: .betaSlow) ?? 1.0
}
}
`
}

function generatePropertyDeclarations(features?: ModelFeatures): string {
  const lines: string[] = []

  // Essential fields
  lines.push("public var hiddenSize: Int")
  lines.push("public var numHiddenLayers: Int")
  lines.push("public var numAttentionHeads: Int")
  lines.push("public var numKeyValueHeads: Int")

  if (features?.hasPerLayerIntermediateSize) {
    lines.push("public var intermediateSizes: [Int]  // Per-layer intermediate sizes")
  } else {
    lines.push("public var intermediateSize: Int")
  }

  lines.push("public var vocabSize: Int")
  lines.push("public var headDim: Int")
  lines.push("public var rmsNormEps: Float")
  lines.push("public var ropeTheta: Float")
  lines.push("public var maxPositionEmbeddings: Int")
  lines.push("public var attentionBias: Bool")
  lines.push("public var mlpBias: Bool")

  // MoE configuration
  if (features?.hasMoE) {
    lines.push("")
    lines.push("// Mixture of Experts configuration")
    lines.push("public var numLocalExperts: Int")
    lines.push("public var numExpertsPerTok: Int")
  }

  // Sliding window
  if (features?.useSlidingWindow) {
    lines.push("public var slidingWindow: Int")
    if (!features.hasAltUp) {
      lines.push("public var slidingWindowPattern: Int")
    }
  }

  // Layer types
  if (features?.hasAltUp || features?.hasMoE) {
    lines.push("public var layerTypes: [String]")
  }

  // Local RoPE theta
  if (features?.hasLocalRopeTheta) {
    lines.push("public var ropeLocalBaseFreq: Float")
  }

  // KV-cache sharing
  if (features?.hasKVSharing) {
    lines.push("public var numKVSharedLayers: Int")
  }

  // AltUp specific
  if (features?.hasAltUp) {
    lines.push("")
    lines.push("// AltUp configuration")
    lines.push("public var altupNumInputs: Int")
    lines.push("public var altupActiveIdx: Int")
    lines.push("public var altupCorrectScale: Bool")
    lines.push("public var altupCoefClip: Float?")
  }

  // Laurel
  if (features?.hasLaurel) {
    lines.push("public var laurelRank: Int")
  }

  // Per-layer inputs
  if (features?.hasPerLayerInputs) {
    lines.push("public var hiddenSizePerLayerInput: Int")
    lines.push("public var vocabSizePerLayerInput: Int")
  }

  // Sparse activation
  if (features?.hasSparseActivation) {
    lines.push("public var activationSparsityPattern: [Float]")
  }

  // Logit softcapping
  if (features?.hasLogitSoftcapping) {
    lines.push("public var finalLogitSoftcapping: Float?")
  }

  // SmolLM3 no-rope layers
  if (features?.hasNoRopeLayers) {
    lines.push("public var noRopeLayers: [Int]  // Layers that skip RoPE (1 = skip, 0 = use)")
  }

  // YaRN RoPE
  if (features?.hasYarnRope) {
    lines.push("public var ropeParameters: RoPEParameters?")
  }

  // Optional fields
  lines.push("public var ropeScaling: [String: StringOrNumber]?")
  lines.push("public var modelType: String?")

  return lines.join("\n")
}

function generateHelperMethods(features?: ModelFeatures): string {
  const lines: string[] = [""]

  if (features?.hasPerLayerIntermediateSize) {
    lines.push(`
/// Get intermediate size for a specific layer
public func intermediateSize(forLayer idx: Int) -> Int {
if idx < intermediateSizes.count {
return intermediateSizes[idx]
}
return intermediateSizes.first ?? 16384
}
`)
  }

  if (features?.hasKVSharing) {
    lines.push(`
/// First KV shared layer index
public var firstKVSharedLayerIdx: Int {
return numHiddenLayers - numKVSharedLayers
}

/// Check if a layer uses shared KV cache
public func isKVSharedLayer(_ layerIdx: Int) -> Bool {
return layerIdx >= firstKVSharedLayerIdx
}
`)
  }

  if (features?.hasNoRopeLayers) {
    lines.push(`
/// Check if a layer should skip RoPE
public func shouldSkipRope(_ layerIdx: Int) -> Bool {
if layerIdx < noRopeLayers.count {
return noRopeLayers[layerIdx] == 1
}
return false
}
`)
  }

  if (features?.useSlidingWindow) {
    if (features.hasAltUp || features.hasMoE) {
      lines.push(`
/// Check if a layer is a global attention layer
public func isGlobalLayer(_ layerIdx: Int) -> Bool {
if layerIdx < layerTypes.count {
let layerType = layerTypes[layerIdx].lowercased()
return layerType == "full_attention" || layerType == "global"
}
return false
}
`)
    } else {
      lines.push(`
/// Check if a layer is a global attention layer
public func isGlobalLayer(_ layerIdx: Int) -> Bool {
return (layerIdx % slidingWindowPattern) == (slidingWindowPattern - 1)
}
`)
    }
  }

  return lines.join("")
}

function generateCodingKeys(features?: ModelFeatures): string {
  const keys: string[] = []

  keys.push('case textConfig = "text_config"')
  keys.push('case hiddenSize = "hidden_size"')
  keys.push('case numHiddenLayers = "num_hidden_layers"')
  keys.push('case numAttentionHeads = "num_attention_heads"')
  keys.push('case numKeyValueHeads = "num_key_value_heads"')
  keys.push('case intermediateSize = "intermediate_size"')
  keys.push('case vocabSize = "vocab_size"')
  keys.push('case headDim = "head_dim"')
  keys.push('case rmsNormEps = "rms_norm_eps"')
  keys.push('case ropeTheta = "rope_theta"')
  keys.push('case maxPositionEmbeddings = "max_position_embeddings"')
  keys.push('case attentionBias = "attention_bias"')
  keys.push('case mlpBias = "mlp_bias"')

  if (features?.hasMoE) {
    keys.push('case numLocalExperts = "num_local_experts"')
    keys.push('case numExpertsPerTok = "num_experts_per_tok"')
  }

  if (features?.useSlidingWindow) {
    keys.push('case slidingWindow = "sliding_window"')
    if (!features.hasAltUp) {
      keys.push('case slidingWindowPattern = "sliding_window_pattern"')
    }
  }

  if (features?.hasAltUp || features?.hasMoE) {
    keys.push('case layerTypes = "layer_types"')
  }

  if (features?.hasLocalRopeTheta) {
    keys.push('case ropeLocalBaseFreq = "rope_local_base_freq"')
  }

  if (features?.hasKVSharing) {
    keys.push('case numKVSharedLayers = "num_kv_shared_layers"')
  }

  if (features?.hasAltUp) {
    keys.push('case altupNumInputs = "altup_num_inputs"')
    keys.push('case altupActiveIdx = "altup_active_idx"')
    keys.push('case altupCorrectScale = "altup_correct_scale"')
    keys.push('case altupCoefClip = "altup_coef_clip"')
  }

  if (features?.hasLaurel) {
    keys.push('case laurelRank = "laurel_rank"')
  }

  if (features?.hasPerLayerInputs) {
    keys.push('case hiddenSizePerLayerInput = "hidden_size_per_layer_input"')
    keys.push('case vocabSizePerLayerInput = "vocab_size_per_layer_input"')
  }

  if (features?.hasSparseActivation) {
    keys.push('case activationSparsityPattern = "activation_sparsity_pattern"')
  }

  if (features?.hasLogitSoftcapping) {
    keys.push('case finalLogitSoftcapping = "final_logit_softcapping"')
  }

  if (features?.hasNoRopeLayers) {
    keys.push('case noRopeLayers = "no_rope_layers"')
  }

  if (features?.hasYarnRope) {
    keys.push('case ropeParameters = "rope_parameters"')
  }

  keys.push('case ropeScaling = "rope_scaling"')
  keys.push('case modelType = "model_type"')

  return `
enum CodingKeys: String, CodingKey {
${keys.join("\n")}
}
`
}

function generateDecoder(features?: ModelFeatures): string {
  const lines: string[] = []

  lines.push(`
public init(from decoder: Swift.Decoder) throws {
let container = try decoder.container(keyedBy: CodingKeys.self)

// Helper to decode from text_config or top level
func decode<T: Decodable>(_ key: CodingKeys, default defaultValue: T? = nil) throws -> T {
if let nested = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig),
   let value = try? nested.decode(T.self, forKey: key) {
return value
}
if let value = try? container.decode(T.self, forKey: key) {
return value
}
if let defaultValue = defaultValue {
return defaultValue
}
throw DecodingError.keyNotFound(key, DecodingError.Context(codingPath: [], debugDescription: "Missing \\(key)"))
}

hiddenSize = try decode(.hiddenSize)
numHiddenLayers = try decode(.numHiddenLayers)
numAttentionHeads = try decode(.numAttentionHeads)
numKeyValueHeads = try decode(.numKeyValueHeads, default: numAttentionHeads)
`)

  // Per-layer intermediate sizes
  if (features?.hasPerLayerIntermediateSize) {
    lines.push(`
// intermediate_size can be a single Int or [Int] array
if let sizes: [Int] = try? decode(.intermediateSize) {
intermediateSizes = sizes
} else if let size: Int = try? decode(.intermediateSize) {
intermediateSizes = Array(repeating: size, count: numHiddenLayers)
} else {
intermediateSizes = Array(repeating: 16384, count: numHiddenLayers)
}
`)
  } else {
    lines.push("intermediateSize = try decode(.intermediateSize)")
  }

  const defaultTheta = features?.defaultRopeTheta ?? 10000
  const defaultAttnBias = features?.hasAttentionBias ?? false
  const defaultMlpBias = features?.hasMlpBias ?? false
  const defaultRmsNormEps = features?.defaultRmsNormEps ?? 1e-6

  lines.push(`
vocabSize = try decode(.vocabSize)
headDim = try decode(.headDim, default: hiddenSize / numAttentionHeads)
rmsNormEps = try decode(.rmsNormEps, default: ${String(defaultRmsNormEps)})
ropeTheta = try decode(.ropeTheta, default: ${String(defaultTheta)}.0)
maxPositionEmbeddings = try decode(.maxPositionEmbeddings, default: 32768)
attentionBias = try decode(.attentionBias, default: ${String(defaultAttnBias)})
mlpBias = try decode(.mlpBias, default: ${String(defaultMlpBias)})
`)

  if (features?.hasMoE) {
    const numExperts = String(features.numExperts ?? 32)
    const numExpertsPerTok = String(features.numExpertsPerTok ?? 4)
    lines.push(`
// MoE configuration
numLocalExperts = try decode(.numLocalExperts, default: ${numExperts})
numExpertsPerTok = try decode(.numExpertsPerTok, default: ${numExpertsPerTok})
`)
  }

  if (features?.useSlidingWindow) {
    const defaultSlidingWindow = features.defaultSlidingWindow ?? 512
    lines.push(
      `slidingWindow = try decode(.slidingWindow, default: ${String(defaultSlidingWindow)})`
    )
    if (!features.hasAltUp) {
      lines.push("slidingWindowPattern = try decode(.slidingWindowPattern, default: 6)")
    }
  }

  if (features?.hasAltUp || features?.hasMoE) {
    const defaultPattern =
      features.hasMoE && !features.hasAltUp
        ? '(0..<numHiddenLayers).map { $0 % 2 == 0 ? "sliding_attention" : "full_attention" }'
        : "[]"

    lines.push(`
if let types: [String] = try? decode(.layerTypes) {
layerTypes = types
} else {
layerTypes = ${defaultPattern}
}
`)
  }

  if (features?.hasLocalRopeTheta) {
    lines.push("ropeLocalBaseFreq = try decode(.ropeLocalBaseFreq, default: 10000.0)")
  }

  if (features?.hasKVSharing) {
    lines.push("numKVSharedLayers = try decode(.numKVSharedLayers, default: 0)")
  }

  if (features?.hasAltUp) {
    lines.push(`
// AltUp configuration with defaults
altupNumInputs = try decode(.altupNumInputs, default: 4)
altupActiveIdx = try decode(.altupActiveIdx, default: 0)
altupCorrectScale = try decode(.altupCorrectScale, default: true)
altupCoefClip = try? decode(.altupCoefClip) as Float
`)
  }

  if (features?.hasLaurel) {
    lines.push("laurelRank = try decode(.laurelRank, default: 64)")
  }

  if (features?.hasPerLayerInputs) {
    lines.push("hiddenSizePerLayerInput = try decode(.hiddenSizePerLayerInput, default: 256)")
    lines.push("vocabSizePerLayerInput = try decode(.vocabSizePerLayerInput, default: 262144)")
  }

  if (features?.hasSparseActivation) {
    lines.push(`
if let pattern: [Float] = try? decode(.activationSparsityPattern) {
activationSparsityPattern = pattern
} else {
activationSparsityPattern = []
}
`)
  }

  if (features?.hasLogitSoftcapping) {
    lines.push("finalLogitSoftcapping = try? decode(.finalLogitSoftcapping)")
  }

  if (features?.hasNoRopeLayers) {
    lines.push(`
// SmolLM3 no_rope_layers configuration
if let layers: [Int] = try? decode(.noRopeLayers) {
noRopeLayers = layers
} else {
// Default: apply RoPE to all layers
noRopeLayers = Array(repeating: 0, count: numHiddenLayers)
}
`)
  }

  if (features?.hasYarnRope) {
    lines.push(
      "ropeParameters = try? container.decode(RoPEParameters.self, forKey: .ropeParameters)"
    )
  }

  lines.push(
    "ropeScaling = try? container.decode([String: StringOrNumber].self, forKey: .ropeScaling)"
  )
  lines.push("modelType = try? container.decode(String.self, forKey: .modelType)")
  lines.push("}")

  return lines.join("\n")
}

/**
 * Infer Swift type from a JSON value (for reference)
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
    const first: unknown = value[0]
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
