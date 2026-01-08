/**
 * Parsed module attribute (e.g., self.q_proj = nn.Linear(...))
 */
export interface ModuleAttribute {
  name: string
  swiftName: string
  moduleType: string | null
  initArgs: string[]
  key: string
  isParameter: boolean
}

/**
 * Parsed module property (e.g., self.num_heads = config.num_attention_heads)
 */
export interface ModuleProperty {
  name: string
  swiftName: string
  swiftType: string
  initExpr: string
}

/**
 * Parsed method argument
 */
export interface MethodArg {
  name: string
  type: string
}

/**
 * Parsed module method (e.g., forward)
 */
export interface ModuleMethod {
  name: string
  swiftName: string
  args: MethodArg[]
  body: string[]
  returnType: string
}

/**
 * Parsed Python module class
 */
export interface ParsedModule {
  name: string
  swiftName: string
  attributes: ModuleAttribute[]
  methods: ModuleMethod[]
  properties: ModuleProperty[]
  baseClasses: string[]
}

/**
 * Configuration field for Swift struct generation
 */
export interface ConfigField {
  name: string
  swiftName: string
  swiftType: string
  default: string | null
  optional: boolean
  codingKey: string
}

/**
 * Generator options
 */
export interface GeneratorOptions {
  modelName: string
  configJson?: Record<string, unknown>
  sourceCode?: string
}

/**
 * Known nn.Module mappings to Swift types
 */
export const NN_MODULES: Record<string, [string | null, string[]]> = {
  "nn.Linear": ["Linear", ["in_features", "out_features", "bias"]],
  "nn.Embedding": ["Embedding", ["num_embeddings", "embedding_dim"]],
  "nn.LayerNorm": ["LayerNorm", ["normalized_shape", "eps"]],
  RMSNorm: ["RMSNorm", ["dimensions", "eps"]],
  "nn.RMSNorm": ["RMSNorm", ["dimensions", "eps"]],
  "nn.Dropout": [null, []],
  "nn.ModuleList": ["Array", []],
  "nn.Conv1d": ["Conv1d", ["in_channels", "out_channels", "kernel_size"]],
  "nn.Conv2d": ["Conv2d", ["in_channels", "out_channels", "kernel_size"]]
}

/**
 * Expression conversion patterns (Python → Swift)
 */
export const EXPR_CONVERSIONS: [RegExp, string][] = [
  [/getattr\((\w+),\s*['"](\w+)['"],\s*([^)]+)\)/g, "$1.$2 ?? $3"],
  [/getattr\((\w+),\s*['"](\w+)['"]\)/g, "$1.$2"],
  [/isinstance\((\w+),\s*(\w+)\)/g, "$1 is $2"],
  [/\bint\(([^)]+)\)/g, "Int($1)"],
  [/\bfloat\(([^)]+)\)/g, "Float($1)"],
  [/\band\b/g, "&&"],
  [/\bor\b/g, "||"],
  [/\bTrue\b/g, "true"],
  [/\bFalse\b/g, "false"],
  [/\bNone\b/g, "nil"],
  // Order matters: "is not nil" must come before "not" conversion
  [/ is not nil/g, " != nil"],
  [/ is nil/g, " == nil"],
  [/\bnot\b/g, "!"],
  [/\bself\./g, ""],
  [/\.view\(/g, ".reshaped(["],
  [/\.reshape\(/g, ".reshaped(["],
  [/\.transpose\((\d+),\s*(\d+)\)/g, ".transposed($1, $2)"],
  [/\.contiguous\(\)/g, ""],
  [/\.unsqueeze\((\d+)\)/g, ".expandedDimensions(axis: $1)"],
  [/\.squeeze\((\d+)\)/g, ".squeezed(axis: $1)"],
  [/'([^']*)'/g, '"$1"']
]
