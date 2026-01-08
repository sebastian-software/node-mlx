/**
 * hf2swift - HuggingFace Transformers to MLX Swift Code Generator
 *
 * This package provides tools to parse Python model code from HuggingFace
 * Transformers and generate equivalent MLX Swift code.
 */

export { HFModelParser } from "./parser.js"
export { SwiftGenerator, formatSwift } from "./generator.js"
export { generateConfigFromJson, inferSwiftType } from "./config.js"
export { toCamel, toPascal, capitalize, convertExpr } from "./naming.js"
export type {
  ParsedModule,
  ModuleAttribute,
  ModuleProperty,
  ModuleMethod,
  MethodArg,
  ConfigField,
  GeneratorOptions
} from "./types.js"
export { NN_MODULES, EXPR_CONVERSIONS } from "./types.js"
