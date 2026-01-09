/**
 * Swift Code Generator
 *
 * Generates MLX Swift model code from parsed Python modules.
 * Output is formatted with SwiftFormat for consistent style.
 *
 * Production quality: Uses MLXFast, RoPE providers, proper caching,
 * sliding window attention, geluApproximate, clipResidual, etc.
 *
 * Advanced features (Gemma3n and future models):
 * - AltUp (Alternating Updates)
 * - Laurel (Learned Augmented Residual) blocks
 * - Per-layer inputs
 * - KV-cache sharing
 * - Sparse activation
 */

import { execSync } from "node:child_process"
import type { ParsedModule } from "../types.js"
import { toPascal } from "../naming.js"
import { generateConfigFromJson } from "../config.js"
import { type ModelFeatures, getModelFeatures } from "./features.js"
import {
  generateHeader,
  generateHelpers,
  generateLaurelBlock,
  generateAltUpBlock
} from "./helpers.js"
import {
  generateRmsNorm,
  generateAttention,
  generateMlp,
  generateDecoderLayer,
  generateModelInner,
  generateModel
} from "./components/index.js"

// Re-export types
export { type ModelFeatures, getModelFeatures } from "./features.js"

/**
 * Format Swift code using swiftformat
 */
export function formatSwift(code: string): string {
  try {
    return execSync("swiftformat stdin", {
      input: code,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024
    })
  } catch {
    console.warn("Warning: swiftformat not available, returning unformatted code")
    return code
  }
}

/**
 * Swift Model Generator - Production Quality
 */
export class SwiftGenerator {
  private modelName: string
  private configClass: string
  private features: ModelFeatures

  constructor(modelName: string, features?: ModelFeatures) {
    this.modelName = toPascal(modelName)
    this.configClass = `${this.modelName}Configuration`
    this.features = features ?? getModelFeatures(modelName)
  }

  /**
   * Generate complete Swift file
   */
  generate(_modules: ParsedModule[], configJson?: Record<string, unknown>): string {
    const normType = `${this.modelName}RMSNorm`

    const parts: string[] = [
      generateHeader(this.modelName),
      configJson ? generateConfigFromJson(configJson, this.modelName, this.features) : "",
      generateRmsNorm(this.modelName, this.features),
      generateHelpers(this.features)
    ]

    // Add AltUp block if needed (must come before DecoderLayer)
    if (this.features.hasAltUp) {
      parts.push(generateAltUpBlock(this.modelName, normType))
    }

    // Add Laurel block if needed (must come before DecoderLayer)
    if (this.features.hasLaurel) {
      parts.push(generateLaurelBlock(this.modelName, this.configClass, normType))
    }

    // Core components
    parts.push(generateAttention(this.modelName, this.configClass, this.features))
    parts.push(generateMlp(this.modelName, this.configClass, this.features))
    parts.push(generateDecoderLayer(this.modelName, this.configClass, this.features))
    parts.push(generateModelInner(this.modelName, this.configClass, this.features))
    parts.push(generateModel(this.modelName, this.configClass, this.features))

    const code = parts.filter(Boolean).join("\n\n")
    return formatSwift(code)
  }
}
