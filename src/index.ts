import { createRequire } from "node:module"
import { platform, arch } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"
import { existsSync } from "node:fs"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// Native binding interface
interface NativeBinding {
  initialize(dylibPath: string): boolean
  isInitialized(): boolean
  loadModel(modelId: string): number
  unloadModel(handle: number): void
  generate(
    handle: number,
    prompt: string,
    options?: { maxTokens?: number; temperature?: number; topP?: number }
  ): string // Returns JSON string
  isAvailable(): boolean
  getVersion(): string
}

// JSON response from Swift
interface JSONGenerationResult {
  success: boolean
  text?: string
  tokenCount?: number
  tokensPerSecond?: number
  error?: string
}

// Load the native addon
let binding: NativeBinding | null = null
let initialized = false

function loadBinding(): NativeBinding {
  if (binding && initialized) {
    return binding
  }

  if (platform() !== "darwin" || arch() !== "arm64") {
    throw new Error("node-mlx is only supported on macOS Apple Silicon (arm64)")
  }

  const require = createRequire(import.meta.url)

  // Try different paths for the native addon
  const possibleAddonPaths = [
    join(__dirname, "..", "native", "build", "Release", "node_mlx.node"),
    join(__dirname, "..", "..", "native", "build", "Release", "node_mlx.node"),
    join(process.cwd(), "native", "build", "Release", "node_mlx.node")
  ]

  let addonPath: string | null = null
  for (const p of possibleAddonPaths) {
    if (existsSync(p)) {
      addonPath = p
      break
    }
  }

  if (!addonPath) {
    throw new Error(
      "Native addon not found. Run 'pnpm build:native' first.\n" +
        `Searched paths:\n${possibleAddonPaths.join("\n")}`
    )
  }

  binding = require(addonPath) as NativeBinding

  // Find and initialize with dylib path
  const possibleDylibPaths = [
    join(__dirname, "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    join(__dirname, "..", "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    join(process.cwd(), "swift", ".build", "release", "libNodeMLX.dylib")
  ]

  let dylibPath: string | null = null
  for (const p of possibleDylibPaths) {
    if (existsSync(p)) {
      dylibPath = p
      break
    }
  }

  if (!dylibPath) {
    throw new Error(
      "Swift library not found. Run 'pnpm build:swift' first.\n" +
        `Searched paths:\n${possibleDylibPaths.join("\n")}`
    )
  }

  const success = binding.initialize(dylibPath)
  if (!success) {
    throw new Error("Failed to initialize node-mlx native library")
  }

  initialized = true
  return binding
}

// MARK: - Public Types

export interface GenerationOptions {
  maxTokens?: number
  temperature?: number
  topP?: number
}

export interface GenerationResult {
  text: string
  tokenCount: number
  tokensPerSecond: number
}

export interface Model {
  /** Generate text from a prompt */
  generate(prompt: string, options?: GenerationOptions): GenerationResult
  /** Unload the model from memory */
  unload(): void
  /** Model handle (internal use) */
  readonly handle: number
}

// MARK: - Recommended Models

export const RECOMMENDED_MODELS = {
  // Gemma 3n (Google) - Efficient on-device model
  "gemma-3n-2b": "mlx-community/gemma-3n-E2B-it-lm-4bit",
  "gemma-3n-4b": "mlx-community/gemma-3n-E4B-it-lm-4bit",

  // Llama 3.2 (Meta) - Fast and capable
  "llama-3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
  "llama-3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",

  // Qwen 2.5 (Alibaba) - Great multilingual support
  "qwen-2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "qwen-2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "qwen-2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",

  // Phi-3 (Microsoft) - Efficient reasoning
  "phi-3-mini": "mlx-community/Phi-3-mini-4k-instruct-4bit"
} as const

export type RecommendedModelKey = keyof typeof RECOMMENDED_MODELS

// MARK: - Public API

/**
 * Check if MLX is available on this system
 * (requires macOS 14+ on Apple Silicon)
 */
export function isSupported(): boolean {
  if (platform() !== "darwin" || arch() !== "arm64") {
    return false
  }

  try {
    const b = loadBinding()
    return b.isAvailable()
  } catch {
    return false
  }
}

/**
 * Get the library version
 */
export function getVersion(): string {
  const b = loadBinding()
  return b.getVersion()
}

/**
 * Load a model from HuggingFace or local path
 *
 * @param modelId - HuggingFace model ID (e.g., "mlx-community/gemma-3n-E2B-it-4bit") or local path
 * @returns Model instance
 *
 * @example
 * ```typescript
 * import { loadModel, RECOMMENDED_MODELS } from "node-mlx"
 *
 * const model = loadModel(RECOMMENDED_MODELS["gemma-3n-2b"])
 * const result = model.generate("Hello, world!")
 * console.log(result.text)
 * model.unload()
 * ```
 */
export function loadModel(modelId: string): Model {
  const b = loadBinding()
  const handle = b.loadModel(modelId)

  return {
    handle,

    generate(prompt: string, options?: GenerationOptions): GenerationResult {
      const jsonStr = b.generate(handle, prompt, {
        maxTokens: options?.maxTokens ?? 256,
        temperature: options?.temperature ?? 0.7,
        topP: options?.topP ?? 0.9
      })

      const result: JSONGenerationResult = JSON.parse(jsonStr)

      if (!result.success) {
        throw new Error(result.error ?? "Generation failed")
      }

      return {
        text: result.text ?? "",
        tokenCount: result.tokenCount ?? 0,
        tokensPerSecond: result.tokensPerSecond ?? 0
      }
    },

    unload(): void {
      b.unloadModel(handle)
    }
  }
}

/**
 * Generate text using a model (one-shot, loads and unloads model)
 *
 * @param modelId - HuggingFace model ID or local path
 * @param prompt - Input text
 * @param options - Generation options
 * @returns Generation result
 *
 * @example
 * ```typescript
 * import { generate } from "node-mlx"
 *
 * const result = generate(
 *   "mlx-community/gemma-3n-E2B-it-4bit",
 *   "Explain quantum computing",
 *   { maxTokens: 100 }
 * )
 * console.log(result.text)
 * ```
 */
export function generate(
  modelId: string,
  prompt: string,
  options?: GenerationOptions
): GenerationResult {
  const model = loadModel(modelId)
  try {
    return model.generate(prompt, options)
  } finally {
    model.unload()
  }
}
