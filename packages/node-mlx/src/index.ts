import { platform, arch } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"
import { existsSync } from "node:fs"
import { createRequire } from "node:module"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const require = createRequire(import.meta.url)

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

/**
 * Load native addon using node-gyp-build (prebuilds) or fallback to built addon
 */
function loadNativeAddon(): NativeBinding {
  // Try node-gyp-build first (prebuilds)
  try {
    const gypBuild = require("node-gyp-build") as (dir: string) => NativeBinding
    const nativeDir = join(__dirname, "..", "native")
    if (existsSync(join(__dirname, "..", "prebuilds"))) {
      return gypBuild(join(__dirname, ".."))
    }
    // Fallback to native/build if no prebuilds
    if (existsSync(join(nativeDir, "build"))) {
      return gypBuild(nativeDir)
    }
  } catch {
    // node-gyp-build failed, try manual loading
  }

  // Manual fallback: try different paths for the native addon
  const possibleAddonPaths = [
    // From package dist/ (npm installed)
    join(__dirname, "..", "prebuilds", "darwin-arm64", "node.napi.node"),
    // From native/build (local development)
    join(__dirname, "..", "native", "build", "Release", "node_mlx.node"),
    // From project root (monorepo development)
    join(process.cwd(), "packages", "node-mlx", "native", "build", "Release", "node_mlx.node")
  ]

  for (const p of possibleAddonPaths) {
    if (existsSync(p)) {
      return require(p) as NativeBinding
    }
  }

  throw new Error(
    "Native addon not found. Run 'pnpm build:native' first.\n" +
      `Searched paths:\n${possibleAddonPaths.join("\n")}`
  )
}

/**
 * Find Swift library path
 */
function findSwiftLibrary(): string {
  const possibleDylibPaths = [
    // From packages/swift/.build (monorepo dev, running from src/)
    join(__dirname, "..", "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    // From packages/swift/.build (monorepo dev, running from dist/)
    join(__dirname, "..", "..", "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    // From project root (monorepo development)
    join(process.cwd(), "packages", "swift", ".build", "release", "libNodeMLX.dylib"),
    // From package swift/ (npm installed, running from dist/)
    join(__dirname, "..", "swift", "libNodeMLX.dylib")
  ]

  for (const p of possibleDylibPaths) {
    if (existsSync(p)) {
      return p
    }
  }

  throw new Error(
    "Swift library not found. Run 'pnpm build:swift' first.\n" +
      `Searched paths:\n${possibleDylibPaths.join("\n")}`
  )
}

function loadBinding(): NativeBinding {
  if (binding && initialized) {
    return binding
  }

  if (platform() !== "darwin" || arch() !== "arm64") {
    throw new Error("node-mlx is only supported on macOS Apple Silicon (arm64)")
  }

  binding = loadNativeAddon()
  const dylibPath = findSwiftLibrary()

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
  // Qwen 2.5 (Alibaba) - Working with proper RoPE support
  // Using non-quantized models - quantized models have loading issues
  qwen: "Qwen/Qwen2.5-1.5B-Instruct",
  "qwen-2.5": "Qwen/Qwen2.5-1.5B-Instruct",
  "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
  "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
  "qwen-2.5-3b": "Qwen/Qwen2.5-3B-Instruct",

  // Phi (Microsoft) - Working with fused QKV and RoPE
  phi: "microsoft/phi-4", // Default to latest
  phi4: "microsoft/phi-4",
  "phi-4": "microsoft/phi-4",
  phi3: "microsoft/Phi-3-mini-4k-instruct",
  "phi-3": "microsoft/Phi-3-mini-4k-instruct",
  "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",

  // Llama 3.2 (Meta) - Requires HuggingFace authentication
  // Note: meta-llama models require accepting license at huggingface.co
  llama: "meta-llama/Llama-3.2-1B-Instruct",
  "llama-3.2": "meta-llama/Llama-3.2-1B-Instruct",
  "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
  "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"

  // TODO: These models need fixes:
  // - Gemma3n: Complex AltUp/Laurel architecture
  // - Mistral: GQA head count compatibility
} as const

export type RecommendedModelKey = keyof typeof RECOMMENDED_MODELS

// MARK: - Public API

/**
 * Check if the platform is Apple Silicon Mac
 */
export function isPlatformSupported(): boolean {
  return platform() === "darwin" && arch() === "arm64"
}

/**
 * Check if MLX is available on this system
 * (requires macOS 14+ on Apple Silicon with built binaries)
 */
export function isSupported(): boolean {
  if (!isPlatformSupported()) {
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

      const result = JSON.parse(jsonStr) as JSONGenerationResult

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
