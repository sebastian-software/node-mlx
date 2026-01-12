import { platform, arch } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"
import { existsSync, readFileSync } from "node:fs"
import { createRequire } from "node:module"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const require = createRequire(import.meta.url)

// Read version from package.json
const packageJsonPath = join(__dirname, "..", "package.json")
const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf-8")) as { version: string }

/** Package version */
export const VERSION = packageJson.version

// Native binding interface
interface NativeBinding {
  initialize(dylibPath: string): boolean
  isInitialized(): boolean
  loadModel(modelId: string): number
  unloadModel(handle: number): void
  generate(
    handle: number,
    prompt: string,
    options?: {
      maxTokens?: number
      temperature?: number
      topP?: number
      repetitionPenalty?: number
      repetitionContextSize?: number
    }
  ): string // Returns JSON string
  generateStreaming(
    handle: number,
    prompt: string,
    options?: {
      maxTokens?: number
      temperature?: number
      topP?: number
      repetitionPenalty?: number
      repetitionContextSize?: number
    }
  ): string // Streams to stdout, returns JSON stats
  generateWithImage(
    handle: number,
    prompt: string,
    imagePath: string,
    options?: {
      maxTokens?: number
      temperature?: number
      topP?: number
      repetitionPenalty?: number
      repetitionContextSize?: number
    }
  ): string // VLM: Streams to stdout, returns JSON stats
  isVLM(handle: number): boolean
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
 * Note: The library is expected to be in a directory with mlx.metallib for MLX to find it
 */
function findSwiftLibrary(): string {
  const possibleDylibPaths = [
    // From package swift/ (preferred - has metallib co-located)
    join(__dirname, "..", "swift", "libNodeMLX.dylib"),
    // From project root packages/node-mlx/swift/ (monorepo development)
    join(process.cwd(), "packages", "node-mlx", "swift", "libNodeMLX.dylib"),
    // Fallback to packages/swift/.build (monorepo dev)
    join(__dirname, "..", "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    join(__dirname, "..", "..", "..", "swift", ".build", "release", "libNodeMLX.dylib"),
    join(process.cwd(), "packages", "swift", ".build", "release", "libNodeMLX.dylib")
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
  /** Penalty for repeating tokens (1.0 = no penalty, 1.1-1.2 recommended) */
  repetitionPenalty?: number
  /** Number of recent tokens to consider for penalty (default: 20) */
  repetitionContextSize?: number
}

export interface GenerationResult {
  text: string
  tokenCount: number
  tokensPerSecond: number
}

export interface StreamingResult {
  tokenCount: number
  tokensPerSecond: number
}

export interface Model {
  /** Generate text from a prompt */
  generate(prompt: string, options?: GenerationOptions): GenerationResult

  /** Generate text with streaming - tokens are written directly to stdout */
  generateStreaming(prompt: string, options?: GenerationOptions): StreamingResult

  /** Generate text from a prompt with an image (VLM only) */
  generateWithImage(prompt: string, imagePath: string, options?: GenerationOptions): StreamingResult

  /** Check if this model supports images (is a Vision-Language Model) */
  isVLM(): boolean

  /** Unload the model from memory */
  unload(): void

  /** Model handle (internal use) */
  readonly handle: number
}

// MARK: - Recommended Models

export const RECOMMENDED_MODELS = {
  // Qwen 3 (Alibaba) - MLX Community 4-bit quantized
  qwen: "mlx-community/Qwen3-4B-4bit",
  qwen3: "mlx-community/Qwen3-4B-4bit",
  "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
  "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
  "qwen3-32b": "mlx-community/Qwen3-32B-4bit",

  // Phi 4 (Microsoft) - Working with fused QKV and RoPE
  phi: "mlx-community/phi-4-4bit",
  phi4: "mlx-community/phi-4-4bit",

  // Llama 4 (Meta) - 4-bit quantized from mlx-community
  llama: "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
  llama4: "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
  "llama4-17b": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",

  // Gemma 3 (Google) - Standard transformer architecture with sliding window
  gemma: "mlx-community/gemma-3-1b-it-4bit",
  gemma3: "mlx-community/gemma-3-1b-it-4bit",
  "gemma3-1b": "mlx-community/gemma-3-1b-it-4bit",
  "gemma3-1b-bf16": "mlx-community/gemma-3-1b-it-bf16",
  "gemma3-4b": "mlx-community/gemma-3-4b-it-4bit",
  "gemma3-4b-bf16": "mlx-community/gemma-3-4b-it-bf16",
  "gemma3-12b": "mlx-community/gemma-3-12b-it-4bit",
  "gemma3-27b": "mlx-community/gemma-3-27b-it-4bit",

  // Gemma 3n (Google) - Efficient architecture with AltUp and Laurel
  // Note: Use -lm variants (language model only, no audio/vision)
  gemma3n: "mlx-community/gemma-3n-E4B-it-lm-4bit",
  "gemma3n-e2b": "mlx-community/gemma-3n-E2B-it-lm-4bit",
  "gemma3n-e4b": "mlx-community/gemma-3n-E4B-it-lm-4bit"
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
 * const model = loadModel(RECOMMENDED_MODELS["gemma-3n"])
 * const result = model.generate("Hello, world!")
 * console.log(result.text)
 * model.unload()
 * ```
 */
/**
 * Resolve a model ID or alias to a full HuggingFace model ID
 * @param modelId - Either a full HuggingFace model ID (e.g., "mlx-community/phi-4-4bit") or a short alias (e.g., "phi4")
 * @returns The full HuggingFace model ID
 */
function resolveModelId(modelId: string): string {
  // Check if it's an alias in RECOMMENDED_MODELS
  if (modelId in RECOMMENDED_MODELS) {
    return RECOMMENDED_MODELS[modelId as keyof typeof RECOMMENDED_MODELS]
  }

  // Otherwise assume it's already a full model ID
  return modelId
}

export function loadModel(modelId: string): Model {
  const b = loadBinding()
  const resolvedId = resolveModelId(modelId)
  const handle = b.loadModel(resolvedId)

  return {
    handle,

    generate(prompt: string, options?: GenerationOptions): GenerationResult {
      const jsonStr = b.generate(handle, prompt, {
        maxTokens: options?.maxTokens ?? 256,
        temperature: options?.temperature ?? 0.7,
        topP: options?.topP ?? 0.9,
        repetitionPenalty: options?.repetitionPenalty ?? 1.1,
        repetitionContextSize: options?.repetitionContextSize ?? 20
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

    generateStreaming(prompt: string, options?: GenerationOptions): StreamingResult {
      // Tokens are written directly to stdout by Swift
      const jsonStr = b.generateStreaming(handle, prompt, {
        maxTokens: options?.maxTokens ?? 256,
        temperature: options?.temperature ?? 0.7,
        topP: options?.topP ?? 0.9,
        repetitionPenalty: options?.repetitionPenalty ?? 1.1,
        repetitionContextSize: options?.repetitionContextSize ?? 20
      })

      const result = JSON.parse(jsonStr) as JSONGenerationResult

      if (!result.success) {
        throw new Error(result.error ?? "Generation failed")
      }

      return {
        tokenCount: result.tokenCount ?? 0,
        tokensPerSecond: result.tokensPerSecond ?? 0
      }
    },

    generateWithImage(
      prompt: string,
      imagePath: string,
      options?: GenerationOptions
    ): StreamingResult {
      // VLM generation with image - tokens are written directly to stdout by Swift
      const jsonStr = b.generateWithImage(handle, prompt, imagePath, {
        maxTokens: options?.maxTokens ?? 256,
        temperature: options?.temperature ?? 0.7,
        topP: options?.topP ?? 0.9,
        repetitionPenalty: options?.repetitionPenalty ?? 1.1,
        repetitionContextSize: options?.repetitionContextSize ?? 20
      })

      const result = JSON.parse(jsonStr) as JSONGenerationResult

      if (!result.success) {
        throw new Error(result.error ?? "Generation failed")
      }

      return {
        tokenCount: result.tokenCount ?? 0,
        tokensPerSecond: result.tokensPerSecond ?? 0
      }
    },

    isVLM(): boolean {
      return b.isVLM(handle)
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
 *   RECOMMENDED_MODELS["gemma-3n"],
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
