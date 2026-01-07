import { spawn } from "node:child_process"
import { createInterface } from "node:readline"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { existsSync } from "node:fs"

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Find the llm-cli binary path
 */
function findCliBinary(): string {
  // In development: swift/.build/release/llm-cli
  const devPath = join(__dirname, "..", "swift", ".build", "release", "llm-cli")
  if (existsSync(devPath)) {
    return devPath
  }

  // In published package: swift/.build/release/llm-cli (relative to dist)
  const pkgPath = join(__dirname, "..", "swift", ".build", "release", "llm-cli")
  if (existsSync(pkgPath)) {
    return pkgPath
  }

  throw new Error(
    "llm-cli binary not found. Please run 'npm run build:swift' or install the package correctly."
  )
}

/**
 * Generation result from the LLM
 */
export interface GenerationResult {
  text: string
  generatedTokens: number
  tokensPerSecond: number
}

/**
 * Options for text generation
 */
export interface GenerateOptions {
  /** Model name or HuggingFace path (default: mlx-community/gemma-3n-E2B-it-4bit) */
  model?: string
  /** Maximum tokens to generate (default: 256) */
  maxTokens?: number
  /** Temperature for sampling (default: 0.7) */
  temperature?: number
  /** Top-p sampling (default: 0.9) */
  topP?: number
}

/**
 * Options for chat
 */
export interface ChatOptions {
  /** Model name or HuggingFace path (default: mlx-community/gemma-3n-E2B-it-4bit) */
  model?: string
  /** System prompt */
  system?: string
  /** Maximum tokens per response (default: 512) */
  maxTokens?: number
  /** Temperature for sampling (default: 0.7) */
  temperature?: number
}

/**
 * Chat message
 */
export interface ChatMessage {
  role: "system" | "user" | "assistant"
  content: string
}

/**
 * Generate text from a prompt
 */
export async function generate(
  prompt: string,
  options: GenerateOptions = {}
): Promise<GenerationResult> {
  const cliBinary = findCliBinary()

  const args = ["generate", "--prompt", prompt, "--json"]

  if (options.model) {
    args.push("--model", options.model)
  }
  if (options.maxTokens !== undefined) {
    args.push("--max-tokens", String(options.maxTokens))
  }
  if (options.temperature !== undefined) {
    args.push("--temperature", String(options.temperature))
  }
  if (options.topP !== undefined) {
    args.push("--top-p", String(options.topP))
  }

  return new Promise((resolve, reject) => {
    const child = spawn(cliBinary, args)

    let stdout = ""
    let stderr = ""

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString()
    })

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString()
    })

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`llm-cli exited with code ${code}: ${stderr}`))
        return
      }

      try {
        const result = JSON.parse(stdout) as GenerationResult
        resolve(result)
      } catch {
        reject(new Error(`Failed to parse llm-cli output: ${stdout}`))
      }
    })

    child.on("error", (error) => {
      reject(new Error(`Failed to spawn llm-cli: ${error.message}`))
    })
  })
}

/**
 * Generate text with streaming output
 */
export async function* generateStream(
  prompt: string,
  options: GenerateOptions = {}
): AsyncGenerator<string, GenerationResult, unknown> {
  const cliBinary = findCliBinary()

  const args = ["generate", "--prompt", prompt]

  if (options.model) {
    args.push("--model", options.model)
  }
  if (options.maxTokens !== undefined) {
    args.push("--max-tokens", String(options.maxTokens))
  }
  if (options.temperature !== undefined) {
    args.push("--temperature", String(options.temperature))
  }
  if (options.topP !== undefined) {
    args.push("--top-p", String(options.topP))
  }

  const child = spawn(cliBinary, args)
  let fullText = ""

  const rl = createInterface({
    input: child.stdout,
    crlfDelay: Infinity
  })

  for await (const chunk of rl) {
    fullText += chunk
    yield chunk
  }

  // Wait for process to complete
  await new Promise<void>((resolve, reject) => {
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`llm-cli exited with code ${code}`))
      } else {
        resolve()
      }
    })
  })

  // Return approximate stats (actual stats would need JSON mode)
  return {
    text: fullText,
    generatedTokens: 0,
    tokensPerSecond: 0
  }
}

/**
 * Create a chat session
 */
export function createChat(options: ChatOptions = {}): Chat {
  return new Chat(options)
}

/**
 * Chat session class for multi-turn conversations
 */
export class Chat {
  private messages: ChatMessage[] = []
  private options: ChatOptions

  constructor(options: ChatOptions = {}) {
    this.options = options

    if (options.system) {
      this.messages.push({ role: "system", content: options.system })
    }
  }

  /**
   * Send a message and get a response
   */
  async send(message: string): Promise<string> {
    this.messages.push({ role: "user", content: message })

    // For now, we reconstruct the prompt from messages
    // A more sophisticated implementation would use the chat subcommand
    const prompt = this.formatMessagesAsPrompt()

    const result = await generate(prompt, {
      model: this.options.model,
      maxTokens: this.options.maxTokens,
      temperature: this.options.temperature
    })

    this.messages.push({ role: "assistant", content: result.text })
    return result.text
  }

  /**
   * Get conversation history
   */
  getHistory(): ChatMessage[] {
    return [...this.messages]
  }

  /**
   * Clear conversation history
   */
  clear(): void {
    this.messages = []
    if (this.options.system) {
      this.messages.push({ role: "system", content: this.options.system })
    }
  }

  private formatMessagesAsPrompt(): string {
    return this.messages
      .map((msg) => {
        switch (msg.role) {
          case "system":
            return `System: ${msg.content}`
          case "user":
            return `User: ${msg.content}`
          case "assistant":
            return `Assistant: ${msg.content}`
        }
      })
      .join("\n\n")
  }
}

/**
 * Check if the current platform is supported (macOS on Apple Silicon)
 */
export function isSupported(): boolean {
  return process.platform === "darwin" && process.arch === "arm64"
}

/**
 * List of recommended models
 */
export const RECOMMENDED_MODELS = {
  // Gemma 3n (Google) - Optimized for on-device
  "gemma-3n-2b": "mlx-community/gemma-3n-E2B-it-4bit",
  "gemma-3n-4b": "mlx-community/gemma-3n-E4B-it-4bit",

  // Gemma 3 (Google)
  "gemma-3-4b": "mlx-community/gemma-3-4b-it-4bit",
  "gemma-3-12b": "mlx-community/gemma-3-12b-it-4bit",

  // Qwen 3 (Alibaba)
  "qwen-3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
  "qwen-3-4b": "mlx-community/Qwen3-4B-4bit",

  // Phi 4 (Microsoft)
  "phi-4": "mlx-community/phi-4-4bit",

  // Llama 4 (Meta)
  "llama-4-scout": "mlx-community/Llama-4-Scout-17B-4bit"
} as const

export type ModelAlias = keyof typeof RECOMMENDED_MODELS
