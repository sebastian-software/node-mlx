/**
 * MLX CLI - Interactive chat with LLMs on Apple Silicon
 *
 * Usage:
 *   mlx                          # Interactive mode with default model
 *   mlx --model llama-3.2-1b     # Use a specific model
 *   mlx "What is 2+2?"           # One-shot query
 *   mlx --list                   # List available models
 */

import * as readline from "node:readline"
import {
  loadModel,
  RECOMMENDED_MODELS,
  isSupported,
  isPlatformSupported,
  type Model,
  type GenerationOptions,
  type RecommendedModelKey
} from "./index.js"

// ANSI colors
const colors = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  magenta: "\x1b[35m",
  red: "\x1b[31m"
}

function log(msg: string) {
  console.log(msg)
}

function error(msg: string) {
  console.error(`${colors.red}Error:${colors.reset} ${msg}`)
}

function printHeader() {
  log("")
  log(`${colors.bold}${colors.cyan}╔══════════════════════════════════════╗${colors.reset}`)
  log(
    `${colors.bold}${colors.cyan}║${colors.reset}  ${colors.bold}MLX CLI${colors.reset} - LLMs on Apple Silicon   ${colors.cyan}║${colors.reset}`
  )
  log(`${colors.bold}${colors.cyan}╚══════════════════════════════════════╝${colors.reset}`)
  log("")
}

function printHelp() {
  log(`${colors.bold}Usage:${colors.reset}`)
  log(`  mlx                              Interactive chat`)
  log(`  mlx "prompt"                     One-shot generation`)
  log(`  mlx --model <name>               Use specific model`)
  log(`  mlx --list                       List available models`)
  log(`  mlx --help                       Show this help`)
  log("")
  log(`${colors.bold}Interactive commands:${colors.reset}`)
  log(`  /model <name>                    Switch model`)
  log(`  /temp <0-2>                      Set temperature`)
  log(`  /tokens <n>                      Set max tokens`)
  log(`  /clear                           Clear conversation`)
  log(`  /help                            Show commands`)
  log(`  /quit                            Exit`)
  log("")
}

function printModels() {
  log(`${colors.bold}Available models:${colors.reset}`)
  log("")

  // Group models by family, showing unique HuggingFace IDs with all their aliases
  const modelsByHfId = new Map<string, string[]>()

  for (const [alias, hfId] of Object.entries(RECOMMENDED_MODELS)) {
    if (!modelsByHfId.has(hfId)) {
      modelsByHfId.set(hfId, [])
    }

    modelsByHfId.get(hfId)!.push(alias)
  }

  // Organize by family
  const families = [
    {
      name: "Phi (Microsoft)",
      prefix: "Phi",
      desc: "Reasoning & coding"
    },
    {
      name: "Gemma (Google)",
      prefix: "gemma",
      desc: "Efficient on-device"
    },
    {
      name: "Llama (Meta)",
      prefix: "Llama",
      desc: "General purpose"
    },
    {
      name: "Qwen (Alibaba)",
      prefix: "Qwen",
      desc: "Multilingual"
    },
    {
      name: "Mistral",
      prefix: "Mistral",
      desc: "Balanced performance"
    },
    {
      name: "Ministral",
      prefix: "Ministral",
      desc: "Fast inference"
    }
  ]

  for (const family of families) {
    const familyModels = Array.from(modelsByHfId.entries()).filter(([hfId]) =>
      hfId.toLowerCase().includes(family.prefix.toLowerCase())
    )

    if (familyModels.length === 0) continue

    log(`${colors.bold}${family.name}${colors.reset} ${colors.dim}— ${family.desc}${colors.reset}`)

    for (const [hfId, aliases] of familyModels) {
      // Sort aliases: shortest first, then alphabetically
      const sortedAliases = aliases.sort((a, b) => a.length - b.length || a.localeCompare(b))
      const primary = sortedAliases[0]
      const others = sortedAliases.slice(1)

      const aliasStr =
        others.length > 0
          ? `${colors.green}${primary}${colors.reset} ${colors.dim}(${others.join(", ")})${colors.reset}`
          : `${colors.green}${primary}${colors.reset}`

      log(`  ${aliasStr.padEnd(45)} ${colors.dim}${hfId}${colors.reset}`)
    }

    log("")
  }

  log(`${colors.dim}Or use any mlx-community model:${colors.reset}`)
  log(`  ${colors.cyan}node-mlx --model mlx-community/YourModel-4bit${colors.reset}`)
  log("")
}

function resolveModel(name: string): string {
  // Check if it's a shortcut
  if (name in RECOMMENDED_MODELS) {
    return RECOMMENDED_MODELS[name as RecommendedModelKey]
  }

  // Check if it's already a full model ID
  if (name.includes("/")) {
    return name
  }

  // Assume it's from mlx-community
  return `mlx-community/${name}`
}

interface ChatState {
  model: Model | null
  modelName: string
  options: GenerationOptions
  history: Array<{ role: "user" | "assistant"; content: string }>
}

async function runInteractive(initialModel: string) {
  const state: ChatState = {
    model: null,
    modelName: initialModel,
    options: {
      maxTokens: 512,
      temperature: 0.7,
      topP: 0.9
    },
    history: []
  }

  // Load initial model
  log(`${colors.dim}Loading ${state.modelName}...${colors.reset}`)
  const modelId = resolveModel(state.modelName)

  try {
    state.model = loadModel(modelId)
    log(`${colors.green}✓${colors.reset} Model loaded`)
  } catch (err) {
    error(`Failed to load model: ${err instanceof Error ? err.message : String(err)}`)
    process.exit(1)
  }

  log("")
  log(`${colors.dim}Type your message or /help for commands${colors.reset}`)
  log("")

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  })

  const prompt = () => {
    rl.question(`${colors.cyan}You:${colors.reset} `, async (input) => {
      const trimmed = input.trim()

      if (!trimmed) {
        prompt()

        return
      }

      // Handle commands
      if (trimmed.startsWith("/")) {
        await handleCommand(trimmed, state, rl)
        prompt()

        return
      }

      // Generate response
      if (!state.model) {
        error("No model loaded")
        prompt()

        return
      }

      // Build prompt with history (simple format)
      const fullPrompt = buildPrompt(state.history, trimmed)
      state.history.push({ role: "user", content: trimmed })

      process.stdout.write(`${colors.magenta}AI:${colors.reset} `)

      try {
        const result = state.model.generate(fullPrompt, state.options)

        log(result.text)
        log(
          `${colors.dim}(${result.tokenCount} tokens, ${result.tokensPerSecond.toFixed(1)} tok/s)${colors.reset}`
        )
        log("")

        state.history.push({ role: "assistant", content: result.text })
      } catch (err) {
        log("")
        error(err instanceof Error ? err.message : String(err))
      }

      prompt()
    })
  }

  rl.on("close", () => {
    log("")
    log(`${colors.dim}Goodbye!${colors.reset}`)

    if (state.model) {
      state.model.unload()
    }

    process.exit(0)
  })

  prompt()
}

function buildPrompt(
  history: Array<{ role: "user" | "assistant"; content: string }>,
  current: string
): string {
  // Simple chat format
  let prompt = ""

  for (const msg of history.slice(-6)) {
    // Keep last 3 exchanges
    if (msg.role === "user") {
      prompt += `User: ${msg.content}\n`
    } else {
      prompt += `Assistant: ${msg.content}\n`
    }
  }

  prompt += `User: ${current}\nAssistant:`

  return prompt
}

async function handleCommand(input: string, state: ChatState, rl: readline.Interface) {
  const [cmd, ...args] = input.slice(1).split(" ")
  const arg = args.join(" ")

  switch (cmd) {
    case "help":
    case "h":
      printHelp()
      break

    case "quit":
    case "q":
    case "exit":
      rl.close()
      break

    case "clear":
    case "c":
      state.history = []
      log(`${colors.dim}Conversation cleared${colors.reset}`)
      break

    case "model":
    case "m":
      if (!arg) {
        log(`${colors.dim}Current model: ${state.modelName}${colors.reset}`)
        log(`${colors.dim}Use /model <name> to switch${colors.reset}`)
      } else {
        log(`${colors.dim}Loading ${arg}...${colors.reset}`)

        if (state.model) {
          state.model.unload()
        }

        try {
          state.model = loadModel(resolveModel(arg))
          state.modelName = arg
          state.history = []
          log(`${colors.green}✓${colors.reset} Switched to ${arg}`)
        } catch (err) {
          error(err instanceof Error ? err.message : String(err))
        }
      }
      break

    case "temp":
    case "t":
      if (!arg) {
        log(`${colors.dim}Temperature: ${state.options.temperature}${colors.reset}`)
      } else {
        const temp = parseFloat(arg)

        if (isNaN(temp) || temp < 0 || temp > 2) {
          error("Temperature must be between 0 and 2")
        } else {
          state.options.temperature = temp
          log(`${colors.dim}Temperature set to ${temp}${colors.reset}`)
        }
      }
      break

    case "tokens":
    case "n":
      if (!arg) {
        log(`${colors.dim}Max tokens: ${state.options.maxTokens}${colors.reset}`)
      } else {
        const tokens = parseInt(arg, 10)

        if (isNaN(tokens) || tokens < 1) {
          error("Tokens must be a positive number")
        } else {
          state.options.maxTokens = tokens
          log(`${colors.dim}Max tokens set to ${tokens}${colors.reset}`)
        }
      }
      break

    case "list":
    case "l":
      printModels()
      break

    default:
      error(`Unknown command: /${cmd}. Type /help for commands.`)
  }
}

async function runOneShot(modelName: string, prompt: string, options: GenerationOptions) {
  log(`${colors.dim}Loading ${modelName}...${colors.reset}`)

  const modelId = resolveModel(modelName)

  try {
    const model = loadModel(modelId)

    log(`${colors.dim}Generating...${colors.reset}`)
    log("")

    const result = model.generate(prompt, options)

    log(result.text)
    log("")
    log(
      `${colors.dim}(${result.tokenCount} tokens, ${result.tokensPerSecond.toFixed(1)} tok/s)${colors.reset}`
    )

    model.unload()
  } catch (err) {
    error(err instanceof Error ? err.message : String(err))
    process.exit(1)
  }
}

// Parse CLI arguments
function parseArgs(): {
  model: string
  prompt: string | null
  options: GenerationOptions
  command: "chat" | "oneshot" | "list" | "help" | "version"
} {
  const args = process.argv.slice(2)
  let model = "llama-3.2-1b"
  let prompt: string | null = null
  const options: GenerationOptions = {
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.9
  }
  let command: "chat" | "oneshot" | "list" | "help" | "version" = "chat"

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]

    if (arg === "--help" || arg === "-h") {
      command = "help"
    } else if (arg === "--version" || arg === "-v") {
      command = "version"
    } else if (arg === "--list" || arg === "-l") {
      command = "list"
    } else if (arg === "--model" || arg === "-m") {
      model = args[++i] || model
    } else if (arg === "--temp" || arg === "-t") {
      options.temperature = parseFloat(args[++i] || "0.7")
    } else if (arg === "--tokens" || arg === "-n") {
      options.maxTokens = parseInt(args[++i] || "512", 10)
    } else if (!arg.startsWith("-")) {
      prompt = arg
      command = "oneshot"
    }
  }

  return { model, prompt, options, command }
}

// Main
async function main() {
  const { model, prompt, options, command } = parseArgs()

  // Commands that don't need Apple Silicon
  switch (command) {
    case "help":
      printHeader()
      printHelp()

      return

    case "version":
      log(`node-mlx v0.1.0`)

      return

    case "list":
      printHeader()
      printModels()

      return
  }

  // Check platform for commands that need the runtime
  if (!isPlatformSupported()) {
    error("node-mlx requires macOS on Apple Silicon (M1/M2/M3/M4)")
    process.exit(1)
  }

  if (!isSupported()) {
    error("Native libraries not found. Run 'pnpm build:swift && pnpm build:native' first.")
    process.exit(1)
  }

  switch (command) {
    case "oneshot":
      await runOneShot(model, prompt!, options)
      break

    case "chat":
      printHeader()
      await runInteractive(model)
      break
  }
}

main().catch((err) => {
  error(err instanceof Error ? err.message : String(err))
  process.exit(1)
})
