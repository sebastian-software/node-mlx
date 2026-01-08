#!/usr/bin/env node
/**
 * hf2swift CLI
 *
 * Generate MLX Swift model code from HuggingFace Transformers Python source
 */

import { readFileSync, writeFileSync } from "fs"
import { HFModelParser } from "./parser.js"
import { SwiftGenerator } from "./generator.js"

interface CliArgs {
  model: string
  source?: string
  config?: string
  output?: string
}

function parseArgs(): CliArgs {
  const args: CliArgs = { model: "" }
  const argv = process.argv.slice(2)

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === "--model" && argv[i + 1]) {
      args.model = argv[++i]
    } else if (arg === "--source" && argv[i + 1]) {
      args.source = argv[++i]
    } else if (arg === "--config" && argv[i + 1]) {
      args.config = argv[++i]
    } else if (arg === "--output" && argv[i + 1]) {
      args.output = argv[++i]
    } else if (arg === "--help" || arg === "-h") {
      printHelp()
      process.exit(0)
    }
  }

  if (!args.model) {
    console.error("Error: --model is required")
    printHelp()
    process.exit(1)
  }

  return args
}

function printHelp(): void {
  console.log(`
hf2swift - HuggingFace to MLX Swift Code Generator

Usage:
  hf2swift --model <name> [options]

Options:
  --model <name>     Model name (required)
  --source <path>    Python source file path (modeling_*.py)
  --config <id>      HuggingFace model ID for config.json
  --output <path>    Output Swift file path (prints to stdout if not specified)
  --help, -h         Show this help message

Examples:
  hf2swift --model qwen2 --source modeling_qwen2.py --output Qwen2.swift
  hf2swift --model gemma3n --source modeling_gemma3n.py --config mlx-community/gemma-3n-E4B-it-lm-4bit
`)
}

async function fetchConfig(modelId: string): Promise<Record<string, unknown> | null> {
  const url = `https://huggingface.co/${modelId}/raw/main/config.json`
  try {
    const response = await fetch(url)
    if (!response.ok) {
      console.error(`Warning: Could not fetch config from ${url}`)
      return null
    }
    return (await response.json()) as Record<string, unknown>
  } catch (error) {
    console.error(`Warning: Could not load config: ${error}`)
    return null
  }
}

async function main(): Promise<void> {
  const args = parseArgs()

  // Read source file if provided
  let source = ""
  if (args.source) {
    try {
      source = readFileSync(args.source, "utf-8")
    } catch (error) {
      console.error(`Error reading source file: ${error}`)
      process.exit(1)
    }
  }

  // Parse Python source
  const parser = new HFModelParser(args.model)
  const modules = source ? parser.parse(source) : []

  console.error(`Parsed ${modules.length} modules:`)
  for (const m of modules) {
    console.error(
      `  - ${m.name}: ${m.attributes.length} attrs, bases=[${m.baseClasses.join(", ")}]`
    )
  }

  // Fetch config if specified
  let configJson: Record<string, unknown> | undefined
  if (args.config) {
    configJson = (await fetchConfig(args.config)) ?? undefined
  }

  // Generate Swift code
  const generator = new SwiftGenerator(args.model)
  const code = generator.generate(modules, configJson)

  // Output
  if (args.output) {
    writeFileSync(args.output, code, "utf-8")
    console.error(`Generated: ${args.output}`)
  } else {
    console.log(code)
  }
}

main().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
