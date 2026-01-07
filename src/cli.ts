#!/usr/bin/env node

import { spawn } from "node:child_process"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { existsSync } from "node:fs"

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Find the llm-cli binary path
 */
function findCliBinary(): string {
  const devPath = join(__dirname, "..", "swift", ".build", "release", "llm-cli")
  if (existsSync(devPath)) {
    return devPath
  }

  throw new Error(
    "llm-cli binary not found. Please run 'npm run build:swift' or install the package correctly."
  )
}

/**
 * Pass through all arguments to the Swift CLI
 */
function main(): void {
  const args = process.argv.slice(2)

  try {
    const cliBinary = findCliBinary()

    const child = spawn(cliBinary, args, {
      stdio: "inherit"
    })

    child.on("close", (code) => {
      process.exit(code ?? 0)
    })

    child.on("error", (error) => {
      console.error(`Failed to start llm-cli: ${error.message}`)
      process.exit(1)
    })
  } catch (error) {
    console.error((error as Error).message)
    process.exit(1)
  }
}

main()
