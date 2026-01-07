#!/usr/bin/env npx tsx

/**
 * Benchmark: node-mlx vs node-llama-cpp
 *
 * Compares LLM inference performance on Apple Silicon using:
 * - node-mlx (MLX backend)
 * - node-llama-cpp (llama.cpp backend)
 *
 * Both use Gemma 3n E4B with 4-bit quantization.
 */

import { spawn } from "node:child_process"
import { writeFileSync } from "node:fs"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = dirname(fileURLToPath(import.meta.url))

// Configuration
const PROMPT = "Explain quantum computing in simple terms. Be concise."
const MAX_TOKENS = 100
const WARMUP_RUNS = 1
const BENCHMARK_RUNS = 3

// Models
const MLX_MODEL = "mlx-community/gemma-3n-E4B-it-4bit"
const GGUF_MODEL = "unsloth/gemma-3n-E4B-it-GGUF"

interface BenchmarkResult {
  library: string
  model: string
  runs: RunResult[]
  average: {
    tokensPerSecond: number
    totalTimeMs: number
    loadTimeMs: number
  }
}

interface RunResult {
  tokensPerSecond: number
  totalTimeMs: number
  generatedTokens: number
}

async function runNodeMlx(): Promise<RunResult> {
  const cliPath = join(__dirname, "..", "swift", ".build", "release", "llm-cli")

  return new Promise((resolve, reject) => {
    const startTime = Date.now()

    const child = spawn(cliPath, [
      "generate",
      "--model",
      MLX_MODEL,
      "--prompt",
      PROMPT,
      "--max-tokens",
      String(MAX_TOKENS),
      "--json"
    ])

    let stdout = ""
    let stderr = ""

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString()
    })

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString()
    })

    child.on("close", (code) => {
      const totalTimeMs = Date.now() - startTime

      if (code !== 0) {
        reject(new Error(`node-mlx failed: ${stderr}`))
        return
      }

      try {
        const result = JSON.parse(stdout) as {
          text: string
          generatedTokens: number
          tokensPerSecond: number
        }

        resolve({
          tokensPerSecond: result.tokensPerSecond,
          totalTimeMs,
          generatedTokens: result.generatedTokens
        })
      } catch {
        reject(new Error(`Failed to parse node-mlx output: ${stdout}`))
      }
    })
  })
}

async function runNodeLlamaCpp(): Promise<RunResult> {
  // node-llama-cpp benchmark using CLI
  // Requires: npm install -g node-llama-cpp (or use npx)
  return new Promise((resolve, reject) => {
    const startTime = Date.now()

    const child = spawn(
      "npx",
      [
        "--yes",
        "node-llama-cpp",
        "chat",
        "--model",
        `hf:${GGUF_MODEL}:Q4_K_M`,
        "--prompt",
        PROMPT,
        "--maxTokens",
        String(MAX_TOKENS),
        "--no-interactive"
      ],
      {
        env: { ...process.env, FORCE_COLOR: "0" }
      }
    )

    let stdout = ""
    let stderr = ""

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString()
    })

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString()
    })

    child.on("close", (code) => {
      const totalTimeMs = Date.now() - startTime

      if (code !== 0 && code !== null) {
        // node-llama-cpp might exit with non-zero for various reasons
        console.warn(`node-llama-cpp exited with code ${code}`)
      }

      // Parse tokens/sec from output (format varies)
      const tokensMatch = stdout.match(/(\d+\.?\d*)\s*tokens?\/s/i)
      const tokensPerSecond = tokensMatch ? parseFloat(tokensMatch[1]) : 0

      // Estimate generated tokens from output length
      const generatedTokens = Math.round(stdout.length / 4) // rough estimate

      resolve({
        tokensPerSecond,
        totalTimeMs,
        generatedTokens
      })
    })

    child.on("error", (error) => {
      reject(new Error(`Failed to run node-llama-cpp: ${error.message}`))
    })
  })
}

async function benchmark(
  name: string,
  model: string,
  runFn: () => Promise<RunResult>
): Promise<BenchmarkResult> {
  console.log(`\n${"=".repeat(60)}`)
  console.log(`Benchmarking: ${name}`)
  console.log(`Model: ${model}`)
  console.log(`${"=".repeat(60)}`)

  // Warmup
  console.log(`\nWarmup (${WARMUP_RUNS} run${WARMUP_RUNS > 1 ? "s" : ""})...`)
  for (let i = 0; i < WARMUP_RUNS; i++) {
    try {
      await runFn()
      console.log(`  Warmup ${i + 1} complete`)
    } catch (error) {
      console.error(`  Warmup ${i + 1} failed:`, (error as Error).message)
    }
  }

  // Benchmark runs
  console.log(`\nBenchmark (${BENCHMARK_RUNS} runs)...`)
  const runs: RunResult[] = []

  for (let i = 0; i < BENCHMARK_RUNS; i++) {
    try {
      const result = await runFn()
      runs.push(result)
      console.log(
        `  Run ${i + 1}: ${result.tokensPerSecond.toFixed(1)} tok/s, ${result.totalTimeMs}ms total`
      )
    } catch (error) {
      console.error(`  Run ${i + 1} failed:`, (error as Error).message)
    }
  }

  // Calculate averages
  const avgTokensPerSecond =
    runs.reduce((sum, r) => sum + r.tokensPerSecond, 0) / runs.length
  const avgTotalTimeMs = runs.reduce((sum, r) => sum + r.totalTimeMs, 0) / runs.length

  return {
    library: name,
    model,
    runs,
    average: {
      tokensPerSecond: avgTokensPerSecond,
      totalTimeMs: avgTotalTimeMs,
      loadTimeMs: 0 // Would need separate measurement
    }
  }
}

async function main(): Promise<void> {
  console.log("╔════════════════════════════════════════════════════════════╗")
  console.log("║         node-mlx vs node-llama-cpp Benchmark               ║")
  console.log("╠════════════════════════════════════════════════════════════╣")
  console.log(`║  Prompt: "${PROMPT.substring(0, 45)}..."`)
  console.log(`║  Max Tokens: ${MAX_TOKENS}`)
  console.log(`║  Warmup: ${WARMUP_RUNS} | Runs: ${BENCHMARK_RUNS}`)
  console.log("╚════════════════════════════════════════════════════════════╝")

  const results: BenchmarkResult[] = []

  // Run node-mlx benchmark
  try {
    const mlxResult = await benchmark("node-mlx", MLX_MODEL, runNodeMlx)
    results.push(mlxResult)
  } catch (error) {
    console.error("node-mlx benchmark failed:", (error as Error).message)
  }

  // Run node-llama-cpp benchmark
  try {
    const llamaResult = await benchmark("node-llama-cpp", GGUF_MODEL, runNodeLlamaCpp)
    results.push(llamaResult)
  } catch (error) {
    console.error("node-llama-cpp benchmark failed:", (error as Error).message)
  }

  // Summary
  console.log("\n")
  console.log("╔════════════════════════════════════════════════════════════╗")
  console.log("║                        RESULTS                             ║")
  console.log("╠════════════════════════════════════════════════════════════╣")

  for (const result of results) {
    console.log(`║  ${result.library.padEnd(20)} ${result.average.tokensPerSecond.toFixed(1).padStart(8)} tok/s`)
  }

  if (results.length === 2) {
    const [mlx, llama] = results
    const speedup = mlx.average.tokensPerSecond / llama.average.tokensPerSecond
    console.log("╠════════════════════════════════════════════════════════════╣")
    if (speedup > 1) {
      console.log(`║  node-mlx is ${speedup.toFixed(2)}x faster than node-llama-cpp`)
    } else {
      console.log(`║  node-llama-cpp is ${(1 / speedup).toFixed(2)}x faster than node-mlx`)
    }
  }

  console.log("╚════════════════════════════════════════════════════════════╝")

  // Save results
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
  const resultsPath = join(__dirname, "results", `benchmark-${timestamp}.json`)

  try {
    const { mkdirSync } = await import("node:fs")
    mkdirSync(join(__dirname, "results"), { recursive: true })
    writeFileSync(
      resultsPath,
      JSON.stringify(
        {
          timestamp: new Date().toISOString(),
          config: {
            prompt: PROMPT,
            maxTokens: MAX_TOKENS,
            warmupRuns: WARMUP_RUNS,
            benchmarkRuns: BENCHMARK_RUNS
          },
          results
        },
        null,
        2
      )
    )
    console.log(`\nResults saved to: ${resultsPath}`)
  } catch {
    console.log("\nCould not save results to file")
  }
}

main().catch(console.error)

