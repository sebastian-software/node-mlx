/**
 * Robust Benchmark: node-mlx vs node-llama-cpp
 *
 * - Multiple runs with warmup
 * - Statistical analysis (mean, std, min, max)
 * - Different token counts
 * - Cold and warm start measurements
 */

import { loadModel as loadMLX } from "node-mlx"
import * as fs from "fs"
import * as path from "path"

// ============ Configuration ============
const CONFIG = {
  warmupRuns: 2,
  benchmarkRuns: 5,
  tokenCounts: [50, 100, 200],
  prompts: [
    "Explain quantum entanglement in simple terms:",
    "Write a short story about a robot learning to paint:",
    "What are the main differences between Python and JavaScript?"
  ]
}

interface RunResult {
  tokens: number
  timeMs: number
  tokensPerSec: number
}

interface BenchmarkResult {
  library: string
  model: string
  loadTimeMs: number
  runs: RunResult[]
  stats: {
    meanTps: number
    stdTps: number
    minTps: number
    maxTps: number
    medianTps: number
  }
}

function calculateStats(values: number[]): {
  mean: number
  std: number
  min: number
  max: number
  median: number
} {
  const n = values.length
  const mean = values.reduce((a, b) => a + b, 0) / n
  const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n
  const std = Math.sqrt(variance)
  const sorted = [...values].sort((a, b) => a - b)
  const median = sorted[Math.floor(n / 2)]

  return {
    mean,
    std,
    min: Math.min(...values),
    max: Math.max(...values),
    median
  }
}

async function benchmarkMLX(modelId: string, modelName: string): Promise<BenchmarkResult> {
  console.log(`\n🍎 Benchmarking node-mlx: ${modelName}`)
  console.log("-".repeat(50))

  // Load model
  const loadStart = Date.now()
  const model = loadMLX(modelId)
  const loadTimeMs = Date.now() - loadStart
  console.log(`  Load time: ${(loadTimeMs / 1000).toFixed(2)}s`)

  const runs: RunResult[] = []

  // Warmup
  console.log(`  Warmup (${CONFIG.warmupRuns} runs)...`)
  for (let i = 0; i < CONFIG.warmupRuns; i++) {
    model.generate(CONFIG.prompts[0], { maxTokens: 50, temperature: 0.7 })
  }

  // Benchmark runs
  console.log(
    `  Benchmark (${CONFIG.benchmarkRuns} runs × ${CONFIG.tokenCounts.length} token counts)...`
  )

  for (const maxTokens of CONFIG.tokenCounts) {
    for (let run = 0; run < CONFIG.benchmarkRuns; run++) {
      const prompt = CONFIG.prompts[run % CONFIG.prompts.length]
      const start = Date.now()
      const result = model.generate(prompt, { maxTokens, temperature: 0.7 })
      const timeMs = Date.now() - start

      runs.push({
        tokens: result.tokenCount,
        timeMs,
        tokensPerSec: result.tokensPerSecond
      })

      process.stdout.write(".")
    }
  }
  console.log(" done")

  model.unload()

  // Calculate stats
  const tpsValues = runs.map((r) => r.tokensPerSec)
  const stats = calculateStats(tpsValues)

  return {
    library: "node-mlx",
    model: modelName,
    loadTimeMs,
    runs,
    stats: {
      meanTps: stats.mean,
      stdTps: stats.std,
      minTps: stats.min,
      maxTps: stats.max,
      medianTps: stats.median
    }
  }
}

async function benchmarkLlamaCpp(
  modelPath: string,
  modelName: string
): Promise<BenchmarkResult | null> {
  console.log(`\n🦙 Benchmarking node-llama-cpp: ${modelName}`)
  console.log("-".repeat(50))

  const expandedPath = modelPath.startsWith("~")
    ? modelPath.replace("~", process.env.HOME || "")
    : path.join(process.cwd(), modelPath)

  if (!fs.existsSync(expandedPath)) {
    console.log(`  ❌ Model not found: ${expandedPath}`)
    return null
  }

  try {
    const { getLlama, LlamaChatSession } = await import("node-llama-cpp")

    // Load model
    const loadStart = Date.now()
    const llama = await getLlama()
    const model = await llama.loadModel({ modelPath: expandedPath })
    const context = await model.createContext()
    const loadTimeMs = Date.now() - loadStart
    console.log(`  Load time: ${(loadTimeMs / 1000).toFixed(2)}s`)

    const runs: RunResult[] = []

    // Warmup - create fresh context each time
    console.log(`  Warmup (${CONFIG.warmupRuns} runs)...`)
    for (let i = 0; i < CONFIG.warmupRuns; i++) {
      const ctx = await model.createContext()
      const session = new LlamaChatSession({ contextSequence: ctx.getSequence() })
      await session.prompt(CONFIG.prompts[0], { maxTokens: 50 })
      await ctx.dispose()
    }

    // Benchmark runs - fresh context for each to avoid sequence exhaustion
    console.log(
      `  Benchmark (${CONFIG.benchmarkRuns} runs × ${CONFIG.tokenCounts.length} token counts)...`
    )

    for (const maxTokens of CONFIG.tokenCounts) {
      for (let run = 0; run < CONFIG.benchmarkRuns; run++) {
        const prompt = CONFIG.prompts[run % CONFIG.prompts.length]

        const ctx = await model.createContext()
        const session = new LlamaChatSession({ contextSequence: ctx.getSequence() })

        const start = Date.now()
        let tokens = 0

        await session.prompt(prompt, {
          maxTokens,
          temperature: 0.7,
          onTextChunk: () => {
            tokens++
          }
        })

        const timeMs = Date.now() - start
        const tokensPerSec = (tokens / timeMs) * 1000

        runs.push({
          tokens,
          timeMs,
          tokensPerSec
        })

        await ctx.dispose()
        process.stdout.write(".")
      }
    }
    console.log(" done")

    await context.dispose()
    await model.dispose()

    // Calculate stats
    const tpsValues = runs.map((r) => r.tokensPerSec)
    const stats = calculateStats(tpsValues)

    return {
      library: "node-llama-cpp",
      model: modelName,
      loadTimeMs,
      runs,
      stats: {
        meanTps: stats.mean,
        stdTps: stats.std,
        minTps: stats.min,
        maxTps: stats.max,
        medianTps: stats.median
      }
    }
  } catch (error) {
    console.log(`  ❌ Error: ${error}`)
    return null
  }
}

function printResults(results: BenchmarkResult[]) {
  console.log("\n")
  console.log("═".repeat(80))
  console.log("📊 BENCHMARK RESULTS")
  console.log("═".repeat(80))

  console.log("\nConfiguration:")
  console.log(`  Warmup runs: ${CONFIG.warmupRuns}`)
  console.log(`  Benchmark runs: ${CONFIG.benchmarkRuns}`)
  console.log(`  Token counts: ${CONFIG.tokenCounts.join(", ")}`)
  console.log(
    `  Total measurements per library: ${CONFIG.benchmarkRuns * CONFIG.tokenCounts.length}`
  )

  console.log("\n┌─────────────────┬──────────────┬─────────────────────────────────────────────┐")
  console.log("│ Library         │ Load Time    │ Tokens/sec (mean ± std, min-max)            │")
  console.log("├─────────────────┼──────────────┼─────────────────────────────────────────────┤")

  for (const r of results) {
    const name = r.library.padEnd(15)
    const load = `${(r.loadTimeMs / 1000).toFixed(1)}s`.padEnd(12)
    const stats =
      `${r.stats.meanTps.toFixed(1)} ± ${r.stats.stdTps.toFixed(1)} (${r.stats.minTps.toFixed(0)}-${r.stats.maxTps.toFixed(0)})`.padEnd(
        43
      )
    console.log(`│ ${name} │ ${load} │ ${stats} │`)
  }

  console.log("└─────────────────┴──────────────┴─────────────────────────────────────────────┘")

  // Comparison
  if (results.length >= 2) {
    const mlx = results.find((r) => r.library === "node-mlx")
    const llama = results.find((r) => r.library === "node-llama-cpp")

    if (mlx && llama) {
      const speedup = mlx.stats.meanTps / llama.stats.meanTps
      const winner = speedup > 1 ? "node-mlx" : "node-llama-cpp"
      const ratio = speedup > 1 ? speedup : 1 / speedup

      console.log(`\n🏆 Winner: ${winner} (${ratio.toFixed(2)}x faster on average)`)
      console.log(
        `   Confidence: Based on ${CONFIG.benchmarkRuns * CONFIG.tokenCounts.length} measurements each`
      )
    }
  }
}

// ============ Main ============
async function main() {
  console.log("🚀 node-mlx vs node-llama-cpp - Robust Benchmark")
  console.log("═".repeat(50))

  const model = process.argv[2] || "phi4"

  const models: Record<string, { mlx: string; gguf: string; name: string; quant: string }> = {
    phi4: {
      mlx: "mlx-community/phi-4-4bit",
      gguf: ".models/phi-4-Q4_K_S.gguf",
      name: "Phi-4 14B",
      quant: "4-bit"
    },
    gemma3n: {
      mlx: "mlx-community/gemma-3n-E4B-it-lm-4bit",
      gguf: ".models/gemma-3n-E4B-it-Q4_K_S.gguf",
      name: "Gemma 3n E4B",
      quant: "4-bit"
    },
    gptoss: {
      mlx: "NexaAI/gpt-oss-20b-MLX-4bit",
      gguf: ".models/gpt-oss-20b-Q4_K_S.gguf",
      name: "GPT-OSS 20B",
      quant: "4-bit"
    },
    qwen3: {
      mlx: "mlx-community/Qwen3-30B-A3B-4bit",
      gguf: ".models/Qwen3-30B-A3B-Q4_K_M.gguf",
      name: "Qwen3 30B A3B",
      quant: "4-bit"
    },
    ministral: {
      mlx: "mlx-community/Ministral-8B-Instruct-2410-4bit",
      gguf: ".models/Ministral-8B-Instruct-2410-Q4_K_M.gguf",
      name: "Ministral 8B",
      quant: "4-bit"
    }
  }

  const selected = models[model]
  if (!selected) {
    console.log(`Unknown model: ${model}`)
    console.log(`Available: ${Object.keys(models).join(", ")}`)
    process.exit(1)
  }

  console.log(`\nModel: ${selected.name}`)
  console.log(`Quantization: ${selected.quant} (both)`)
  console.log(`MLX: ${selected.mlx}`)
  console.log(`GGUF: ${selected.gguf}`)

  const results: BenchmarkResult[] = []

  // Run benchmarks
  const mlxResult = await benchmarkMLX(selected.mlx, selected.name)
  results.push(mlxResult)

  const llamaResult = await benchmarkLlamaCpp(selected.gguf, selected.name)
  if (llamaResult) {
    results.push(llamaResult)
  }

  // Print results
  printResults(results)

  // Save raw data
  const outputPath = `results-${model}-${Date.now()}.json`
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        config: CONFIG,
        model: selected,
        results,
        timestamp: new Date().toISOString()
      },
      null,
      2
    )
  )
  console.log(`\n📁 Raw data saved to: ${outputPath}`)
}

main().catch(console.error)
