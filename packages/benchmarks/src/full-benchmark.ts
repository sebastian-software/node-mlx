/**
 * Full Benchmark: node-mlx vs node-llama-cpp
 * Tests all currently supported model families with multiple sizes
 */

import { loadModel as loadMLX } from "node-mlx"
import * as fs from "fs"
import * as path from "path"

// ============ Configuration ============
const CONFIG = {
  warmupRuns: 1,
  benchmarkRuns: 3,
  maxTokens: 100,
  prompt: "Explain quantum entanglement in simple terms:"
}

interface BenchmarkResult {
  model: string
  family: string
  size: string
  mlxId: string
  ggufPath: string | null
  mlx: {
    loadTimeMs: number
    tokensPerSec: number
    runs: number[]
  } | null
  llamacpp: {
    loadTimeMs: number
    tokensPerSec: number
    runs: number[]
  } | null
  speedup: number | null
}

// All models to benchmark
const MODELS: Array<{
  family: string
  name: string
  size: string
  mlx: string
  gguf: string | null
}> = [
  // Qwen 3 (latest)
  {
    family: "Qwen3",
    name: "Qwen3 0.6B",
    size: "0.6B",
    mlx: "mlx-community/Qwen3-0.6B-4bit",
    gguf: ".models/Qwen3-0.6B-Q4_K_M.gguf"
  },
  {
    family: "Qwen3",
    name: "Qwen3 1.7B",
    size: "1.7B",
    mlx: "mlx-community/Qwen3-1.7B-4bit",
    gguf: ".models/Qwen3-1.7B-Q4_K_M.gguf"
  },
  {
    family: "Qwen3",
    name: "Qwen3 4B",
    size: "4B",
    mlx: "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit",
    gguf: ".models/Qwen3-4B-Instruct-Q4_K_M.gguf"
  },

  // Phi (Microsoft)
  {
    family: "Phi",
    name: "Phi-3.5 Mini",
    size: "3.8B",
    mlx: "mlx-community/Phi-3.5-mini-instruct-4bit",
    gguf: ".models/Phi-3.5-mini-instruct-Q4_K_M.gguf"
  },
  {
    family: "Phi",
    name: "Phi-4",
    size: "14B",
    mlx: "mlx-community/phi-4-4bit",
    gguf: ".models/phi-4-Q4_K_M.gguf"
  },

  // Gemma 3 (Google)
  {
    family: "Gemma3",
    name: "Gemma 3 1B",
    size: "1B",
    mlx: "mlx-community/gemma-3-1b-it-4bit",
    gguf: ".models/gemma-3-1b-it-Q4_K_M.gguf"
  },
  {
    family: "Gemma3",
    name: "Gemma 3 4B",
    size: "4B",
    mlx: "mlx-community/gemma-3-4b-it-4bit",
    gguf: ".models/gemma-3-4b-it-Q4_K_M.gguf"
  },
  {
    family: "Gemma3",
    name: "Gemma 3 12B",
    size: "12B",
    mlx: "mlx-community/gemma-3-12b-it-4bit",
    gguf: ".models/gemma-3-12b-it-Q4_K_M.gguf"
  },

  // Gemma 3n (Google - efficient)
  {
    family: "Gemma3n",
    name: "Gemma 3n E2B",
    size: "E2B",
    mlx: "mlx-community/gemma-3n-E2B-it-lm-4bit",
    gguf: ".models/gemma-3n-E2B-it-Q4_K_M.gguf"
  },
  {
    family: "Gemma3n",
    name: "Gemma 3n E4B",
    size: "E4B",
    mlx: "mlx-community/gemma-3n-E4B-it-lm-4bit",
    gguf: ".models/gemma-3n-E4B-it-Q4_K_M.gguf"
  },

  // Mistral
  {
    family: "Mistral",
    name: "Mistral 7B v0.3",
    size: "7B",
    mlx: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    gguf: ".models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
  }
]

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

async function benchmarkMLX(
  mlxId: string,
  modelName: string
): Promise<{ loadTimeMs: number; tokensPerSec: number; runs: number[] } | null> {
  console.log(`  🍎 MLX: ${mlxId}`)

  try {
    const loadStart = Date.now()
    const model = loadMLX(mlxId)
    const loadTimeMs = Date.now() - loadStart

    // Warmup
    for (let i = 0; i < CONFIG.warmupRuns; i++) {
      model.generate(CONFIG.prompt, { maxTokens: 20, temperature: 0.7 })
    }

    // Benchmark
    const runs: number[] = []
    for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
      const result = model.generate(CONFIG.prompt, {
        maxTokens: CONFIG.maxTokens,
        temperature: 0.7
      })
      runs.push(result.tokensPerSecond)
      process.stdout.write(".")
    }
    console.log()

    model.unload()

    return { loadTimeMs, tokensPerSec: mean(runs), runs }
  } catch (error) {
    console.log(`    ❌ Error: ${error}`)
    return null
  }
}

async function benchmarkLlamaCpp(
  ggufPath: string | null,
  modelName: string
): Promise<{ loadTimeMs: number; tokensPerSec: number; runs: number[] } | null> {
  if (!ggufPath) {
    console.log(`  🦙 llama.cpp: skipped (no GGUF)`)
    return null
  }

  const expandedPath = ggufPath.startsWith("~")
    ? ggufPath.replace("~", process.env.HOME || "")
    : path.join(process.cwd(), ggufPath)

  if (!fs.existsSync(expandedPath)) {
    console.log(`  🦙 llama.cpp: skipped (file not found)`)
    return null
  }

  console.log(`  🦙 llama.cpp: ${ggufPath}`)

  try {
    const { getLlama, LlamaChatSession } = await import("node-llama-cpp")

    const loadStart = Date.now()
    const llama = await getLlama()
    const model = await llama.loadModel({ modelPath: expandedPath })
    const loadTimeMs = Date.now() - loadStart

    // Warmup
    for (let i = 0; i < CONFIG.warmupRuns; i++) {
      const ctx = await model.createContext()
      const session = new LlamaChatSession({ contextSequence: ctx.getSequence() })
      await session.prompt(CONFIG.prompt, { maxTokens: 20 })
      await ctx.dispose()
    }

    // Benchmark
    const runs: number[] = []
    for (let i = 0; i < CONFIG.benchmarkRuns; i++) {
      const ctx = await model.createContext()
      const session = new LlamaChatSession({ contextSequence: ctx.getSequence() })

      const start = Date.now()
      let tokens = 0
      await session.prompt(CONFIG.prompt, {
        maxTokens: CONFIG.maxTokens,
        temperature: 0.7,
        onTextChunk: () => {
          tokens++
        }
      })
      const timeMs = Date.now() - start
      runs.push((tokens / timeMs) * 1000)

      await ctx.dispose()
      process.stdout.write(".")
    }
    console.log()

    await model.dispose()

    return { loadTimeMs, tokensPerSec: mean(runs), runs }
  } catch (error) {
    console.log(`    ❌ Error: ${error}`)
    return null
  }
}

function printResults(results: BenchmarkResult[]) {
  console.log("\n")
  console.log("═".repeat(100))
  console.log("📊 BENCHMARK RESULTS - node-mlx vs node-llama-cpp")
  console.log("═".repeat(100))
  console.log(`Config: ${CONFIG.benchmarkRuns} runs, ${CONFIG.maxTokens} tokens/run`)
  console.log("")

  // Group by family
  const families = [...new Set(results.map((r) => r.family))]

  for (const family of families) {
    console.log(`\n${family}`)
    console.log("-".repeat(100))
    console.log(
      "Model".padEnd(25) +
        "MLX (tok/s)".padEnd(15) +
        "llama.cpp".padEnd(15) +
        "Speedup".padEnd(12) +
        "Winner"
    )
    console.log("-".repeat(100))

    const familyResults = results.filter((r) => r.family === family)
    for (const r of familyResults) {
      const mlxStr = r.mlx ? r.mlx.tokensPerSec.toFixed(1) : "N/A"
      const llamaStr = r.llamacpp ? r.llamacpp.tokensPerSec.toFixed(1) : "N/A"

      let speedupStr = "N/A"
      let winner = ""
      if (r.mlx && r.llamacpp) {
        const speedup = r.mlx.tokensPerSec / r.llamacpp.tokensPerSec
        speedupStr = `${speedup.toFixed(2)}x`
        winner = speedup > 1 ? "🍎 MLX" : "🦙 llama"
      } else if (r.mlx) {
        winner = "🍎 MLX only"
      }

      console.log(
        `${r.name.padEnd(25)}${mlxStr.padEnd(15)}${llamaStr.padEnd(15)}${speedupStr.padEnd(12)}${winner}`
      )
    }
  }

  // Summary
  const withBoth = results.filter((r) => r.mlx && r.llamacpp)
  if (withBoth.length > 0) {
    const avgSpeedup =
      withBoth.reduce((sum, r) => sum + r.mlx!.tokensPerSec / r.llamacpp!.tokensPerSec, 0) /
      withBoth.length
    console.log("\n" + "═".repeat(100))
    console.log(
      `📈 Average speedup: ${avgSpeedup.toFixed(2)}x (node-mlx vs node-llama-cpp, ${withBoth.length} models)`
    )
  }
}

async function main() {
  console.log("🚀 Full Benchmark: node-mlx vs node-llama-cpp")
  console.log("═".repeat(60))
  console.log(`Testing ${MODELS.length} models...\n`)

  const results: BenchmarkResult[] = []

  for (const model of MODELS) {
    console.log(`\n📦 ${model.name} (${model.size})`)

    const mlxResult = await benchmarkMLX(model.mlx, model.name)
    const llamaResult = await benchmarkLlamaCpp(model.gguf, model.name)

    let speedup: number | null = null
    if (mlxResult && llamaResult) {
      speedup = mlxResult.tokensPerSec / llamaResult.tokensPerSec
    }

    results.push({
      model: model.name,
      family: model.family,
      size: model.size,
      mlxId: model.mlx,
      ggufPath: model.gguf,
      mlx: mlxResult,
      llamacpp: llamaResult,
      speedup
    })
  }

  printResults(results)

  // Save results
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
  const outputPath = `benchmark-results-${timestamp}.json`
  fs.writeFileSync(
    outputPath,
    JSON.stringify({ config: CONFIG, results, timestamp: new Date().toISOString() }, null, 2)
  )
  console.log(`\n📁 Results saved to: ${outputPath}`)
}

main().catch(console.error)
