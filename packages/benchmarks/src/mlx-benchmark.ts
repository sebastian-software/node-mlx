/**
 * Quick MLX Benchmark - Tests all supported models
 */

import { loadModel } from "node-mlx"

const PROMPT = "Explain quantum entanglement in simple terms:"
const MAX_TOKENS = 100
const RUNS = 3

interface Result {
  model: string
  family: string
  size: string
  tokensPerSec: number
  loadTimeMs: number
}

const MODELS = [
  // Qwen 3
  { family: "Qwen3", name: "Qwen3 0.6B", size: "0.6B", id: "mlx-community/Qwen3-0.6B-4bit" },
  { family: "Qwen3", name: "Qwen3 1.7B", size: "1.7B", id: "mlx-community/Qwen3-1.7B-4bit" },
  {
    family: "Qwen3",
    name: "Qwen3 4B",
    size: "4B",
    id: "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
  },

  // Phi 4
  { family: "Phi", name: "Phi-4", size: "14B", id: "mlx-community/phi-4-4bit" },

  // Gemma 3
  { family: "Gemma3", name: "Gemma 3 1B", size: "1B", id: "mlx-community/gemma-3-1b-it-4bit" },
  { family: "Gemma3", name: "Gemma 3 4B", size: "4B", id: "mlx-community/gemma-3-4b-it-4bit" },
  { family: "Gemma3", name: "Gemma 3 12B", size: "12B", id: "mlx-community/gemma-3-12b-it-4bit" },

  // Gemma 3n
  {
    family: "Gemma3n",
    name: "Gemma 3n E2B",
    size: "E2B",
    id: "mlx-community/gemma-3n-E2B-it-lm-4bit"
  },
  {
    family: "Gemma3n",
    name: "Gemma 3n E4B",
    size: "E4B",
    id: "mlx-community/gemma-3n-E4B-it-lm-4bit"
  },

  // Mistral
  {
    family: "Mistral",
    name: "Mistral 7B",
    size: "7B",
    id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  }
]

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

async function benchmark(
  modelDef: (typeof MODELS)[0]
): Promise<{ tokensPerSec: number; loadTimeMs: number } | null> {
  try {
    console.log(`  Loading ${modelDef.id}...`)
    const loadStart = Date.now()
    const model = loadModel(modelDef.id)
    const loadTimeMs = Date.now() - loadStart
    console.log(`  Loaded in ${(loadTimeMs / 1000).toFixed(1)}s`)

    // Warmup
    model.generate(PROMPT, { maxTokens: 20, temperature: 0.7 })

    // Benchmark
    const runs: number[] = []
    for (let i = 0; i < RUNS; i++) {
      const result = model.generate(PROMPT, { maxTokens: MAX_TOKENS, temperature: 0.7 })
      runs.push(result.tokensPerSecond)
      process.stdout.write(`  Run ${i + 1}: ${result.tokensPerSecond.toFixed(1)} tok/s\n`)
    }

    model.unload()

    return { tokensPerSec: mean(runs), loadTimeMs }
  } catch (error) {
    console.log(`  ❌ Error: ${error}`)
    return null
  }
}

async function main() {
  console.log("🍎 node-mlx Benchmark")
  console.log("═".repeat(60))
  console.log(`Config: ${RUNS} runs, ${MAX_TOKENS} tokens/run\n`)

  const results: Result[] = []

  for (const modelDef of MODELS) {
    console.log(`\n📦 ${modelDef.name} (${modelDef.size})`)

    const result = await benchmark(modelDef)
    if (result) {
      results.push({
        model: modelDef.name,
        family: modelDef.family,
        size: modelDef.size,
        tokensPerSec: result.tokensPerSec,
        loadTimeMs: result.loadTimeMs
      })
    }
  }

  // Print results table
  console.log("\n\n" + "═".repeat(70))
  console.log("📊 RESULTS")
  console.log("═".repeat(70))
  console.log("Model".padEnd(25) + "Size".padEnd(10) + "tok/s".padEnd(12) + "Load Time")
  console.log("-".repeat(70))

  // Sort by family, then by tok/s descending
  const sorted = [...results].sort((a, b) => {
    if (a.family !== b.family) return a.family.localeCompare(b.family)
    return b.tokensPerSec - a.tokensPerSec
  })

  let currentFamily = ""
  for (const r of sorted) {
    if (r.family !== currentFamily) {
      currentFamily = r.family
      console.log(`\n${currentFamily}:`)
    }
    console.log(
      `  ${r.model.padEnd(23)}${r.size.padEnd(10)}${r.tokensPerSec.toFixed(1).padEnd(12)}${(r.loadTimeMs / 1000).toFixed(1)}s`
    )
  }

  // Find fastest
  const fastest = sorted.reduce((a, b) => (a.tokensPerSec > b.tokensPerSec ? a : b))
  console.log("\n" + "═".repeat(70))
  console.log(`🏆 Fastest: ${fastest.model} at ${fastest.tokensPerSec.toFixed(1)} tok/s`)
}

main().catch(console.error)
