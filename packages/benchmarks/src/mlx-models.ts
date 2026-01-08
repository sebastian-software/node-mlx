/**
 * Quick MLX Model Benchmark
 *
 * Tests different model sizes to show performance scaling.
 */

import { loadModel } from "node-mlx"

const PROMPT = "Write a haiku about programming:"
const MAX_TOKENS = 30

interface Result {
  model: string
  size: string
  loadMs: number
  genMs: number
  tokens: number
  tokensPerSec: number
  response: string
}

const MODELS = [
  { id: "mlx-community/Qwen2.5-0.5B-Instruct-4bit", size: "0.5B" },
  { id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit", size: "1.5B" },
  { id: "mlx-community/Llama-3.2-1B-Instruct-4bit", size: "1B" },
  { id: "mlx-community/Phi-3-mini-4k-instruct-4bit", size: "3.8B" }
]

async function benchmark(modelId: string, size: string): Promise<Result> {
  console.log(`\n  Testing ${size} model...`)

  const loadStart = Date.now()
  const model = loadModel(modelId)
  const loadMs = Date.now() - loadStart

  const genStart = Date.now()
  const result = model.generate(PROMPT, { maxTokens: MAX_TOKENS, temperature: 0.7 })
  const genMs = Date.now() - genStart

  model.unload()

  return {
    model: modelId.split("/")[1],
    size,
    loadMs,
    genMs,
    tokens: result.tokenCount,
    tokensPerSec: result.tokensPerSecond,
    response: result.text.trim().replace(/\n/g, " ").slice(0, 60)
  }
}

async function main() {
  console.log("🍎 node-mlx Model Benchmark")
  console.log("═".repeat(50))
  console.log(`Prompt: "${PROMPT}"`)
  console.log(`Max tokens: ${MAX_TOKENS}`)

  const results: Result[] = []

  for (const m of MODELS) {
    try {
      results.push(await benchmark(m.id, m.size))
    } catch (e) {
      console.log(`  ❌ Failed: ${e}`)
    }
  }

  // Print table
  console.log("\n" + "═".repeat(70))
  console.log("📊 RESULTS")
  console.log("═".repeat(70))
  console.log("")
  console.log("┌────────────────────────────────┬───────┬─────────┬─────────────┐")
  console.log("│ Model                          │ Size  │ Load    │ Tokens/sec  │")
  console.log("├────────────────────────────────┼───────┼─────────┼─────────────┤")

  for (const r of results) {
    const model = r.model.slice(0, 30).padEnd(30)
    const size = r.size.padEnd(5)
    const load = `${(r.loadMs / 1000).toFixed(1)}s`.padEnd(7)
    const tps = `${r.tokensPerSec.toFixed(0)} t/s`.padEnd(11)
    console.log(`│ ${model} │ ${size} │ ${load} │ ${tps} │`)
  }

  console.log("└────────────────────────────────┴───────┴─────────┴─────────────┘")

  // Best performer
  if (results.length > 0) {
    const fastest = results.reduce((a, b) => (a.tokensPerSec > b.tokensPerSec ? a : b))
    console.log(`\n🏆 Fastest: ${fastest.model} @ ${fastest.tokensPerSec.toFixed(0)} tokens/sec`)
  }

  // Sample outputs
  console.log("\n📝 Sample outputs:")
  for (const r of results.slice(0, 2)) {
    console.log(`\n${r.size}: ${r.response}...`)
  }
}

main().catch(console.error)
