/**
 * Quick Benchmark: node-mlx vs node-llama-cpp
 *
 * Compares inference speed on similar models.
 */

import { loadModel as loadMLX } from "node-mlx"

const PROMPT = "Explain quantum computing in one sentence:"
const MAX_TOKENS = 50

interface BenchmarkResult {
  name: string
  model: string
  loadTimeMs: number
  generateTimeMs: number
  tokens: number
  tokensPerSecond: number
  response: string
}

async function benchmarkMLX(): Promise<BenchmarkResult> {
  console.log("\n🍎 Testing node-mlx (MLX)...")

  const modelId = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

  const loadStart = Date.now()
  const model = loadMLX(modelId)
  const loadTime = Date.now() - loadStart

  const genStart = Date.now()
  const result = model.generate(PROMPT, { maxTokens: MAX_TOKENS, temperature: 0.7 })
  const genTime = Date.now() - genStart

  model.unload()

  return {
    name: "node-mlx",
    model: modelId,
    loadTimeMs: loadTime,
    generateTimeMs: genTime,
    tokens: result.tokenCount,
    tokensPerSecond: result.tokensPerSecond,
    response: result.text.slice(0, 100) + "..."
  }
}

async function benchmarkLlamaCpp(): Promise<BenchmarkResult | null> {
  console.log("\n🦙 Testing node-llama-cpp (llama.cpp)...")

  try {
    const { getLlama, LlamaChatSession } = await import("node-llama-cpp")

    // Use Qwen GGUF for fair comparison
    const modelPath =
      "~/.cache/lm-studio/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    const expandedPath = modelPath.replace("~", process.env.HOME || "")

    // Check if model exists
    const fs = await import("fs")
    if (!fs.existsSync(expandedPath)) {
      console.log(`  ⚠️ Model not found: ${expandedPath}`)
      console.log("  Download from: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
      return null
    }

    const loadStart = Date.now()
    const llama = await getLlama()
    const model = await llama.loadModel({ modelPath: expandedPath })
    const context = await model.createContext()
    const session = new LlamaChatSession({ contextSequence: context.getSequence() })
    const loadTime = Date.now() - loadStart

    const genStart = Date.now()
    let response = ""
    let tokens = 0

    await session.prompt(PROMPT, {
      maxTokens: MAX_TOKENS,
      temperature: 0.7,
      onTextChunk: (text) => {
        response += text
        tokens++
      }
    })
    const genTime = Date.now() - genStart

    await model.dispose()

    return {
      name: "node-llama-cpp",
      model: "Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)",
      loadTimeMs: loadTime,
      generateTimeMs: genTime,
      tokens: tokens,
      tokensPerSecond: (tokens / genTime) * 1000,
      response: response.slice(0, 100) + "..."
    }
  } catch (error) {
    console.log(`  ❌ Error: ${error}`)
    return null
  }
}

function printResults(results: (BenchmarkResult | null)[]) {
  console.log("\n" + "═".repeat(60))
  console.log("📊 BENCHMARK RESULTS")
  console.log("═".repeat(60))
  console.log(`Prompt: "${PROMPT}"`)
  console.log(`Max tokens: ${MAX_TOKENS}`)
  console.log("")

  const valid = results.filter((r): r is BenchmarkResult => r !== null)

  // Table header
  console.log("┌─────────────────┬──────────────┬──────────────┬─────────────┐")
  console.log("│ Library         │ Load Time    │ Gen Time     │ Tokens/sec  │")
  console.log("├─────────────────┼──────────────┼──────────────┼─────────────┤")

  for (const r of valid) {
    const name = r.name.padEnd(15)
    const load = `${(r.loadTimeMs / 1000).toFixed(1)}s`.padEnd(12)
    const gen = `${(r.generateTimeMs / 1000).toFixed(2)}s`.padEnd(12)
    const tps = `${r.tokensPerSecond.toFixed(1)}`.padEnd(11)
    console.log(`│ ${name} │ ${load} │ ${gen} │ ${tps} │`)
  }

  console.log("└─────────────────┴──────────────┴──────────────┴─────────────┘")

  // Responses
  console.log("\n📝 Responses:")
  for (const r of valid) {
    console.log(`\n${r.name}:`)
    console.log(`  ${r.response}`)
  }

  // Winner
  if (valid.length >= 2) {
    const fastest = valid.reduce((a, b) => (a.tokensPerSecond > b.tokensPerSecond ? a : b))
    const speedup = (
      fastest.tokensPerSecond / valid.find((r) => r !== fastest)!.tokensPerSecond
    ).toFixed(1)
    console.log(`\n🏆 Winner: ${fastest.name} (${speedup}x faster)`)
  }
}

async function main() {
  console.log("🚀 node-mlx vs node-llama-cpp Benchmark")
  console.log("=".repeat(40))

  const results: (BenchmarkResult | null)[] = []

  // Run benchmarks
  results.push(await benchmarkMLX())
  results.push(await benchmarkLlamaCpp())

  // Print results
  printResults(results)
}

main().catch(console.error)
