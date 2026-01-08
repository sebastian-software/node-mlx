/**
 * Direct Comparison: Phi-4 on node-mlx vs node-llama-cpp
 */

import { loadModel as loadMLX } from "node-mlx"

const PROMPT = "Explain quantum entanglement in simple terms:"
const MAX_TOKENS = 100

console.log("🔬 Phi-4: node-mlx (MLX) vs node-llama-cpp (GGUF)")
console.log("═".repeat(50))
console.log(`Prompt: "${PROMPT}"`)
console.log(`Max tokens: ${MAX_TOKENS}`)

// ============ node-mlx (MLX) ============
console.log("\n🍎 node-mlx (MLX Format)")
console.log("-".repeat(40))

const mlxModel = "mlx-community/phi-4-4bit"
console.log(`Model: ${mlxModel}`)

const mlxLoadStart = Date.now()
const mlx = loadMLX(mlxModel)
const mlxLoadTime = (Date.now() - mlxLoadStart) / 1000

console.log(`Load time: ${mlxLoadTime.toFixed(1)}s`)

const mlxGenStart = Date.now()
const mlxResult = mlx.generate(PROMPT, { maxTokens: MAX_TOKENS, temperature: 0.7 })
const mlxGenTime = (Date.now() - mlxGenStart) / 1000

console.log(`Generation: ${mlxGenTime.toFixed(2)}s`)
console.log(`Tokens: ${mlxResult.tokenCount}`)
console.log(`Speed: ${mlxResult.tokensPerSecond.toFixed(1)} tokens/sec`)
console.log(`\nResponse:\n${mlxResult.text.slice(0, 200)}...`)

mlx.unload()

// ============ node-llama-cpp (GGUF) ============
console.log("\n\n🦙 node-llama-cpp (GGUF Format)")
console.log("-".repeat(40))

try {
  const { getLlama, LlamaChatSession } = await import("node-llama-cpp")
  const path = await import("path")
  const fs = await import("fs")

  // Local GGUF model path
  const ggufPath = path.join(process.cwd(), ".models/phi-4-Q2_K.gguf")

  if (!fs.existsSync(ggufPath)) {
    throw new Error(`GGUF model not found at ${ggufPath}. Download it first.`)
  }

  console.log(`Model: phi-4-Q2_K.gguf (local)`)

  const llamaLoadStart = Date.now()
  const llama = await getLlama()

  const model = await llama.loadModel({
    modelPath: ggufPath
  })

  const context = await model.createContext()
  const session = new LlamaChatSession({ contextSequence: context.getSequence() })
  const llamaLoadTime = (Date.now() - llamaLoadStart) / 1000

  console.log(`Load time: ${llamaLoadTime.toFixed(1)}s`)

  const llamaGenStart = Date.now()
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

  const llamaGenTime = (Date.now() - llamaGenStart) / 1000
  const llamaTps = tokens / llamaGenTime

  console.log(`Generation: ${llamaGenTime.toFixed(2)}s`)
  console.log(`Tokens: ${tokens}`)
  console.log(`Speed: ${llamaTps.toFixed(1)} tokens/sec`)
  console.log(`\nResponse:\n${response.slice(0, 200)}...`)

  await model.dispose()

  // ============ Summary ============
  console.log("\n\n" + "═".repeat(50))
  console.log("📊 SUMMARY: Phi-4 Comparison")
  console.log("═".repeat(50))
  console.log("")
  console.log("┌─────────────────┬─────────────┬─────────────┐")
  console.log("│ Library         │ Load Time   │ Tokens/sec  │")
  console.log("├─────────────────┼─────────────┼─────────────┤")
  console.log(
    `│ node-mlx (MLX)  │ ${mlxLoadTime.toFixed(1)}s`.padEnd(15) +
      `│ ${mlxResult.tokensPerSecond.toFixed(0)} t/s`.padEnd(14) +
      "│"
  )
  console.log(
    `│ node-llama-cpp  │ ${llamaLoadTime.toFixed(1)}s`.padEnd(15) +
      `│ ${llamaTps.toFixed(0)} t/s`.padEnd(14) +
      "│"
  )
  console.log("└─────────────────┴─────────────┴─────────────┘")

  const winner = mlxResult.tokensPerSecond > llamaTps ? "node-mlx" : "node-llama-cpp"
  const speedup =
    Math.max(mlxResult.tokensPerSecond, llamaTps) / Math.min(mlxResult.tokensPerSecond, llamaTps)
  console.log(`\n🏆 Winner: ${winner} (${speedup.toFixed(1)}x faster)`)
} catch (error) {
  console.log(`\n❌ node-llama-cpp failed: ${error}`)
  console.log("\nTo run this benchmark, you need to:")
  console.log("1. Download a Phi-4 GGUF model")
  console.log("2. Or use: npx --yes node-llama-cpp pull microsoft/phi-4-gguf")

  console.log("\n\n📊 node-mlx Results Only:")
  console.log(`  Speed: ${mlxResult.tokensPerSecond.toFixed(1)} tokens/sec`)
}
