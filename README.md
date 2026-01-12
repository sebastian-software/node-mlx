# node-mlx

**The fastest way to run LLMs in Node.js on Apple Silicon.**

[![CI](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/node-mlx.svg)](https://www.npmjs.com/package/node-mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

```bash
npm install node-mlx
npx node-mlx "What is 2+2?"
```

**Requirements:** macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4), Node.js 20+

## Why node-mlx?

🚀 **2× faster** than node-llama-cpp on Apple Silicon
🧠 **Unified memory** – no CPU↔GPU copying overhead
📦 **Zero config** – just npm install and you're ready

## Usage

```typescript
import { generate } from "node-mlx"

const result = generate("qwen", "Explain quantum computing:", {
  maxTokens: 200,
  temperature: 0.7
})

console.log(result.text)
console.log(`${result.tokensPerSecond} tok/s`)
```

## Documentation

📚 **[Full Documentation](https://sebastian-software.github.io/node-mlx/)**

- [Getting Started](https://sebastian-software.github.io/node-mlx/docs)
- [Model Guide](https://sebastian-software.github.io/node-mlx/docs/models)
- [API Reference](https://sebastian-software.github.io/node-mlx/docs/api)

## Supported Models

| Provider  | Models           | Status             |
| --------- | ---------------- | ------------------ |
| Qwen      | Qwen3 0.6B–4B    | ✅ **Recommended** |
| Microsoft | Phi-4            | ✅ High Quality    |
| Google    | Gemma 3 1B–27B   | ✅ Latest          |
| Meta      | Llama 4          | ✅ Auth required   |
| Mistral   | Ministral 3B–14B | ✅                 |
| OpenAI    | GPT-OSS 20B/120B | ✅ MoE             |

Use short aliases or any model from [mlx-community](https://huggingface.co/mlx-community):

```typescript
loadModel("qwen") // Qwen3-4B (default)
loadModel("phi4") // Phi-4 (high quality)
loadModel("gemma-3-12b") // Gemma-3-12B
loadModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
```

## Performance

Benchmarks on Mac Studio M1 Ultra (64GB):

| Model       | node-mlx  | node-llama-cpp | Winner             |
| ----------- | --------- | -------------- | ------------------ |
| Mistral 7B  | 101 tok/s | 51 tok/s       | **2× faster** 🏆   |
| Phi-4 14B   | 56 tok/s  | 32 tok/s       | **1.8× faster** 🏆 |
| Qwen3 4B    | 120 tok/s | 65 tok/s       | **1.8× faster** 🏆 |
| Gemma-3 12B | 78 tok/s  | 42 tok/s       | **1.9× faster** 🏆 |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup.

## Credits

Built on [MLX](https://github.com/ml-explore/mlx) by Apple, [mlx-swift](https://github.com/ml-explore/mlx-swift), and [swift-transformers](https://github.com/huggingface/swift-transformers) by HuggingFace.

## License

MIT © 2026 [Sebastian Software GmbH](https://sebastian-software.de)
