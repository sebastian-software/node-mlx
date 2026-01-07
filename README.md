# node-mlx

> Native LLM inference for Node.js powered by Apple MLX on Apple Silicon

[![CI](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/node-mlx.svg)](https://www.npmjs.com/package/node-mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Run large language models locally on your Mac with **native** Apple Silicon performance. Built on Apple's [MLX](https://github.com/ml-explore/mlx) framework via [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

## Features

- 🚀 **True Native Binding** - No subprocess, no CLI wrapper - direct Swift ↔ Node.js bridge
- 🧠 **Unified Memory** - Models stay loaded in MLX's unified CPU/GPU memory
- ⚡ **Maximum Performance** - Zero serialization overhead between calls
- 📦 **Simple API** - Load once, generate many times
- 🤗 **HuggingFace Integration** - Load any MLX-compatible model directly

## Requirements

- macOS 14.0+ (Sonoma or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Node.js 20+

### For Development

- Xcode Command Line Tools (`xcode-select --install`)

## Installation

```bash
npm install node-mlx
```

## Quick Start

```typescript
import { loadModel, RECOMMENDED_MODELS } from "node-mlx"

// Load model (stays in memory)
const model = loadModel(RECOMMENDED_MODELS["llama-3.2-1b"])

// Generate text (fast - model already loaded)
const result = model.generate("Explain quantum computing in simple terms", {
  maxTokens: 256,
  temperature: 0.7
})

console.log(result.text)
console.log(`${result.tokenCount} tokens at ${result.tokensPerSecond.toFixed(1)} tok/s`)

// Unload when done
model.unload()
```

## API Reference

### `loadModel(modelId)`

Load a model into memory. Returns a `Model` instance.

```typescript
const model = loadModel("mlx-community/gemma-3n-E2B-it-4bit")
```

### `model.generate(prompt, options?)`

Generate text from a prompt.

**Options:**

- `maxTokens` (number) - Maximum tokens to generate (default: 256)
- `temperature` (number) - Sampling temperature (default: 0.7)
- `topP` (number) - Top-p sampling (default: 0.9)

**Returns:** `{ text, tokenCount, tokensPerSecond }`

### `model.unload()`

Free the model from memory.

### `generate(modelId, prompt, options?)`

One-shot generation (loads model, generates, unloads).

```typescript
import { generate } from "node-mlx"

const result = generate("mlx-community/Llama-3.2-1B-Instruct-4bit", "Hello!", { maxTokens: 50 })
```

### `isSupported()`

Check if the current platform supports MLX.

### `getVersion()`

Get the library version.

## Recommended Models

| Model         | ID                                            | Size   | Speed\* |
| ------------- | --------------------------------------------- | ------ | ------- |
| Llama 3.2 1B  | `mlx-community/Llama-3.2-1B-Instruct-4bit`    | ~0.7GB | 370 t/s |
| Llama 3.2 3B  | `mlx-community/Llama-3.2-3B-Instruct-4bit`    | ~1.8GB | 200 t/s |
| Qwen 2.5 0.5B | `mlx-community/Qwen2.5-0.5B-Instruct-4bit`    | ~0.4GB | 400 t/s |
| Qwen 2.5 1.5B | `mlx-community/Qwen2.5-1.5B-Instruct-4bit`    | ~1GB   | 200 t/s |
| Qwen 2.5 7B   | `mlx-community/Qwen2.5-7B-Instruct-4bit`      | ~4GB   | 80 t/s  |
| Phi-3 Mini    | `mlx-community/Phi-3-mini-4k-instruct-4bit`   | ~2GB   | 140 t/s |
| Mistral 7B    | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4GB   | 80 t/s  |

\*Speed measured on M3 Pro. Your results may vary.

Models are automatically downloaded from HuggingFace on first use.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Node.js App                      │
└─────────────────────────────────────────────────────┘
                         │
                    N-API Binding
                         │
┌─────────────────────────────────────────────────────┐
│              Swift Library (libNodeMLX)             │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              mlx-swift-lm                    │   │
│  │  • Model Loading & Caching                  │   │
│  │  • Tokenization                             │   │
│  │  • Generation Loop                          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │              MLX Framework                   │   │
│  │  • Unified Memory (CPU + GPU)               │   │
│  │  • Metal Acceleration                       │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

The model lives entirely in Swift/MLX memory. Only strings (prompts and responses) cross the binding boundary - no tensor serialization overhead.

## Comparison with node-llama-cpp

| Feature      | node-mlx              | node-llama-cpp |
| ------------ | --------------------- | -------------- |
| Backend      | Apple MLX             | llama.cpp      |
| Platform     | macOS (Apple Silicon) | Cross-platform |
| Memory       | Unified (CPU+GPU)     | Separate       |
| Model Format | MLX/Safetensors       | GGUF           |
| Binding      | Native Swift          | Native C++     |

**Use node-mlx when:**

- You're on Apple Silicon
- You want optimal memory efficiency
- You prefer MLX model ecosystem

**Use node-llama-cpp when:**

- You need cross-platform support
- You want GGUF model compatibility

## Development

```bash
# Clone repository
git clone https://github.com/sebastian-software/node-mlx.git
cd node-mlx

# Install dependencies
pnpm install

# Build Swift library
pnpm build:swift

# Build native addon
pnpm build:native

# Build TypeScript
pnpm build:ts

# Or build everything
pnpm build

# Run tests
pnpm test
```

## Credits & Acknowledgments

This project is built on top of Apple's excellent MLX ecosystem:

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework for Apple Silicon
- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Swift bindings for MLX
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - Swift LLM library (MIT License)
- [mlx-community](https://huggingface.co/mlx-community) - Community-maintained MLX model hub

All Apple MLX projects are released under the **MIT License** by the [ml-explore](https://github.com/ml-explore) team.

See [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md) for full license texts.

## License

MIT © [Sebastian Werner](https://github.com/sebastian-software)
