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
const model = loadModel(RECOMMENDED_MODELS["gemma-3n-2b"])

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

const result = generate("mlx-community/gemma-3n-E2B-it-4bit", "Hello!", { maxTokens: 50 })
```

### `isSupported()`

Check if the current platform supports MLX.

### `getVersion()`

Get the library version.

## Recommended Models

| Model         | ID                                     | Size   |
| ------------- | -------------------------------------- | ------ |
| Gemma 3n 2B   | `mlx-community/gemma-3n-E2B-it-4bit`   | ~1.5GB |
| Gemma 3n 4B   | `mlx-community/gemma-3n-E4B-it-4bit`   | ~2.5GB |
| Qwen 3 1.7B   | `mlx-community/Qwen3-1.7B-4bit`        | ~1GB   |
| Qwen 3 4B     | `mlx-community/Qwen3-4B-4bit`          | ~2.5GB |
| Phi 4         | `mlx-community/phi-4-4bit`             | ~8GB   |
| Llama 4 Scout | `mlx-community/Llama-4-Scout-17B-4bit` | ~10GB  |

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

## Credits

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - Swift LLM library
- [mlx-community](https://huggingface.co/mlx-community) - MLX model hub

## License

MIT © [Sebastian Werner](https://github.com/sebastian-software)
