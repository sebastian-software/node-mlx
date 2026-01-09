# node-mlx

**The fastest way to run LLMs in Node.js on Apple Silicon.**

[![CI](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/node-mlx.svg)](https://www.npmjs.com/package/node-mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Installation

```bash
npm install node-mlx
```

**Requirements:** macOS 14+ (Sonoma) on Apple Silicon (M1/M2/M3/M4), Node.js 20+

### Try it with the CLI

```bash
npx node-mlx "What is 2+2?"
npx node-mlx --model phi-3-mini   # Interactive chat
```

---

## Usage

### Basic Example

```typescript
import { loadModel, RECOMMENDED_MODELS } from "node-mlx"

// Load a model (downloads automatically on first use)
const model = loadModel(RECOMMENDED_MODELS["llama-3.2-1b"])

// Generate text
const result = model.generate("Explain quantum computing in simple terms:", {
  maxTokens: 200,
  temperature: 0.7
})

console.log(result.text)
console.log(`${result.tokensPerSecond} tokens/sec`)

// Clean up when done
model.unload()
```

### Using Phi-4

```typescript
import { loadModel } from "node-mlx"

// Use any model from mlx-community
const model = loadModel("mlx-community/Phi-4-mini-instruct-4bit")

const result = model.generate("Write a haiku about coding:", {
  maxTokens: 50,
  temperature: 0.8
})

console.log(result.text)
model.unload()
```

### Available Models

The `RECOMMENDED_MODELS` constant provides shortcuts to tested models:

```typescript
import { RECOMMENDED_MODELS } from "node-mlx"

// Small & Fast
RECOMMENDED_MODELS["qwen-2.5-0.5b"] // Qwen 2.5 0.5B - Great for simple tasks
RECOMMENDED_MODELS["llama-3.2-1b"] // Llama 3.2 1B - Fast general purpose
RECOMMENDED_MODELS["qwen-2.5-1.5b"] // Qwen 2.5 1.5B - Good balance

// Medium
RECOMMENDED_MODELS["llama-3.2-3b"] // Llama 3.2 3B - Better quality
RECOMMENDED_MODELS["qwen-2.5-3b"] // Qwen 2.5 3B - Multilingual
RECOMMENDED_MODELS["phi-3-mini"] // Phi-3 Mini - Reasoning tasks

// Multimodal (text-only mode)
RECOMMENDED_MODELS["gemma-3n-2b"] // Gemma 3n 2B - Efficient
RECOMMENDED_MODELS["gemma-3n-4b"] // Gemma 3n 4B - Higher quality
```

You can also use **any model** from [mlx-community](https://huggingface.co/mlx-community):

```typescript
loadModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
loadModel("mlx-community/Qwen3-30B-A3B-4bit") // MoE model
```

### Model Quantization (4-bit vs bf16)

Most models on mlx-community come in two variants:

| Variant   | Memory      | Download | Quality          | Speed        |
| --------- | ----------- | -------- | ---------------- | ------------ |
| **bf16**  | ~2× size    | Larger   | 100% (reference) | Baseline     |
| **4-bit** | ~4× smaller | Faster   | ~97-99%          | Often faster |

**When to use 4-bit:**

- Limited RAM (8-16 GB)
- Larger models (7B+)
- General conversation, creative writing, summaries

**When to use bf16:**

- Math, logic, coding tasks where precision matters
- Smaller models where memory isn't a concern
- Maximum quality is critical

```typescript
// 4-bit: ~2 GB RAM, slightly less precise
loadModel("mlx-community/Llama-3.2-3B-Instruct-4bit")

// bf16: ~6 GB RAM, full precision
loadModel("mlx-community/Llama-3.2-3B-Instruct-bf16")
```

> **Tip:** For most use cases, 4-bit is indistinguishable from bf16. Start with 4-bit and only switch to bf16 if you notice quality issues.

### How Model Loading Works

1. **First use:** Model downloads from HuggingFace (~2-8 GB depending on model)
2. **Cached:** Models are stored in `~/.cache/huggingface/` for future use
3. **GPU ready:** Model loads directly into Apple Silicon unified memory

```typescript
// First call - downloads and caches
const model = loadModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
// ⏳ Downloading... (one time only)

// Second call - instant from cache
const model2 = loadModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
// ⚡ Ready immediately
```

### One-Shot Generation

For single generations without keeping the model loaded:

```typescript
import { generate } from "node-mlx"

// Loads, generates, unloads automatically
const result = generate("mlx-community/Llama-3.2-1B-Instruct-4bit", "Hello, world!", {
  maxTokens: 100
})
```

### API Reference

#### `loadModel(modelId: string): Model`

Loads a model from HuggingFace or local path.

#### `model.generate(prompt, options): GenerationResult`

| Option        | Type   | Default | Description                             |
| ------------- | ------ | ------- | --------------------------------------- |
| `maxTokens`   | number | 256     | Maximum tokens to generate              |
| `temperature` | number | 0.7     | Sampling randomness (0 = deterministic) |
| `topP`        | number | 0.9     | Nucleus sampling threshold              |

#### `GenerationResult`

```typescript
{
  text: string // Generated text
  tokenCount: number // Tokens generated
  tokensPerSecond: number // Generation speed
}
```

#### Utilities

```typescript
import { isSupported, getVersion } from "node-mlx"

isSupported() // true on Apple Silicon Mac
getVersion() // Library version
```

---

## Performance

Benchmarks on Mac Studio M1 Ultra (64GB):

| Model               | node-mlx  | node-llama-cpp | Winner             |
| ------------------- | --------- | -------------- | ------------------ |
| **Qwen3 30B** (MoE) | 67 tok/s  | 1 tok/s        | **60x faster** 🏆  |
| **GPT-OSS 20B**     | 58 tok/s  | 5 tok/s        | **11x faster** 🏆  |
| **Ministral 8B**    | 101 tok/s | 51 tok/s       | **2x faster** 🏆   |
| **Phi-4 14B**       | 56 tok/s  | 32 tok/s       | **1.8x faster** 🏆 |

<details>
<summary>Why is MLX faster?</summary>

1. **Unified Memory** – No data copying between CPU and GPU
2. **Metal Optimization** – Native Apple GPU kernels
3. **Lazy Evaluation** – Fused operations, minimal memory bandwidth
4. **Native Quantization** – 4-bit optimized for Apple Silicon

</details>

---

## Supported Architectures

| Architecture | Example Models                | Status          |
| ------------ | ----------------------------- | --------------- |
| **Qwen2**    | Qwen 2.5, Qwen3 (MoE)         | ✅ Full support |
| **Llama**    | Llama 3.2, Mistral, Ministral | ✅ Full support |
| **Phi3**     | Phi-3, Phi-4                  | ✅ Full support |
| **GPT-OSS**  | GPT-OSS 20B (MoE)             | ✅ Full support |
| **Gemma3n**  | Gemma 3n (VLM text-only)      | 🔧 Experimental |

---

## vs. node-llama-cpp

|                  | node-mlx            | node-llama-cpp |
| ---------------- | ------------------- | -------------- |
| **Platform**     | macOS Apple Silicon | Cross-platform |
| **Backend**      | Apple MLX           | llama.cpp      |
| **Memory**       | Unified CPU+GPU     | Separate       |
| **Model Format** | MLX/Safetensors     | GGUF           |
| **MoE Support**  | ✅ Excellent        | ⚠️ Limited     |

**Choose node-mlx** for maximum performance on Apple Silicon.
**Choose node-llama-cpp** for cross-platform or GGUF compatibility.

---

# Development

Everything below is for contributors and maintainers.

## Setup

```bash
git clone https://github.com/sebastian-software/node-mlx.git
cd node-mlx
pnpm install
```

## Build

```bash
# Build everything
pnpm build:swift    # Swift library
pnpm build:native   # N-API addon (uses local Swift build)
pnpm build          # TypeScript (all packages)

# Or for development with prebuilds
cd packages/node-mlx
pnpm prebuildify    # Create prebuilt binaries for Node 20/22/24
```

## Test

```bash
pnpm test           # All packages
pnpm test:coverage  # With coverage

# Swift tests
cd packages/swift
swift test
```

## Project Structure

```
node-mlx/
├── package.json                 # Workspace root (private)
├── pnpm-workspace.yaml
├── turbo.json                   # Task orchestration
│
└── packages/
    ├── node-mlx/                # 📦 The npm package
    │   ├── package.json         # Published as "node-mlx"
    │   ├── src/                 # TypeScript API
    │   ├── test/                # TypeScript tests
    │   ├── native/              # C++ N-API binding
    │   ├── prebuilds/           # Prebuilt binaries (generated)
    │   └── swift/               # Swift artifacts (generated)
    │
    ├── swift/                   # Swift Package
    │   ├── Package.swift
    │   ├── Sources/NodeMLXCore/ # Swift implementation
    │   └── Tests/               # Swift tests
    │
    ├── hf2swift/                # Model code generator
    │   ├── src/                 # TypeScript generator
    │   └── tests/               # Generator tests
    │
    └── benchmarks/              # Performance benchmarks
        └── src/                 # Benchmark scripts
```

## Publishing

```bash
# 1. Build Swift (copies to packages/node-mlx/swift/)
pnpm build:swift

# 2. Create prebuilds for Node 20/22/24
cd packages/node-mlx
pnpm prebuildify

# 3. Build TypeScript
pnpm build

# 4. Publish
npm publish
```

The published package includes:

- `dist/` – TypeScript (ESM + CJS)
- `prebuilds/darwin-arm64/node.node` – N-API binary (72 KB)
- `swift/libNodeMLX.dylib` – Swift ML library
- `swift/mlx-swift_Cmlx.bundle/` – Metal shaders

## Adding New Models

```bash
pnpm hf2swift \
  --model MyModel \
  --source path/to/modeling_mymodel.py \
  --config organization/model-name \
  --output packages/swift/Sources/NodeMLXCore/Models/MyModel.swift
```

The `hf2swift` generator parses Python model code and produces Swift using MLX primitives.

---

## Credits

Built on [MLX](https://github.com/ml-explore/mlx) by Apple, [mlx-swift](https://github.com/ml-explore/mlx-swift), and [swift-transformers](https://github.com/huggingface/swift-transformers) by HuggingFace.

**Special thanks to [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)** – we adopted and adapted several core components from their excellent implementation:

- KV Cache management (`KVCacheSimple`, `RotatingKVCache`)
- Token sampling strategies (temperature, top-p, repetition penalty)
- RoPE implementations (Llama3, Yarn, LongRoPE)
- Attention utilities and quantization support

---

## License

MIT © 2026 [Sebastian Software GmbH](https://sebastian-software.de)
