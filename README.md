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
npx node-mlx --model qwen   # Interactive chat with Qwen3
npx node-mlx --model phi    # Interactive chat with Phi-3.5
```

---

## Usage

### Basic Example

```typescript
import { generate } from "node-mlx"

// One-shot generation (loads, generates, unloads automatically)
const result = generate("qwen", "Explain quantum computing in simple terms:", {
  maxTokens: 200,
  temperature: 0.7
})

console.log(result.text)
console.log(`${result.tokensPerSecond} tokens/sec`)
```

### Keeping the Model Loaded

```typescript
import { loadModel } from "node-mlx"

// Load a model (downloads automatically on first use)
const model = loadModel("phi")

// Generate multiple responses
const result1 = model.generate("What is 2+2?")
const result2 = model.generate("What is the capital of France?")

// Clean up when done
model.unload()
```

### Using Phi-4

```typescript
import { loadModel } from "node-mlx"

// Use Phi-4 (8GB, high quality)
const model = loadModel("phi4")

const result = model.generate("Write a haiku about coding:", {
  maxTokens: 50,
  temperature: 0.8
})

console.log(result.text)
model.unload()
```

### Available Models

Use short aliases or full HuggingFace model IDs:

```typescript
import { loadModel } from "node-mlx"

// Qwen 3 (recommended default)
loadModel("qwen") // Qwen3-4B-Instruct (best balance)
loadModel("qwen-3-0.6b") // Smallest, fastest
loadModel("qwen-3-1.7b") // Small, good quality

// Qwen 2.5 (legacy)
loadModel("qwen-2.5") // Qwen2.5-1.5B-Instruct
loadModel("qwen-2.5-3b") // Qwen2.5-3B-Instruct

// Phi (Microsoft)
loadModel("phi") // Phi-3.5-mini (default)
loadModel("phi3") // Phi-3-mini
loadModel("phi4") // Phi-4 (8GB, high quality)

// Gemma 3 (Google)
loadModel("gemma-3") // Gemma-3-1B
loadModel("gemma-3-4b") // Gemma-3-4B
loadModel("gemma-3-12b") // Gemma-3-12B
loadModel("gemma-3-27b") // Gemma-3-27B

// Gemma 3n (Google) - Efficient architecture
loadModel("gemma-3n") // Gemma-3n-E4B (default)
loadModel("gemma-3n-e2b") // Gemma-3n-E2B (smaller)

// Llama 3.2 (Meta) - Requires HuggingFace authentication
loadModel("llama") // Llama-3.2-1B-Instruct
loadModel("llama-3.2-3b") // Llama-3.2-3B-Instruct
```

You can also use **any model** from [mlx-community](https://huggingface.co/mlx-community):

```typescript
loadModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
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

### Full HuggingFace Model IDs

You can use any compatible model from HuggingFace:

```typescript
import { loadModel } from "node-mlx"

// MLX-Community pre-quantized models (recommended)
loadModel("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
loadModel("mlx-community/gemma-3-4b-it-4bit")
loadModel("mlx-community/Phi-3.5-mini-instruct-4bit")

// Full precision models (more RAM required)
loadModel("Qwen/Qwen2.5-3B-Instruct")
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

| Model           | node-mlx  | node-llama-cpp | Winner             |
| --------------- | --------- | -------------- | ------------------ |
| **Mistral 7B**  | 101 tok/s | 51 tok/s       | **2x faster** 🏆   |
| **Phi-4 14B**   | 56 tok/s  | 32 tok/s       | **1.8x faster** 🏆 |
| **Qwen3 4B**    | 120 tok/s | 65 tok/s       | **1.8x faster** 🏆 |
| **Gemma-3 12B** | 78 tok/s  | 42 tok/s       | **1.9x faster** 🏆 |

<details>
<summary>Why is MLX faster?</summary>

1. **Unified Memory** – No data copying between CPU and GPU
2. **Metal Optimization** – Native Apple GPU kernels
3. **Lazy Evaluation** – Fused operations, minimal memory bandwidth
4. **Native Quantization** – 4-bit optimized for Apple Silicon

</details>

---

## Supported Architectures

| Architecture | Example Models        | Status          |
| ------------ | --------------------- | --------------- |
| **Qwen2**    | Qwen 2.5              | ✅ Full support |
| **Qwen3**    | Qwen3 0.6B–4B         | ✅ Full support |
| **Llama**    | Llama 3.2, Mistral    | ✅ Full support |
| **Phi3**     | Phi-3, Phi-3.5, Phi-4 | ✅ Full support |
| **Gemma3**   | Gemma 3 (1B–27B)      | ✅ Full support |
| **Gemma3n**  | Gemma 3n E2B/E4B      | ✅ Full support |
| **Mistral3** | Ministral 3 (3B–14B)  | ✅ Full support |
| **SmolLM3**  | SmolLM3 3B            | ✅ Full support |
| **GPT-OSS**  | GPT-OSS 20B/120B MoE  | ✅ Full support |

> **Note:** Llama models require HuggingFace authentication. Run `huggingface-cli login` first.

---

## vs. node-llama-cpp

|                  | node-mlx            | node-llama-cpp |
| ---------------- | ------------------- | -------------- |
| **Platform**     | macOS Apple Silicon | Cross-platform |
| **Backend**      | Apple MLX           | llama.cpp      |
| **Memory**       | Unified CPU+GPU     | Separate       |
| **Model Format** | MLX/Safetensors     | GGUF           |

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
