# node-mlx

**The fastest way to run LLMs in Node.js on Apple Silicon.**

[![CI](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/node-mlx.svg)](https://www.npmjs.com/package/node-mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript Coverage](https://codecov.io/gh/sebastian-software/node-mlx/branch/main/graph/badge.svg?flag=typescript)](https://codecov.io/gh/sebastian-software/node-mlx)
[![Swift Coverage](https://codecov.io/gh/sebastian-software/node-mlx/branch/main/graph/badge.svg?flag=swift)](https://codecov.io/gh/sebastian-software/node-mlx)

---

## Why node-mlx?

<table>
<tr>
<td width="25%" align="center">
<h3>⚡ 60x Faster</h3>
<p>Up to <strong>60x faster</strong> than llama.cpp on MoE models.</p>
</td>
<td width="25%" align="center">
<h3>🧠 Unified Memory</h3>
<p>Models live in Apple Silicon's unified memory. No CPU↔GPU copies.</p>
</td>
<td width="25%" align="center">
<h3>🔗 True Native</h3>
<p>Direct Swift↔Node.js bridge. No subprocess. No CLI wrapper.</p>
</td>
<td width="25%" align="center">
<h3>🤖 Auto-Generated</h3>
<p>Model code generated from HuggingFace. New models in minutes.</p>
</td>
</tr>
</table>

---

## Quick Start

```bash
npm install node-mlx
```

```typescript
import { loadModel } from "node-mlx"

const model = loadModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

const result = model.generate("Explain quantum computing:", {
  maxTokens: 200,
  temperature: 0.7
})

console.log(result.text)
// → 101 tokens/sec on M1 Ultra

model.unload()
```

**That's it.** Model downloads automatically. Runs on GPU immediately.

---

## Performance

Real benchmarks on Mac Studio M1 Ultra (64GB):

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

## Supported Models

Tested model architectures with auto-generated Swift code:

| Architecture | Example Models                | Status          |
| ------------ | ----------------------------- | --------------- |
| **Qwen2**    | Qwen 2.5, Qwen3 (MoE)         | ✅ Full support |
| **Llama**    | Llama 3.2, Mistral, Ministral | ✅ Full support |
| **Phi3**     | Phi-3, Phi-4                  | ✅ Full support |
| **GPT-OSS**  | GPT-OSS 20B (MoE)             | ✅ Full support |
| **Gemma3n**  | Gemma 3n (VLM text-only)      | 🔧 Experimental |

Any 4-bit quantized model from [mlx-community](https://huggingface.co/mlx-community) with a supported architecture works. Models download automatically on first use.

---

## API

### Load & Generate (Recommended)

```typescript
import { loadModel } from "node-mlx"

// Load once
const model = loadModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

// Generate many times (fast - model stays in memory)
const r1 = model.generate("Hello!", { maxTokens: 100 })
const r2 = model.generate("Another prompt", { maxTokens: 100 })

// Clean up
model.unload()
```

### One-Shot (Convenience)

```typescript
import { generate } from "node-mlx"

// Loads, generates, unloads automatically
const result = generate("mlx-community/Llama-3.2-1B-Instruct-4bit", "Hello!")
```

### Options

| Option        | Type   | Default | Description                             |
| ------------- | ------ | ------- | --------------------------------------- |
| `maxTokens`   | number | 256     | Maximum tokens to generate              |
| `temperature` | number | 0.7     | Sampling randomness (0 = deterministic) |
| `topP`        | number | 0.9     | Nucleus sampling threshold              |

### Response

```typescript
{
  text: string // Generated text
  tokenCount: number // Tokens generated
  tokensPerSecond: number // Generation speed
}
```

### Utilities

```typescript
import { isSupported, getVersion } from "node-mlx"

isSupported() // true on Apple Silicon Mac
getVersion() // "1.0.0"
```

---

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Node.js 20+

For development: Xcode Command Line Tools (`xcode-select --install`)

---

## Architecture

```
Node.js  →  N-API  →  Swift (.dylib)  →  MLX  →  Metal GPU
                           │
                    NodeMLXCore
                    ├── LLMEngine
                    ├── Auto-generated Models (hf2swift)
                    └── HuggingFace Tokenizers
```

### vs. mlx-swift-lm

|                   | node-mlx                   | mlx-swift-lm          |
| ----------------- | -------------------------- | --------------------- |
| **Model Code**    | Auto-generated from Python | Hand-written Swift    |
| **New Model**     | Run generator → done       | Manual implementation |
| **Dependencies**  | Only mlx-swift             | Full mlx-swift-lm     |
| **Customization** | Full control               | Use as-is             |

**node-mlx** uses `hf2swift` to automatically generate Swift model code from HuggingFace Transformers Python sources. This means:

- ✅ **New models in minutes** – Just run the generator
- ✅ **Stays current** – Tracks upstream HuggingFace changes
- ✅ **Full transparency** – Generated code is readable and debuggable
- ✅ **Zero runtime dependency** on mlx-swift-lm

<details>
<summary>Adding a new model</summary>

```bash
# Generate Swift code from any HuggingFace model
pnpm hf2swift \
  --model MyModel \
  --source path/to/modeling_mymodel.py \
  --config organization/model-name \
  --output packages/swift/Sources/NodeMLXCore/Models/MyModel.swift
```

The TypeScript-based `hf2swift` generator parses the Python model code using [py-ast](https://www.npmjs.com/package/py-ast) and produces equivalent Swift using MLX primitives.

</details>

---

## vs. node-llama-cpp

|                  | node-mlx            | node-llama-cpp |
| ---------------- | ------------------- | -------------- |
| **Platform**     | macOS Apple Silicon | Cross-platform |
| **Backend**      | Apple MLX           | llama.cpp      |
| **Memory**       | Unified CPU+GPU     | Separate       |
| **Model Format** | MLX/Safetensors     | GGUF           |
| **MoE Support**  | ✅ Excellent        | ⚠️ Limited     |

**Choose node-mlx** if you're on Apple Silicon and want maximum performance.

**Choose node-llama-cpp** if you need cross-platform or GGUF compatibility.

---

## Development

```bash
git clone https://github.com/sebastian-software/node-mlx.git
cd node-mlx
pnpm install
pnpm build        # Build everything
pnpm test         # Run tests
```

<details>
<summary>Project structure</summary>

```
node-mlx/
├── packages/
│   ├── node-mlx/           # TypeScript + Native binding
│   │   ├── src/            # TypeScript source
│   │   ├── test/           # TypeScript tests
│   │   └── native/         # C++ N-API binding
│   ├── swift/              # Swift package
│   │   ├── Sources/        # Swift source
│   │   └── Tests/          # Swift tests
│   ├── hf2swift/           # TypeScript code generator
│   │   ├── src/            # Generator source
│   │   └── tests/          # Generator tests
│   └── benchmarks/         # Performance benchmarks
│       └── src/            # Benchmark scripts
└── dist/                   # Built output
```

</details>

<details>
<summary>Build steps</summary>

```bash
pnpm build:swift   # Swift library → packages/swift/.build/
pnpm build:native  # N-API addon → packages/node-mlx/native/build/
pnpm build:ts      # TypeScript → dist/
```

</details>

---

## Credits

Built on [MLX](https://github.com/ml-explore/mlx) by Apple, [mlx-swift](https://github.com/ml-explore/mlx-swift), and [swift-transformers](https://github.com/huggingface/swift-transformers) by HuggingFace.

**Special thanks to [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)** – while our model code is auto-generated, we adopted and adapted several core components from their excellent implementation:

- KV Cache management (`KVCacheSimple`, `RotatingKVCache`)
- Token sampling strategies (temperature, top-p, repetition penalty)
- RoPE implementations (Llama3, Yarn, LongRoPE)
- Attention utilities and quantization support

Their work provided an invaluable foundation for building reliable LLM infrastructure in Swift.

---

## License

MIT © 2026 [Sebastian Software GmbH](https://sebastian-software.de)
