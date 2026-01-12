# hf2swift

Swift code generator for HuggingFace transformer models.

## Overview

`hf2swift` generates Swift model implementations from HuggingFace model patterns. It produces code that integrates with Apple's MLX framework for efficient inference on Apple Silicon.

## Features

- **Feature-based generation**: Uses architectural features, not model names
- **Shared components**: Generates typealiases to reduce code duplication
- **Type-safe configs**: Generates Decodable configuration structs
- **SwiftFormat integration**: Consistent code formatting

## Installation

```bash
pnpm install
pnpm build
```

## Usage

### CLI

```bash
# Generate a model
pnpm hf2swift --model llama --output ./LlamaGenerated.swift

# From config.json
pnpm hf2swift --config ./config.json --model llama --output ./LlamaGenerated.swift
```

### Programmatic

```typescript
import { SwiftGenerator } from "@node-mlx/hf2swift"

const generator = new SwiftGenerator("llama")
const swiftCode = generator.generate([])
```

## Architecture

```
src/
├── generator/
│   ├── model-defs/         # Model family definitions
│   │   ├── types.ts        # Interfaces + defaults
│   │   ├── llama.ts        # Llama family
│   │   ├── qwen.ts         # Qwen2, Qwen3
│   │   ├── gemma.ts        # Gemma3, Gemma3n
│   │   ├── phi.ts          # Phi3, Phi4
│   │   ├── mistral.ts      # Mistral, Mistral3
│   │   ├── gpt-oss.ts      # GPT-OSS MoE
│   │   └── smollm.ts       # SmolLM3
│   ├── components/         # Swift code generators
│   │   ├── attention.ts    # Attention layer
│   │   ├── mlp.ts          # MLP layer
│   │   ├── decoder-layer.ts # Decoder layer
│   │   ├── model.ts        # Model wrapper
│   │   └── rms-norm.ts     # RMSNorm
│   ├── features.ts         # Feature merging
│   ├── helpers.ts          # Utility generators
│   └── index.ts            # Main generator class
├── config.ts               # Config struct generator
├── naming.ts               # Name conversion utilities
└── cli.ts                  # CLI entry point
```

## Model Definitions

Each model family is defined in `model-defs/`:

```typescript
// model-defs/llama.ts
export const llama: ModelDefinition = {
  name: "Llama",
  matches: (modelType) => modelType.toLowerCase().includes("llama"),
  architectural: {
    rmsNormStyle: "standard",
    activation: "silu",
    hasQKNorms: false,
    normsPerLayer: 2
  },
  configDefaults: {
    ropeTheta: 10000,
    rmsNormEps: 1e-5
  }
}
```

## Supported Models

| Model    | Type       | Features                |
| -------- | ---------- | ----------------------- |
| Llama    | `llama`    | Standard transformer    |
| Qwen2    | `qwen2`    | Attention bias          |
| Qwen3    | `qwen3`    | Q/K norms, weight tying |
| Phi-3/4  | `phi3`     | Fused QKV/gate_up       |
| Gemma3   | `gemma3`   | 4 norms, Gemma RMSNorm  |
| Gemma3n  | `gemma3n`  | AltUp, Laurel, VLM      |
| Mistral  | `mistral`  | Sliding window          |
| Mistral3 | `mistral3` | YaRN RoPE               |
| SmolLM3  | `smollm3`  | No-RoPE layers          |
| GPT-OSS  | `gpt_oss`  | MoE, attention sinks    |

## Feature Flags

### Architectural Features

| Feature                 | Effect                              |
| ----------------------- | ----------------------------------- |
| `rmsNormStyle: "gemma"` | Uses (1+weight) scaling             |
| `activation: "silu"`    | SiLU activation in MLP              |
| `hasFusedQKV: true`     | Single qkv_proj instead of separate |
| `hasMoE: true`          | Mixture of Experts MLP              |
| `hasAltUp: true`        | Alternating Updates (Gemma3n)       |
| `hasQKNorms: true`      | Q/K normalization                   |

### Config Values (from config.json)

| Value           | Source              |
| --------------- | ------------------- |
| `ropeTheta`     | `rope_theta`        |
| `slidingWindow` | `sliding_window`    |
| `numExperts`    | `num_local_experts` |

## Generated Output

### Simple Models (Llama, Qwen2)

~195 lines using shared components:

```swift
// MARK: - Attention
typealias LlamaAttention = StandardAttention<LlamaConfiguration>

// MARK: - MLP
typealias LlamaMLP = StandardMLP<LlamaConfiguration>

// MARK: - Decoder Layer
typealias LlamaDecoderLayer = StandardDecoderLayer<LlamaConfiguration>
```

### Complex Models (Gemma3n)

~700+ lines with custom implementations for advanced features.

## Adding a New Model

1. **Create definition**: `model-defs/newmodel.ts`

   ```typescript
   export const newModel: ModelDefinition = {
     name: "NewModel",
     matches: (t) => t.includes("newmodel"),
     architectural: { ...DEFAULT_ARCHITECTURAL, ... },
     configDefaults: { ...DEFAULT_CONFIG, ... }
   }
   ```

2. **Register**: Add to `model-defs/index.ts`:

   ```typescript
   import { newModel } from "./newmodel.js"
   const MODEL_REGISTRY = [..., newModel]
   ```

3. **Test**:
   ```bash
   pnpm hf2swift --model newmodel
   ```

## Development

```bash
# Build
pnpm build

# Test
pnpm test

# Lint
pnpm lint

# Watch mode
pnpm dev
```

## License

MIT
