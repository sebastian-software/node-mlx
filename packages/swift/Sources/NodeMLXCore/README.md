# NodeMLXCore

Swift implementation of MLX-based language model inference for Node.js.

## Architecture

```
NodeMLXCore/
├── generated/          # Auto-generated model code (DO NOT EDIT)
│   └── models/         # One Swift file per model
├── ported/             # Code ported from mlx-lm Python
│   ├── KVCache.swift   # KV cache implementations
│   ├── RoPEUtils.swift # Rotary position embeddings
│   └── ...
├── shared/             # Reusable Swift components
│   ├── Protocols.swift # Base configuration protocols
│   ├── Standard*.swift # Generic model components
│   └── ...
└── (root)              # Hand-written integration code
    ├── Generate.swift  # Text generation
    ├── LLMModel.swift  # Model protocol
    ├── NodeMLXCore.swift # C-interface bridge
    └── Tokenizer.swift # Tokenization
```

## Three-Layer Design

| Directory    | Source          | Edit Policy     | Purpose                        |
| ------------ | --------------- | --------------- | ------------------------------ |
| `generated/` | `hf2swift`      | ❌ Never edit   | Model-specific implementations |
| `ported/`    | `mlx-lm` Python | 🔄 Re-port only | Core MLX infrastructure        |
| `shared/`    | Hand-written    | ✅ Free to edit | Reusable components            |
| Root files   | Hand-written    | ✅ Free to edit | Node.js integration            |

## Supported Models

| Model        | Type           | Features                         |
| ------------ | -------------- | -------------------------------- |
| Llama 3.x    | Standard       | Uses shared components           |
| Qwen2, Qwen3 | Standard       | Qwen3 has Q/K norms              |
| Phi-3, Phi-4 | Fused QKV      | Fused projections                |
| Gemma3       | 4-norm         | Gemma-style RMSNorm              |
| Gemma3n      | VLM            | AltUp, Laurel, sparse activation |
| Mistral      | Sliding window | Window attention                 |
| GPT-OSS      | MoE            | Mixture of Experts               |
| SmolLM3      | No-RoPE layers | Selective RoPE                   |

## Quick Start

### Regenerate a Model

```bash
pnpm hf2swift --model llama --output packages/swift/Sources/NodeMLXCore/generated/models/LlamaGenerated.swift
```

### Build and Test

```bash
cd packages/swift
swift build -c release
swift test
```

## Documentation

- **[PORTING_DECISIONS.md](../../PORTING_DECISIONS.md)** - Architectural decisions
- **[generated/README.md](generated/README.md)** - Generated code guidelines
- **[ported/README.md](ported/README.md)** - Porting process
- **[shared/README.md](shared/README.md)** - Shared component catalog
