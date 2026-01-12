# Shared Components

Reusable Swift implementations shared across all generated models. These reduce generated code by ~70% and provide a single source of truth for common patterns.

## Protocols

Configuration protocols enable generic components:

| Protocol                     | Properties                            | Used By               |
| ---------------------------- | ------------------------------------- | --------------------- |
| `BaseModelConfiguration`     | hiddenSize, numHeads, ropeTheta, etc. | All models            |
| `AttentionConfiguration`     | + attentionScale                      | Phi (fused attention) |
| `SlidingWindowConfiguration` | + slidingWindow, isGlobalLayer()      | Mistral               |
| `MoEConfiguration`           | + numExperts, numExpertsPerTok        | GPT-OSS               |
| `AltUpConfiguration`         | + altupNumInputs, altupActiveIdx      | Gemma3n               |
| `LaurelConfiguration`        | + laurelRank                          | Gemma3n               |
| `SparseMLPConfiguration`     | + intermediateSizes, sparsityPattern  | Gemma3n               |

## Standard Components

Generic implementations for common transformer patterns:

| Component                 | Description                    | Used By      |
| ------------------------- | ------------------------------ | ------------ |
| `RMSNorm`                 | Root Mean Square normalization | Most models  |
| `StandardAttention<C>`    | GQA attention with RoPE        | Llama, Qwen2 |
| `StandardMLP<C>`          | SwiGLU MLP (gate/up/down)      | Llama, Qwen2 |
| `StandardDecoderLayer<C>` | Pre-norm decoder (2 norms)     | Llama, Qwen2 |
| `FusedQKVAttention<C>`    | Fused Q/K/V projection         | Phi3, Phi4   |

## Specialized Components

For advanced architectures:

| Component         | Description                            | Used By     |
| ----------------- | -------------------------------------- | ----------- |
| `AltUpBlock<C>`   | Alternating Updates for sparse compute | Gemma3n     |
| `LaurelBlock<C>`  | Low-rank residual layer                | Gemma3n     |
| `SparseMLP<C>`    | gelu_topk sparse activation            | Gemma3n     |
| `MoESanitizer`    | MoE weight transformation              | GPT-OSS     |
| `WeightSanitizer` | Standard weight cleanup                | Most models |

## Utilities

| File              | Functions                              | Purpose              |
| ----------------- | -------------------------------------- | -------------------- |
| `MathUtils.swift` | `erfinv()`, `clipResidual()`, `topK()` | Mathematical helpers |
| `Protocols.swift` | `ConfigDecoder`                        | JSON decoding helper |

## Usage in Generated Code

### Simple Models (Llama, Qwen2)

Generator produces typealiases:

```swift
// MARK: - Attention
typealias LlamaAttention = StandardAttention<LlamaConfiguration>

// MARK: - MLP
typealias LlamaMLP = StandardMLP<LlamaConfiguration>

// MARK: - Decoder Layer
typealias LlamaDecoderLayer = StandardDecoderLayer<LlamaConfiguration>
```

### Complex Models (Gemma3n, GPT-OSS)

Generator produces custom code but still uses shared components:

```swift
// Uses shared AltUpBlock
extension Gemma3nConfiguration: AltUpConfiguration {}
typealias Gemma3nAltUp = AltUpBlock<Gemma3nConfiguration>

// Uses shared MathUtils
private func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
    MathUtils.clipResidual(x, y)
}
```

## Adding New Components

1. Create Swift file in `shared/`
2. Define protocol if configuration-dependent
3. Implement as generic class: `class MyComponent<C: MyConfiguration>: Module`
4. Update generator to use component when features match
5. Add tests

## Benefits

- **Less code**: ~195 lines vs ~350 lines per simple model
- **Testable**: Components tested once, used everywhere
- **Consistent**: Same behavior across all models
- **Maintainable**: Fix once, applies to all models
