# RFC 001: GPT-OSS Mixture of Experts Support

**Status**: Implemented
**Created**: 2026-01-09
**Author**: node-mlx team

## Summary

Add support for OpenAI's GPT-OSS models (gpt-oss-20b, gpt-oss-120b) which use a Mixture of Experts (MoE) architecture.

## Motivation

GPT-OSS is OpenAI's first open-weight model family (released August 2025) under Apache 2.0 license. The 20B model is particularly attractive for local inference as it can run on systems with 16GB RAM while providing strong performance.

**Available MLX Models**:

- `mlx-community/gpt-oss-20b-MXFP4-Q8` (630k+ downloads)
- `mlx-community/gpt-oss-120b-MXFP4-Q8`
- Various quantization levels (4-bit, 8-bit)

## Architecture Overview

### Model Configuration

```json
{
  "model_type": "gpt_oss",
  "architectures": ["GptOssForCausalLM"],
  "hidden_size": 2880,
  "intermediate_size": 2880,
  "num_hidden_layers": 24,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "num_local_experts": 32,
  "num_experts_per_tok": 4,
  "sliding_window": 128,
  "attention_bias": true,
  "layer_types": ["sliding_attention", "full_attention", ...]
}
```

### Key Components

#### 1. SwitchGLU (Mixture of Experts Layer)

The core MoE component that routes tokens to selected experts:

```python
class SwitchGLU:
    def __init__(self, input_dims, hidden_dims, num_experts, activation, bias):
        # Creates num_experts independent expert networks
        # Each expert is a GLU (Gated Linear Unit)
        pass

    def __call__(self, x, indices):
        # Routes input x to experts specified by indices
        # Returns weighted combination of expert outputs
        pass
```

**Swift Implementation Required**:

- `SwitchGLU` module with expert routing
- Batched expert computation for efficiency
- Weight loading for `experts.gate_proj`, `experts.up_proj`, `experts.down_proj`

#### 2. Custom SwiGLU Activation

GPT-OSS uses a modified SwiGLU with specific parameters:

```python
def swiglu(x_linear, x_glu, alpha=1.702, limit=7.0):
    x_glu = clip(x_glu, max=limit)
    x_linear = clip(x_linear, min=-limit, max=limit)
    glu_scaled = alpha * x_glu
    sig = sigmoid(glu_scaled)
    out_glu = x_glu * sig
    return out_glu * (x_linear + 1)  # Note: +1 bias
```

#### 3. Expert Router

```python
class Router:
    def __init__(self, hidden_size, num_experts):
        self.linear = Linear(hidden_size, num_experts, bias=True)

    def __call__(self, x):
        logits = self.linear(x)
        # Select top-k experts
        values, indices = topk(logits, k=num_experts_per_tok)
        weights = softmax(values)
        return weights, indices
```

#### 4. Attention with Sinks

```python
class Attention:
    def __init__(self):
        self.sinks = zeros((num_attention_heads,))  # Learnable attention sinks

    def __call__(self, x, mask, cache):
        # Standard attention with sink tokens for long context
        output = scaled_dot_product_attention(q, k, v, sinks=self.sinks)
        return output
```

#### 5. Mixed Attention Pattern

Alternating between sliding window and full attention:

```python
layer_types = ["sliding_attention", "full_attention"] * (num_layers // 2)
```

## Implementation Plan

### Phase 1: Core MoE Infrastructure

1. **Add `SwitchGLU` module** to Swift
   - Implement expert weight storage
   - Implement batched expert forward pass
   - Handle quantized expert weights

2. **Add TopK operator** for expert selection
   - MLX Swift binding for `argpartition`
   - Extract top-k indices and values

3. **Implement SwiGLU activation**
   - Custom activation with α=1.702, limit=7.0
   - Clipping and bias handling

### Phase 2: GPT-OSS Model

4. **Add `GptOssConfiguration`** struct
   - All MoE-specific fields
   - Layer type patterns

5. **Add `GptOssAttention`** with sinks
   - Learnable sink parameters
   - Sliding/full attention switching

6. **Add `GptOssMLP`** (Router + Experts)
   - Expert routing logic
   - SwitchGLU forward pass

7. **Add `GptOssModel`** wrapper
   - Weight sanitization for fused projections
   - Cache creation with mixed types

### Phase 3: Generator Support

8. **Update `hf2swift` generator**
   - Add MoE feature flags
   - Generate SwitchGLU components
   - Handle expert weight patterns

## Estimated Effort

| Component            | Complexity | Time Estimate    |
| -------------------- | ---------- | ---------------- |
| SwitchGLU module     | High       | 4-6 hours        |
| TopK operator        | Medium     | 1-2 hours        |
| SwiGLU activation    | Low        | 1 hour           |
| Configuration        | Low        | 1 hour           |
| Attention with sinks | Medium     | 2-3 hours        |
| MLP with routing     | High       | 3-4 hours        |
| Model wrapper        | Medium     | 2 hours          |
| Generator updates    | Medium     | 2-3 hours        |
| Testing & debugging  | High       | 4-6 hours        |
| **Total**            |            | **~20-28 hours** |

## Open Questions

1. **Expert parallelism**: Should we support multi-GPU expert sharding?
2. **Memory optimization**: Expert caching strategies for large models?
3. **Quantization**: How to handle per-expert quantization parameters?

## References

- [OpenAI GPT-OSS Announcement](https://openai.com/index/introducing-gpt-oss)
- [mlx-lm gpt_oss.py](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gpt_oss.py)
- [mlx-lm switch_layers.py](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py)

## Implementation Notes

The GPT-OSS MoE support has been implemented in the following files:

### Swift Components

1. **`MoELayers.swift`** - Core MoE infrastructure (manual implementation):
   - `gptOssSwiGLU()` - Custom SwiGLU activation (α=1.702, limit=7.0)
   - `MoERouter` - Token-to-expert routing with top-k selection via `argPartition`
   - `SwitchGLU` - Batched expert computation
   - `MoEMLP` - Complete MoE MLP layer

2. **`GptOssGenerated.swift`** - **AUTO-GENERATED** by hf2swift:
   - `GptOSSConfiguration` - Model configuration with MoE fields
   - `GptOSSAttention` - Attention with learnable sinks
   - `GptOSSDecoderLayer` - Decoder layer with MoE MLP
   - `GptOSSModel` - Top-level model wrapper

3. **`LLMModel.swift`** - Model registry updates:
   - Added `gptOss` architecture case
   - Model factory integration

### Generator Updates (hf2swift)

1. **`features.ts`** - MoE feature flags:
   - `hasMoE`, `numExperts`, `numExpertsPerTok`
   - `hasAttentionSinks`, `useCustomSwiGLU`

2. **`config.ts`** - MoE configuration fields:
   - `numLocalExperts`, `numExpertsPerTok`, `layerTypes`

3. **`mlp.ts`** - MoE MLP generation:
   - `generateMoEMlp()` function for MoE MLP components

4. **`attention.ts`** - Attention sinks support:
   - `sinks` parameter declaration and initialization

5. **`model.ts`** - MoE-specific handling:
   - `newCache()` using `layerTypes` for cache creation
   - `sanitize()` with MoE expert weight mapping

### Regenerate Command

```bash
pnpm hf2swift --model gpt_oss --output packages/swift/Sources/NodeMLXCore/Models/GptOssGenerated.swift
```

### Usage

```typescript
import { loadModel, generate } from "node-mlx"

const model = await loadModel("mlx-community/gpt-oss-20b-MXFP4-Q8")
const response = await generate(model, "Hello, world!")
```

## Appendix: Weight Structure

```
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.{weight,bias}
model.layers.0.self_attn.k_proj.{weight,bias}
model.layers.0.self_attn.v_proj.{weight,bias}
model.layers.0.self_attn.o_proj.{weight,bias}
model.layers.0.self_attn.sinks
model.layers.0.mlp.router.{weight,bias}
model.layers.0.mlp.experts.gate_proj.{weight,bias}  # [num_experts, hidden, intermediate]
model.layers.0.mlp.experts.up_proj.{weight,bias}
model.layers.0.mlp.experts.down_proj.{weight,bias}
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
lm_head.weight
```
