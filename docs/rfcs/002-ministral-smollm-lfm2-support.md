# RFC 002: Ministral 3, SmolLM 3 & LFM2 Support

**Status**: Implemented (Ministral 3 & SmolLM 3) / Deferred (LFM2)
**Created**: 2026-01-10
**Author**: node-mlx team

## Summary

Add support for three new model families:

1. **Ministral 3** (Mistral AI) - Multimodal edge-optimized models
2. **SmolLM 3** (Hugging Face) - Compact multilingual reasoning model
3. **LFM2** (Liquid AI) - Hybrid SSM/Transformer architecture

## Model Overview

### 1. Ministral 3 (Mistral AI)

| Variant         | Parameters | Context | Features             |
| --------------- | ---------- | ------- | -------------------- |
| Ministral 3 3B  | 3.4B       | 256k    | Vision, Multilingual |
| Ministral 3 8B  | 8B         | 256k    | Vision, Multilingual |
| Ministral 3 14B | 14B        | 256k    | Vision, Multilingual |

**Architecture**: Mistral-based with sliding window attention
**License**: Apache 2.0
**Variants**: Base, Instruct, Reasoning

**Expected config.json**:

```json
{
  "model_type": "mistral",
  "architectures": ["MistralForCausalLM"],
  "hidden_size": 2560,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "sliding_window": 4096,
  "vocab_size": 131072
}
```

### 2. SmolLM 3 (Hugging Face)

| Variant    | Parameters | Context | Features                         |
| ---------- | ---------- | ------- | -------------------------------- |
| SmolLM3-3B | 3B         | 128k    | 6 Languages, Think/NoThink modes |

**Architecture**: Llama-based (likely `llama` or `smollm` model_type)
**License**: Apache 2.0
**Languages**: English, French, Spanish, German, Italian, Portuguese

**Expected Features**:

- Long context (128k tokens)
- Dual reasoning modes ("think" vs "no_think")
- Efficient edge deployment

**Expected config.json**:

```json
{
  "model_type": "llama",
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 3072,
  "num_hidden_layers": 36,
  "num_attention_heads": 24,
  "num_key_value_heads": 8,
  "rope_theta": 1000000,
  "max_position_embeddings": 131072
}
```

### 3. LFM2 (Liquid AI)

| Variant   | Parameters | Features               |
| --------- | ---------- | ---------------------- |
| LFM2-350M | 350M       | Hybrid SSM/Transformer |
| LFM2-700M | 700M       | Hybrid SSM/Transformer |
| LFM2-1.2B | 1.2B       | Hybrid SSM/Transformer |
| LFM2-2.6B | 2.6B       | Hybrid SSM/Transformer |

**Architecture**: **Hybrid State Space Model + Transformer**
**License**: TBD (likely proprietary or restricted)
**Languages**: English, Japanese + 8 more

⚠️ **Critical**: LFM2 uses a fundamentally different architecture combining:

- State Space Model (SSM) layers (similar to Mamba)
- Transformer attention layers
- Custom hybrid routing

This is NOT a standard Transformer architecture and requires significant new implementation work.

## Implementation Analysis

### Ministral 3

**Status**: ✅ Should work with existing Mistral support

The generator already recognizes `ministral` in `features.ts` (line 230):

```typescript
if (lower.includes("mistral") || lower.includes("ministral")) {
  return {
    rmsNormStyle: "standard",
    activation: "silu",
    useSlidingWindow: true
    // ...
  }
}
```

**Required Work**:

1. Verify model loads correctly with existing `MistralGenerated.swift`
2. Test quantized variants from `mlx-community`
3. Add Vision support (separate VLM implementation)

**Estimated Effort**: 2-4 hours (mostly testing)

### SmolLM 3

**Status**: ⚠️ May need minor adjustments

SmolLM 3 is likely Llama-based but may have custom features.

**Required Work**:

1. Download model and inspect `config.json` for `model_type`
2. If `model_type: "llama"` → Should work with existing `LlamaGenerated.swift`
3. If custom `model_type: "smollm"` → Add feature flags in generator
4. Verify long context (128k) works with existing RoPE scaling

**Potential Additions**:

```typescript
// features.ts
if (lower.includes("smollm")) {
  return {
    rmsNormStyle: "standard",
    activation: "silu",
    useSlidingWindow: false,
    defaultRopeTheta: 1000000, // Long context
    hasQKNorms: false,
    normsPerLayer: 2
    // SmolLM specific if needed
  }
}
```

**Estimated Effort**: 4-8 hours

### LFM2

**Status**: ❌ Requires major new implementation

LFM2 uses a Hybrid architecture that is NOT supported by the current codebase:

#### State Space Model (SSM) Components Needed

1. **Mamba/SSM Core**:
   - Selective state space mechanism
   - Hardware-efficient recurrence
   - Different computational pattern than attention

2. **Hybrid Layer Types**:

   ```python
   layer_types = ["ssm", "attention", "ssm", "attention", ...]
   ```

3. **New Modules Required**:
   - `SSMLayer` - State space computation
   - `SelectiveSSM` - Input-dependent state selection
   - `CausalConv1d` - Causal convolution for SSM
   - Hybrid model wrapper

#### Architecture Comparison

| Component   | Transformer       | SSM (Mamba-style)    |
| ----------- | ----------------- | -------------------- |
| Core Op     | Attention (O(n²)) | Recurrence (O(n))    |
| Memory      | KV Cache          | Hidden State         |
| Parallelism | Fully parallel    | Sequential (or scan) |

**Estimated Effort**: 40-60 hours (new architecture)

## Implementation Plan

### Phase 1: Ministral 3 (Low effort)

1. Test existing Mistral support with Ministral 3 models
2. Verify quantized variants work
3. Document any config differences
4. (Optional) Add Vision encoder support

**Timeline**: 1 day

### Phase 2: SmolLM 3 (Medium effort)

1. Download and analyze SmolLM 3 config
2. Add `smollm` feature detection if needed
3. Generate and test Swift model
4. Verify 128k context support

**Timeline**: 2-3 days

### Phase 3: LFM2 (High effort) - Optional/Deferred

⚠️ **Recommendation**: Defer LFM2 until:

- Architecture details are publicly documented
- mlx-lm adds official support
- Community demand justifies the effort

If proceeding:

1. Research LFM2/Liquid architecture in detail
2. Implement SSM core modules
3. Add hybrid layer support to generator
4. Extensive testing and optimization

**Timeline**: 2-3 weeks

## Estimated Total Effort

| Model                 | Complexity | Time          | Priority    |
| --------------------- | ---------- | ------------- | ----------- |
| Ministral 3           | Low        | 2-4 hours     | High        |
| SmolLM 3              | Medium     | 4-8 hours     | High        |
| LFM2                  | Very High  | 40-60 hours   | Low (defer) |
| **Total (Phase 1+2)** |            | **~1-2 days** |             |

## Open Questions

1. **SmolLM 3 model_type**: Is it `llama`, `smollm`, or something else?
2. **Ministral 3 Vision**: Should we add multimodal support in this RFC?
3. **LFM2 Availability**: Are weights publicly available? What license?
4. **SSM Priority**: Is there community demand for Mamba/SSM support?

## Recommendations

1. **Proceed immediately** with Ministral 3 and SmolLM 3
2. **Defer LFM2** until:
   - mlx-lm adds official support (follow their implementation)
   - Public weights and documentation available
   - Clear demand from users

3. **Consider separate RFC** for SSM/Mamba architecture support if LFM2 becomes priority

## References

- [Ministral 3 Collection](https://huggingface.co/collections/mistralai/ministral-3)
- [SmolLM 3 Repository](https://github.com/huggingface/smollm)
- [SmolLM 3 Website](https://smollm3.com/)
- [Liquid AI LFM2 Blog](https://www.liquid.ai/blog/introducing-lfm2-2-6b-redefining-efficiency-in-language-models)
- [Mamba Paper](https://arxiv.org/abs/2312.00752) (for SSM architecture reference)

## Appendix: Quick Verification Commands

### Test Ministral 3

```bash
# Check if existing Mistral support works
pnpm hf2swift --model mistral --output test-ministral.swift
cd packages/swift && swift build
```

### Inspect SmolLM 3

```bash
# Download and check config
huggingface-cli download HuggingFaceTB/SmolLM3-3B-Instruct config.json --local-dir ./tmp
cat ./tmp/config.json | jq '.model_type'
```

## Implementation Notes

### Ministral 3 (Mistral 3)

Implemented via generator with the following key features:

**config.json Analysis**:

```json
{
  "model_type": "mistral3",
  "text_config": {
    "model_type": "ministral3",
    "hidden_size": 4096,
    "num_hidden_layers": 34,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 1000000.0,
    "rope_parameters": {
      "rope_type": "yarn",
      "factor": 16.0,
      "mscale": 1.0,
      "original_max_position_embeddings": 16384
    }
  }
}
```

**Generator Features** (`features.ts`):

- `hasYarnRope: true` - YaRN RoPE scaling for long context
- `defaultRopeTheta: 1000000` - 1M theta
- Standard Mistral-style attention and MLP

**Generated Files**:

- `Mistral3Generated.swift` - Full model implementation
- `RoPEParameters` struct for YaRN configuration

### SmolLM 3

Implemented via generator with the following key features:

**config.json Analysis**:

```json
{
  "model_type": "smollm3",
  "hidden_size": 2048,
  "num_hidden_layers": 36,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "rope_theta": 5000000.0,
  "tie_word_embeddings": true,
  "no_rope_layers": [1, 1, 1, 0, 1, 1, 1, 0, ...]
}
```

**Unique Feature**: `no_rope_layers` - Some layers skip RoPE entirely (1 = skip, 0 = use)

**Generator Features** (`features.ts`):

- `hasNoRopeLayers: true` - Layer-specific RoPE skipping
- `defaultRopeTheta: 5000000` - 5M theta for long context
- `hasWeightTying: true` - Shared embed/lm_head weights

**Generated Files**:

- `SmolLM3Generated.swift` - Full model implementation
- `shouldSkipRope(layerIdx)` helper in config
- Conditional RoPE application in attention

### Regeneration Commands

```bash
# Regenerate Ministral 3
pnpm hf2swift --model mistral3 --output packages/swift/Sources/NodeMLXCore/Models/Mistral3Generated.swift

# Regenerate SmolLM3
pnpm hf2swift --model smollm3 --output packages/swift/Sources/NodeMLXCore/Models/SmolLM3Generated.swift

# Verify build
cd packages/swift && swift build -c release
```

### Usage

```typescript
import { loadModel, generate } from "node-mlx"

// Ministral 3
const ministral = await loadModel("mlx-community/Ministral-3-8B-Instruct-2512")
const response1 = await generate(ministral, "Hello!")

// SmolLM3
const smollm = await loadModel("HuggingFaceTB/SmolLM3-3B")
const response2 = await generate(smollm, "Explain quantum computing")
```
