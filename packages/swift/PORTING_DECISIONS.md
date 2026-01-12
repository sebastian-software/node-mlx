# Porting Decisions: mlx-lm (Python) → NodeMLXCore (Swift)

This document tracks architectural decisions made during the port from Apple's `mlx-lm` Python library to Swift.

## Source of Truth

**Decision**: Port directly from `mlx-lm` (Python), not from `mlx-swift-lm` (Swift).

**Why**:

- `mlx-lm` is updated more frequently and supports more models
- `mlx-swift-lm` lags behind in features and model support
- Direct Python→Swift porting gives us full control

**Reference**: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models

**Current Version**:

- Git Hash: `7585c142a6be9c9245f4ce61d087839776cb8275`
- Ported: 2026-01-12

---

## Directory Structure

**Date**: 2026-01-12

### Layout

```
Sources/NodeMLXCore/
├── generated/          # Auto-generated code (DO NOT EDIT)
│   └── models/         # Model implementations from hf2swift
├── ported/             # Code ported from mlx-lm Python (LLM-assisted)
│   ├── KVCache.swift
│   ├── RoPEUtils.swift
│   └── SwitchLayers.swift
└── (root)              # Hand-written integration code
    ├── Generate.swift
    ├── LLMModel.swift
    ├── NodeMLXCore.swift
    ├── StringOrNumber.swift
    └── Tokenizer.swift
```

### Design Decisions

1. **Clear separation**: Generated, ported, and hand-written code in distinct directories
2. **README in each folder**: Documents purpose and maintenance guidelines
3. **Co-located tests**: Tests will live alongside source files (not in separate `Tests/` folder)

---

## KVCache (cache.py → ported/KVCache.swift)

**Date**: 2026-01-12

### Ported

| Python Class              | Swift Class             | Notes                                          |
| ------------------------- | ----------------------- | ---------------------------------------------- |
| `KVCache`                 | `StandardKVCache`       | Grow-in-place strategy with step=256           |
| `RotatingKVCache`         | `RotatingKVCache`       | Sliding window with `keep` for attention sinks |
| `QuantizedKVCache`        | `QuantizedKVCache`      | 8-bit quantized KV storage                     |
| `create_causal_mask()`    | `createCausalMask()`    | With optional window size                      |
| `create_attention_mask()` | `createAttentionMask()` | Returns MLXFast mask mode                      |

### Not Ported (Low Priority)

| Python Class           | Reason                                                         |
| ---------------------- | -------------------------------------------------------------- |
| `BatchKVCache`         | Server/batch processing - not needed for single-user inference |
| `BatchRotatingKVCache` | Server/batch processing                                        |
| `MambaCache`           | SSM models (Mamba, Jamba) - niche use case                     |
| `ArraysCache`          | Generic container for SSM                                      |
| `ChunkedKVCache`       | Chunked attention - specialized use case                       |
| `CacheList`            | Container for mixed caches                                     |
| `ConcatenateKVCache`   | Simple concat - rarely used, KVCache is better                 |
| `save_prompt_cache()`  | Serialization - can add later if needed                        |
| `load_prompt_cache()`  | Serialization                                                  |

### Design Decisions

1. **Protocol-based architecture**: `KVCacheProtocol` enables polymorphism
2. **Static step constant**: `step = 256` is a static constant, not instance variable
3. **Renamed main class**: `KVCache` → `StandardKVCache` to avoid name conflicts with protocol alias
4. **Type aliases for compatibility**: `KVCache` = `KVCacheProtocol`, `KVCacheSimple` = `StandardKVCache`

---

## RoPE Utils (rope_utils.py → ported/RoPEUtils.swift)

**Date**: 2026-01-12

### Ported

| Python Class        | Swift Class        | Notes                                 |
| ------------------- | ------------------ | ------------------------------------- |
| `nn.RoPE`           | `StandardRoPE`     | Wrapper with RoPEProvider conformance |
| `Llama3RoPE`        | `Llama3RoPE`       | Smooth frequency interpolation        |
| `YarnRoPE`          | `YarnRoPE`         | Beta-based correction, mscale         |
| `SuScaledRoPE`      | `SuScaledRoPE`     | Long context (longrope)               |
| `initialize_rope()` | `initializeRope()` | Factory function                      |

### Supported rope_type values

- `"default"` → Standard RoPE
- `"linear"` → Linearly scaled (scale = 1/factor)
- `"llama3"` → Llama 3 with smooth interpolation
- `"yarn"` → Yet Another RoPE for extended context
- `"longrope"` → Su-scaled for very long context
- `"mrope"` → Multimodal (returns basic RoPE)

### Design Decisions

1. **RoPEProvider protocol**: All RoPE variants conform to common interface
2. **callAsFunction signature**: `(_ x: MLXArray, offset: Int) -> MLXArray`

---

## SwitchLayers (switch_layers.py → ported/SwitchLayers.swift)

**Date**: 2026-01-12

### Ported

| Python Class/Function   | Swift                   | Notes                                    |
| ----------------------- | ----------------------- | ---------------------------------------- |
| `_gather_sort()`        | `gatherSort()`          | Sort tokens by expert for batched access |
| `_scatter_unsort()`     | `scatterUnsort()`       | Restore original token order             |
| `SwitchLinear`          | `SwitchLinear`          | Expert-specific linear layer             |
| `QuantizedSwitchLinear` | `QuantizedSwitchLinear` | Quantized version                        |
| `SwitchGLU`             | `SwitchGLU`             | Gated linear unit with experts           |
| `SwitchMLP`             | `SwitchMLP`             | Simple MLP with experts                  |
| `swiglu()`              | `swiGLU()`              | SwiGLU activation function               |

### GPT-OSS Specific

| Python         | Swift             | Notes                   |
| -------------- | ----------------- | ----------------------- |
| Clipped SwiGLU | `gptOssSwiGLU()`  | With limit=7.0 clipping |
| SwiGLU variant | `SwiGLUSwitchGLU` | Uses clipped activation |

### Design Decisions

1. **Sort threshold**: `indices.size >= 64` (same as Python)
2. **Compiled activation**: Using lazy closure for compiled SwiGLU
3. **Module initialization**: Using `_property.wrappedValue` pattern

---

## TODO: Remaining Work

### API Compatibility

The generated models use APIs that need to be aligned:

1. **RoPE**: Models use `rope.apply(x, offset:)` but our port uses `rope(x, offset:)`
2. **createAttentionMask**: Parameter signature mismatch
3. **KVCache interface**: Ensure protocol methods match generated model expectations

### Options to Fix

1. **Update generator**: Modify `hf2swift` to use new API signatures
2. **Add compatibility layer**: Create wrapper functions that match old signatures
3. **Gradual migration**: Update models one by one

---

## General Principles

1. **Focus on popular models**: Llama, Qwen, Phi, Gemma, Mistral, GPT-OSS
2. **Skip niche features**: Batch processing, SSM models, prompt caching
3. **Premium Swift quality**: Protocols, proper documentation, type safety
4. **Co-located tests**: Tests live next to source files
5. **Minimal dependencies**: Only port what's actually used
