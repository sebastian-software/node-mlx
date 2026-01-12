# Porting Decisions: mlx-lm (Python) → NodeMLXCore (Swift)

This document tracks architectural decisions made during the port from Apple's `mlx-lm` Python library to Swift.

## Source of Truth

**Decision**: Port directly from `mlx-lm` (Python), not from `mlx-swift-lm` (Swift).

**Why**:

- `mlx-lm` is updated more frequently and supports more models
- `mlx-swift-lm` lags behind in features and model support
- Direct Python→Swift porting gives us full control

**Reference**: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models

---

## KVCache (cache.py → KVCache.swift)

**Date**: 2026-01-12

### Ported

| Python Class              | Swift Class             | Notes                                          |
| ------------------------- | ----------------------- | ---------------------------------------------- |
| `KVCache`                 | `KVCacheSimple`         | Grow-in-place strategy with step=256           |
| `RotatingKVCache`         | `RotatingKVCache`       | Sliding window with `keep` for attention sinks |
| `QuantizedKVCache`        | `QuantizedKVCache`      | 8-bit quantized KV storage                     |
| `create_causal_mask()`    | `createCausalMask()`    | With optional window size                      |
| `create_attention_mask()` | `createAttentionMask()` | Delegates to cache.makeMask()                  |

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

1. **Protocol-based architecture**: `KVCache` is a Swift protocol, not a base class
   - Enables better composition and testing
   - Default implementations via protocol extension

2. **Static step constant**: `step` is `static let` instead of instance variable
   - More Swift-idiomatic
   - Prevents accidental modification

3. **Method naming**: `update(keys:values:)` instead of `updateAndFetch`
   - Matches existing generated model code
   - Shorter, Swift-idiomatic

4. **Mask return type**: `MLXFast.ScaledDotProductAttentionMaskMode`
   - Integrates directly with MLX's optimized SDPA
   - Supports `.none`, `.causal`, and `.array(MLXArray)`

---

## RoPE Utils (rope_utils.py → RoPEUtils.swift)

**Date**: 2026-01-12

### Ported

| Python Class        | Swift Class        | Notes                                |
| ------------------- | ------------------ | ------------------------------------ |
| `nn.RoPE`           | `RoPE` (MLXNN)     | Built-in, extended with RoPEProvider |
| `Llama3RoPE`        | `Llama3RoPE`       | Smooth frequency interpolation       |
| `YarnRoPE`          | `YarnRoPE`         | Beta-based correction, mscale        |
| `SuScaledRoPE`      | `SuScaledRoPE`     | Long context (longrope)              |
| `initialize_rope()` | `initializeRope()` | Factory function                     |

### Supported rope_type values

- `"default"` → Standard RoPE
- `"linear"` → Linearly scaled (scale = 1/factor)
- `"llama3"` → Llama 3 with smooth interpolation
- `"yarn"` → Yet Another RoPE for extended context
- `"longrope"` → Su-scaled for very long context
- `"mrope"` → Multimodal (returns basic RoPE, modal logic in attention)

### Design Decisions

1. **RoPEProvider protocol**: All RoPE variants conform to `RoPEProvider`
   - Enables polymorphic usage: `any RoPEProvider`
   - Simple interface: `apply(_ x: MLXArray, offset: Int) -> MLXArray`

2. **Simplified SuScaledRoPE**: Original Python has short/long factor switching
   - Our version focuses on long context (the common use case)
   - Short factor is optional with default `[1.0]`

3. **Private computed properties**: `computedMscale`, `computedFreqs` instead of stored
   - Clearer intent: these are derived from init parameters
   - Slightly more Swift-idiomatic

---

## SwitchLayers (switch_layers.py → SwitchLayers.swift)

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
| `swiglu()`              | `gptOssSwiGLU()`        | Clipped SwiGLU for GPT-OSS               |
| `SwiGLU`                | (inlined)               | Simple wrapper, not needed               |

### Not Ported

| Python         | Reason                                  |
| -------------- | --------------------------------------- |
| `SwiGLU` class | Trivial wrapper, function is sufficient |

### Design Decisions

1. **Compiled activation**: `compiledGptOssSwiGLU()` returns compiled closure
   - Matches Python's `@partial(mx.compile, shapeless=True)`
   - Lazy compilation on first call

2. **Sort threshold**: `indices.size > 64`
   - Same as Python: only sort when many tokens
   - Balances sorting overhead vs. memory access efficiency

3. **GPT-OSS specific activation**: Separate `gptOssSwiGLU` function
   - With clipping for numerical stability
   - `alpha=1.702`, `limit=7.0` defaults match GPT-OSS

---

## Base Model (base.py → LLMModel.swift)

**Date**: 2026-01-12

### Ported

| Python                      | Swift                   | Notes                        |
| --------------------------- | ----------------------- | ---------------------------- |
| `BaseModelArgs.from_dict()` | `Decodable` protocol    | Swift's native JSON decoding |
| `create_causal_mask()`      | `createCausalMask()`    | Already in KVCache.swift     |
| `create_attention_mask()`   | `createAttentionMask()` | Already in KVCache.swift     |
| Model interface             | `LLMModel` protocol     | Custom protocol for node-mlx |
| Model factory               | `ModelFactory`          | Type-safe model creation     |

### Not Ported

| Python                                     | Reason                           |
| ------------------------------------------ | -------------------------------- |
| `create_ssm_mask()`                        | SSM models (Mamba) not supported |
| `quantized_scaled_dot_product_attention()` | Advanced feature - can add later |
| `scaled_dot_product_attention()`           | MLXFast.SDPA is used directly    |

### Design Decisions

1. **Protocol-based architecture**: `LLMModel` protocol instead of base class
   - All models conform to common interface
   - Enables type-safe factory pattern

2. **Type-safe model factory**: `ModelFactory.createModel()`
   - Uses `ModelArchitecture` enum
   - Automatic VLM detection via `vision_config`

3. **Decodable configs**: Model configurations use Swift Codable
   - Automatic JSON parsing
   - No manual `from_dict` needed

4. **Cache integration**: `newCache()` method on models
   - Models can provide custom cache types
   - Default uses `createLayerCaches()`

---

## General Principles

1. **Focus on popular models**: Llama, Qwen, Phi, Gemma, Mistral, GPT-OSS
2. **Skip niche features**: Batch processing, SSM models, prompt caching
3. **Premium Swift quality**: Protocols, proper documentation, type safety
4. **Co-located tests**: Tests live next to source files
5. **Minimal dependencies**: Only port what's actually used
