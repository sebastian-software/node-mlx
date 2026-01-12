# Ported Code

Code in this directory is ported from Apple's `mlx-lm` Python library using LLM assistance.

## Source

- **Repository**: https://github.com/ml-explore/mlx-lm
- **Path**: `mlx_lm/models/`
- **Git Hash**: `7585c142a6be9c9245f4ce61d087839776cb8275`
- **Date**: 2026-01-12

## Ported Files

| Python Source      | Swift File           | Description                                              |
| ------------------ | -------------------- | -------------------------------------------------------- |
| `cache.py`         | `KVCache.swift`      | KV cache implementations (Standard, Rotating, Quantized) |
| `rope_utils.py`    | `RoPEUtils.swift`    | Rotary position embeddings (Standard, Llama3, Yarn, Su)  |
| `switch_layers.py` | `SwitchLayers.swift` | MoE switch layers (SwitchLinear, SwitchGLU, etc.)        |
| `gemma.py`         | `GemmaRMSNorm.swift` | Gemma-style (1+weight) RMSNorm                           |

## Porting Guidelines

### File Header

Every ported file must include:

```swift
// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/<filename>.py
// Git Hash: <full-40-char-hash> (<YYYY-MM-DD>)
```

### Update Process

1. **Check latest mlx-lm**:

   ```bash
   curl -s "https://api.github.com/repos/ml-explore/mlx-lm/commits/main" | grep '"sha"' | head -1
   ```

2. **Download Python source**:

   ```bash
   curl -s "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/<file>.py" -o /tmp/<file>.py
   ```

3. **Use Cursor command**: `/port-python-to-swift`

4. **Update documentation**: Update hash in file header and PORTING_DECISIONS.md

## Design Decisions

See [PORTING_DECISIONS.md](../../PORTING_DECISIONS.md) for detailed architectural decisions.

### Key Patterns

| Python        | Swift             |
| ------------- | ----------------- |
| `snake_case`  | `camelCase`       |
| `mx.array`    | `MLXArray`        |
| `nn.Module`   | `Module` (MLXNN)  |
| `__init__`    | `init`            |
| `@property`   | computed property |
| `Optional[T]` | `T?`              |

### What We Skip

- Batch processing (BatchKVCache, etc.)
- SSM models (MambaCache)
- Serialization (save/load prompt cache)
- Server-specific features
- Niche use cases (< 5% of users)

## Testing

Tests live in `packages/swift/Tests/NodeMLXCoreTests/`:

- `KVCacheTests.swift`
- `RoPEUtilsTests.swift`
- `SwitchLayersTests.swift`

Run tests:

```bash
cd packages/swift
swift test
```
