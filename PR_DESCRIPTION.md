# Port MLX infrastructure directly from Python

## Summary

Switches the MLX Swift infrastructure from ad-hoc implementations to systematic ports from Apple's `mlx-lm` Python library. This provides access to the latest model architectures faster and ensures compatibility with the canonical MLX implementation.

## Key Changes

- **Direct Python→Swift ports** for KVCache, RoPE variants, and MoE layers (SwitchLayers)
- **New directory structure** separating generated, ported, and hand-written code
- **Version tracking** with git hash in all ported files for reproducible updates
- **Swift unit tests** for all ported components (49 tests)
- **CI integration** running Swift tests on macOS with Metal GPU

## Architecture Decisions

### Why port from Python instead of mlx-swift-lm?

`mlx-lm` (Python) is Apple's primary implementation, updated more frequently, and supports models like Llama 4 MoE before `mlx-swift-lm` catches up. Direct porting gives us full control over the timeline.

### Directory structure

```
Sources/NodeMLXCore/
├── generated/models/   # hf2swift generator output (DO NOT EDIT)
├── ported/             # LLM-assisted ports from Python (version tracked)
└── (root)              # Hand-written integration code
```

### What was ported

| Component    | Source             | Notes                                     |
| ------------ | ------------------ | ----------------------------------------- |
| KVCache      | `cache.py`         | Standard, Rotating, Quantized variants    |
| RoPE         | `rope_utils.py`    | Standard, Llama3, Yarn, SuScaled          |
| SwitchLayers | `switch_layers.py` | MoE support for GPT-OSS and future models |

### What was intentionally skipped

Batch processing (server use case), SSM/Mamba support, prompt cache serialization — these can be added when needed.

## Reference

- **mlx-lm version**: `7585c142a6be9c9245f4ce61d087839776cb8275`
- **Porting guide**: `.cursor/prompts/port-python-to-swift.md`
- **Decisions log**: `packages/swift/PORTING_DECISIONS.md`
