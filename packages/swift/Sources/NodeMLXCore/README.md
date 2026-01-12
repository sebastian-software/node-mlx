# NodeMLXCore

Swift implementation of MLX-based language model inference for Node.js.

## Directory Structure

```
NodeMLXCore/
├── generated/          # Auto-generated code (DO NOT EDIT)
│   └── models/         # Model implementations from hf2swift
├── ported/             # Code ported from mlx-lm Python (LLM-assisted)
│   ├── KVCache.swift
│   ├── RoPEUtils.swift
│   └── ...
└── (root)              # Hand-written code
    └── ...
```

## Code Origins

### `/generated/models/`

Auto-generated Swift model implementations. These files are created by the
`hf2swift` generator and should **never be edited manually**.

To regenerate a model:

```bash
pnpm hf2swift --model <name> --output packages/swift/Sources/NodeMLXCore/generated/models/<Name>Generated.swift
```

### `/ported/`

Code ported from Apple's `mlx-lm` Python library using LLM assistance.
These files follow the patterns and logic from the Python originals but
are written in idiomatic Swift.

Source: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models

See `PORTING_DECISIONS.md` in the swift package root for architectural decisions.

### Root Directory

Hand-written Swift code specific to node-mlx that doesn't have a Python
equivalent or requires custom implementation.
