# Shared Model Components

This directory contains reusable Swift implementations shared across all generated models.

## Purpose

Reduce code duplication in generated model files by extracting common patterns into shared, well-tested components.

## Components

| File                      | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| `Protocols.swift`         | Base configuration protocols (`BaseModelConfiguration`, etc.) |
| `RMSNorm.swift`           | Root Mean Square Layer Normalization                          |
| `StandardAttention.swift` | Multi-Head Attention with GQA and RoPE                        |
| `StandardMLP.swift`       | SwiGLU MLP block                                              |
| `StandardDecoder.swift`   | Pre-norm decoder layer                                        |
| `WeightSanitizer.swift`   | Common weight sanitization logic                              |

## Usage in Generated Models

Generated models should:

1. Have their config conform to `BaseModelConfiguration`
2. Use `StandardAttention<Config>`, `StandardMLP<Config>`, etc. for standard components
3. Only generate custom code for model-specific features

## Benefits

- **~70% less generated code** per model
- **Single source of truth** for common patterns
- **Easier testing** - shared components are tested once
- **Consistent behavior** across all models
