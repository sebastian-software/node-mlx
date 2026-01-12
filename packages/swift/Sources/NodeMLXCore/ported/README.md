# Ported Code

Code in this directory is ported from Apple's `mlx-lm` Python library.

## Source

- Repository: https://github.com/ml-explore/mlx-lm
- Path: `mlx_lm/models/`
- Git Hash: `7585c142a6be9c9245f4ce61d087839776cb8275`
- Ported: 2026-01-12

## Porting Process

These files are ported using LLM assistance following the guidelines in
`.cursor/prompts/port-python-to-swift.md`.

### To port or update a file:

1. Download the latest Python source:

   ```bash
   curl -s "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/<file>.py" -o /tmp/<file>.py
   ```

2. Use the `/port-python-to-swift` slash command in Cursor

3. Follow the porting guidelines for idiomatic Swift

## Ported Files

| Python Source      | Swift File           | Description                |
| ------------------ | -------------------- | -------------------------- |
| `cache.py`         | `KVCache.swift`      | KV cache implementations   |
| `rope_utils.py`    | `RoPEUtils.swift`    | Rotary position embeddings |
| `switch_layers.py` | `SwitchLayers.swift` | MoE switch layers          |
| `gemma.py`         | `GemmaRMSNorm.swift` | Gemma (1+weight) RMSNorm   |

## Design Decisions

See `../../PORTING_DECISIONS.md` for architectural decisions made during porting.
