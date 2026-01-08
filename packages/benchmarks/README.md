# @anthropic/benchmarks

Benchmarks comparing node-mlx with node-llama-cpp.

## Usage

```bash
# From the benchmarks package directory
cd packages/benchmarks

# Run main benchmark (default: phi4)
pnpm benchmark [model]

# Available models: phi4, gemma3n, gptoss, qwen3, ministral

# Quick comparison
pnpm compare

# Model-specific benchmarks
pnpm phi4
pnpm gemma3n
pnpm mlx-models
```

## Prerequisites

- Built node-mlx (`pnpm build` from root)
- For llama.cpp comparison: GGUF models in `.models/` directory

## Benchmark Files

| File                 | Description                                   |
| -------------------- | --------------------------------------------- |
| `benchmark.ts`       | Full statistical benchmark with multiple runs |
| `compare.ts`         | Quick side-by-side comparison                 |
| `phi4-compare.ts`    | Phi-4 specific benchmark                      |
| `gemma3n-compare.ts` | Gemma 3n specific benchmark                   |
| `mlx-models.ts`      | Test multiple MLX model sizes                 |
