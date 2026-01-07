# Benchmark: node-mlx vs node-llama-cpp

This benchmark compares LLM inference performance on Apple Silicon between:

- **node-mlx** - Uses Apple's MLX framework (Metal GPU)
- **node-llama-cpp** - Uses llama.cpp (Metal GPU)

Both libraries use GPU acceleration via Metal on Apple Silicon.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Gemma 3n E4B (4-bit) |
| node-mlx model | `mlx-community/gemma-3n-E4B-it-4bit` |
| node-llama-cpp model | `unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M` |
| Max Tokens | 100 |
| Warmup Runs | 1 |
| Benchmark Runs | 3 |

## Running the Benchmark

### Prerequisites

1. Build node-mlx:
   ```bash
   pnpm build
   ```

2. Ensure you have network access (models are downloaded on first run)

### Run

```bash
npx tsx benchmark/run.ts
```

The benchmark will:
1. Download models if not cached (~2.5GB for MLX, ~4.5GB for GGUF)
2. Run warmup iterations
3. Run benchmark iterations
4. Save results to `benchmark/results/`

## Results

> **Note**: Results will vary based on your hardware (M1/M2/M3/M4) and system load.

### Expected Metrics

| Metric | node-mlx | node-llama-cpp |
|--------|----------|----------------|
| Tokens/sec | TBD | TBD |
| Model Size | ~2.5 GB | ~4.5 GB |
| Memory Usage | TBD | TBD |

### Why MLX?

| Advantage | Explanation |
|-----------|-------------|
| **Unified Memory** | MLX uses Apple's unified memory architecture more efficiently |
| **Smaller Models** | MLX quantization typically produces smaller files |
| **Apple Optimized** | Developed by Apple specifically for Apple Silicon |
| **New Architectures** | Faster support for new model architectures (e.g., MatFormer) |

## Interpreting Results

- **Tokens/sec**: Higher is better. Measures generation speed.
- **Model Size**: Smaller is better for storage/download.
- **Memory Usage**: Lower is better. MLX's unified memory can be more efficient.

## Hardware Tested

Results should be collected on:

- [ ] M1 MacBook Air
- [ ] M1 Pro MacBook Pro
- [ ] M2 MacBook Air
- [ ] M3 MacBook Pro
- [ ] M3 Max Mac Studio
- [ ] M4 MacBook Pro

## Contributing Results

If you run this benchmark on your hardware, please share your results by:

1. Running `npx tsx benchmark/run.ts`
2. Opening an issue or PR with your results JSON
3. Include your Mac model and macOS version

