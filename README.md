# node-mlx

> LLM inference for Node.js powered by Apple MLX on Apple Silicon

[![CI](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastian-software/node-mlx/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/node-mlx.svg)](https://www.npmjs.com/package/node-mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Run large language models locally on your Mac with native Apple Silicon performance. Built on Apple's [MLX](https://github.com/ml-explore/mlx) framework via [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm).

## Features

- 🚀 **Native Performance** - Optimized for Apple Silicon (M1/M2/M3/M4)
- 🧠 **Unified Memory** - Efficient memory usage via MLX's unified architecture
- 📦 **Easy to Use** - Simple API for text generation and chat
- 🤗 **HuggingFace Integration** - Load any MLX-compatible model
- 💬 **Chat Support** - Multi-turn conversations with context
- 🔄 **Streaming** - Real-time token streaming

## Requirements

- macOS 14.0+ (Sonoma or later)
- Apple Silicon Mac (M1/M2/M3/M4)
- Node.js 20+
- Xcode Command Line Tools

## Installation

```bash
npm install node-mlx
```

## Quick Start

### Text Generation

```typescript
import { generate } from "node-mlx"

const result = await generate("Explain quantum computing in simple terms", {
  model: "mlx-community/gemma-3n-E2B-it-4bit",
  maxTokens: 256,
  temperature: 0.7
})

console.log(result.text)
console.log(`Generated ${result.generatedTokens} tokens at ${result.tokensPerSecond} tok/s`)
```

### Streaming Generation

```typescript
import { generateStream } from "node-mlx"

for await (const token of generateStream("Write a haiku about coding", {
  model: "mlx-community/gemma-3n-E2B-it-4bit"
})) {
  process.stdout.write(token)
}
```

### Chat

```typescript
import { createChat } from "node-mlx"

const chat = createChat({
  model: "mlx-community/gemma-3n-E2B-it-4bit",
  system: "You are a helpful assistant."
})

const response1 = await chat.send("What is TypeScript?")
console.log(response1)

const response2 = await chat.send("How does it compare to JavaScript?")
console.log(response2)
```

## CLI Usage

```bash
# Generate text
npx node-mlx generate --prompt "Hello, world!" --model mlx-community/gemma-3n-E2B-it-4bit

# Interactive chat
npx node-mlx chat --model mlx-community/gemma-3n-E2B-it-4bit

# List recommended models
npx node-mlx models
```

## Recommended Models

| Model         | ID                                     | Size   | Description                        |
| ------------- | -------------------------------------- | ------ | ---------------------------------- |
| Gemma 3n 2B   | `mlx-community/gemma-3n-E2B-it-4bit`   | ~1.5GB | Google's efficient on-device model |
| Gemma 3n 4B   | `mlx-community/gemma-3n-E4B-it-4bit`   | ~2.5GB | Larger Gemma 3n variant            |
| Qwen 3 1.7B   | `mlx-community/Qwen3-1.7B-4bit`        | ~1GB   | Alibaba's compact model            |
| Qwen 3 4B     | `mlx-community/Qwen3-4B-4bit`          | ~2.5GB | Alibaba's mid-size model           |
| Phi 4         | `mlx-community/phi-4-4bit`             | ~8GB   | Microsoft's reasoning model        |
| Llama 4 Scout | `mlx-community/Llama-4-Scout-17B-4bit` | ~10GB  | Meta's latest model                |

Models are automatically downloaded from HuggingFace on first use.

## API Reference

### `generate(prompt, options?)`

Generate text from a prompt.

**Parameters:**

- `prompt` (string) - Input text
- `options.model` (string) - Model ID (default: `mlx-community/gemma-3n-E2B-it-4bit`)
- `options.maxTokens` (number) - Maximum tokens to generate (default: 256)
- `options.temperature` (number) - Sampling temperature (default: 0.7)
- `options.topP` (number) - Top-p sampling (default: 0.9)

**Returns:** `Promise<GenerationResult>`

### `generateStream(prompt, options?)`

Generate text with streaming output.

**Returns:** `AsyncGenerator<string, GenerationResult>`

### `createChat(options?)`

Create a chat session.

**Parameters:**

- `options.model` (string) - Model ID
- `options.system` (string) - System prompt
- `options.maxTokens` (number) - Max tokens per response
- `options.temperature` (number) - Sampling temperature

**Returns:** `Chat` instance

### `isSupported()`

Check if the current platform is supported.

**Returns:** `boolean`

## How It Works

node-mlx uses a thin Node.js wrapper around a Swift CLI built with [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm). This approach provides:

1. **Native Performance** - Swift code runs directly on Apple Silicon
2. **Automatic Updates** - Benefits from Apple's MLX improvements
3. **Minimal Maintenance** - No complex C++ bindings to maintain
4. **Full Compatibility** - Access to all mlx-swift-lm supported models

## Comparison with node-llama-cpp

| Feature      | node-mlx              | node-llama-cpp      |
| ------------ | --------------------- | ------------------- |
| Platform     | macOS (Apple Silicon) | Cross-platform      |
| Backend      | Apple MLX             | llama.cpp           |
| Memory       | Unified (CPU+GPU)     | Separate            |
| GPU Support  | Native Metal          | Metal, CUDA, etc.   |
| Model Format | MLX/Safetensors       | GGUF                |
| Installation | npm install           | npm install + build |

**When to use node-mlx:**

- You're on Apple Silicon
- You want optimal memory efficiency
- You prefer MLX model ecosystem

**When to use node-llama-cpp:**

- You need cross-platform support
- You want GGUF model compatibility
- You're on Intel Mac or other platforms

## Development

```bash
# Clone repository
git clone https://github.com/sebastian-software/node-mlx.git
cd node-mlx

# Install dependencies
pnpm install

# Build Swift CLI
pnpm build:swift

# Build TypeScript
pnpm build:ts

# Run tests
pnpm test
```

## Credits

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - Swift LLM library
- [mlx-community](https://huggingface.co/mlx-community) - MLX model hub

## License

MIT © [Sebastian Werner](https://github.com/sebastian-software)
