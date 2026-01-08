# @anthropic/mlx-cli

Interactive CLI for chatting with LLMs on Apple Silicon using node-mlx.

## Installation

```bash
# From npm (when published)
npm install -g @anthropic/mlx-cli

# Or run directly with npx
npx @anthropic/mlx-cli
```

## Usage

### Interactive Chat

```bash
mlx                              # Chat with default model (llama-3.2-1b)
mlx --model phi-3-mini           # Use a specific model
mlx --model mlx-community/Phi-4  # Use any mlx-community model
```

### One-Shot Query

```bash
mlx "What is 2+2?"
mlx "Explain quantum computing" --tokens 500
mlx "Write a poem" --model qwen-2.5-3b --temp 0.9
```

### Commands

```bash
mlx --list        # List available models
mlx --help        # Show help
mlx --version     # Show version
```

## Interactive Commands

Once in chat mode:

| Command         | Description             |
| --------------- | ----------------------- |
| `/model <name>` | Switch to another model |
| `/temp <0-2>`   | Set temperature         |
| `/tokens <n>`   | Set max tokens          |
| `/clear`        | Clear conversation      |
| `/list`         | List available models   |
| `/help`         | Show commands           |
| `/quit`         | Exit                    |

## Examples

```bash
# Quick question
$ mlx "What's the capital of France?"
Loading llama-3.2-1b...
Generating...

The capital of France is Paris.

(12 tokens, 98.5 tok/s)

# Interactive chat
$ mlx --model qwen-2.5-3b
╔══════════════════════════════════════╗
║  MLX CLI - LLMs on Apple Silicon     ║
╚══════════════════════════════════════╝

Loading qwen-2.5-3b...
✓ Model loaded

Type your message or /help for commands

You: Hello!
AI: Hello! How can I help you today?
(8 tokens, 112.3 tok/s)

You: /temp 0.9
Temperature set to 0.9

You: Write a haiku about coding
AI: Lines of code appear,
    Bugs hide in the morning light,
    Coffee saves the day.
(18 tokens, 105.7 tok/s)

You: /quit
Goodbye!
```

## Available Models

| Shortcut      | Model                                     |
| ------------- | ----------------------------------------- |
| llama-3.2-1b  | mlx-community/Llama-3.2-1B-Instruct-4bit  |
| llama-3.2-3b  | mlx-community/Llama-3.2-3B-Instruct-4bit  |
| qwen-2.5-0.5b | mlx-community/Qwen2.5-0.5B-Instruct-4bit  |
| qwen-2.5-1.5b | mlx-community/Qwen2.5-1.5B-Instruct-4bit  |
| qwen-2.5-3b   | mlx-community/Qwen2.5-3B-Instruct-4bit    |
| phi-3-mini    | mlx-community/Phi-3-mini-4k-instruct-4bit |
| gemma-3n-2b   | mlx-community/gemma-3n-E2B-it-lm-4bit     |
| gemma-3n-4b   | mlx-community/gemma-3n-E4B-it-lm-4bit     |

Or use any model from [mlx-community](https://huggingface.co/mlx-community):

```bash
mlx --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```
