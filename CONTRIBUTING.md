# Contributing to node-mlx

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- macOS 14.0+ on Apple Silicon (M1/M2/M3/M4)
- Node.js 20+
- pnpm 10+
- Xcode Command Line Tools (for Swift)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/sebastian-software/node-mlx.git
cd node-mlx

# Install dependencies
pnpm install

# Build the Swift CLI
pnpm build:swift

# Build TypeScript
pnpm build:ts

# Run tests
pnpm test
```

### Project Structure

```
node-mlx/
├── swift/
│   ├── Package.swift       # Swift package definition
│   └── Sources/
│       └── llm-cli/
│           └── main.swift  # Swift CLI implementation
├── src/
│   ├── index.ts           # Main TypeScript API
│   └── cli.ts             # Node.js CLI wrapper
├── test/
│   └── index.test.ts      # Unit tests
└── dist/                  # Built TypeScript
```

## Development Workflow

### Commands

| Command              | Description                  |
| -------------------- | ---------------------------- |
| `pnpm build`         | Build Swift CLI + TypeScript |
| `pnpm build:swift`   | Build only Swift CLI         |
| `pnpm build:ts`      | Build only TypeScript        |
| `pnpm test`          | Run unit tests               |
| `pnpm test:coverage` | Run tests with coverage      |
| `pnpm lint`          | Run ESLint                   |
| `pnpm format`        | Format code with Prettier    |
| `pnpm typecheck`     | TypeScript type checking     |

### Code Style

- We use ESLint with TypeScript rules
- Prettier for formatting
- Conventional Commits for commit messages

Commit message format:

```
type(scope): description

feat: add new feature
fix: bug fix
docs: documentation changes
chore: maintenance tasks
test: adding tests
refactor: code refactoring
```

### Pre-commit Hooks

The project uses Husky to run:

- **pre-commit**: Prettier formatting via lint-staged
- **commit-msg**: Conventional commit validation

## Submitting Changes

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run tests (`pnpm test`)
5. Commit using conventional commits
6. Push to your fork
7. Open a Pull Request

### PR Guidelines

- Keep PRs focused on a single change
- Include tests for new features
- Update documentation if needed
- Ensure CI passes

## Reporting Issues

### Bug Reports

Please include:

- macOS version and chip (e.g., macOS 14.2, M3 Pro)
- Node.js version (`node --version`)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

### Feature Requests

- Describe the use case
- Explain why it would be useful
- Consider if it fits the project scope (macOS/MLX focused)

## Questions?

Feel free to open a Discussion or Issue if you have questions!
