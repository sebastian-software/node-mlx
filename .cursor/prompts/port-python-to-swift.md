# Port Python mlx-lm to Swift

You are porting Python code from Apple's `mlx-lm` library to Swift for the `node-mlx` project.

## Source Repository

- **Primary**: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models
- **Reference only**: https://github.com/ml-explore/mlx-swift-lm

**IMPORTANT**: Always record the exact git hash. Get it with:

```bash
curl -s "https://api.github.com/repos/ml-explore/mlx-lm/commits/main" | grep '"sha"' | head -1
```

## File Locations

| Type              | Directory                                              |
| ----------------- | ------------------------------------------------------ |
| Ported code       | `packages/swift/Sources/NodeMLXCore/ported/`           |
| Shared components | `packages/swift/Sources/NodeMLXCore/shared/`           |
| Tests             | `packages/swift/Tests/NodeMLXCoreTests/`               |
| Generated models  | `packages/swift/Sources/NodeMLXCore/generated/models/` |

## Core Principles

### 1. Clean Cut Philosophy

- Start fresh, don't patch existing code
- Port with understanding, not blind translation
- Premium architect-level Swift: idiomatic, elegant, maintainable

### 2. Focus on Popular Models

| Priority     | Models                                    | Notes                     |
| ------------ | ----------------------------------------- | ------------------------- |
| ✅ Essential | Llama, Qwen, Phi, Gemma, Mistral, GPT-OSS | Mainstream                |
| ⏸️ Defer     | Mamba, Jamba, DBRX                        | SSM/unusual architectures |
| ❌ Skip      | Batch processing, server features         | Not needed for inference  |

### 3. Minimal Viable Port

- Port core functionality, not edge cases
- Skip features that < 5% of users need
- Add extensibility points for future additions

## File Header Template

Every ported file **must** include:

```swift
// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/<filename>.py
// Git Hash: <full-40-char-hash> (<YYYY-MM-DD>)
```

## Swift Style Guide

### Naming Conventions

| Python                 | Swift                 |
| ---------------------- | --------------------- |
| `snake_case`           | `camelCase`           |
| `class KVCache`        | `class KVCache`       |
| `def update_and_fetch` | `func updateAndFetch` |
| `__init__`             | `init`                |
| `__len__`              | `var count: Int`      |
| `_private_method`      | `private func method` |

### Type Mappings

| Python        | Swift             |
| ------------- | ----------------- |
| `mx.array`    | `MLXArray`        |
| `nn.Module`   | `Module` (MLXNN)  |
| `Optional[T]` | `T?`              |
| `List[T]`     | `[T]`             |
| `Dict[K, V]`  | `[K: V]`          |
| `Tuple[A, B]` | `(A, B)`          |
| `None`        | `nil`             |
| `@property`   | computed property |

### MLX Operations

| Python                           | Swift                           |
| -------------------------------- | ------------------------------- |
| `mx.zeros(shape)`                | `MLXArray.zeros(shape)`         |
| `mx.concatenate([a, b], axis=2)` | `concatenated([a, b], axis: 2)` |
| `mx.quantize(x, ...)`            | `MLX.quantized(x, ...)`         |
| `x[..., :n, :]`                  | `x[.ellipsis, ..<n, 0...]`      |
| `x.shape[0]`                     | `x.dim(0)`                      |
| `x.dtype`                        | `x.dtype`                       |

## Code Structure

### Protocol-First Design

```swift
/// Protocol for all KV cache implementations
public protocol KVCacheProtocol: AnyObject {
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    var offset: Int { get }
    func makeMask(queryLength: Int, windowSize: Int?) -> MLXFast.ScaledDotProductAttentionMaskMode
}
```

### Class Structure

```swift
/// KV cache with grow-in-place strategy
public class StandardKVCache: KVCacheProtocol {
    // MARK: - Properties

    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0

    public static let step = 256

    // MARK: - Initialization

    public init() {}

    // MARK: - Cache Operations

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // Implementation
    }
}
```

## What NOT to Port

### From cache.py

- ❌ `BatchKVCache`, `BatchRotatingKVCache` - Server/batch processing
- ❌ `MambaCache`, `ArraysCache` - SSM models
- ❌ `ChunkedKVCache`, `CacheList` - Specialized use cases
- ❌ `save_prompt_cache`, `load_prompt_cache` - Serialization

### General

- ❌ Batch processing features
- ❌ Prompt caching to disk
- ❌ Speculative decoding caches
- ❌ Multi-modal (initially)

## Shared Components

Before porting, check if a shared component already exists in `shared/`:

| Component         | File                        | Use When                   |
| ----------------- | --------------------------- | -------------------------- |
| RMSNorm           | `RMSNorm.swift`             | Standard RMS normalization |
| GemmaRMSNorm      | `ported/GemmaRMSNorm.swift` | (1+weight) scaling         |
| StandardAttention | `StandardAttention.swift`   | Basic GQA attention        |
| StandardMLP       | `StandardMLP.swift`         | SwiGLU MLP                 |
| MathUtils         | `MathUtils.swift`           | erfinv, clipResidual, topK |

## Testing

### Test File Location

Tests go in `packages/swift/Tests/NodeMLXCoreTests/`:

```swift
import XCTest
@testable import NodeMLXCore
import MLX

final class KVCacheTests: XCTestCase {
    func testUpdateAndFetch() {
        let cache = StandardKVCache()
        let keys = MLXArray.zeros([1, 4, 8, 64])
        let values = MLXArray.zeros([1, 4, 8, 64])

        let (k, v) = cache.update(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(k.dim(2), 8)
    }
}
```

### Running Tests

```bash
cd packages/swift
swift test
```

## Workflow

1. **Download Python source**:

   ```bash
   curl -s "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/<file>.py" -o /tmp/<file>.py
   ```

2. **Analyze**: Essential vs. optional features

3. **Check shared components**: Reuse if exists

4. **Design Swift API**: Protocols, classes

5. **Implement**: Premium Swift patterns

6. **Test**: Comprehensive coverage

7. **Document**: Update PORTING_DECISIONS.md

8. **Build**:
   ```bash
   cd packages/swift && swift build -c release && swift test
   ```

## Documentation Updates

After porting, update:

1. **File header**: Git hash, date
2. **PORTING_DECISIONS.md**: What was ported, decisions made
3. **ported/README.md**: Add to ported files table
4. **Tests**: Add test file

## Quick Reference

```bash
# Get latest mlx-lm hash
curl -s "https://api.github.com/repos/ml-explore/mlx-lm/commits/main" | grep '"sha"' | head -1

# Download Python source
curl -s "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py" -o /tmp/cache.py

# Build and test
cd packages/swift && swift build -c release && swift test

# Regenerate models (to ensure compatibility)
pnpm hf2swift --model llama --output packages/swift/Sources/NodeMLXCore/generated/models/LlamaGenerated.swift
```
