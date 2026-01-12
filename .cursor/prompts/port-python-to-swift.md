# Port Python mlx-lm to Swift

You are porting Python code from Apple's `mlx-lm` library to Swift for the `node-mlx` project.

## Source Repository

- Python (Primary): https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models
- Swift (Reference only): https://github.com/ml-explore/mlx-swift-lm

**Important**: Always record the exact git hash when porting. Get it with:

```bash
curl -s "https://api.github.com/repos/ml-explore/mlx-lm/commits/main" | grep '"sha"' | head -1
```

## Core Principles

### 1. Clean Cut Philosophy

- Start fresh, don't patch existing code
- Port with understanding, not blind translation
- Premium architect-level Swift: idiomatic, elegant, maintainable

### 2. Focus on Popular Models

Only port what's needed for mainstream models:

- ✅ **Essential**: Llama, Qwen, Phi, Gemma, Mistral, GPT-OSS
- ⏸️ **Defer**: Mamba, Jamba (SSM), DBRX, unusual architectures
- ❌ **Skip**: Batch processing, server-specific features

### 3. Minimal Viable Port

- Port core functionality, not edge cases
- Skip features that < 5% of users need
- Add extensibility points for future additions

## File Structure

- Place Swift files in `packages/swift/Sources/NodeMLXCore/ported/`
- Tests in `packages/swift/Tests/NodeMLXCoreTests/`
- Use `// MARK: -` comments for logical sections

### File Header Template

Every ported file must include the source git hash:

```swift
// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/<filename>.py
// Git Hash: <full-40-char-hash> (<YYYY-MM-DD>)
```

## Swift Style Guide

### Naming

| Python                 | Swift                            |
| ---------------------- | -------------------------------- |
| `snake_case`           | `camelCase`                      |
| `class KVCache`        | `class KVCache`                  |
| `def update_and_fetch` | `func updateAndFetch`            |
| `__init__`             | `init`                           |
| `__len__`              | `var count: Int` (Sequence-like) |
| `_private_method`      | `private func method`            |

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

### Protocol Design

```swift
/// Protocol for all KV cache implementations
public protocol KVCacheProtocol: AnyObject {
    /// Update cache with new keys/values and return full sequence
    func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Number of cached tokens
    var offset: Int { get }

    /// Create attention mask for current cache state
    func makeMask(queryLength: Int, windowSize: Int?) -> MLXArray?
}
```

### Class Structure

```swift
/// KV cache with grow-in-place strategy for efficient memory use
///
/// Ported from mlx-lm/mlx_lm/models/cache.py
public class KVCache: KVCacheProtocol {
    // MARK: - Properties

    private var keys: MLXArray?
    private var values: MLXArray?
    public private(set) var offset: Int = 0

    /// Growth step size for buffer allocation
    public static let step = 256

    // MARK: - Initialization

    public init() {}

    // MARK: - Cache Operations

    public func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // ... implementation
    }
}
```

## What NOT to Port

### From cache.py

- ❌ `ConcatenateKVCache` - Simple concatenation, rarely used
- ❌ `ArraysCache` - Generic container
- ❌ `MambaCache` - SSM models only
- ❌ `ChunkedKVCache` - Chunked attention
- ❌ `CacheList` - Container for mixed caches
- ❌ `BatchKVCache` - Batch processing
- ❌ `BatchRotatingKVCache` - Batch processing
- ❌ `save_prompt_cache` / `load_prompt_cache` - Serialization (add later if needed)
- ❌ `dynamic_roll` - Batch-specific helper

### From all files

- ❌ Batch processing features
- ❌ Prompt caching to disk
- ❌ Multi-modal extensions (initially)
- ❌ Speculative decoding caches

## Testing

### Co-located Test Structure

```swift
// File: KVCacheTests.swift (same directory as KVCache.swift)
import XCTest
@testable import NodeMLXCore
import MLX

final class KVCacheTests: XCTestCase {
    func testUpdateAndFetch() {
        let cache = KVCache()
        let keys = MLXArray.zeros([1, 4, 8, 64])
        let values = MLXArray.zeros([1, 4, 8, 64])

        let (k, v) = cache.updateAndFetch(keys: keys, values: values)

        XCTAssertEqual(cache.offset, 8)
        XCTAssertEqual(k.dim(2), 8)
    }

    func testGrowthBehavior() {
        // Test that cache grows in steps
    }

    func testTrim() {
        // Test cache trimming
    }
}
```

## Documentation

### Header Template

```swift
// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/cache.py
// SPDX-License-Identifier: MIT
```

### Public API Documentation

```swift
/// Updates the cache with new key/value pairs and returns the full sequence.
///
/// This method uses a grow-in-place strategy: the internal buffer grows
/// in steps of `Self.step` (256) to avoid frequent reallocations.
///
/// - Parameters:
///   - keys: New keys to add, shape [B, H, S, D]
///   - values: New values to add, shape [B, H, S, D]
/// - Returns: Tuple of (allKeys, allValues) including new and cached entries
public func updateAndFetch(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
```

## Workflow

1. Download latest Python source
2. Analyze: What's essential vs. what's optional?
3. Design Swift API (protocols, classes)
4. Implement with Premium Swift patterns
5. Add comprehensive tests
6. Run `swift build -c release` and `swift test`
7. Document decisions in code comments

## Quick Reference Commands

```bash
# Download Python sources
curl -s "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/cache.py" -o /tmp/cache.py

# Build and test
cd packages/swift && swift build -c release && swift test

# Regenerate models (to ensure compatibility)
pnpm hf2swift --model llama --output packages/swift/Sources/NodeMLXCore/Models/LlamaGenerated.swift
```
