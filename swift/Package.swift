// swift-tools-version: 5.9

import PackageDescription

let package = Package(
  name: "NodeMLX",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .library(
      name: "NodeMLX",
      type: .dynamic,
      targets: ["NodeMLX"]
    )
  ],
  dependencies: [
    // Using fork with Gemma 3n fix until PR #46 is merged
    // https://github.com/ml-explore/mlx-swift-lm/pull/46
    .package(url: "https://github.com/swernerx/mlx-swift-lm.git", branch: "fix/gemma3n-intermediate-size-array"),

    // Direct dependencies for mlx-swift-lm replacement
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6")
  ],
  targets: [
    .target(
      name: "NodeMLX",
      dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm")
      ],
      path: "Sources/NodeMLX"
    ),
    // Standalone core without mlx-swift-lm (experimental)
    .target(
      name: "NodeMLXCore",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "Hub", package: "swift-transformers")
      ],
      path: "Sources/NodeMLXCore"
    ),
    // Tests for NodeMLXCore
    .testTarget(
      name: "NodeMLXCoreTests",
      dependencies: [
        "NodeMLXCore",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "Hub", package: "swift-transformers")
      ]
    )
  ]
)

// ═══════════════════════════════════════════════════════════════════════════════
// ROADMAP: Pure MLX without mlx-swift-lm
// ═══════════════════════════════════════════════════════════════════════════════
//
// The hf2swift generator can produce ~95% complete Swift model code from
// HuggingFace Transformers. To remove the mlx-swift-lm dependency:
//
// 1. Change dependency to:
//    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
//
// 2. Generate model files using:
//    python tools/hf2swift/generator_v4.py --model <name> --config <hf-id>
//
// 3. Add generated files to Sources/NodeMLX/Models/
//
// 4. Implement minimal infrastructure (~350 lines):
//    - Generate.swift: Token generation with sampling
//    - ModelLoader.swift: Load weights from SafeTensors
//    - KVCache.swift: Key-value caching
//
// See: tools/hf2swift/ and swift/Sources/NodeMLXCore/ for prototypes
// ═══════════════════════════════════════════════════════════════════════════════
