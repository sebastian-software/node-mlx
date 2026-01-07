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
        // Direct MLX dependencies (no mlx-swift-lm!)
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6")
    ],
    targets: [
        // Main binding library - uses NodeMLXCore
        .target(
            name: "NodeMLX",
            dependencies: [
                "NodeMLXCore"
            ],
            path: "Sources/NodeMLX"
        ),
        // Core LLM functionality (replaces mlx-swift-lm)
        .target(
            name: "NodeMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
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
// node-mlx: Independent MLX bindings for Node.js
// ═══════════════════════════════════════════════════════════════════════════════
//
// This package NO LONGER depends on mlx-swift-lm!
//
// Model support is now powered by:
// - hf2swift generator: Converts HuggingFace models to MLX Swift
// - NodeMLXCore: Custom LLM infrastructure (ModelLoader, Generate, KVCache)
//
// Supported models (auto-generated):
// - Phi3
// - Llama
// - Qwen2
// - GPT-OSS (MoE)
// - Gemma3n (VLM)
//
// To add new models:
//   python tools/hf2swift/generator_v5.py --model <name> --config <hf-model-id>
//
// ═══════════════════════════════════════════════════════════════════════════════
