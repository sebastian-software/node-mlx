// swift-tools-version: 5.9

import PackageDescription

let package = Package(
  name: "llm-cli",
  platforms: [
    .macOS(.v14)
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.29.0"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0")
  ],
  targets: [
    .executableTarget(
      name: "llm-cli",
      dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "ArgumentParser", package: "swift-argument-parser")
      ],
      path: "Sources/llm-cli"
    )
  ]
)

