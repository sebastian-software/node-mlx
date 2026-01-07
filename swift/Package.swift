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
    .package(url: "https://github.com/swernerx/mlx-swift-lm.git", branch: "fix/gemma3n-intermediate-size-array")
  ],
  targets: [
    .target(
      name: "NodeMLX",
      dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm")
      ],
      path: "Sources/NodeMLX"
    )
  ]
)
