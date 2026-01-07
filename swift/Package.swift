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
    .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "2.29.0")
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
