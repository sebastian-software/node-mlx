//
//  NodeMLXCore.swift
//  NodeMLXCore
//
//  Pure MLX implementation for LLM inference without mlx-swift-lm dependency.
//
//  This module provides:
//  - Generate: Token generation with temperature/topP sampling
//  - ModelLoader: Load models from HuggingFace Hub cache
//  - KVCache: Key-value caching for efficient autoregressive generation
//
//  To use, set `usePureMLX = true` in Package.swift
//

import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXFast

// Re-export all public APIs
@_exported import struct Foundation.URL

// MARK: - Version

public let nodeMLXCoreVersion = "0.1.0"

// MARK: - Module Info

/// Information about the NodeMLXCore module
public struct NodeMLXCoreInfo {
    public static let version = nodeMLXCoreVersion
    public static let mlxVersion = "0.21.0"  // Minimum MLX version
    public static let description = "Pure MLX implementation for LLM inference"
    
    public static func printInfo() {
        print("""
        NodeMLXCore v\(version)
        - MLX Swift: \(mlxVersion)+
        - Description: \(description)
        - Features: Generate, ModelLoader, KVCache
        """)
    }
}

// MARK: - Convenience Functions

/// Check if MLX is available on this system
public func isMLXAvailable() -> Bool {
    // MLX requires macOS 14+ and Apple Silicon
    #if os(macOS)
    if #available(macOS 14.0, *) {
        return true
    }
    #endif
    return false
}

/// Get default MLX device
public func getDefaultDevice() -> String {
    return "gpu"  // MLX uses GPU by default on Apple Silicon
}

