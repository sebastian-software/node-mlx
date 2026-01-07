//
//  NodeMLXCore.swift
//  NodeMLXCore
//
//  Standalone utilities for LLM inference.
//  Phase 1: Tokenizer support via swift-transformers
//  Phase 2: Full MLX integration (requires mlx-swift dependency)
//

import Foundation
import Tokenizers

// Re-export Tokenizers
@_exported import Tokenizers

// MARK: - Version

public let nodeMLXCoreVersion = "0.1.0"

// MARK: - Module Info

public struct NodeMLXCoreInfo {
    public static let version = nodeMLXCoreVersion
    public static let description = "Standalone LLM utilities for node-mlx"
    
    public static func printInfo() {
        print("""
        NodeMLXCore v\(version)
        - Features: Tokenizer (via swift-transformers)
        - Status: Phase 1 (Tokenizer only)
        """)
    }
}

// MARK: - Platform Check

/// Check if the current platform supports MLX
public func isMLXSupported() -> Bool {
    #if os(macOS) && arch(arm64)
    if #available(macOS 14.0, *) {
        return true
    }
    #endif
    return false
}

