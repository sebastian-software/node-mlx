//
//  Tokenizer.swift
//  NodeMLXCore
//
//  Tokenizer wrapper using HuggingFace swift-transformers.
//  swift-transformers is Apache 2.0 licensed by HuggingFace.
//  See: https://github.com/huggingface/swift-transformers
//

import Foundation
import Tokenizers
import Hub

// MARK: - Tokenizer Protocol

/// Protocol for tokenizers that can encode and decode text
public protocol TokenizerProtocol: Sendable {
    /// Encode text to token IDs
    func encode(_ text: String) -> [Int]

    /// Decode token IDs to text
    func decode(_ tokens: [Int]) -> String

    /// Get special token IDs
    var bosTokenId: Int? { get }
    var eosTokenId: Int? { get }
    var padTokenId: Int? { get }
}

// MARK: - HuggingFace Tokenizer Wrapper

/// Wrapper around HuggingFace's Tokenizer from swift-transformers
public class HFTokenizer: TokenizerProtocol, @unchecked Sendable {
    private let tokenizer: any Tokenizer

    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?

    /// Load tokenizer from a HuggingFace model directory
    public init(modelDirectory: URL) async throws {
        // Load using swift-transformers AutoTokenizer
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)

        // Extract special tokens from config
        let configURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        if let configData = try? Data(contentsOf: configURL),
           let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
            self.bosTokenId = config["bos_token_id"] as? Int
            self.eosTokenId = config["eos_token_id"] as? Int
            self.padTokenId = config["pad_token_id"] as? Int
        } else {
            self.bosTokenId = nil
            self.eosTokenId = nil
            self.padTokenId = nil
        }
    }

    /// Load tokenizer from HuggingFace Hub model ID
    public convenience init(modelId: String) async throws {
        // Use Hub to get model directory
        let hub = HubApi()
        let repo = Hub.Repo(id: modelId)

        // Download tokenizer files
        let filePatterns = ["tokenizer.json", "tokenizer_config.json", "vocab.*", "merges.txt"]
        let modelDir = try await hub.snapshot(from: repo, matching: filePatterns)

        try await self.init(modelDirectory: modelDir)
    }

    // MARK: - TokenizerProtocol

    public func encode(_ text: String) -> [Int] {
        return tokenizer.encode(text: text)
    }

    public func decode(_ tokens: [Int]) -> String {
        return tokenizer.decode(tokens: tokens)
    }
}

// MARK: - Convenience Extensions

extension HFTokenizer {
    /// Encode text with special tokens (BOS/EOS)
    public func encodeWithSpecialTokens(
        _ text: String,
        addBos: Bool = true,
        addEos: Bool = false
    ) -> [Int] {
        var tokens = encode(text)

        if addBos, let bos = bosTokenId {
            tokens.insert(bos, at: 0)
        }

        if addEos, let eos = eosTokenId {
            tokens.append(eos)
        }

        return tokens
    }

    /// Decode tokens, optionally skipping special tokens
    public func decode(_ tokens: [Int], skipSpecialTokens: Bool) -> String {
        return tokenizer.decode(tokens: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    /// Apply chat template to format a user message for the model
    /// Returns token IDs ready for model input
    public func applyChatTemplate(userMessage: String) throws -> [Int] {
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": userMessage]
        ]
        return try tokenizer.applyChatTemplate(messages: messages)
    }

    /// Apply chat template with conversation history
    public func applyChatTemplate(messages: [[String: any Sendable]]) throws -> [Int] {
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}

// MARK: - HuggingFace Hub Utilities

/// Simple HuggingFace Hub cache utilities
public struct HFHubCache {
    public static let cacheDir: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }()

    /// Get the local cache path for a HuggingFace model
    public static func modelPath(for modelId: String) -> URL {
        let sanitized = modelId.replacingOccurrences(of: "/", with: "--")
        return cacheDir
            .appendingPathComponent("models--\(sanitized)")
            .appendingPathComponent("snapshots")
    }

    /// Check if a model is cached locally
    public static func isCached(_ modelId: String) -> Bool {
        let path = modelPath(for: modelId)
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir) && isDir.boolValue
    }

    /// Get the latest snapshot directory for a cached model
    public static func latestSnapshot(for modelId: String) -> URL? {
        let snapshotsDir = modelPath(for: modelId)

        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        // Return the most recent snapshot
        return contents.sorted { a, b in
            let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return aDate > bDate
        }.first
    }
}
