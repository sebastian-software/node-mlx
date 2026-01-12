// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tokenizer wrapper for HuggingFace tokenizers via swift-transformers.

import Foundation
import Hub
import MLX
import Tokenizers

// MARK: - Tokenizer Protocol

/// Protocol for text tokenization.
public protocol TokenizerProtocol {
    /// Encodes text into token IDs.
    func encode(text: String) -> [Int]

    /// Decodes token IDs back to text.
    func decode(tokens: [Int]) -> String

    /// The vocabulary size.
    var vocabularySize: Int { get }

    /// End of sequence token ID.
    var eosTokenId: Int? { get }

    /// Beginning of sequence token ID.
    var bosTokenId: Int? { get }
}

// MARK: - HuggingFace Tokenizer

/// Tokenizer loaded from HuggingFace Hub.
public class HFTokenizer: TokenizerProtocol {
    private let tokenizer: Tokenizer
    private let config: TokenizerConfig?

    public let vocabularySize: Int
    public let eosTokenId: Int?
    public let bosTokenId: Int?

    /// Loads a tokenizer from a local directory (async version).
    ///
    /// - Parameter path: Path to directory containing tokenizer.json
    /// - Throws: Error if tokenizer files cannot be loaded
    public init(path: String) async throws {
        let url = URL(fileURLWithPath: path)
        tokenizer = try await AutoTokenizer.from(modelFolder: url)

        // Try to load tokenizer_config.json for special tokens
        let configPath = url.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: configPath) {
            config = try? JSONDecoder().decode(TokenizerConfig.self, from: data)
        } else {
            config = nil
        }

        // Get vocabulary size - use a reasonable default if not available
        vocabularySize = 128_000 // Common default for modern models

        // Extract special token IDs
        eosTokenId = config?.eosTokenId ?? tokenizer.eosTokenId
        bosTokenId = config?.bosTokenId ?? tokenizer.bosTokenId
    }

    public func encode(text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}

// MARK: - Tokenizer Config

/// Configuration for tokenizer special tokens.
private struct TokenizerConfig: Decodable {
    let eosTokenId: Int?
    let bosTokenId: Int?
    let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case padTokenId = "pad_token_id"
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Handle both single int and array format
        if let id = try? container.decode(Int.self, forKey: .eosTokenId) {
            eosTokenId = id
        } else if let ids = try? container.decode([Int].self, forKey: .eosTokenId), let first = ids.first {
            eosTokenId = first
        } else {
            eosTokenId = nil
        }

        if let id = try? container.decode(Int.self, forKey: .bosTokenId) {
            bosTokenId = id
        } else if let ids = try? container.decode([Int].self, forKey: .bosTokenId), let first = ids.first {
            bosTokenId = first
        } else {
            bosTokenId = nil
        }

        if let id = try? container.decode(Int.self, forKey: .padTokenId) {
            padTokenId = id
        } else if let ids = try? container.decode([Int].self, forKey: .padTokenId), let first = ids.first {
            padTokenId = first
        } else {
            padTokenId = nil
        }
    }
}

// MARK: - Chat Template

/// Applies chat template to messages.
public func applyChatTemplate(
    messages: [[String: String]],
    addGenerationPrompt: Bool = true
) -> String {
    // Default template for models without explicit chat template
    var result = ""

    for message in messages {
        guard let role = message["role"], let content = message["content"] else {
            continue
        }

        switch role {
        case "system":
            result += "<|system|>\n\(content)\n"
        case "user":
            result += "<|user|>\n\(content)\n"
        case "assistant":
            result += "<|assistant|>\n\(content)\n"
        default:
            result += "\(content)\n"
        }
    }

    if addGenerationPrompt {
        result += "<|assistant|>\n"
    }

    return result
}
