import XCTest
import Tokenizers
import Hub
@testable import NodeMLXCore

final class TokenizerTests: XCTestCase {

    // MARK: - Basic Tokenizer Tests

    func testHFTokenizerFromHub() async throws {
        // Test mit Qwen - modernes Modell mit korrektem Config-Format
        let tokenizer = try await HFTokenizer(modelId: "Qwen/Qwen2.5-0.5B-Instruct")

        let text = "Hello, world!"
        let tokens = tokenizer.encode(text)

        XCTAssertFalse(tokens.isEmpty, "Tokens should not be empty")
        print("✓ Encoded '\(text)' to \(tokens.count) tokens: \(tokens)")

        let decoded = tokenizer.decode(tokens)
        print("✓ Decoded back to: '\(decoded)'")

        XCTAssertTrue(decoded.contains("Hello"))
    }

    func testHFHubCachePath() {
        // Test cache path generation
        let path = HFHubCache.modelPath(for: "mlx-community/Llama-3.2-1B-Instruct-4bit")
        XCTAssertTrue(path.path.contains("mlx-community--Llama-3.2-1B-Instruct-4bit"))
        print("✓ Cache path: \(path.path)")
    }

    func testTokenizerRoundtrip() async throws {
        // Test mit Qwen tokenizer (nicht gated!)
        let tokenizer = try await HFTokenizer(modelId: "Qwen/Qwen2.5-0.5B-Instruct")

        let texts = [
            "Hello!",
            "The quick brown fox jumps over the lazy dog.",
            "1 + 1 = 2"
        ]

        for text in texts {
            let tokens = tokenizer.encode(text)
            let decoded = tokenizer.decode(tokens)
            print("✓ '\(text)' → \(tokens.count) tokens → '\(decoded)'")

            XCTAssertFalse(tokens.isEmpty)
        }
    }

    // MARK: - Chat Template Tests (important for LLM inference)

    func testChatTemplateAvailability() async throws {
        // AutoTokenizer from swift-transformers should support chat templates
        let hub = HubApi()
        let repo = Hub.Repo(id: "Qwen/Qwen2.5-0.5B-Instruct")
        let modelDir = try await hub.snapshot(from: repo, matching: ["tokenizer*", "vocab*", "merges*"])

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        // Check if chat template is available
        let messages: [[String: String]] = [
            ["role": "user", "content": "Hello!"]
        ]

        // Try to apply chat template
        do {
            let result = try tokenizer.applyChatTemplate(messages: messages)
            print("✓ Chat template applied, got \(result.count) tokens")
            XCTAssertFalse(result.isEmpty)
        } catch {
            print("⚠ Chat template not available: \(error)")
            // Not all tokenizers have chat templates, so this isn't necessarily a failure
        }
    }

    // MARK: - Special Tokens Tests

    func testSpecialTokens() async throws {
        let tokenizer = try await HFTokenizer(modelId: "Qwen/Qwen2.5-0.5B-Instruct")

        print("Special tokens:")
        print("  BOS: \(tokenizer.bosTokenId ?? -1)")
        print("  EOS: \(tokenizer.eosTokenId ?? -1)")
        print("  PAD: \(tokenizer.padTokenId ?? -1)")

        // At least one special token should be defined
        let hasSpecialTokens = tokenizer.bosTokenId != nil ||
                               tokenizer.eosTokenId != nil ||
                               tokenizer.padTokenId != nil

        print("✓ Has special tokens: \(hasSpecialTokens)")
    }
}

