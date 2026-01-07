import XCTest
@testable import NodeMLXCore

final class TokenizerTests: XCTestCase {
    
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
}

