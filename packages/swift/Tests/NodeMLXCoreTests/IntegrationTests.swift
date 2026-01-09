//
//  IntegrationTests.swift
//  NodeMLXCoreTests
//
//  Integration tests for the LLMEngine.
//

@testable import NodeMLXCore
import XCTest

final class IntegrationTests: XCTestCase {
    // Use a small, fast model for testing
    let testModelId = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    func testModelArchitectureDetection() {
        // Test architecture detection from model_type
        XCTAssertEqual(ModelArchitecture.from(modelType: "llama"), .llama)
        XCTAssertEqual(ModelArchitecture.from(modelType: "phi3"), .phi3)
        XCTAssertEqual(ModelArchitecture.from(modelType: "gemma3n"), .gemma3n)
        XCTAssertEqual(ModelArchitecture.from(modelType: "qwen2"), .qwen2)

        // Case insensitive
        XCTAssertEqual(ModelArchitecture.from(modelType: "LLAMA"), .llama)
        XCTAssertEqual(ModelArchitecture.from(modelType: "Phi3"), .phi3)

        // Unknown returns nil
        XCTAssertNil(ModelArchitecture.from(modelType: "unknown_model"))
    }

    func testLLMEngineInitialization() {
        let engine = LLMEngine()
        XCTAssertNotNil(engine)
    }

    func testModelLoading() async throws {
        let engine = LLMEngine()

        // This test requires network and downloads a model
        // Skip in CI if needed
        guard ProcessInfo.processInfo.environment["SKIP_INTEGRATION_TESTS"] == nil else {
            throw XCTSkip("Skipping integration test (SKIP_INTEGRATION_TESTS set)")
        }

        // Note: Metal shaders won't work in SPM tests (xcodebuild required)
        // This test will fail with "Failed to load the default metallib"
        // The code is correct - it's a test environment limitation

        var progressUpdates: [Float] = []

        try await engine.loadModel(modelId: testModelId) { progress in
            progressUpdates.append(progress)
            print("Loading: \(Int(progress * 100))%")
        }

        // Verify progress was reported
        XCTAssertGreaterThan(progressUpdates.count, 0)

        // Clean up
        engine.unload()
    }

    func testGeneration() async throws {
        guard ProcessInfo.processInfo.environment["SKIP_INTEGRATION_TESTS"] == nil else {
            throw XCTSkip("Skipping integration test (SKIP_INTEGRATION_TESTS set)")
        }

        let engine = LLMEngine()

        print("Loading model...")
        try await engine.loadModel(modelId: testModelId)

        print("Generating...")
        let result = try engine.generate(
            prompt: "What is 2+2?",
            maxTokens: 50,
            temperature: 0.7
        )

        print("Generated: \(result.text)")
        print("Tokens: \(result.tokenCount)")
        print("Speed: \(result.tokensPerSecond) tok/s")

        XCTAssertGreaterThan(result.tokenCount, 0)
        XCTAssertFalse(result.text.isEmpty)
        XCTAssertGreaterThan(result.tokensPerSecond, 0)

        engine.unload()
    }

    func testStreamingGeneration() async throws {
        guard ProcessInfo.processInfo.environment["SKIP_INTEGRATION_TESTS"] == nil else {
            throw XCTSkip("Skipping integration test (SKIP_INTEGRATION_TESTS set)")
        }

        let engine = LLMEngine()

        print("Loading model...")
        try await engine.loadModel(modelId: testModelId)

        print("Streaming generation...")
        var streamedTokens: [String] = []

        let result = try engine.generateStream(
            prompt: "Count from 1 to 5:",
            maxTokens: 30,
            temperature: 0.3
        ) { token in
            streamedTokens.append(token)
            print(token, terminator: "")
            return true // Continue
        }

        print("\n\nTotal tokens: \(result.tokenCount)")
        print("Streamed \(streamedTokens.count) token strings")

        XCTAssertGreaterThan(streamedTokens.count, 0)
        XCTAssertEqual(streamedTokens.joined(), result.text)

        engine.unload()
    }

    func testGenerationWithoutModel() {
        let engine = LLMEngine()

        // Should throw when no model is loaded
        XCTAssertThrowsError(try engine.generate(prompt: "test")) { error in
            XCTAssertTrue(error is LLMEngineError)
        }
    }
}
