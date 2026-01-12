// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Integration tests for LLMEngine with real models.
// These tests download models from HuggingFace Hub if not cached.

@testable import NodeMLXCore
import XCTest

final class IntegrationTests: XCTestCase {
    // MARK: - Test Models

    /// Models to test - quantized models for CI
    static let testModels: [(id: String, architecture: ModelArchitecture)] = [
        ("mlx-community/Qwen3-4B-4bit", .qwen3),
        ("mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit", .llama),
        // Add more models as needed
    ]

    // Use Qwen3 4B as default test model
    let defaultTestModelId = "mlx-community/Qwen3-4B-4bit"

    // MARK: - Architecture Detection Tests

    func testModelArchitectureDetection() {
        // Test architecture detection from model_type
        XCTAssertEqual(ModelArchitecture(modelType: "llama"), .llama)
        XCTAssertEqual(ModelArchitecture(modelType: "phi3"), .phi3)
        XCTAssertEqual(ModelArchitecture(modelType: "gemma3n"), .gemma3n)
        XCTAssertEqual(ModelArchitecture(modelType: "qwen3"), .qwen3)

        // Case insensitive
        XCTAssertEqual(ModelArchitecture(modelType: "LLAMA"), .llama)
        XCTAssertEqual(ModelArchitecture(modelType: "Phi3"), .phi3)

        // Handle variations - all qwen variants map to qwen3
        XCTAssertEqual(ModelArchitecture(modelType: "qwen"), .qwen3)
        XCTAssertEqual(ModelArchitecture(modelType: "qwen2"), .qwen3)
        XCTAssertEqual(ModelArchitecture(modelType: "qwen2.5"), .qwen3)
        XCTAssertEqual(ModelArchitecture(modelType: "llama3.2"), .llama)

        // Unknown returns nil
        XCTAssertNil(ModelArchitecture(modelType: "unknown_model"))
    }

    func testAllSupportedArchitectures() {
        // Verify all supported architectures can be created
        let supportedTypes = [
            "llama", "qwen3", "phi3",
            "gemma3", "gemma3n", "mistral", "mistral3",
            "smollm3", "gpt_oss",
        ]

        for modelType in supportedTypes {
            XCTAssertNotNil(
                ModelArchitecture(modelType: modelType),
                "Should support \(modelType)"
            )
        }
    }

    // MARK: - Engine Tests

    func testLLMEngineInitialization() {
        let engine = LLMEngine()
        XCTAssertNotNil(engine)
        XCTAssertFalse(engine.isLoaded)
        XCTAssertFalse(engine.isVLM)
    }

    func testGenerationWithoutModel() {
        let engine = LLMEngine()

        // Should throw when no model is loaded
        XCTAssertThrowsError(try engine.generate(prompt: "test")) { error in
            XCTAssertTrue(error is LLMEngineError)
            if case LLMEngineError.modelNotLoaded = error {
                // Expected
            } else {
                XCTFail("Expected modelNotLoaded error")
            }
        }
    }

    // MARK: - Integration Tests (require Metal GPU)

    /// Helper to skip tests that require Metal GPU
    func skipIfNoMetal() throws {
        // Check if we're in an environment where Metal tests should be skipped
        if ProcessInfo.processInfo.environment["SKIP_METAL_TESTS"] != nil {
            throw XCTSkip("Skipping Metal-dependent test (SKIP_METAL_TESTS set)")
        }
    }

    func testModelLoading() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        try await engine.loadModel(modelId: defaultTestModelId)

        XCTAssertTrue(engine.isLoaded)

        engine.unload()
        XCTAssertFalse(engine.isLoaded)
    }

    func testBasicGeneration() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        try await engine.loadModel(modelId: defaultTestModelId)

        let config = GenerationConfig(
            maxTokens: 20,
            temperature: 0.7
        )
        let result = try engine.generate(prompt: "What is 2+2?", config: config)

        XCTAssertFalse(result.isEmpty)
        print("Generated: \(result)")

        engine.unload()
    }

    func testStreamingGeneration() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        try await engine.loadModel(modelId: defaultTestModelId)

        var streamedTokens: [String] = []

        let result = try engine.generateStream(
            prompt: "Count from 1 to 5:",
            maxTokens: 30,
            temperature: 0.3,
            topP: 0.9
        ) { token in
            streamedTokens.append(token)
            return true // Continue
        }

        XCTAssertGreaterThan(streamedTokens.count, 0)
        XCTAssertGreaterThan(result.tokenCount, 0)
        XCTAssertGreaterThan(result.tokensPerSecond, 0)
        XCTAssertEqual(streamedTokens.joined(), result.text)

        print("Generated \(result.tokenCount) tokens at \(result.tokensPerSecond) tok/s")
        print("Text: \(result.text)")

        engine.unload()
    }

    func testMultipleGenerations() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        try await engine.loadModel(modelId: defaultTestModelId)

        let config = GenerationConfig(maxTokens: 10, temperature: 0.5)

        // Generate multiple times without reloading
        for i in 1 ... 3 {
            let result = try engine.generate(prompt: "Say '\(i)':", config: config)
            XCTAssertFalse(result.isEmpty, "Generation \(i) should produce output")
        }

        engine.unload()
    }

    func testEarlyStopGeneration() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        try await engine.loadModel(modelId: defaultTestModelId)

        var tokenCount = 0
        let maxTokensBeforeStop = 5

        _ = try engine.generateStream(
            prompt: "Write a long story:",
            maxTokens: 100,
            temperature: 0.7,
            topP: 0.9
        ) { _ in
            tokenCount += 1
            return tokenCount < maxTokensBeforeStop // Stop after N tokens
        }

        XCTAssertEqual(tokenCount, maxTokensBeforeStop)

        engine.unload()
    }

    // MARK: - Multi-Model Tests

    func testMultipleModels() async throws {
        try skipIfNoMetal()

        let engine = LLMEngine()

        // Test loading and unloading multiple models
        for (modelId, expectedArch) in testModels.prefix(2) {
            print("Testing \(modelId)...")

            try await engine.loadModel(modelId: modelId)
            XCTAssertTrue(engine.isLoaded)

            let config = GenerationConfig(maxTokens: 10, temperature: 0.5)
            let result = try engine.generate(prompt: "Hello", config: config)
            XCTAssertFalse(result.isEmpty, "\(expectedArch) should generate output")

            engine.unload()
            XCTAssertFalse(engine.isLoaded)
        }
    }
}

// MARK: - Generation Result Helper

extension IntegrationTests {
    struct BenchmarkResult {
        let modelId: String
        let tokensPerSecond: Float
        let timeToFirstToken: Double
    }
}
