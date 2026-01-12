// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for text generation utilities.
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: tests/test_generate.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import MLX
import XCTest

@testable import NodeMLXCore

final class GenerateTests: XCTestCase {
    // MARK: - Generation Config Tests

    func testDefaultConfig() {
        let config = GenerationConfig()

        XCTAssertEqual(config.maxTokens, 256)
        XCTAssertEqual(config.temperature, 0.7, accuracy: 1e-5)
        XCTAssertEqual(config.topP, 0.9, accuracy: 1e-5)
        XCTAssertEqual(config.repetitionPenalty, 1.0, accuracy: 1e-5)
        XCTAssertTrue(config.stopTokens.isEmpty)
    }

    func testCustomConfig() {
        let config = GenerationConfig(
            maxTokens: 100,
            temperature: 0.5,
            topP: 0.8,
            repetitionPenalty: 1.1,
            stopTokens: [1, 2, 3]
        )

        XCTAssertEqual(config.maxTokens, 100)
        XCTAssertEqual(config.temperature, 0.5, accuracy: 1e-5)
        XCTAssertEqual(config.topP, 0.8, accuracy: 1e-5)
        XCTAssertEqual(config.repetitionPenalty, 1.1, accuracy: 1e-5)
        XCTAssertEqual(config.stopTokens, [1, 2, 3])
    }

    // MARK: - Token Sampling Tests

    func testGreedySampling() {
        // With temperature 0, should always pick highest probability
        let logits = MLXArray([Float(1.0), 2.0, 5.0, 3.0])

        let token = sampleToken(logits: logits, temperature: 0)
        XCTAssertEqual(token, 2) // Index of 5.0 (highest)
    }

    func testGreedySamplingConsistent() {
        // Multiple calls with temp=0 should be deterministic
        let logits = MLXArray([Float(1.0), 2.0, 5.0, 3.0])

        for _ in 0 ..< 10 {
            let token = sampleToken(logits: logits, temperature: 0)
            XCTAssertEqual(token, 2)
        }
    }

    func testSamplingWithTemperature() {
        // With temperature 0, greedy decoding picks highest logit
        let logits = MLXArray([Float(0.0), 0.0, 1.0, 0.0])

        // Temperature 0 = greedy, should always pick index 2
        let greedyToken = sampleToken(logits: logits, temperature: 0)
        XCTAssertEqual(greedyToken, 2)

        // With higher temperature, sampling is more random
        // Just verify it returns a valid token index
        let sampledToken = sampleToken(logits: logits, temperature: 1.0)
        XCTAssertTrue(sampledToken >= 0 && sampledToken < 4)
    }

    func testSamplingWithTopP() {
        // Create logits where one token dominates
        let logits = log(MLXArray([Float(0.01), 0.01, 0.97, 0.01]))

        // With low topP and low temp, should pick dominant token (index 2)
        // Note: Using temperature 0 for deterministic greedy selection
        let token = sampleToken(logits: logits, temperature: 0, topP: 0.5)
        XCTAssertEqual(token, 2)
    }

    // MARK: - Streaming Generator Tests

    func testGenerationStepStructure() {
        let step = GenerationStep(tokenId: 42, isComplete: false, text: "hello")

        XCTAssertEqual(step.tokenId, 42)
        XCTAssertFalse(step.isComplete)
        XCTAssertEqual(step.text, "hello")
    }

    func testGenerationStepComplete() {
        let step = GenerationStep(tokenId: 0, isComplete: true, text: nil)

        XCTAssertEqual(step.tokenId, 0)
        XCTAssertTrue(step.isComplete)
        XCTAssertNil(step.text)
    }

    // MARK: - Edge Cases

    func testSamplingUniformLogits() {
        // All equal logits should sample uniformly
        let logits = MLXArray([Float(1.0), 1.0, 1.0, 1.0])

        // With greedy, should return first (or consistent) result
        let token = sampleToken(logits: logits, temperature: 0)
        XCTAssertTrue(token >= 0 && token < 4)
    }

    func testSamplingNegativeLogits() {
        // Test with negative logits (normal case after processing)
        let logits = MLXArray([Float(-10.0), -5.0, -1.0, -3.0])

        let token = sampleToken(logits: logits, temperature: 0)
        XCTAssertEqual(token, 2) // -1.0 is highest
    }

    func testSamplingLargeVocab() {
        // Test with larger vocabulary
        var logitsArray = Array(repeating: Float(0.0), count: 10000)
        logitsArray[5000] = 10.0
        let logits = MLXArray(logitsArray)

        let token = sampleToken(logits: logits, temperature: 0)
        XCTAssertEqual(token, 5000)
    }
}
