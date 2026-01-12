// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for SamplingUtils.
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: tests/test_sample_utils.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import MLX
import XCTest

@testable import NodeMLXCore

final class SamplingUtilsTests: XCTestCase {
    // MARK: - Top-P Tests

    func testApplyTopPHighConfidence() {
        // When top token has 0.9 probability and threshold is 0.3,
        // only the top token should remain
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopP(logits, p: 0.3)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        XCTAssertEqual(actualProbs[0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[2].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.0, accuracy: 1e-5)
    }

    func testApplyTopPHighThreshold() {
        // When threshold is 0.95, all tokens should remain
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopP(logits, p: 0.95)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        XCTAssertEqual(actualProbs[0].item(Float.self), 0.9, accuracy: 1e-4)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.1, accuracy: 1e-4)
    }

    func testApplyTopPMultipleTokens() {
        let probs = MLXArray([Float(0.0), 0.5, 0.4, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        // With p=0.4, only the top token should remain
        var newLogits = SamplingUtils.applyTopP(logits, p: 0.4)
        var actualProbs = softmax(newLogits, axis: -1).squeezed()
        XCTAssertEqual(actualProbs[0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[2].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.0, accuracy: 1e-5)

        // With p=0.6, top two tokens should remain
        newLogits = SamplingUtils.applyTopP(logits, p: 0.6)
        actualProbs = softmax(newLogits, axis: -1).squeezed()
        XCTAssertEqual(actualProbs[0].item(Float.self), 0.0, accuracy: 1e-4)
        XCTAssertEqual(actualProbs[1].item(Float.self), 0.5556, accuracy: 1e-3)
        XCTAssertEqual(actualProbs[2].item(Float.self), 0.4444, accuracy: 1e-3)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.0, accuracy: 1e-4)
    }

    func testApplyTopPBatchMode() {
        // Create 2x4 batch: [[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.1, 0.1]]
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1, 0.0, 0.8, 0.1, 0.1]).reshaped([2, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopP(logits, p: 0.5)
        let actualProbs = softmax(newLogits, axis: -1)

        // First batch: only first token
        XCTAssertEqual(actualProbs[0, 0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[0, 1].item(Float.self), 0.0, accuracy: 1e-5)

        // Second batch: only second token
        XCTAssertEqual(actualProbs[1, 0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1, 1].item(Float.self), 1.0, accuracy: 1e-5)
    }

    // MARK: - Top-K Tests

    func testApplyTopKSingle() {
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopK(logits, k: 1)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        XCTAssertEqual(actualProbs[0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[2].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.0, accuracy: 1e-5)
    }

    func testApplyTopKTwo() {
        let probs = MLXArray([Float(0.6), 0.0, 0.1, 0.3]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopK(logits, k: 2)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        // Renormalized: 0.6/(0.6+0.3) = 0.6667, 0.3/(0.6+0.3) = 0.3333
        XCTAssertEqual(actualProbs[0].item(Float.self), 0.6667, accuracy: 1e-3)
        XCTAssertEqual(actualProbs[1].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[2].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.3333, accuracy: 1e-3)
    }

    func testApplyTopKBatchMode() {
        // Create 2x4 batch: [[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]]
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.1]).reshaped([2, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyTopK(logits, k: 1)
        let actualProbs = softmax(newLogits, axis: -1)

        // First batch: only first token
        XCTAssertEqual(actualProbs[0, 0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[0, 3].item(Float.self), 0.0, accuracy: 1e-5)

        // Second batch: only second token
        XCTAssertEqual(actualProbs[1, 0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1, 1].item(Float.self), 1.0, accuracy: 1e-5)
    }

    // MARK: - Min-P Tests

    func testApplyMinPHighThreshold() {
        // With minP=0.8, only tokens with prob >= 0.8 * maxProb remain
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyMinP(logits, minP: 0.8)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        // Only first token (0.9) passes: 0.1 < 0.8 * 0.9 = 0.72
        XCTAssertEqual(actualProbs[0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.0, accuracy: 1e-5)
    }

    func testApplyMinPLowThreshold() {
        // With minP=0.05, tokens with prob >= 0.05 * maxProb remain
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        let newLogits = SamplingUtils.applyMinP(logits, minP: 0.05)
        let actualProbs = softmax(newLogits, axis: -1).squeezed()

        // Both first and last pass: 0.1 >= 0.05 * 0.9 = 0.045
        XCTAssertEqual(actualProbs[0].item(Float.self), 0.9, accuracy: 1e-4)
        XCTAssertEqual(actualProbs[3].item(Float.self), 0.1, accuracy: 1e-4)
    }

    func testApplyMinPBatchMode() {
        // Create 2x4 batch: [[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]]
        let probs = MLXArray([Float(0.9), 0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.1]).reshaped([2, 4])
        let logits = log(probs)

        // With minP=0.7, threshold is 0.7 * maxProb
        let newLogits = SamplingUtils.applyMinP(logits, minP: 0.7)
        let actualProbs = softmax(newLogits, axis: -1)

        // First batch: threshold = 0.7 * 0.9 = 0.63, only first passes
        XCTAssertEqual(actualProbs[0, 0].item(Float.self), 1.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[0, 3].item(Float.self), 0.0, accuracy: 1e-5)

        // Second batch: threshold = 0.7 * 0.8 = 0.56, only second passes
        XCTAssertEqual(actualProbs[1, 0].item(Float.self), 0.0, accuracy: 1e-5)
        XCTAssertEqual(actualProbs[1, 1].item(Float.self), 1.0, accuracy: 1e-5)
    }

    // MARK: - Combined Sampling Tests

    func testSampleTokenGreedy() {
        let probs = MLXArray([Float(0.1), 0.2, 0.5, 0.2]).reshaped([1, 4])
        let logits = log(probs)

        // With temperature 0, should always pick highest probability
        let token = SamplingUtils.sampleToken(logits: logits, temperature: 0)
        XCTAssertEqual(token, 2) // Index of 0.5
    }

    func testSampleTokenWithTopK() {
        let probs = MLXArray([Float(0.1), 0.2, 0.5, 0.2]).reshaped([1, 4])
        let logits = log(probs)

        // With topK=1 and temp=0, should pick highest
        let token = SamplingUtils.sampleToken(logits: logits, temperature: 0, topK: 1)
        XCTAssertEqual(token, 2)
    }

    func testSampleTokenWithTopP() {
        let probs = MLXArray([Float(0.1), 0.2, 0.6, 0.1]).reshaped([1, 4])
        let logits = log(probs)

        // With topP=0.5 and temp=0, should pick highest (only token in nucleus)
        let token = SamplingUtils.sampleToken(logits: logits, temperature: 0, topP: 0.5)
        XCTAssertEqual(token, 2)
    }
}
