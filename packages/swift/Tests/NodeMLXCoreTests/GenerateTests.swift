//
//  GenerateTests.swift
//  NodeMLXCoreTests
//
//  Tests for sampling strategies in Generate.swift
//

import MLX
@testable import NodeMLXCore
import XCTest

class GenerateTests: XCTestCase {
    // MARK: - GenerateParameters Tests

    func testGenerateParametersDefaults() {
        let params = GenerateParameters()

        XCTAssertEqual(params.maxTokens, 256)
        XCTAssertEqual(params.temperature, 0.7)
        XCTAssertEqual(params.topP, 0.9)
        XCTAssertNil(params.repetitionPenalty)
        XCTAssertEqual(params.repetitionContextSize, 20)
    }

    func testGenerateParametersCustom() {
        let params = GenerateParameters(
            maxTokens: 100,
            temperature: 0.5,
            topP: 0.95,
            repetitionPenalty: 1.2,
            repetitionContextSize: 50
        )

        XCTAssertEqual(params.maxTokens, 100)
        XCTAssertEqual(params.temperature, 0.5)
        XCTAssertEqual(params.topP, 0.95)
        XCTAssertEqual(params.repetitionPenalty, 1.2)
        XCTAssertEqual(params.repetitionContextSize, 50)
    }

    // MARK: - Argmax Sampling Tests

    func testSampleArgmax() {
        // Create logits where position 5 has the highest value
        var logits = MLXArray.zeros([10])
        logits[5] = MLXArray(Float32(100.0))
        eval(logits)

        let token = sampleArgmax(logits)
        XCTAssertEqual(token, 5, "Argmax should return index of maximum value")
    }

    func testSampleArgmaxWithNegativeValues() {
        // All negative values, position 2 is least negative
        let logits = MLXArray([-10.0, -5.0, -1.0, -8.0, -3.0] as [Float32])
        eval(logits)

        let token = sampleArgmax(logits)
        XCTAssertEqual(token, 2, "Argmax should return index of maximum (least negative) value")
    }

    func testSampleArgmaxRepeatable() {
        var logits = MLXArray.zeros([100])
        logits[42] = MLXArray(Float32(1000.0))
        eval(logits)

        // Argmax should always return the same result
        for _ in 0 ..< 10 {
            let token = sampleArgmax(logits)
            XCTAssertEqual(token, 42, "Argmax should be deterministic")
        }
    }

    // MARK: - Temperature Sampling Tests

    func testSampleTemperatureHighTemp() {
        // With very high temperature, distribution should be more uniform
        let logits = MLXArray([10.0, 0.0, 0.0, 0.0, 0.0] as [Float32])
        eval(logits)

        var counts = [0, 0, 0, 0, 0]
        for _ in 0 ..< 100 {
            let token = sampleTemperature(logits, temperature: 10.0)
            counts[token] += 1
        }

        // With high temp, non-max tokens should also get sampled
        let nonMaxSamples = counts[1] + counts[2] + counts[3] + counts[4]
        XCTAssertGreaterThan(nonMaxSamples, 0, "High temperature should sample non-max tokens")
    }

    func testSampleTemperatureLowTemp() {
        // With very low temperature, should almost always pick max
        let logits = MLXArray([10.0, 0.0, 0.0, 0.0, 0.0] as [Float32])
        eval(logits)

        var maxCount = 0
        for _ in 0 ..< 50 {
            let token = sampleTemperature(logits, temperature: 0.01)
            if token == 0 {
                maxCount += 1
            }
        }

        // With very low temp, should almost always get max token
        XCTAssertGreaterThan(maxCount, 45, "Low temperature should mostly sample max token")
    }

    // MARK: - Top-P Sampling Tests

    func testSampleTopPNarrow() {
        // Create logits where one token dominates
        let logits = MLXArray([100.0, 0.0, 0.0, 0.0, 0.0] as [Float32])
        eval(logits)

        var maxCount = 0
        for _ in 0 ..< 50 {
            let token = sampleTopP(logits, temperature: 1.0, topP: 0.5)
            if token == 0 {
                maxCount += 1
            }
        }

        // With dominating logit and low topP, should almost always pick it
        XCTAssertGreaterThan(maxCount, 45, "Top-P with dominating logit should mostly pick max")
    }

    func testSampleTopPWide() {
        // Uniform-ish logits
        let logits = MLXArray([1.0, 1.0, 1.0, 1.0, 1.0] as [Float32])
        eval(logits)

        var uniqueTokens = Set<Int>()
        for _ in 0 ..< 100 {
            let token = sampleTopP(logits, temperature: 1.0, topP: 0.99)
            uniqueTokens.insert(token)
        }

        // With uniform logits and high topP, should sample multiple tokens
        XCTAssertGreaterThan(uniqueTokens.count, 1, "Top-P with uniform logits should sample varied tokens")
    }

    // MARK: - Sample Dispatch Tests

    func testSampleDispatchGreedy() {
        var logits = MLXArray.zeros([10])
        logits[7] = MLXArray(Float32(100.0))
        eval(logits)

        // temperature = 0 should use argmax
        let params = GenerateParameters(temperature: 0, topP: 0.9)
        let token = sample(logits, params: params)
        XCTAssertEqual(token, 7, "Temperature 0 should use greedy (argmax) sampling")
    }

    func testSampleDispatchTopP() {
        let logits = MLXArray([10.0, 0.0, 0.0, 0.0, 0.0] as [Float32])
        eval(logits)

        // topP < 1 should use top-p sampling
        let params = GenerateParameters(temperature: 1.0, topP: 0.5)

        // Just verify it runs without crashing
        let token = sample(logits, params: params)
        XCTAssertGreaterThanOrEqual(token, 0)
        XCTAssertLessThan(token, 5)
    }

    func testSampleDispatchTemperature() {
        let logits = MLXArray([10.0, 0.0, 0.0, 0.0, 0.0] as [Float32])
        eval(logits)

        // topP = 1 should use temperature sampling
        let params = GenerateParameters(temperature: 0.5, topP: 1.0)

        // Just verify it runs without crashing
        let token = sample(logits, params: params)
        XCTAssertGreaterThanOrEqual(token, 0)
        XCTAssertLessThan(token, 5)
    }

    // MARK: - Repetition Penalty Tests

    func testRepetitionPenaltyNoOp() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float32])
        eval(logits)

        // No penalty (1.0) should return unchanged logits
        let result = applyRepetitionPenalty(logits, generatedTokens: [0, 1, 2], penalty: 1.0, contextSize: 10)
        eval(result)

        XCTAssertEqual(result[0].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(result[4].item(Float.self), 5.0, accuracy: 0.001)
    }

    func testRepetitionPenaltyEmptyTokens() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float32])
        eval(logits)

        // Empty tokens should return unchanged logits
        let result = applyRepetitionPenalty(logits, generatedTokens: [], penalty: 2.0, contextSize: 10)
        eval(result)

        XCTAssertEqual(result[0].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(result[4].item(Float.self), 5.0, accuracy: 0.001)
    }

    func testRepetitionPenaltyPositiveLogits() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float32])
        eval(logits)

        // Penalize tokens 1 and 3
        let result = applyRepetitionPenalty(logits, generatedTokens: [1, 3], penalty: 2.0, contextSize: 10)
        eval(result)

        // Token 1: 2.0 / 2.0 = 1.0
        XCTAssertEqual(result[1].item(Float.self), 1.0, accuracy: 0.001, "Positive logits should be divided by penalty")

        // Token 3: 4.0 / 2.0 = 2.0
        XCTAssertEqual(result[3].item(Float.self), 2.0, accuracy: 0.001, "Positive logits should be divided by penalty")

        // Unpenalized tokens should be unchanged
        XCTAssertEqual(result[0].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(result[2].item(Float.self), 3.0, accuracy: 0.001)
        XCTAssertEqual(result[4].item(Float.self), 5.0, accuracy: 0.001)
    }

    func testRepetitionPenaltyNegativeLogits() {
        let logits = MLXArray([-1.0, -2.0, -3.0, -4.0, -5.0] as [Float32])
        eval(logits)

        // Penalize token 2
        let result = applyRepetitionPenalty(logits, generatedTokens: [2], penalty: 2.0, contextSize: 10)
        eval(result)

        // Token 2: -3.0 * 2.0 = -6.0 (negative logits get multiplied)
        XCTAssertEqual(result[2].item(Float.self), -6.0, accuracy: 0.001, "Negative logits should be multiplied by penalty")

        // Unpenalized tokens should be unchanged
        XCTAssertEqual(result[0].item(Float.self), -1.0, accuracy: 0.001)
        XCTAssertEqual(result[4].item(Float.self), -5.0, accuracy: 0.001)
    }

    func testRepetitionPenaltyContextWindow() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float32])
        eval(logits)

        // Only recent tokens within context should be penalized
        let tokens = [0, 1, 2, 3, 4] // 5 tokens
        let result = applyRepetitionPenalty(logits, generatedTokens: tokens, penalty: 2.0, contextSize: 2)
        eval(result)

        // Only tokens 3 and 4 (last 2) should be penalized
        XCTAssertEqual(result[0].item(Float.self), 1.0, accuracy: 0.001, "Token outside context should be unchanged")
        XCTAssertEqual(result[1].item(Float.self), 2.0, accuracy: 0.001, "Token outside context should be unchanged")
        XCTAssertEqual(result[2].item(Float.self), 3.0, accuracy: 0.001, "Token outside context should be unchanged")
        XCTAssertEqual(result[3].item(Float.self), 2.0, accuracy: 0.001, "Token in context should be penalized")
        XCTAssertEqual(result[4].item(Float.self), 2.5, accuracy: 0.001, "Token in context should be penalized")
    }

    func testRepetitionPenaltyDuplicateTokens() {
        let logits = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0] as [Float32])
        eval(logits)

        // Duplicate tokens should be handled (unique set)
        let tokens = [1, 1, 1, 3, 3]
        let result = applyRepetitionPenalty(logits, generatedTokens: tokens, penalty: 2.0, contextSize: 10)
        eval(result)

        // Should only penalize each unique token once
        XCTAssertEqual(result[1].item(Float.self), 1.0, accuracy: 0.001)
        XCTAssertEqual(result[3].item(Float.self), 2.0, accuracy: 0.001)
    }
}
