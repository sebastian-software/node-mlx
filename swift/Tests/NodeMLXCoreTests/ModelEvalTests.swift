// Copyright © 2026 Sebastian Software GmbH.
// Tests adapted from mlx-swift-lm patterns (MIT License, Apple Inc.)

import XCTest
import MLX
import MLXNN
@testable import NodeMLXCore

final class ModelEvalTests: XCTestCase {

    // MARK: - Qwen2 Model Tests

    func testQwen2ModelForward() throws {
        // Create a small Qwen2 model for testing
        let config = try makeTestQwen2Config()
        let model = Qwen2Model(config)

        // Quantize the model (like production usage)
        quantize(model: model, groupSize: 64, bits: 4)

        // Forward pass with batch of tokens
        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]  // [1, 5]
        var cache: [KVCache]? = nil
        let output = model(input, cache: &cache)
        eval(output)

        XCTAssertEqual(output.shape, [1, 5, 100])  // [batch, seq, vocab]
    }

    func testQwen2ModelWithCache() throws {
        let config = try makeTestQwen2Config()
        let model = Qwen2Model(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Create cache
        var cache: [KVCache]? = model.newCache()
        XCTAssertEqual(cache?.count, 2)  // One per layer

        // Prefill with 5 tokens
        let prefill = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let out1 = model(prefill, cache: &cache)
        eval(out1)

        XCTAssertEqual(out1.shape, [1, 5, 100])
        XCTAssertEqual(cache?[0].offset, 5)

        // Incremental generation - single token
        let nextToken = MLXArray([6])[.newAxis, .ellipsis]
        let out2 = model(nextToken, cache: &cache)
        eval(out2)

        XCTAssertEqual(out2.shape, [1, 1, 100])
        XCTAssertEqual(cache?[0].offset, 6)
    }

    func testQwen2ModelWithTiedEmbeddings() throws {
        let config = try makeTestQwen2Config(tieWordEmbeddings: true)
        let model = Qwen2Model(config)

        let input = MLXArray([1, 2, 3])[.newAxis, .ellipsis]
        let output = model(input)
        eval(output)

        XCTAssertEqual(output.shape, [1, 3, 100])
    }

    // MARK: - Concurrent Evaluation Tests

    func testConcurrentModelEvaluation() async throws {
        let config = try makeTestQwen2Config(
            hiddenSize: 32,
            intermediateSize: 64,
            vocabularySize: 50
        )
        let model = Qwen2Model(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        eval(model)

        let numTasks = 3
        let results = await withTaskGroup(of: [Int].self) { group in
            var allResults: [[Int]] = []

            for taskId in 0..<numTasks {
                group.addTask {
                    let input = MLXArray([1 + taskId, 2 + taskId, 3 + taskId])[.newAxis, .ellipsis]
                    let output = model(input)
                    eval(output)
                    return output.shape
                }
            }

            for await result in group {
                allResults.append(result)
            }

            return allResults
        }

        XCTAssertEqual(results.count, numTasks)

        for result in results {
            XCTAssertEqual(result, [1, 3, 50])
        }
    }

    // MARK: - Sampling Tests

    func testGreedySampling() throws {
        // Test that greedy sampling (argmax) works correctly
        let vocabSize = 100
        let logits = MLXArray.zeros([vocabSize])

        // Set one token to have highest probability
        var logitsArray = logits.asArray(Float.self)
        logitsArray[42] = 10.0  // Token 42 should be selected
        let modifiedLogits = MLXArray(logitsArray)

        let token = argMax(modifiedLogits, axis: -1)
        eval(token)

        XCTAssertEqual(Int(token.item(Int32.self)), 42)
    }

    func testCategoricalSampling() throws {
        // Test that categorical sampling produces valid tokens
        let vocabSize = 100
        let logits = MLXRandom.normal([vocabSize])

        // Sample multiple times and verify all tokens are in valid range
        for _ in 0..<10 {
            let probs = softmax(logits, axis: -1)
            let token = MLXRandom.categorical(probs)
            eval(token)

            let tokenValue = Int(token.item(Int32.self))
            XCTAssertGreaterThanOrEqual(tokenValue, 0)
            XCTAssertLessThan(tokenValue, vocabSize)
        }
    }

    // MARK: - Configuration Tests

    func testQwen2ConfigurationDecoding() throws {
        let json = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false
        }
        """

        let config = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: json.data(using: .utf8)!
        )

        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.hiddenLayers, 24)
        XCTAssertEqual(config.intermediateSize, 4096)
        XCTAssertEqual(config.attentionHeads, 16)
        XCTAssertEqual(config.vocabularySize, 32000)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.ropeTheta, 1000000.0)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testQwen2ConfigurationWithRopeScaling() throws {
        let json = """
        {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 10000,
            "num_key_value_heads": 2,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }
        """

        let config = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: json.data(using: .utf8)!
        )

        XCTAssertNotNil(config.ropeScaling)

        if case .string(let type) = config.ropeScaling?["type"] {
            XCTAssertEqual(type, "linear")
        } else {
            XCTFail("Expected rope_scaling type to be 'linear'")
        }

        XCTAssertEqual(config.ropeScaling?["factor"]?.asFloat(), 2.0)
    }
}

// MARK: - Helper Functions for Testing

/// Create a Qwen2Configuration from parameters (for testing)
func makeTestQwen2Config(
    hiddenSize: Int = 64,
    hiddenLayers: Int = 2,
    intermediateSize: Int = 128,
    attentionHeads: Int = 4,
    rmsNormEps: Float = 1e-6,
    vocabularySize: Int = 100,
    kvHeads: Int = 2,
    ropeTheta: Float = 10000.0,
    tieWordEmbeddings: Bool = false
) throws -> Qwen2Configuration {
    let json = """
    {
        "hidden_size": \(hiddenSize),
        "num_hidden_layers": \(hiddenLayers),
        "intermediate_size": \(intermediateSize),
        "num_attention_heads": \(attentionHeads),
        "rms_norm_eps": \(rmsNormEps),
        "vocab_size": \(vocabularySize),
        "num_key_value_heads": \(kvHeads),
        "rope_theta": \(ropeTheta),
        "tie_word_embeddings": \(tieWordEmbeddings)
    }
    """
    return try JSONDecoder().decode(Qwen2Configuration.self, from: json.data(using: .utf8)!)
}

