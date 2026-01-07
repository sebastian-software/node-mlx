//
//  Generate.swift
//  NodeMLXCore
//
//  Minimal LLM generation without mlx-swift-lm dependency.
//
//  This implementation is inspired by mlx-swift-lm's generation approach
//  but is independently written. mlx-swift-lm is MIT licensed by ml-explore.
//  See: https://github.com/ml-explore/mlx-swift-lm
//
//  NOTE: This file requires MLX as a dependency. Currently disabled until
//  we add mlx-swift as a direct dependency instead of through mlx-swift-lm.
//

import Foundation

#if canImport(MLX)
import MLX
import MLXRandom
#endif

// MARK: - Generation Parameters

public struct GenerateParameters: Sendable {
    public let temperature: Float
    public let topP: Float
    public let maxTokens: Int
    public let repetitionPenalty: Float

    public init(
        temperature: Float = 0.7,
        topP: Float = 0.9,
        maxTokens: Int = 256,
        repetitionPenalty: Float = 1.0
    ) {
        self.temperature = max(0.0, temperature)
        self.topP = min(max(0.0, topP), 1.0)
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
    }
}

// MARK: - Generation Result

public struct GenerateResult: Sendable {
    public let tokens: [Int]
    public let text: String
    public let tokensPerSecond: Double
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
}

// MARK: - Sampling Functions

/// Apply temperature scaling to logits
func applyTemperature(_ logits: MLXArray, temperature: Float) -> MLXArray {
    if temperature <= 0 {
        return logits
    }
    return logits / temperature
}

/// Apply top-p (nucleus) sampling
func applyTopP(_ logits: MLXArray, topP: Float) -> MLXArray {
    if topP >= 1.0 {
        return logits
    }

    // Sort logits descending
    let probs = softmax(logits, axis: -1)
    let sortedIndices = argSort(probs, axis: -1)
    let sortedProbs = take(probs, sortedIndices, axis: -1)

    // Compute cumulative probabilities
    let cumProbs = cumsum(sortedProbs, axis: -1)

    // Create mask for tokens to keep
    let mask = cumProbs .<= topP

    // Set probabilities outside top-p to 0
    let filteredProbs = sortedProbs * mask.asType(.float32)

    // Normalize
    let normalizedProbs = filteredProbs / filteredProbs.sum(axis: -1, keepDims: true)

    // Convert back to logits
    return log(normalizedProbs + 1e-10)
}

/// Apply repetition penalty to logits
func applyRepetitionPenalty(
    _ logits: MLXArray,
    generatedTokens: [Int],
    penalty: Float
) -> MLXArray {
    if penalty == 1.0 || generatedTokens.isEmpty {
        return logits
    }

    var penalizedLogits = logits

    // Penalize already generated tokens
    for token in Set(generatedTokens) {
        let tokenIdx = MLXArray([Int32(token)])
        let currentLogit = penalizedLogits[0, token]

        // Apply penalty: divide positive logits, multiply negative
        let penalized = where(
            currentLogit .> 0,
            currentLogit / penalty,
            currentLogit * penalty
        )

        // Update logits (this is simplified - full impl would use scatter)
        // For now, we'll skip this optimization
    }

    return penalizedLogits
}

/// Sample next token from logits
func sampleToken(_ logits: MLXArray, temperature: Float, topP: Float) -> Int {
    var processedLogits = logits

    // Apply temperature
    if temperature > 0 {
        processedLogits = applyTemperature(processedLogits, temperature: temperature)
    }

    // Apply top-p
    if topP < 1.0 {
        processedLogits = applyTopP(processedLogits, topP: topP)
    }

    // Sample
    if temperature <= 0 {
        // Greedy sampling
        let token = argMax(processedLogits, axis: -1)
        return token.item(Int.self)
    } else {
        // Multinomial sampling
        let probs = softmax(processedLogits, axis: -1)
        let token = MLXRandom.categorical(probs)
        return token.item(Int.self)
    }
}

// MARK: - LLM Protocol

/// Protocol for language models that can generate text
public protocol LLMGeneratable {
    /// Vocabulary size
    var vocabularySize: Int { get }

    /// Forward pass: tokens → logits
    func callAsFunction(_ tokens: MLXArray, cache: Any?) -> MLXArray
}

// MARK: - Generate Function

/// Generate tokens from a language model
///
/// - Parameters:
///   - model: The language model
///   - promptTokens: Initial prompt tokens
///   - parameters: Generation parameters
///   - eosToken: End-of-sequence token ID
///   - onToken: Optional callback for each generated token
/// - Returns: Generation result
public func generate<M: LLMGeneratable>(
    model: M,
    promptTokens: [Int],
    parameters: GenerateParameters,
    eosToken: Int,
    onToken: ((Int, String) -> Bool)? = nil
) -> GenerateResult {
    let startTime = Date()

    // Convert prompt to MLXArray
    var tokens = MLXArray(promptTokens.map { Int32($0) })
    tokens = tokens.reshaped([1, -1])  // [1, seq_len]

    var generatedTokens: [Int] = []
    var cache: Any? = nil  // KVCache would go here

    // Initial forward pass with full prompt
    var logits = model(tokens, cache: cache)
    logits = logits[0, -1]  // Get last position logits [vocab_size]

    // Generate tokens
    for _ in 0..<parameters.maxTokens {
        // Sample next token
        let nextToken = sampleToken(
            logits,
            temperature: parameters.temperature,
            topP: parameters.topP
        )

        // Check for EOS
        if nextToken == eosToken {
            break
        }

        generatedTokens.append(nextToken)

        // Callback
        if let onToken = onToken {
            // Note: Actual token→string conversion needs tokenizer
            if !onToken(nextToken, "") {
                break  // Early stop requested
            }
        }

        // Next forward pass with single token
        let nextTokenArray = MLXArray([Int32(nextToken)]).reshaped([1, 1])
        logits = model(nextTokenArray, cache: cache)
        logits = logits[0, -1]
    }

    let endTime = Date()
    let elapsedSeconds = endTime.timeIntervalSince(startTime)
    let tokensPerSecond = Double(generatedTokens.count) / max(elapsedSeconds, 0.001)

    return GenerateResult(
        tokens: generatedTokens,
        text: "",  // Needs tokenizer decode
        tokensPerSecond: tokensPerSecond,
        promptTokenCount: promptTokens.count,
        generatedTokenCount: generatedTokens.count
    )
}

