//
//  Generate.swift
//  NodeMLXCore
//
//  Token generation with sampling strategies.
//
//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).
//  See: https://github.com/ml-explore/mlx-swift-lm
//

import Foundation
import MLX
import MLXRandom

// MARK: - Generation Parameters

public struct GenerateParameters: Sendable {
    /// Maximum tokens to generate
    public var maxTokens: Int

    /// Sampling temperature (0 = greedy/argmax)
    public var temperature: Float

    /// Top-p (nucleus) sampling threshold
    public var topP: Float

    /// Penalty for repeating tokens
    public var repetitionPenalty: Float?

    /// Context size for repetition penalty
    public var repetitionContextSize: Int

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}

// MARK: - Sampling Strategies

/// Sample from logits using argmax (greedy decoding)
public func sampleArgmax(_ logits: MLXArray) -> Int {
    let token = argMax(logits, axis: -1)
    return token.item(Int.self)
}

/// Sample from logits using temperature
public func sampleTemperature(_ logits: MLXArray, temperature: Float) -> Int {
    let scaled = logits / MLXArray(temperature)
    let probs = softmax(scaled, axis: -1)

    // Sample from categorical distribution
    let uniform = MLXRandom.uniform(low: 0, high: 1, [1])
    let cumsum = cumsum(probs, axis: -1)
    let token = argMax(cumsum .>= uniform, axis: -1)
    return token.item(Int.self)
}

/// Sample from logits using top-p (nucleus) sampling
public func sampleTopP(_ logits: MLXArray, temperature: Float, topP: Float) -> Int {
    // Apply temperature
    let scaled = logits / MLXArray(temperature)
    let probs = softmax(scaled, axis: -1)

    // Sort probabilities in descending order
    let sortedIndices = argSort(probs, axis: -1)
    // Reverse to get descending order
    let reversedIndices = sortedIndices[.ellipsis, .stride(by: -1)]
    let sortedProbs = take(probs, reversedIndices, axis: -1)

    // Compute cumulative probabilities
    let cumProbs = cumsum(sortedProbs, axis: -1)

    // Find cutoff index where cumulative prob exceeds topP
    let mask = cumProbs .<= MLXArray(topP)
    let numTokens = sum(mask.asType(.int32)).item(Int.self) + 1

    // Keep only top-p tokens
    let topIndices = reversedIndices[0 ..< numTokens]
    let topProbs = sortedProbs[0 ..< numTokens]

    // Renormalize
    let normalizedProbs = topProbs / sum(topProbs)

    // Sample from truncated distribution
    let uniform = MLXRandom.uniform(low: 0, high: 1, [1])
    let cumsum2 = cumsum(normalizedProbs, axis: -1)
    let sampleIdx = argMax(cumsum2 .>= uniform, axis: -1).item(Int.self)

    return topIndices[sampleIdx].item(Int.self)
}

/// Main sampling function that dispatches to the right strategy
public func sample(_ logits: MLXArray, params: GenerateParameters) -> Int {
    if params.temperature == 0 {
        sampleArgmax(logits)
    } else if params.topP > 0, params.topP < 1 {
        sampleTopP(logits, temperature: params.temperature, topP: params.topP)
    } else {
        sampleTemperature(logits, temperature: params.temperature)
    }
}

// MARK: - Repetition Penalty

/// Apply repetition penalty to logits
public func applyRepetitionPenalty(
    _ logits: MLXArray,
    generatedTokens: [Int],
    penalty: Float,
    contextSize: Int
) -> MLXArray {
    guard penalty != 1.0, !generatedTokens.isEmpty else {
        return logits
    }

    // Get recent tokens within context window
    let recentTokens = Array(generatedTokens.suffix(contextSize))
    guard !recentTokens.isEmpty else {
        return logits
    }

    // Create penalty mask
    let uniqueTokens = Array(Set(recentTokens))
    let indices = MLXArray(uniqueTokens.map { Int32($0) })

    // Get logits at penalized positions
    let selectedLogits = take(logits, indices, axis: -1)

    // Apply penalty: divide positive logits, multiply negative
    let positiveLogits = maximum(selectedLogits, MLXArray(0))
    let negativeLogits = minimum(selectedLogits, MLXArray(0))
    let penalized = positiveLogits / MLXArray(penalty) + negativeLogits * MLXArray(penalty)

    // Scatter back into original logits
    var result = logits
    for (i, token) in uniqueTokens.enumerated() {
        result[token] = penalized[i]
    }

    return result
}
