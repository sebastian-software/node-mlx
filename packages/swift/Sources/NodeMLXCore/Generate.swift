// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Text generation loop for autoregressive language models.

import Foundation
import MLX
import MLXNN

// MARK: - Generation Configuration

/// Configuration for text generation.
public struct GenerationConfig {
    /// Maximum number of tokens to generate.
    public var maxTokens: Int

    /// Temperature for sampling (0 = greedy, higher = more random).
    public var temperature: Float

    /// Top-p nucleus sampling threshold.
    public var topP: Float

    /// Repetition penalty (1.0 = no penalty).
    public var repetitionPenalty: Float

    /// Token IDs that signal end of generation.
    public var stopTokens: Set<Int>

    /// Creates a generation configuration.
    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.0,
        stopTokens: Set<Int> = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.stopTokens = stopTokens
    }
}

// MARK: - Token Sampling

/// Samples the next token from logits.
///
/// - Parameters:
///   - logits: Model output logits [vocab_size]
///   - temperature: Sampling temperature
///   - topP: Nucleus sampling threshold
/// - Returns: Sampled token ID
public func sampleToken(
    logits: MLXArray,
    temperature: Float,
    topP: Float = 1.0
) -> Int {
    // Greedy decoding for temperature 0
    if temperature == 0 {
        return argMax(logits).item(Int.self)
    }

    // Apply temperature
    var scaledLogits = logits / temperature

    // Apply top-p (nucleus) sampling if needed
    if topP < 1.0 {
        scaledLogits = applyTopP(scaledLogits, topP: topP)
    }

    // Sample from the distribution
    let probs = softmax(scaledLogits)
    let token = categorical(probs)
    return token.item(Int.self)
}

/// Applies top-p (nucleus) sampling by zeroing low-probability tokens.
private func applyTopP(_ logits: MLXArray, topP: Float) -> MLXArray {
    let probs = softmax(logits)
    let sortedIndices = argSort(probs)
    let sortedProbs = probs[sortedIndices]

    // Find cumulative probabilities
    let cumProbs = cumsum(sortedProbs)

    // Find tokens below threshold
    let belowThreshold = cumProbs .<= (1.0 - topP)

    // Mask out tokens below threshold
    var result = logits
    let maskValue = Float.leastNormalMagnitude
    result = which(belowThreshold, MLXArray(maskValue), sortedProbs)

    // Unsort back to original order
    var unsorted = MLXArray.zeros(like: logits)
    unsorted[sortedIndices] = result

    return unsorted
}

// MARK: - Generation Loop

/// Generates text from a language model.
///
/// - Parameters:
///   - model: The language model to use
///   - inputIds: Initial token IDs
///   - config: Generation configuration
///   - onToken: Callback for each generated token
/// - Returns: Array of generated token IDs (excluding input)
public func generate(
    model: any LLMModel,
    inputIds: [Int],
    config: GenerationConfig = GenerationConfig(),
    onToken: ((Int) -> Bool)? = nil
) -> [Int] {
    var generatedTokens: [Int] = []
    var cache: [KVCacheProtocol]? = model.newCache()

    // Convert input to MLXArray
    var currentIds = MLXArray(inputIds.map { Int32($0) }).reshaped([1, inputIds.count])

    // Process prompt (prefill)
    var logits = model(currentIds, cache: &cache)
    eval(logits, cache as Any)

    // Get logits for last token
    var nextLogits = logits[0..., -1, 0...]

    // Generation loop
    for _ in 0 ..< config.maxTokens {
        // Sample next token
        let nextToken = sampleToken(
            logits: nextLogits,
            temperature: config.temperature,
            topP: config.topP
        )

        // Check for stop token
        if config.stopTokens.contains(nextToken) {
            break
        }

        generatedTokens.append(nextToken)

        // Callback for streaming
        if let onToken {
            if !onToken(nextToken) {
                break
            }
        }

        // Prepare next input
        currentIds = MLXArray([Int32(nextToken)]).reshaped([1, 1])

        // Generate next logits
        logits = model(currentIds, cache: &cache)
        eval(logits, cache as Any)

        nextLogits = logits[0..., -1, 0...]
    }

    return generatedTokens
}

// MARK: - Streaming Generation

/// Result of a single generation step.
public struct GenerationStep {
    /// The generated token ID.
    public let tokenId: Int

    /// Whether generation is complete.
    public let isComplete: Bool

    /// Decoded text for this token (if decoder provided).
    public let text: String?
}

/// Streaming generator for incremental text generation.
public class StreamingGenerator {
    private let model: any LLMModel
    private let config: GenerationConfig
    private var cache: [KVCacheProtocol]?
    private var tokenCount: Int = 0

    /// Creates a streaming generator.
    public init(model: any LLMModel, config: GenerationConfig = GenerationConfig()) {
        self.model = model
        self.config = config
    }

    /// Processes the initial prompt and returns the first token.
    public func processPrompt(_ inputIds: [Int]) -> GenerationStep {
        cache = model.newCache()

        let currentIds = MLXArray(inputIds.map { Int32($0) }).reshaped([1, inputIds.count])
        let logits = model(currentIds, cache: &cache)
        eval(logits, cache as Any)

        let nextLogits = logits[0..., -1, 0...]
        let nextToken = sampleToken(
            logits: nextLogits,
            temperature: config.temperature,
            topP: config.topP
        )

        tokenCount = 1

        return GenerationStep(
            tokenId: nextToken,
            isComplete: config.stopTokens.contains(nextToken),
            text: nil
        )
    }

    /// Generates the next token given the previous one.
    public func nextStep(previousToken: Int) -> GenerationStep {
        guard tokenCount < config.maxTokens else {
            return GenerationStep(tokenId: 0, isComplete: true, text: nil)
        }

        let currentIds = MLXArray([Int32(previousToken)]).reshaped([1, 1])
        let logits = model(currentIds, cache: &cache)
        eval(logits, cache as Any)

        let nextLogits = logits[0..., -1, 0...]
        let nextToken = sampleToken(
            logits: nextLogits,
            temperature: config.temperature,
            topP: config.topP
        )

        tokenCount += 1

        return GenerationStep(
            tokenId: nextToken,
            isComplete: config.stopTokens.contains(nextToken) || tokenCount >= config.maxTokens,
            text: nil
        )
    }

    /// Resets the generator state.
    public func reset() {
        cache = nil
        tokenCount = 0
    }
}
