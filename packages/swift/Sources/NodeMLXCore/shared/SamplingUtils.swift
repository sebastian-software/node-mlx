// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Sampling utilities for token generation.
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/sample_utils.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import Foundation
import MLX

// MARK: - Sampling Utilities

/// Sampling utilities for nucleus (top-p), top-k, and min-p sampling.
public enum SamplingUtils {
    // MARK: - Top-P (Nucleus) Sampling

    /// Applies top-p (nucleus) sampling to logits.
    ///
    /// Masks tokens outside the smallest set of tokens whose cumulative
    /// probability exceeds p.
    ///
    /// - Parameters:
    ///   - logits: Input logits, shape [..., vocab_size]
    ///   - p: Probability threshold (0.0-1.0)
    /// - Returns: Filtered logits with low-probability tokens masked to -inf
    public static func applyTopP(_ logits: MLXArray, p: Float) -> MLXArray {
        // Get probabilities
        let probs = softmax(logits, axis: -1)

        // Sort probabilities descending
        let sortedIndices = argSort(-probs, axis: -1)
        let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)

        // Cumulative probabilities
        let cumProbs = cumsum(sortedProbs, axis: -1)

        // Create shifted cumsum: prepend 0 and drop last element
        // This ensures we keep at least the top token even if it exceeds p
        let zerosShape = Array(cumProbs.shape.dropLast()) + [1]
        let zeros = MLXArray.zeros(zerosShape)
        let shiftedCumProbs = concatenated([zeros, cumProbs[.ellipsis, ..<(-1)]], axis: -1)

        // Find cutoff: positions where shifted cumsum > p should be masked
        let topPMask = shiftedCumProbs .> MLXArray(p)

        // Apply mask: set excluded tokens to -inf
        let sortedLogits = takeAlong(logits, sortedIndices, axis: -1)
        let filteredSortedLogits = which(topPMask, MLXArray(-Float.infinity), sortedLogits)

        // Unsort back to original order
        let unsortIndices = argSort(sortedIndices, axis: -1)
        return takeAlong(filteredSortedLogits, unsortIndices, axis: -1)
    }

    // MARK: - Top-K Sampling

    /// Applies top-k sampling to logits.
    ///
    /// Keeps only the k tokens with highest probability, masking the rest.
    ///
    /// - Parameters:
    ///   - logits: Input logits, shape [..., vocab_size]
    ///   - k: Number of tokens to keep
    /// - Returns: Filtered logits with low-probability tokens masked to -inf
    public static func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
        guard k > 0 else { return logits }

        // Get top k indices using partition (more efficient than full sort)
        let topKIndices = argPartition(-logits, kth: k, axis: -1)[.ellipsis, ..<k]

        // Create mask for top k tokens
        let topKValues = takeAlong(logits, topKIndices, axis: -1)

        // Find minimum value in top k
        let threshold = topKValues.min(axis: -1, keepDims: true)

        // Mask tokens below threshold
        let mask = logits .< threshold
        return which(mask, MLXArray(-Float.infinity), logits)
    }

    // MARK: - Min-P Sampling

    /// Applies min-p sampling to logits.
    ///
    /// Masks tokens whose probability is less than minP times the maximum
    /// probability.
    ///
    /// - Parameters:
    ///   - logits: Input logits, shape [..., vocab_size]
    ///   - minP: Minimum probability ratio (0.0-1.0)
    /// - Returns: Filtered logits with low-probability tokens masked to -inf
    public static func applyMinP(_ logits: MLXArray, minP: Float) -> MLXArray {
        guard minP > 0 else { return logits }

        // Get probabilities
        let probs = softmax(logits, axis: -1)

        // Find maximum probability
        let maxProb = probs.max(axis: -1, keepDims: true)

        // Threshold is minP * maxProb
        let threshold = maxProb * MLXArray(minP)

        // Mask tokens below threshold
        let mask = probs .< threshold
        return which(mask, MLXArray(-Float.infinity), logits)
    }

    // MARK: - Combined Sampling

    /// Samples a token from logits with temperature and optional filtering.
    ///
    /// - Parameters:
    ///   - logits: Input logits, shape [vocab_size] or [1, vocab_size]
    ///   - temperature: Temperature for scaling (0 = greedy)
    ///   - topP: Top-p threshold (1.0 = disabled)
    ///   - topK: Top-k count (0 = disabled)
    ///   - minP: Min-p threshold (0.0 = disabled)
    /// - Returns: Sampled token index
    public static func sampleToken(
        logits: MLXArray,
        temperature: Float = 1.0,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0
    ) -> Int {
        // Ensure 2D shape
        var workingLogits = logits.ndim == 1 ? logits.reshaped([1, -1]) : logits

        // Greedy decoding
        if temperature == 0 {
            return argMax(workingLogits, axis: -1).item(Int.self)
        }

        // Apply temperature
        workingLogits = workingLogits / MLXArray(temperature)

        // Apply filters in order
        if topK > 0 {
            workingLogits = applyTopK(workingLogits, k: topK)
        }
        if topP < 1.0 {
            workingLogits = applyTopP(workingLogits, p: topP)
        }
        if minP > 0 {
            workingLogits = applyMinP(workingLogits, minP: minP)
        }

        // Sample from distribution
        let probs = softmax(workingLogits, axis: -1)
        return categorical(probs.squeezed()).item(Int.self)
    }
}
