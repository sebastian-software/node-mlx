// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Mathematical utility functions for neural network operations.

import Foundation
import MLX

// MARK: - Math Utilities

/// Mathematical utility functions used across the codebase.
public enum MathUtils {
    /// Approximate inverse error function.
    ///
    /// Uses a rational approximation that is accurate to about 4 decimal places.
    /// Used primarily for computing gelu_topk sparse activation thresholds.
    ///
    /// - Parameter x: Input value in range (-1, 1)
    /// - Returns: Inverse error function of x
    public static func erfinv(_ x: Float) -> Float {
        let a: Float = 0.147
        let sign: Float = x < 0 ? -1 : 1
        let x2 = x * x
        let lnTerm = log(1 - x2)
        let term1 = 2 / (Float.pi * a) + lnTerm / 2
        let term2 = lnTerm / a
        return sign * sqrt(sqrt(term1 * term1 - term2) - term1)
    }

    /// Clip residual for float16 overflow protection.
    ///
    /// When using float16, residual additions can overflow. This function
    /// converts to float32 for the addition and clips to float16 bounds
    /// before converting back.
    ///
    /// - Parameters:
    ///   - x: First operand
    ///   - y: Second operand to add
    /// - Returns: Clipped sum in original dtype
    public static func clipResidual(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        if x.dtype != .float16 {
            return x + y
        }
        let bound = Float16.greatestFiniteMagnitude
        let sum = (x.asType(.float32) + y.asType(.float32))
        return clip(sum, min: MLXArray(-Float(bound)), max: MLXArray(Float(bound))).asType(.float16)
    }

    /// Top-k selection for MoE routing.
    ///
    /// Efficiently selects the top k values and their indices from an array.
    /// Uses argPartition for O(n) performance instead of O(n log n) full sort.
    ///
    /// - Parameters:
    ///   - a: Input array
    ///   - k: Number of top elements to select
    ///   - axis: Axis along which to select (default: -1)
    /// - Returns: Tuple of (top k values, top k indices)
    public static func topK(_ a: MLXArray, k: Int, axis: Int = -1) -> (values: MLXArray, indices: MLXArray) {
        let partitionedIndices = argPartition(a, kth: -k, axis: axis)
        let topKIndices = partitionedIndices[.ellipsis, (-k)...]
        let topKValues = takeAlong(a, topKIndices, axis: axis)
        return (topKValues, topKIndices)
    }
}
