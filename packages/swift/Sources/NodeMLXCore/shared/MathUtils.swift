// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Mathematical utility functions for neural network operations.

import Foundation

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
}
