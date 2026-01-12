// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/gemma.py
// Git Hash: 7585c142a6be9c9245f4ce61d087839776cb8275 (2026-01-12)

import MLX
import MLXFast
import MLXNN

/// Gemma-style RMSNorm with (1 + weight) scaling.
///
/// Unlike standard RMSNorm which uses `weight` directly, Gemma models
/// use `(1 + weight)` scaling. This means the weight is initialized to
/// zeros and the effective scale is `1 + weight`.
///
/// This is used by Gemma, Gemma2, Gemma3, and Gemma3n models.
///
/// Original Python:
/// ```python
/// class RMSNorm(nn.Module):
///     def __init__(self, dims: int, eps: float = 1e-5):
///         super().__init__()
///         self.weight = mx.ones((dims,))
///         self.eps = eps
///
///     def __call__(self, x):
///         return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
/// ```
public class GemmaRMSNorm: Module {
    public let eps: Float

    @ModuleInfo(key: "weight") public var weight: MLXArray

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        // Initialize to zeros - effective scale will be (1 + weight) = 1
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma uses (1 + weight) scaling
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}
