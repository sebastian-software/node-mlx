// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared RMSNorm implementation used by all transformer models.

import MLX
import MLXFast
import MLXNN

/// Root Mean Square Layer Normalization.
///
/// This is the standard normalization layer used in modern LLMs like
/// Llama, Qwen, Mistral, Gemma, etc.
///
/// RMSNorm is computationally simpler than LayerNorm as it only
/// normalizes by the RMS of activations, without centering.
public class RMSNorm: Module {
    public let eps: Float

    @ModuleInfo(key: "weight") public var weight: MLXArray

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}
