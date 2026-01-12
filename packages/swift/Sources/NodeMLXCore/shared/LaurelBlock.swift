// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Laurel (Learned Augmented Residual) block for efficient low-rank residual computation.
// Used by Gemma3n and potentially future models with similar architecture.

import MLX
import MLXNN

/// Laurel (Learned Augmented Residual) block.
///
/// A low-rank residual layer that adds a learned residual to the input:
/// output = x + postNorm(right(left(x)))
///
/// This is more parameter-efficient than full-rank residual connections
/// while still allowing the model to learn useful residual transformations.
///
/// Architecture:
/// 1. Project down to low-rank: x → Linear(hidden → laurelRank)
/// 2. Project back up: Linear(laurelRank → hidden)
/// 3. Normalize: RMSNorm
/// 4. Add residual: x + normalized
public class LaurelBlock<Config: LaurelConfiguration>: Module {
    @ModuleInfo(key: "linear_left") public var linearLeft: Linear
    @ModuleInfo(key: "linear_right") public var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") public var postLaurelNorm: RMSNorm

    private let hiddenSize: Int
    private let laurelRank: Int

    public init(_ config: Config) {
        hiddenSize = config.hiddenSize
        laurelRank = config.laurelRank

        _linearLeft.wrappedValue = Linear(hiddenSize, laurelRank, bias: false)
        _linearRight.wrappedValue = Linear(laurelRank, hiddenSize, bias: false)
        _postLaurelNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var laurel = linearLeft(x)
        laurel = linearRight(laurel)
        laurel = postLaurelNorm(laurel)
        // Add residual connection
        return x + laurel
    }
}
