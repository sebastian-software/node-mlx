// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Standard MLP (SwiGLU) implementation shared by most transformer models.

import MLX
import MLXNN

/// Standard SwiGLU MLP block used by most modern LLMs.
///
/// SwiGLU (Swish-Gated Linear Unit) is the dominant MLP architecture
/// in models like Llama, Qwen, Mistral, Gemma, etc.
///
/// Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))
public class StandardMLP<Config: BaseModelConfiguration>: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(_ config: Config) {
        let intermediateSize = config.intermediateSize
        let mlpBias = config.mlpBias
        _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: mlpBias)
        _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: mlpBias)
        _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: mlpBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
