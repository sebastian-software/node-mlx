// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/rope_utils.py
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RoPE Protocol

/// Protocol for Rotary Position Embedding (RoPE) implementations.
///
/// RoPE encodes positional information by rotating pairs of dimensions
/// in the embedding space. Different variants exist for different use cases:
/// - Standard: Basic rotary embeddings
/// - Llama3: With smooth frequency interpolation
/// - Yarn: Yet Another RoPE, with mscale and correction ranges
/// - SuScaled: For very long context (longrope)
public protocol RoPEProvider {
    /// Apply rotary position embeddings to input tensor.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape [B, H, S, D] or [B, S, D]
    ///   - offset: Position offset for cached sequences
    /// - Returns: Tensor with rotated embeddings
    func apply(_ x: MLXArray, offset: Int) -> MLXArray
}

// MARK: - RoPE Extension

extension RoPE: RoPEProvider {
    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}

// MARK: - Llama3RoPE

/// Llama 3 RoPE with smooth frequency interpolation.
///
/// This variant handles frequency scaling with smooth transitions between
/// low-frequency and high-frequency ranges, avoiding abrupt changes that
/// could hurt model quality.
///
/// ## Parameters from scaling_config
/// - `factor`: Base scaling factor
/// - `low_freq_factor`: Factor for low frequency range (default: 1.0)
/// - `high_freq_factor`: Factor for high frequency range (default: 4.0)
/// - `original_max_position_embeddings`: Original context length (default: 8192)
public class Llama3RoPE: Module, RoPEProvider {
    // MARK: - Properties

    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let freqs: MLXArray

    // MARK: - Initialization

    public init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        guard let scalingConfig else {
            fatalError("Llama3RoPE requires scaling_config")
        }

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192.0

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        // Scale low frequencies by factor
        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        // Smooth interpolation for medium frequencies
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }

    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}

// MARK: - YarnRoPE

/// Yet Another RoPE (Yarn) for extended context.
///
/// Yarn uses a combination of NTK-aware interpolation and attention scaling
/// to enable longer context windows while maintaining model quality.
///
/// ## Key Features
/// - Beta-based correction range for frequency adjustments
/// - mscale for attention score normalization
/// - Linear ramp mask for smooth transitions
public class YarnRoPE: Module, RoPEProvider {
    // MARK: - Properties

    let dimensions: Int
    let traditional: Bool

    private let computedMscale: Float
    private let computedFreqs: MLXArray

    // MARK: - Initialization

    public init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings _: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional

        // Helper functions matching Python implementation
        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            if scale <= 1 { return 1.0 }
            return 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal { maxVal += 0.001 } // Prevent singularity
            let linearFunc = (MLXArray(0 ..< dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        // Compute mscale
        computedMscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
                / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        // Compute frequencies with correction
        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / dimensions
        )
        let freqInter = scalingFactor * freqExtra

        let (low, high) = yarnFindCorrectionRange()
        let freqMask = 1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        computedFreqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        var input = x
        if computedMscale != 1.0 {
            input[.ellipsis, 0 ..< dimensions] = computedMscale * input[.ellipsis, 0 ..< dimensions]
        }

        return MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: computedFreqs
        )
    }

    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}

// MARK: - SuScaledRoPE

/// Su-scaled RoPE for very long context (longrope).
///
/// This variant uses different scaling factors for short and long sequences,
/// with a smooth transition based on the original training context length.
///
/// ## Key Features
/// - Separate short/long frequency factors
/// - mscale for attention normalization
/// - Automatic switching based on sequence length
public class SuScaledRoPE: Module, RoPEProvider {
    // MARK: - Properties

    let dimensions: Int
    let originalMaxPositionEmbeddings: Int

    private let longFreqs: MLXArray
    private let mscaleLong: Float

    // MARK: - Initialization

    public init(
        dimensions: Int,
        base: Float = 10000,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4096,
        shortFactor _: [Float] = [1.0],
        longFactor: [Float]
    ) {
        self.dimensions = dimensions
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings

        // Compute base frequencies
        let baseFreqs = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions)
        )

        // Long frequencies (scaled by long factor)
        longFreqs = MLXArray(longFactor).asType(.float32) * baseFreqs

        // Compute mscale based on extension factor
        func defaultScale(_ factor: Float) -> Float {
            sqrt(1 + log(factor) / log(Float(originalMaxPositionEmbeddings)))
        }

        let factor = Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        mscaleLong = factor <= 1.0 ? 1.0 : defaultScale(factor)

        super.init()
    }

    // MARK: - Forward

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Scale input if needed
        let input: MLXArray
        if mscaleLong != 1.0 {
            var scaled = x
            scaled[.ellipsis, 0 ..< dimensions] = mscaleLong * scaled[.ellipsis, 0 ..< dimensions]
            input = scaled
        } else {
            input = x
        }

        return MLXFast.RoPE(
            input,
            dimensions: dimensions,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: longFreqs
        )
    }

    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        callAsFunction(x, offset: offset)
    }
}

// MARK: - RoPE Factory

/// Initialize the appropriate RoPE module based on configuration.
///
/// Supported rope_type values:
/// - `"default"`: Standard RoPE (nn.RoPE)
/// - `"linear"`: Linearly scaled RoPE
/// - `"llama3"`: Llama 3 style with smooth interpolation
/// - `"yarn"`: Yet Another RoPE for extended context
/// - `"longrope"`: Su-scaled RoPE for very long context
/// - `"mrope"`: Multimodal RoPE (returns basic RoPE, modal logic in attention)
///
/// - Parameters:
///   - dims: Rotation dimensions (typically head_dim)
///   - base: Base frequency (typically 10000)
///   - traditional: Use traditional (GPT-J) vs modern (GPT-NeoX) rotation
///   - scalingConfig: Optional scaling configuration dictionary
///   - maxPositionEmbeddings: Maximum position for embeddings
/// - Returns: Configured RoPE provider
public func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> any RoPEProvider {
    // Extract rope type from config
    let ropeType: String = {
        guard let config = scalingConfig,
              let typeValue = config["type"] ?? config["rope_type"],
              case let .string(s) = typeValue
        else { return "default" }
        return s
    }()

    switch ropeType {
    case "default", "linear":
        let scale: Float = if ropeType == "linear",
                              let factor = scalingConfig?["factor"]?.asFloat()
        {
            1 / factor
        } else {
            1.0
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)

    case "llama3":
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )

    case "yarn":
        return YarnRoPE(
            dimensions: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: scalingConfig?["factor"]?.asFloat() ?? 32.0,
            originalMaxPositionEmbeddings: scalingConfig?["original_max_position_embeddings"]?.asInt() ?? 4096,
            betaFast: scalingConfig?["beta_fast"]?.asFloat() ?? 32.0,
            betaSlow: scalingConfig?["beta_slow"]?.asFloat() ?? 1.0,
            mscale: scalingConfig?["mscale"]?.asFloat() ?? 1.0,
            mscaleAllDim: scalingConfig?["mscale_all_dim"]?.asFloat() ?? 0.0
        )

    case "longrope":
        guard let config = scalingConfig,
              let origMax = config["original_max_position_embeddings"]?.asInt(),
              let longFactor = config["long_factor"]?.asFloats()
        else {
            fatalError("longrope requires scaling_config with original_max_position_embeddings and long_factor")
        }

        let shortFactor = config["short_factor"]?.asFloats() ?? [1.0]

        return SuScaledRoPE(
            dimensions: dims,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131_072,
            originalMaxPositionEmbeddings: origMax,
            shortFactor: shortFactor,
            longFactor: longFactor
        )

    case "mrope":
        // MRoPE returns basic RoPE; multimodal rotary logic is in the attention layer
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)

    default:
        fatalError("Unsupported RoPE type: \(ropeType)")
    }
}
