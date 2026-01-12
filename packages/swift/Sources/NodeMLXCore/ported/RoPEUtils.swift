// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from mlx-lm (https://github.com/ml-explore/mlx-lm)
// Original: mlx_lm/models/rope_utils.py

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - RoPE Provider Protocol

/// Protocol for all RoPE (Rotary Position Embedding) implementations.
///
/// RoPE variants provide position information by rotating query/key vectors
/// at different frequencies depending on their position in the sequence.
public protocol RoPEProvider {
    /// Applies rotary position embedding to the input tensor.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape [B, H, S, D]
    ///   - offset: Position offset for cached sequence
    /// - Returns: Tensor with rotary embeddings applied
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray
}

// MARK: - Su Scaled RoPE (longrope)

/// Su Scaled Rotary Position Embedding for extended context.
///
/// Uses scaling factors to extend the effective context length beyond
/// the original training length. Primarily used for "longrope" models.
///
/// Ported from: mlx_lm/models/rope_utils.py::SuScaledRoPE
public final class SuScaledRoPE: Module, RoPEProvider {
    private let dim: Int
    private let freqs: MLXArray
    private let scale: Float

    /// Creates a Su-scaled RoPE layer.
    ///
    /// - Parameters:
    ///   - dims: Feature dimensions to rotate
    ///   - base: Base frequency (default: 10000)
    ///   - maxPositionEmbeddings: Extended context length (default: 131072)
    ///   - originalMaxPositionEmbeddings: Original training length (default: 4096)
    ///   - longFactor: Scaling factors for extended positions
    ///   - longMscale: Optional explicit magnitude scale
    public init(
        dims: Int,
        base: Float = 10000.0,
        maxPositionEmbeddings: Int = 131_072,
        originalMaxPositionEmbeddings: Int = 4096,
        longFactor: [Float] = [1.0],
        longMscale: Float? = nil
    ) {
        dim = dims

        // Compute base frequencies
        let indices = MLXArray(stride(from: Float(0), to: Float(dims), by: 2))
        let baseFreqs = pow(Float(base), indices / Float(dims))

        // Apply long scaling factors
        let factors = MLXArray(longFactor)
        freqs = factors * baseFreqs

        // Compute magnitude scale
        let factor = Float(maxPositionEmbeddings) / Float(originalMaxPositionEmbeddings)
        if let mscale = longMscale {
            scale = mscale
        } else if factor <= 1.0 {
            scale = 1.0
        } else {
            // Default scale: sqrt(1 + log(factor) / log(original))
            scale = sqrt(1.0 + log(factor) / log(Float(originalMaxPositionEmbeddings)))
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // Scale the rotated dimensions
        var result = x
        result[.ellipsis, ..<dim] = scale * x[.ellipsis, ..<dim]

        return MLXFast.RoPE(
            result,
            dimensions: dim,
            traditional: false,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}

// MARK: - Llama3 RoPE

/// Llama 3 style RoPE with smooth frequency interpolation.
///
/// Uses a frequency-dependent scaling scheme that smoothly interpolates
/// between scaled and unscaled frequencies based on wavelength.
///
/// Ported from: mlx_lm/models/rope_utils.py::Llama3RoPE
public final class Llama3RoPE: Module, RoPEProvider {
    private let dims: Int
    private let traditional: Bool
    private let freqs: MLXArray

    /// Creates a Llama3-style RoPE layer.
    ///
    /// - Parameters:
    ///   - dims: Feature dimensions to rotate
    ///   - maxPositionEmbeddings: Maximum sequence length
    ///   - traditional: Use traditional RoPE formulation
    ///   - base: Base frequency
    ///   - scalingConfig: Configuration with factor and freq parameters
    public init(
        dims: Int,
        maxPositionEmbeddings _: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000.0,
        scalingConfig: [String: Any]
    ) {
        self.dims = dims
        self.traditional = traditional

        // Extract scaling parameters
        let factor = (scalingConfig["factor"] as? Double).map { Float($0) } ?? 1.0
        let lowFreqFactor = (scalingConfig["low_freq_factor"] as? Double).map { Float($0) } ?? 1.0
        let highFreqFactor = (scalingConfig["high_freq_factor"] as? Double).map { Float($0) } ?? 4.0
        let oldContextLen = (scalingConfig["original_max_position_embeddings"] as? Int) ?? 8192

        // Calculate wavelength boundaries
        let lowFreqWavelen = Float(oldContextLen) / lowFreqFactor
        let highFreqWavelen = Float(oldContextLen) / highFreqFactor

        // Compute base frequencies
        let indices = MLXArray(stride(from: Float(0), to: Float(dims), by: 2))
        var baseFreqs = pow(Float(base), indices / Float(dims))
        let wavelens = 2.0 * Float.pi * baseFreqs

        // Apply frequency-dependent scaling
        // Low frequencies (long wavelength) get full factor scaling
        baseFreqs = which(wavelens .> lowFreqWavelen, baseFreqs * factor, baseFreqs)

        // Medium frequencies get smooth interpolation
        let isMediumFreq = logicalAnd(wavelens .> highFreqWavelen, wavelens .< lowFreqWavelen)
        let smoothFactors = (Float(oldContextLen) / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = baseFreqs / ((1.0 - smoothFactors) / factor + smoothFactors)

        freqs = which(isMediumFreq, smoothFreqs, baseFreqs)

        super.init()
    }

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
}

// MARK: - Yarn RoPE

/// Yet Another RoPE for extended context windows.
///
/// Uses a more sophisticated frequency interpolation scheme with
/// configurable beta parameters and magnitude scaling.
///
/// Ported from: mlx_lm/models/rope_utils.py::YarnRoPE
public final class YarnRoPE: Module, RoPEProvider {
    private let dims: Int
    private let traditional: Bool
    private let freqs: MLXArray
    private let mscale: Float

    /// Creates a YARN RoPE layer.
    ///
    /// - Parameters:
    ///   - dims: Feature dimensions to rotate
    ///   - traditional: Use traditional RoPE formulation
    ///   - maxPositionEmbeddings: Maximum sequence length
    ///   - base: Base frequency
    ///   - scalingFactor: Context extension factor
    ///   - originalMaxPositionEmbeddings: Original training length
    ///   - betaFast: High frequency correction parameter
    ///   - betaSlow: Low frequency correction parameter
    ///   - mscale: Magnitude scaling factor
    ///   - mscaleAllDim: Dimension-wide magnitude scaling
    public init(
        dims: Int,
        traditional: Bool = false,
        maxPositionEmbeddings _: Int = 2048,
        base: Float = 10000.0,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32.0,
        betaSlow: Float = 1.0,
        mscale: Float = 1.0,
        mscaleAllDim: Float = 0.0
    ) {
        self.dims = dims
        self.traditional = traditional

        // Helper functions
        func yarnFindCorrectionDim(_ numRotations: Float) -> Float {
            Float(dims) * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2.0 * Float.pi)) / (2.0 * log(base))
        }

        func yarnFindCorrectionRange() -> (Int, Int) {
            let low = Int(floor(yarnFindCorrectionDim(betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(betaSlow)))
            return (max(low, 0), min(high, dims - 1))
        }

        func yarnGetMscale(scale: Float, m: Float) -> Float {
            if scale <= 1.0 {
                return 1.0
            }
            return 0.1 * m * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxV = maxVal
            if minVal == maxVal {
                maxV += 0.001 // Prevent singularity
            }
            let indices = MLXArray(0 ..< Int32(dim)).asType(.float32)
            let linearFunc = (indices - minVal) / (maxV - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        // Compute mscale
        self.mscale = yarnGetMscale(scale: scalingFactor, m: mscale) / yarnGetMscale(scale: scalingFactor, m: mscaleAllDim)

        // Compute frequencies
        let indices = MLXArray(stride(from: Float(0), to: Float(dims), by: 2))
        let freqExtra = pow(Float(base), indices / Float(dims))
        let freqInter = scalingFactor * freqExtra

        let (low, high) = yarnFindCorrectionRange()
        let freqMask = 1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dims / 2)

        freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1.0 - freqMask))

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        var result = x
        if mscale != 1.0 {
            result[.ellipsis, ..<dims] = mscale * x[.ellipsis, ..<dims]
        }
        return MLXFast.RoPE(
            result,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: freqs
        )
    }
}

// MARK: - Standard RoPE Wrapper

/// Standard RoPE implementation wrapper conforming to RoPEProvider.
public final class StandardRoPE: Module, RoPEProvider {
    @ModuleInfo private var rope: RoPE

    public init(dims: Int, traditional: Bool = false, base: Float = 10000.0, scale: Float = 1.0) {
        _rope = ModuleInfo(wrappedValue: RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        rope(x, offset: offset)
    }
}

// MARK: - Factory Function

/// Initializes the appropriate RoPE implementation based on configuration.
///
/// Supported rope_type values:
/// - "default": Standard RoPE
/// - "linear": Linearly scaled (scale = 1/factor)
/// - "llama3": Llama 3 with smooth frequency interpolation
/// - "yarn": YARN for extended context
/// - "longrope": Su-scaled for very long context
/// - "mrope": Multimodal (returns standard RoPE)
///
/// - Parameters:
///   - dims: Feature dimensions to rotate
///   - base: Base frequency
///   - traditional: Use traditional RoPE formulation
///   - scalingConfig: Optional configuration dictionary
///   - maxPositionEmbeddings: Maximum sequence length
/// - Returns: Configured RoPE implementation
public func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: Any]? = nil,
    maxPositionEmbeddings: Int? = nil
) -> any RoPEProvider {
    let ropeType: String = if let config = scalingConfig {
        (config["type"] as? String) ?? (config["rope_type"] as? String) ?? "default"
    } else {
        "default"
    }

    switch ropeType {
    case "default":
        return StandardRoPE(dims: dims, traditional: traditional, base: base, scale: 1.0)

    case "linear":
        let factor = (scalingConfig?["factor"] as? Double).map { Float($0) } ?? 1.0
        return StandardRoPE(dims: dims, traditional: traditional, base: base, scale: 1.0 / factor)

    case "llama3":
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig ?? [:]
        )

    case "yarn":
        let factor = (scalingConfig?["factor"] as? Double).map { Float($0) } ?? 1.0
        let origMax = (scalingConfig?["original_max_position_embeddings"] as? Int) ?? 4096
        let betaFast = (scalingConfig?["beta_fast"] as? Double).map { Float($0) } ?? 32.0
        let betaSlow = (scalingConfig?["beta_slow"] as? Double).map { Float($0) } ?? 1.0
        let mscale = (scalingConfig?["mscale"] as? Double).map { Float($0) } ?? 1.0
        let mscaleAllDim = (scalingConfig?["mscale_all_dim"] as? Double).map { Float($0) } ?? 0.0

        return YarnRoPE(
            dims: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: factor,
            originalMaxPositionEmbeddings: origMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )

    case "longrope":
        guard let config = scalingConfig else {
            fatalError("longrope requires scaling configuration")
        }
        let origMax = config["original_max_position_embeddings"] as? Int ?? 4096
        let longFactor = config["long_factor"] as? [Double] ?? [1.0]

        return SuScaledRoPE(
            dims: dims,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131_072,
            originalMaxPositionEmbeddings: origMax,
            longFactor: longFactor.map { Float($0) }
        )

    case "mrope":
        // MRoPE: multimodal, position handling in attention
        return StandardRoPE(dims: dims, traditional: traditional, base: base)

    default:
        fatalError("Unsupported RoPE type: \(ropeType)")
    }
}
