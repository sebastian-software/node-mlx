// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// AltUp (Alternating Updates) block for efficient sparse computation.
// Used by Gemma3n and potentially future models with similar architecture.

import Foundation
import MLX
import MLXFast
import MLXNN

/// AltUp (Alternating Updates) module for efficient sparse computation.
///
/// AltUp reduces computation by maintaining multiple "virtual" hidden states
/// but only computing attention/MLP on one active state at a time.
/// The predict step spreads information, and the correct step refines predictions.
///
/// Architecture:
/// 1. **Predict**: Use learned coefficients to predict inactive states from active
/// 2. **Activate**: Run attention/MLP on the active state only
/// 3. **Correct**: Refine predictions based on the activated output
///
/// This allows N× throughput improvement with minimal quality loss.
public class AltUpBlock<Config: AltUpConfiguration>: Module {
    public let numInputs: Int
    public let activeIdx: Int
    public let hiddenSize: Int
    public let altupCoefClip: Float?

    @ModuleInfo(key: "correct_output_scale") public var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") public var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") public var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") public var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") public var routerNorm: RMSNorm

    public init(_ config: Config) {
        numInputs = config.altupNumInputs
        activeIdx = config.altupActiveIdx
        hiddenSize = config.hiddenSize
        altupCoefClip = config.altupCoefClip

        _correctOutputScale.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _correctionCoefs.wrappedValue = Linear(numInputs, numInputs, bias: false)
        _predictionCoefs.wrappedValue = Linear(numInputs, numInputs * numInputs, bias: false)
        _modalityRouter.wrappedValue = Linear(config.hiddenSize, numInputs, bias: false)
        _routerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Compute router modalities from input hidden states.
    public func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        let scale = Foundation.pow(Float(hiddenSize), -1.0)
        let routerInputs = routerNorm(x) * scale
        let routed = modalityRouter(routerInputs).asType(.float32)
        return tanh(routed)
    }

    /// Predict step: modifies input using learned coefficients.
    ///
    /// - Parameter hiddenStates: [numInputs, batch, seq, hidden]
    /// - Returns: Predictions [numInputs, batch, seq, hidden]
    public func predict(_ hiddenStates: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(hiddenStates[activeIdx])

        // Compute prediction coefficients with optional clipping
        var weight = predictionCoefs.weight.asType(.float32)
        if let clipVal = altupCoefClip {
            weight = clip(weight, min: -clipVal, max: clipVal)
        }

        // Manual linear: modalities @ weight.T
        var allCoefs = matmul(modalities.asType(.float32), weight.T)
        let shape = modalities.shape
        allCoefs = allCoefs.reshaped([shape[0], shape[1], numInputs, numInputs])
        allCoefs = allCoefs.transposed(0, 1, 3, 2)

        // Convert to float32 for better precision
        let xUp = hiddenStates.asType(.float32)
        let xPermuted = xUp.transposed(1, 2, 3, 0)
        var predictions = matmul(xPermuted, allCoefs)
        predictions = predictions.transposed(3, 0, 1, 2)
        predictions = predictions + xUp

        return predictions.asType(hiddenStates.dtype)
    }

    /// Correct step: refines predictions based on activated output.
    ///
    /// - Parameters:
    ///   - predictions: Predicted states [numInputs, batch, seq, hidden]
    ///   - activated: Output from attention/MLP [batch, seq, hidden]
    /// - Returns: Corrected states [numInputs, batch, seq, hidden]
    public func correct(_ predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)

        // Compute correction coefficients with optional clipping
        var weight = correctionCoefs.weight.asType(.float32)
        if let clipVal = altupCoefClip {
            weight = clip(weight, min: -clipVal, max: clipVal)
        }

        // Manual linear + 1.0: modalities @ weight.T + 1.0
        var allCoefs = matmul(modalities.asType(.float32), weight.T) + 1.0
        let activeX = predictions[activeIdx]
        let innovation = activated - activeX

        // allCoefs: [batch, seq, numInputs] -> [numInputs, batch, seq]
        allCoefs = allCoefs.transposed(2, 0, 1)

        // innovation: [batch, seq, hidden]
        // Broadcast: [numInputs, batch, seq, 1] * [1, batch, seq, hidden]
        let innovationExpanded = innovation.expandedDimensions(axis: 0)
        let allCoefsExpanded = allCoefs.expandedDimensions(axis: -1)
        let corrected = innovationExpanded * allCoefsExpanded + predictions

        return corrected.asType(activated.dtype)
    }

    /// Scale the correction output (used when altupCorrectScale is enabled).
    public func scaleCorrectOutput(_ corrected: MLXArray) -> MLXArray {
        corrected * correctOutputScale
    }
}
