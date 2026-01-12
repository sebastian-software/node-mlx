// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// MoE (Mixture of Experts) weight sanitization utilities.
// Used by GPT-OSS and similar MoE architectures.

import Foundation
import MLX
import MLXNN

// MARK: - MoE Weight Sanitizer

/// Utilities for sanitizing MoE model weights.
///
/// Handles:
/// - Packed tensor format (blocks + scales) → unpacked bfloat16
/// - Fused gate_up_proj → separate gate_proj + up_proj
/// - Weight key transformations for MLXNN compatibility
public enum MoESanitizer {
    /// Convert packed MoE tensors from blocks+scales format to unpacked bfloat16.
    ///
    /// The packed format uses a 4-bit lookup table encoding with separate scale factors.
    /// This function unpacks them into standard bfloat16 weights.
    ///
    /// - Parameters:
    ///   - blocks: Packed weight blocks
    ///   - scales: Scale factors for each block
    /// - Returns: Unpacked weights in bfloat16 format
    public static func convertPackedTensors(blocks: MLXArray, scales: MLXArray) -> MLXArray {
        precondition(
            blocks.shape.dropLast() == scales.shape,
            "blocks.shape=\(blocks.shape) does not match scales.shape=\(scales.shape)"
        )

        var scales = scales.asType(.int32) - 127
        let lut = MLXArray([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ]).asType(.bfloat16)

        let (prefixShape, G, B) = (Array(blocks.shape.dropLast(2)), blocks.dim(-2), blocks.dim(-1))

        let blocks = blocks.reshaped(-1, B)
        scales = scales.reshaped(-1, 1)

        let idxLo = blocks & 0x0F
        let idxHi = blocks >> 4

        var out = stacked([lut[idxLo], lut[idxHi]], axis: -1).flattened(start: -2)
        out = (2.0 ** scales) * out
        out = out.reshaped(prefixShape + [G * B * 2])
        return out.asType(.bfloat16)
    }

    /// Sanitize MoE model weights for MLXNN compatibility.
    ///
    /// Performs the following transformations:
    /// 1. Unpacks packed tensors (blocks + scales) if present
    /// 2. Splits fused gate_up_proj into separate gate_proj and up_proj
    /// 3. Transforms weight keys to match MLXNN module expectations
    ///
    /// - Parameter weights: Raw weights dictionary from model file
    /// - Returns: Sanitized weights ready for MLXNN module loading
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        // Check if already in expected format
        if weights.keys.contains(where: { $0.contains("gate_proj.weight") }) {
            return weights
        }

        // Handle packed MoE tensor format (blocks + scales)
        if weights.keys.contains(where: { $0.contains("gate_up_proj_scales") }) {
            var newWeights: [String: MLXArray] = [:]
            for (k, v) in weights {
                if k.hasSuffix("_scales") {
                    continue
                } else if k.hasSuffix("_blocks") {
                    let scaleKey = k.replacingOccurrences(of: "_blocks", with: "_scales")
                    if let scales = weights[scaleKey] {
                        let newV = convertPackedTensors(blocks: v, scales: scales)
                        let newK = k.replacingOccurrences(of: "_blocks", with: "")
                        newWeights[newK] = newV
                    }
                } else {
                    newWeights[k] = v
                }
            }
            weights = newWeights
        }

        // Transform weight keys to expected format
        var finalWeights: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("gate_up_proj"), !k.contains("bias") {
                // Split interleaved gate_up_proj into separate gate_proj and up_proj
                finalWeights[k.replacingOccurrences(of: "gate_up_proj", with: "gate_proj.weight")] =
                    v[.ellipsis, .stride(by: 2), 0...]
                finalWeights[k.replacingOccurrences(of: "gate_up_proj", with: "up_proj.weight")] =
                    v[.ellipsis, .stride(from: 1, by: 2), 0...]
            } else if k.contains("down_proj"), !k.contains("bias") {
                finalWeights[k.replacingOccurrences(of: "down_proj", with: "down_proj.weight")] = v
            } else if k.contains("gate_up_proj_bias") {
                finalWeights[k.replacingOccurrences(of: "gate_up_proj_bias", with: "gate_proj.bias")] =
                    v[.ellipsis, .stride(by: 2)]
                finalWeights[k.replacingOccurrences(of: "gate_up_proj_bias", with: "up_proj.bias")] =
                    v[.ellipsis, .stride(from: 1, by: 2)]
            } else if k.contains("down_proj_bias") {
                finalWeights[k.replacingOccurrences(of: "down_proj_bias", with: "down_proj.bias")] = v
            } else {
                finalWeights[k] = v
            }
        }

        return finalWeights
    }
}
