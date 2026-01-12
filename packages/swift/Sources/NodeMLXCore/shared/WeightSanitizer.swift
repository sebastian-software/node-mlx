// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Common weight sanitization logic shared by all models.

import MLX

/// Standard weight sanitization for LLM models.
///
/// Handles common patterns:
/// - Removing "language_model." prefix (for VLM models)
/// - Filtering out vision/audio components
/// - Tied embeddings (copying embed_tokens to lm_head)
public func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]

    for (key, value) in weights {
        var newKey = key

        // Handle VLM prefix patterns
        if newKey.hasPrefix("language_model.model.") {
            newKey = "model." + String(newKey.dropFirst("language_model.model.".count))
        } else if newKey.hasPrefix("language_model.lm_head.") {
            newKey = "lm_head." + String(newKey.dropFirst("language_model.lm_head.".count))
        } else if newKey.hasPrefix("language_model.") {
            newKey = String(newKey.dropFirst("language_model.".count))
        }

        // Skip vision/audio/multimodal components
        if newKey.contains("vision_tower") ||
            newKey.contains("audio_tower") ||
            newKey.contains("multi_modal_projector")
        {
            continue
        }

        result[newKey] = value
    }

    // Handle tied embeddings: if lm_head.weight is missing, copy from embed_tokens
    if result["lm_head.weight"] == nil {
        for suffix in ["weight", "scales", "biases"] {
            if let embedWeight = result["model.embed_tokens.\(suffix)"] {
                result["lm_head.\(suffix)"] = embedWeight
            }
        }
    }

    return result
}
