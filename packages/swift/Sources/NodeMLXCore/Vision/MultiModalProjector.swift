//
//  MultiModalProjector.swift
//  NodeMLXCore
//
//  Multi-Modal Projector for Gemma 3 VLM.
//  Projects vision embeddings into the language model's embedding space.
//
//  Based on HuggingFace transformers Gemma3MultiModalProjector
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Gemma3 RMSNorm (for projector)

/// RMSNorm with Gemma-style (1 + weight) scaling
public class ProjectorRMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma uses (1 + weight) scaling
        return MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

// MARK: - Multi-Modal Projector

/// Projects vision features into language model space
/// Uses average pooling to reduce patch count to mm_tokens_per_image (256 for Gemma 3)
public class Gemma3MultiModalProjector: Module {
    /// Linear projection weight (manual parameter, not wrapped Linear)
    @ModuleInfo(key: "mm_input_projection_weight") var projectionWeight: MLXArray

    /// RMSNorm before projection
    @ModuleInfo(key: "mm_soft_emb_norm") var softEmbNorm: ProjectorRMSNorm

    let patchesPerImage: Int
    let tokensPerSide: Int
    let kernelSize: Int

    /// Initialize the projector
    /// - Parameters:
    ///   - visionHiddenSize: Hidden size of vision encoder (e.g., 1152)
    ///   - textHiddenSize: Hidden size of text model (e.g., 2304)
    ///   - imageSize: Vision model image size (e.g., 896)
    ///   - patchSize: Vision model patch size (e.g., 14)
    ///   - mmTokensPerImage: Target number of image tokens (e.g., 256)
    ///   - layerNormEps: Layer norm epsilon
    public init(
        visionHiddenSize: Int,
        textHiddenSize: Int,
        imageSize: Int = 896,
        patchSize: Int = 14,
        mmTokensPerImage: Int = 256,
        layerNormEps: Float = 1e-6
    ) {
        // Calculate pooling parameters
        patchesPerImage = imageSize / patchSize  // 896/14 = 64
        tokensPerSide = Int(sqrt(Double(mmTokensPerImage)))  // sqrt(256) = 16
        kernelSize = patchesPerImage / tokensPerSide  // 64/16 = 4

        // Initialize projection weight to zeros (following HF init)
        _projectionWeight.wrappedValue = MLXArray.zeros([visionHiddenSize, textHiddenSize])

        // RMSNorm for soft embeddings
        _softEmbNorm.wrappedValue = ProjectorRMSNorm(dimensions: visionHiddenSize, eps: layerNormEps)
    }

    /// Initialize from configs
    public init(visionConfig: SiglipVisionConfiguration, textHiddenSize: Int, mmTokensPerImage: Int = 256) {
        patchesPerImage = visionConfig.patchesPerSide
        tokensPerSide = Int(sqrt(Double(mmTokensPerImage)))
        kernelSize = patchesPerImage / tokensPerSide

        _projectionWeight.wrappedValue = MLXArray.zeros([visionConfig.hiddenSize, textHiddenSize])
        _softEmbNorm.wrappedValue = ProjectorRMSNorm(dimensions: visionConfig.hiddenSize, eps: visionConfig.layerNormEps)
    }

    /// Project vision features to language model space
    /// - Parameter visionOutputs: Vision encoder output [B, num_patches, vision_hidden]
    /// - Returns: Projected features [B, mm_tokens_per_image, text_hidden]
    public func callAsFunction(_ visionOutputs: MLXArray) -> MLXArray {
        let batchSize = visionOutputs.dim(0)
        let seqLength = visionOutputs.dim(2)  // vision_hidden_size

        // Reshape for 2D pooling: [B, num_patches, hidden] -> [B, hidden, patches_h, patches_w]
        var reshaped = visionOutputs.transposed(0, 2, 1)  // [B, hidden, num_patches]
        reshaped = reshaped.reshaped([batchSize, seqLength, patchesPerImage, patchesPerImage])

        // Average pooling to reduce spatial dimensions
        // [B, hidden, 64, 64] -> [B, hidden, 16, 16] with kernel_size=4
        let pooled = avgPool2d(reshaped, kernelSize: kernelSize)

        // Flatten spatial dims: [B, hidden, tokens_h, tokens_w] -> [B, hidden, num_tokens]
        let flattened = pooled.reshaped([batchSize, seqLength, -1])

        // Transpose back: [B, hidden, num_tokens] -> [B, num_tokens, hidden]
        var output = flattened.transposed(0, 2, 1)

        // Apply RMSNorm
        output = softEmbNorm(output)

        // Project to text hidden size: [B, num_tokens, vision_hidden] @ [vision_hidden, text_hidden]
        output = matmul(output, projectionWeight)

        return output
    }
}

// MARK: - Average Pooling Helper

/// 2D Average Pooling
/// - Parameters:
///   - x: Input tensor [B, C, H, W]
///   - kernelSize: Pooling kernel size
/// - Returns: Pooled tensor [B, C, H/kernel, W/kernel]
private func avgPool2d(_ x: MLXArray, kernelSize: Int) -> MLXArray {
    let (B, C, H, W) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let newH = H / kernelSize
    let newW = W / kernelSize

    // Reshape to extract pooling windows
    // [B, C, H, W] -> [B, C, newH, kernel, newW, kernel]
    var reshaped = x.reshaped([B, C, newH, kernelSize, newW, kernelSize])

    // Move kernel dims together and compute mean
    // [B, C, newH, kernel, newW, kernel] -> [B, C, newH, newW, kernel, kernel]
    reshaped = reshaped.transposed(0, 1, 2, 4, 3, 5)

    // Reshape to [B, C, newH, newW, kernel*kernel] and mean over last dim
    reshaped = reshaped.reshaped([B, C, newH, newW, kernelSize * kernelSize])
    let pooled = reshaped.mean(axis: -1)

    return pooled
}
