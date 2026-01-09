//
//  SiglipVision.swift
//  NodeMLXCore
//
//  SigLIP Vision Encoder for Gemma 3 VLM.
//  Converts images into visual embeddings.
//
//  Based on:
//  - HuggingFace transformers (Apache 2.0): models/siglip/modeling_siglip.py
//  - mlx-vlm (MIT): models/siglip/siglip.py
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct SiglipVisionConfiguration: Decodable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numChannels: Int
    public var imageSize: Int
    public var patchSize: Int
    public var layerNormEps: Float
    public var hiddenAct: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 27
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 896
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
    }

    public init(
        hiddenSize: Int = 1152,
        intermediateSize: Int = 4304,
        numHiddenLayers: Int = 27,
        numAttentionHeads: Int = 16,
        numChannels: Int = 3,
        imageSize: Int = 896,
        patchSize: Int = 14,
        layerNormEps: Float = 1e-6,
        hiddenAct: String = "gelu_pytorch_tanh"
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numChannels = numChannels
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.layerNormEps = layerNormEps
        self.hiddenAct = hiddenAct
    }

    /// Number of patches per side
    public var patchesPerSide: Int {
        imageSize / patchSize
    }

    /// Total number of patches
    public var numPatches: Int {
        patchesPerSide * patchesPerSide
    }

    /// Head dimension
    public var headDim: Int {
        hiddenSize / numAttentionHeads
    }
}

// MARK: - Vision Embeddings

/// Converts image pixels to patch embeddings with positional encoding
public class SiglipVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let numPatches: Int

    public init(_ config: SiglipVisionConfiguration) {
        numPatches = config.numPatches

        // Conv2d to extract patches: [B, C, H, W] -> [B, hidden, patches, patches]
        let patchSize = config.patchSize
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair((patchSize, patchSize)),
            stride: IntOrPair((patchSize, patchSize)),
            padding: IntOrPair((0, 0))
        )

        // Learnable position embeddings for each patch
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPatches,
            dimensions: config.hiddenSize
        )
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // pixelValues: [B, C, H, W] or [B, H, W, C]
        var x = pixelValues

        // MLX Conv2d expects NHWC format
        if x.dim(1) == 3, x.dim(2) == x.dim(3) {
            // Convert NCHW to NHWC
            x = x.transposed(0, 2, 3, 1)
        }

        // Apply patch embedding: [B, H, W, C] -> [B, patches_h, patches_w, hidden]
        let patchEmbeds = patchEmbedding(x)

        // Flatten patches: [B, ph, pw, hidden] -> [B, num_patches, hidden]
        let batchSize = patchEmbeds.dim(0)
        let hiddenSize = patchEmbeds.dim(3)
        var embeddings = patchEmbeds.reshaped([batchSize, -1, hiddenSize])

        // Add position embeddings
        let positionIds = MLXArray(0 ..< numPatches)
        embeddings = embeddings + positionEmbedding(positionIds)

        return embeddings
    }
}

// MARK: - Attention

/// Multi-head self-attention for vision
public class SiglipAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    public init(_ config: SiglipVisionConfiguration) {
        numHeads = config.numAttentionHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)

        let hiddenSize = config.hiddenSize
        _qProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _kProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _vProj.wrappedValue = Linear(hiddenSize, hiddenSize)
        _outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Project to Q, K, V
        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        // Reshape: [B, L, hidden] -> [B, L, heads, headDim] -> [B, heads, L, headDim]
        queries = queries.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Scaled dot-product attention (no causal mask for vision)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )

        // Reshape back: [B, heads, L, headDim] -> [B, L, hidden]
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])

        return outProj(outputReshaped)
    }
}

// MARK: - MLP

/// Feed-forward network with GELU activation
public class SiglipMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    let useApproxGelu: Bool

    public init(_ config: SiglipVisionConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)

        // Check if using approximate GELU (pytorch_tanh variant)
        useApproxGelu = config.hiddenAct.contains("tanh")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = fc1(x)
        h = useApproxGelu ? geluApproximate(h) : gelu(h)
        return fc2(h)
    }
}

// MARK: - Encoder Layer

/// Single transformer encoder layer
public class SiglipEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SiglipAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SiglipMLP

    public init(_ config: SiglipVisionConfiguration) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _selfAttn.wrappedValue = SiglipAttention(config)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = SiglipMLP(config)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Pre-norm self-attention
        var residual = hiddenStates
        var h = layerNorm1(hiddenStates)
        h = selfAttn(h)
        h = residual + h

        // Pre-norm MLP
        residual = h
        h = layerNorm2(h)
        h = mlp(h)
        h = residual + h

        return h
    }
}

// MARK: - Vision Encoder

/// Full SigLIP vision encoder
public class SiglipEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [SiglipEncoderLayer]

    public init(_ config: SiglipVisionConfiguration) {
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            SiglipEncoderLayer(config)
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

// MARK: - Vision Model

/// Complete SigLIP Vision Model
public class SiglipVisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SiglipVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: SiglipEncoder
    @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

    let config: SiglipVisionConfiguration

    public init(_ config: SiglipVisionConfiguration) {
        self.config = config
        _embeddings.wrappedValue = SiglipVisionEmbeddings(config)
        _encoder.wrappedValue = SiglipEncoder(config)
        _postLayernorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// Forward pass
    /// - Parameter pixelValues: Image tensor [B, C, H, W] or [B, H, W, C]
    /// - Returns: Hidden states [B, num_patches, hidden_size]
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hiddenStates = embeddings(pixelValues)
        hiddenStates = encoder(hiddenStates)
        hiddenStates = postLayernorm(hiddenStates)
        return hiddenStates
    }
}
