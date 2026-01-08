//
//  Gemma3nSpecific.swift
//  NodeMLXCore
//
//  Gemma 3n specific architecture components.
//  These are unique to Gemma 3n and not found in standard transformers.
//
//  Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modeling_gemma3n.py
//  Based on: "Alternating Updates for Efficient Transformers" (NeurIPS 2023)
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Gemma3n RMSNorm (with optional scale)

/// RMSNorm variant that uses the actual weight dimension from loaded weights
class Gemma3nRMSNorm: Module {
    let eps: Float
    @ModuleInfo(key: "weight") var weight: MLXArray
    
    init(dimensions: Int, eps: Float = 1e-6, withScale: Bool = true) {
        self.eps = eps
        // Initialize with given dimensions - will be overwritten by loaded weights
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Use the actual weight dimensions, not the initialized ones
        let dims = weight.dim(-1)
        let inputDim = x.dim(-1)
        
        // If dimensions match, apply norm directly
        if inputDim == dims {
            let variance = mean(x.pow(2), axis: -1, keepDims: true)
            let normalized = x * rsqrt(variance + eps)
            return normalized * weight
        }
        
        // If input is multi-dimensional (e.g., per-head), reshape weight for broadcast
        let variance = mean(x.pow(2), axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + eps)
        
        // Broadcast weight to match input shape
        if inputDim % dims == 0 {
            // Input has multiple heads, weight is per-head
            return normalized * weight
        }
        
        // Fallback: just normalize without scaling
        return normalized
    }
}

// MARK: - Scaled Word Embedding

/// Embedding that scales output by sqrt(hidden_size)
/// Supports both regular and quantized weights
class Gemma3nTextScaledWordEmbedding: Module, Quantizable {
    // Use @ModuleInfo so quantize() can find and replace this
    @ModuleInfo(key: "inner") var innerEmbedding: Embedding
    let embedScale: Float
    let embeddingCount: Int
    let dimensions: Int
    
    init(embeddingCount: Int, dimensions: Int, embedScale: Float = 1.0) {
        self.embedScale = embedScale
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self._innerEmbedding.wrappedValue = Embedding(embeddingCount: embeddingCount, dimensions: dimensions)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return innerEmbedding(x) * embedScale
    }
    
    // Quantizable protocol implementation
    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        return QuantizedGemma3nTextScaledWordEmbedding(
            from: self,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

/// Quantized version of Gemma3nTextScaledWordEmbedding
class QuantizedGemma3nTextScaledWordEmbedding: Module, Quantized {
    @ModuleInfo(key: "inner") var innerEmbedding: QuantizedEmbedding
    let embedScale: Float
    
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    
    init(from source: Gemma3nTextScaledWordEmbedding, groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.embedScale = source.embedScale
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._innerEmbedding.wrappedValue = QuantizedEmbedding(
            source.innerEmbedding,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return innerEmbedding(x) * embedScale
    }
}

// MARK: - Laurel Block (Learned Augmented Residual Layer)

/// Low-rank residual layer that adds a compressed representation path
class Gemma3nTextLaurelBlock: Module {
    @ModuleInfo(key: "linear_left") var linearLeft: Linear
    @ModuleInfo(key: "linear_right") var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") var postLaurelNorm: Gemma3nRMSNorm
    
    init(hiddenSize: Int, laurelRank: Int, eps: Float) {
        self._linearLeft.wrappedValue = Linear(hiddenSize, laurelRank, bias: false)
        self._linearRight.wrappedValue = Linear(laurelRank, hiddenSize, bias: false)
        self._postLaurelNorm.wrappedValue = Gemma3nRMSNorm(dimensions: hiddenSize, eps: eps)
    }
    
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var laurel = linearLeft(hiddenStates)
        laurel = linearRight(laurel)
        laurel = postLaurelNorm(laurel)
        return hiddenStates + laurel
    }
}

// MARK: - AltUp (Alternating Updates)

/// Alternating Updates module for efficient sparse computation
/// See: https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
class Gemma3nTextAltUp: Module {
    let numInputs: Int
    let activeIdx: Int
    let correctScale: Bool
    let hiddenSize: Int
    
    @ModuleInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: Gemma3nRMSNorm
    
    let routerInputScale: Float
    
    init(hiddenSize: Int, numInputs: Int, activeIdx: Int, correctScale: Bool, eps: Float) {
        self.hiddenSize = hiddenSize
        self.numInputs = numInputs
        self.activeIdx = activeIdx
        self.correctScale = correctScale
        self.routerInputScale = pow(Float(hiddenSize), -1.0)
        
        self._correctOutputScale.wrappedValue = MLXArray.zeros([hiddenSize])
        self._correctionCoefs.wrappedValue = Linear(numInputs, numInputs, bias: false)
        self._predictionCoefs.wrappedValue = Linear(numInputs, numInputs * numInputs, bias: false)
        self._modalityRouter.wrappedValue = Linear(hiddenSize, numInputs, bias: false)
        self._routerNorm.wrappedValue = Gemma3nRMSNorm(dimensions: hiddenSize, eps: eps)
    }
    
    /// Compute router modalities from input
    func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        let routerInputs = routerNorm(x) * routerInputScale
        let routed = modalityRouter(routerInputs)
        return tanh(routed)
    }
    
    /// Predict step: transform hidden_states using learned coefficients
    /// Input: [numInputs, batch, seq, hidden] -> Output: [numInputs, batch, seq, hidden]
    func predict(_ hiddenStates: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(hiddenStates[activeIdx])
        
        // Compute prediction coefficients: [batch, seq, numInputs] -> [batch, seq, numInputs, numInputs]
        var allCoefs = predictionCoefs(modalities)
        let shape = modalities.shape
        allCoefs = allCoefs.reshaped([shape[0], shape[1], numInputs, numInputs])
        allCoefs = allCoefs.transposed(0, 1, 3, 2)  // Transpose last two dims
        
        // Apply coefficients: matmul with permuted hidden states
        // hiddenStates: [numInputs, batch, seq, hidden] -> [batch, seq, hidden, numInputs]
        let hPermuted = hiddenStates.transposed(1, 2, 3, 0)
        var predictions = matmul(hPermuted, allCoefs)
        predictions = predictions.transposed(3, 0, 1, 2)  // Back to [numInputs, batch, seq, hidden]
        
        // Add residual
        return predictions + hiddenStates
    }
    
    /// Correct step: refine predictions based on activated output
    func correct(_ predictions: MLXArray, activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)
        
        // Innovation = difference between activated and prediction at active index
        let innovation = activated - predictions[activeIdx]
        
        // Compute correction coefficients
        var allCoefs = correctionCoefs(modalities) + 1.0
        allCoefs = allCoefs.transposed(2, 0, 1).expandedDimensions(axis: -1)
        
        // Apply correction: repeat innovation for each altup input
        var innovationRepeated = innovation.expandedDimensions(axis: 0)
        innovationRepeated = MLXArray.repeated(innovationRepeated, count: numInputs, axis: 0)
        
        let corrected = innovationRepeated * allCoefs + predictions
        return corrected
    }
    
    /// Scale corrected output if configured
    func scaleCorrectOutput(_ corrected: MLXArray) -> MLXArray {
        return corrected * correctOutputScale
    }
}

// MARK: - Gemma3n Text Attention

class Gemma3nTextAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma3nRMSNorm
    
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let numKVGroups: Int
    let rope: RoPE
    
    init(hiddenSize: Int, numHeads: Int, numKVHeads: Int, headDim: Int, queryPreAttnScalar: Int?, ropeTheta: Float, eps: Float) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.numKVGroups = numHeads / numKVHeads
        
        // Gemma3n uses queryPreAttnScalar for scaling
        if let scalar = queryPreAttnScalar {
            self.scale = 1.0 / sqrt(Float(scalar))
        } else {
            self.scale = 1.0 / sqrt(Float(headDim))
        }
        
        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim
        
        self._qProj.wrappedValue = Linear(hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, hiddenSize, bias: false)
        
        // Per-head norms
        self._qNorm.wrappedValue = Gemma3nRMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = Gemma3nRMSNorm(dimensions: headDim, eps: eps)
        
        // RoPE with Gemma3n's high theta
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
    }
    
    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray? = nil,
        cache: inout KVCache?
    ) -> MLXArray {
        let (B, L, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))
        
        // Project to Q, K, V
        var queries = qProj(hiddenStates).reshaped([B, L, numHeads, headDim])
        var keys = kProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
        var values = vProj(hiddenStates).reshaped([B, L, numKVHeads, headDim])
        
        // Apply per-head norms (Gemma3n normalizes Q and K)
        queries = qNorm(queries)
        keys = kNorm(keys)
        
        // Transpose for attention: [B, heads, L, headDim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        
        // Apply RoPE with cache offset
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)
        
        // Update cache and get full key/value history
        var cachedKeys = keys
        var cachedValues = values
        if let c = cache {
            (cachedKeys, cachedValues) = c.update(keys: keys, values: values)
        }
        
        // Expand KV heads if needed (GQA)
        if numKVGroups > 1 {
            cachedKeys = MLXArray.repeated(cachedKeys, count: numKVGroups, axis: 1)
            cachedValues = MLXArray.repeated(cachedValues, count: numKVGroups, axis: 1)
        }
        
        // Scaled dot-product attention
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask != nil ? .array(mask!) : .none
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: maskMode
        )
        
        // Reshape back: [B, heads, L, headDim] -> [B, L, hidden]
        let outputReshaped = output.transposed(0, 2, 1, 3).reshaped([B, L, -1])
        
        return oProj(outputReshaped)
    }
}

// MARK: - Gemma3n Text MLP

class Gemma3nTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    
    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(gelu(gateProj(x)) * upProj(x))
    }
}
