//
//  NodeMLXCore.swift
//  NodeMLXCore
//
//  Main entry point for LLM inference without mlx-swift-lm dependency.
//
//  Copyright © 2026 Sebastian Software GmbH. All rights reserved.
//

import Foundation
import Hub
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Tokenizers

// MARK: - Public API

/// Main interface for LLM operations
public class LLMEngine {
    private var model: (any LLMModel)?
    private var vlmModel: Gemma3VLMModel? // VLM-specific reference
    private var tokenizer: HFTokenizer?
    private var modelDirectory: URL?
    private var imageProcessor: ImageProcessor?
    private var _isVLM: Bool = false
    private var _isGemma: Bool = false // For enforcing Gemma chat template

    /// Whether the loaded model is a Vision-Language Model
    public var isVLM: Bool { _isVLM }

    public init() {}

    // MARK: - Model Loading

    /// Load a model from HuggingFace Hub
    public func loadModel(
        modelId: String,
        progressHandler: ((Float) -> Void)? = nil
    ) async throws {
        // Download model files
        let hub = HubApi()
        let repo = Hub.Repo(id: modelId)

        let directory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "*.json", "tokenizer*", "vocab*", "merges*"],
            progressHandler: { progress in
                progressHandler?(Float(progress.fractionCompleted))
            }
        )

        modelDirectory = directory

        // Detect architecture from config
        let configPath = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]

        guard configDict["model_type"] as? String != nil else {
            throw LLMEngineError.invalidConfig("model_type not found in config.json")
        }

        // Detect architecture (including VLM detection)
        let architecture = try ModelFactory.detectArchitecture(modelDirectory: directory)

        // Track if this is a VLM
        _isVLM = architecture.isVLM

        // Track if this is a Gemma model (for enforcing chat template)
        _isGemma = architecture == .gemma3 || architecture == .gemma3vlm || architecture == .gemma3n

        // Create model
        var model = try ModelFactory.createModel(
            modelDirectory: directory,
            architecture: architecture
        )

        // Keep VLM-specific reference for image generation
        if let vlm = model as? Gemma3VLMModel {
            vlmModel = vlm
            // Create image processor for VLM
            imageProcessor = ImageProcessor(config: .siglip)
        }

        // Load weights first
        let weights = try loadWeights(from: directory)
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantize if needed - use dynamic quantization based on weight presence
        if let quantizationConfig = configDict["quantization"] as? [String: Any],
           let groupSize = quantizationConfig["group_size"] as? Int,
           let bits = quantizationConfig["bits"] as? Int
        {
            // Quantize modules that have .scales weights
            // The filter returns (groupSize, bits) if the module should be quantized
            quantize(model: model) { path, _ in
                if sanitizedWeights["\(path).scales"] != nil {
                    (groupSize, bits)
                } else {
                    nil
                }
            }
        }

        // Apply weights to model
        model.update(parameters: ModuleParameters.unflattened(sanitizedWeights))

        // Force evaluation of weights to ensure they're loaded to GPU
        eval(model)

        self.model = model

        // Load tokenizer
        tokenizer = try await HFTokenizer(modelDirectory: directory)
    }

    // MARK: - Generation

    /// Generate text from a prompt
    public func generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) throws -> GenerationResult {
        guard let model else {
            throw LLMEngineError.modelNotLoaded
        }
        guard let tokenizer else {
            throw LLMEngineError.tokenizerNotLoaded
        }

        // Apply chat template to format prompt correctly for the model
        var inputTokens: [Int]
        if _isGemma {
            // Gemma models (including Gemma3n) need explicit chat template formatting
            // Some variants don't have chat_template in tokenizer_config.json
            let formattedPrompt = "<bos><start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"
            inputTokens = tokenizer.encode(formattedPrompt)
        } else {
            do {
                inputTokens = try tokenizer.applyChatTemplate(userMessage: prompt)
            } catch {
                // Fallback: use raw prompt
                inputTokens = tokenizer.encode(prompt)
            }
        }
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0) // Add batch dimension

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Create KV cache for efficient generation
        var cache: [KVCache]? = model.newCache()

        // Process prompt (prefill) - all tokens at once
        var logits = model(inputArray, cache: &cache)
        var lastLogits = logits[0, logits.dim(1) - 1]
        eval(lastLogits)

        // Apply repetition penalty if configured
        if let penalty = repetitionPenalty {
            lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: inputTokens, penalty: penalty, contextSize: repetitionContextSize)
        }

        // Sample first token
        var nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)

        // Generation loop - one token at a time with cached context
        for _ in 0 ..< maxTokens {
            // Check for EOS (both <eos> and <end_of_turn> for chat models)
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }
            // Gemma models use <end_of_turn> (106) for chat
            if nextToken == 106 {
                break
            }

            generatedTokens.append(nextToken)

            // Prepare next input - just the single new token
            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)

            // Forward pass with cache - only processes new token
            logits = model(inputArray, cache: &cache)
            lastLogits = logits[0, 0] // Single token output

            // Async eval for pipelining
            eval(lastLogits)

            // Apply repetition penalty before sampling
            if let penalty = repetitionPenalty {
                lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: generatedTokens, penalty: penalty, contextSize: repetitionContextSize)
            }

            // Sample next token
            nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)
        }

        let endTime = Date()
        let duration = Float(endTime.timeIntervalSince(startTime))
        let tokensPerSecond = duration > 0 ? Float(generatedTokens.count) / duration : 0

        // Decode generated tokens, skipping special tokens like <|end|>
        let generatedText = tokenizer.decode(generatedTokens, skipSpecialTokens: true)

        return GenerationResult(
            text: generatedText,
            tokenCount: generatedTokens.count,
            tokensPerSecond: tokensPerSecond
        )
    }

    /// Generate text with streaming callback
    public func generateStream(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        onToken: @escaping (String) -> Bool // Return false to stop
    ) throws -> GenerationResult {
        guard let model else {
            throw LLMEngineError.modelNotLoaded
        }
        guard let tokenizer else {
            throw LLMEngineError.tokenizerNotLoaded
        }

        // Apply chat template to format prompt correctly for the model
        var inputTokens: [Int]
        if _isGemma {
            // Gemma models (including Gemma3n) need explicit chat template formatting
            let formattedPrompt = "<bos><start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"
            inputTokens = tokenizer.encode(formattedPrompt)
        } else {
            do {
                inputTokens = try tokenizer.applyChatTemplate(userMessage: prompt)
            } catch {
                // Fallback: use raw prompt
                inputTokens = tokenizer.encode(prompt)
            }
        }
        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0)

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Create KV cache for efficient generation
        var cache: [KVCache]? = model.newCache()

        // Process prompt (prefill) - all tokens at once
        var logits = model(inputArray, cache: &cache)
        var lastLogits = logits[0, logits.dim(1) - 1]
        eval(lastLogits)

        // Apply repetition penalty if configured
        if let penalty = repetitionPenalty {
            lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: inputTokens, penalty: penalty, contextSize: repetitionContextSize)
        }

        // Sample first token
        var nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)

        // Generation loop with KV cache
        for _ in 0 ..< maxTokens {
            // Check for EOS (both <eos> and <end_of_turn> for chat models)
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }
            // Gemma models use <end_of_turn> (106) for chat
            if nextToken == 106 {
                break
            }

            generatedTokens.append(nextToken)

            // Stream the token (skip special tokens like <|end|>)
            let tokenText = tokenizer.decode([nextToken], skipSpecialTokens: true)
            if !tokenText.isEmpty, !onToken(tokenText) {
                break // User requested stop
            }

            // Prepare next input - just the single new token
            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)

            // Forward pass with cache - only processes new token
            logits = model(inputArray, cache: &cache)
            lastLogits = logits[0, 0] // Single token output
            eval(lastLogits)

            // Apply repetition penalty before sampling
            if let penalty = repetitionPenalty {
                lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: generatedTokens, penalty: penalty, contextSize: repetitionContextSize)
            }

            nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)
        }

        let endTime = Date()
        let duration = Float(endTime.timeIntervalSince(startTime))
        let tokensPerSecond = duration > 0 ? Float(generatedTokens.count) / duration : 0

        return GenerationResult(
            text: tokenizer.decode(generatedTokens, skipSpecialTokens: true),
            tokenCount: generatedTokens.count,
            tokensPerSecond: tokensPerSecond
        )
    }

    // MARK: - VLM Generation

    /// Generate text with image input (for VLMs)
    public func generateStreamWithImage(
        prompt: String,
        imagePath: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        onToken: @escaping (String) -> Bool
    ) throws -> GenerationResult {
        guard let vlmModel else {
            throw LLMEngineError.notAVLM
        }
        guard let tokenizer else {
            throw LLMEngineError.tokenizerNotLoaded
        }
        guard let imageProcessor else {
            throw LLMEngineError.imageProcessingFailed("No image processor available")
        }

        // Load and preprocess image
        let pixelValues: MLXArray
        do {
            pixelValues = try imageProcessor.loadAndPreprocess(path: imagePath)
        } catch {
            throw LLMEngineError.imageProcessingFailed("Failed to load image: \(error.localizedDescription)")
        }

        // For VLM, we need to include the image token ID directly
        // The tokenizer doesn't recognize <image> as a special token, so we insert it manually
        // Gemma 3 VLM image token ID is 262144
        let imageTokenId = 262_144

        // First tokenize the prompt without image
        var inputTokens: [Int]
        do {
            inputTokens = try tokenizer.applyChatTemplate(userMessage: prompt)
        } catch {
            // Fallback: manually construct a VLM-style prompt
            let manualPrompt = "<start_of_turn>user\n\(prompt)<end_of_turn>\n<start_of_turn>model\n"
            inputTokens = tokenizer.encode(manualPrompt)
        }

        // Find position after "user\n" to insert image token
        // The format is: <bos><start_of_turn>user\n[IMAGE_HERE]prompt<end_of_turn><start_of_turn>model\n
        // Token IDs: 2 (bos), 105 (start_of_turn), user tokens, 107 (newline)
        var insertPos = 0
        for (i, token) in inputTokens.enumerated() {
            // Look for the newline token (107) after user
            if token == 107, i > 2 {
                insertPos = i + 1
                break
            }
        }

        // Insert image token at the found position
        if insertPos > 0, insertPos < inputTokens.count {
            inputTokens.insert(imageTokenId, at: insertPos)
        } else {
            // Fallback: insert after BOS token
            inputTokens.insert(imageTokenId, at: 1)
        }

        var inputArray = MLXArray(inputTokens.map { Int32($0) })
        inputArray = inputArray.expandedDimensions(axis: 0)

        let startTime = Date()
        var generatedTokens: [Int] = []

        // Create KV cache
        var cache: [KVCache]? = vlmModel.newCache()

        // Process prompt with image (prefill)
        var logits = vlmModel(inputArray, pixelValues: pixelValues, cache: &cache)
        var lastLogits = logits[0, logits.dim(1) - 1]
        eval(lastLogits)

        // Apply repetition penalty if configured
        if let penalty = repetitionPenalty {
            lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: inputTokens, penalty: penalty, contextSize: repetitionContextSize)
        }

        // Sample first token
        var nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)

        // Generation loop
        for _ in 0 ..< maxTokens {
            if let eosId = tokenizer.eosTokenId, nextToken == eosId {
                break
            }
            if nextToken == 106 { // <end_of_turn>
                break
            }

            generatedTokens.append(nextToken)

            let tokenText = tokenizer.decode([nextToken], skipSpecialTokens: true)
            if !tokenText.isEmpty, !onToken(tokenText) {
                break
            }

            inputArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)

            // Forward without image (already encoded in KV cache)
            logits = vlmModel(inputArray, pixelValues: nil, cache: &cache)
            lastLogits = logits[0, 0]
            eval(lastLogits)

            // Apply repetition penalty before sampling
            if let penalty = repetitionPenalty {
                lastLogits = applyRepetitionPenalty(lastLogits, generatedTokens: generatedTokens, penalty: penalty, contextSize: repetitionContextSize)
            }

            nextToken = sampleToken(logits: lastLogits, temperature: temperature, topP: topP)
        }

        let endTime = Date()
        let duration = Float(endTime.timeIntervalSince(startTime))
        let tokensPerSecond = duration > 0 ? Float(generatedTokens.count) / duration : 0

        return GenerationResult(
            text: tokenizer.decode(generatedTokens, skipSpecialTokens: true),
            tokenCount: generatedTokens.count,
            tokensPerSecond: tokensPerSecond
        )
    }

    // MARK: - Cleanup

    /// Unload the model from memory
    public func unload() {
        model = nil
        vlmModel = nil
        tokenizer = nil
        modelDirectory = nil
        imageProcessor = nil
        _isVLM = false
    }

    // MARK: - Private Helpers

    private func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]

        let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        )!

        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let fileWeights = try loadArrays(url: url)
                for (key, value) in fileWeights {
                    weights[key] = value
                }
            }
        }

        if weights.isEmpty {
            throw LLMEngineError.weightsNotFound
        }

        return weights
    }

    private func sampleToken(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        if temperature == 0 {
            // Greedy decoding - no randomness
            let token = argMax(logits, axis: -1)
            eval(token)
            return Int(token.item(Int32.self))
        }

        // Temperature scaling and convert to probabilities
        let temp = MLXArray(temperature)
        var logitsFloat = logits
        if logitsFloat.dtype == .bfloat16 {
            logitsFloat = logitsFloat.asType(.float32)
        }
        let probs = softmax(logitsFloat / temp, axis: -1)

        // For top-p sampling, use the mlx-swift-lm approach
        if topP > 0, topP < 1 {
            let topPArray = MLXArray(topP)

            // Sort in ascending order (lowest first)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = take(probs, sortedIndices, axis: -1)

            // Cumulative sum (from lowest to highest)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            // Keep only tokens where cumulative prob > (1 - topP)
            // This keeps the top-p highest probability tokens
            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topPArray),
                sortedProbs,
                MLXArray.zeros(like: sortedProbs)
            )

            // Sample using log probabilities (avoid numerical issues)
            let sortedToken = MLXRandom.categorical(log(topProbs))
            eval(sortedToken)

            // Map back to original index
            let originalIdx = sortedIndices[Int(sortedToken.item(Int32.self))]
            eval(originalIdx)
            return Int(originalIdx.item(Int32.self))
        }

        // Simple temperature sampling without top-p
        let token = MLXRandom.categorical(probs)
        eval(token)
        return Int(token.item(Int32.self))
    }
}

// MARK: - Types

public struct GenerationResult {
    public let text: String
    public let tokenCount: Int
    public let tokensPerSecond: Float

    public init(text: String, tokenCount: Int, tokensPerSecond: Float) {
        self.text = text
        self.tokenCount = tokenCount
        self.tokensPerSecond = tokensPerSecond
    }
}

public enum LLMEngineError: Error, LocalizedError {
    case modelNotLoaded
    case tokenizerNotLoaded
    case invalidConfig(String)
    case unsupportedModel(String)
    case weightsNotFound
    case notAVLM
    case imageProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "No model loaded. Call loadModel() first."
        case .tokenizerNotLoaded:
            "No tokenizer loaded."
        case let .invalidConfig(msg):
            "Invalid config: \(msg)"
        case let .unsupportedModel(msg):
            "Unsupported model: \(msg)"
        case .weightsNotFound:
            "No weights found in model directory."
        case .notAVLM:
            "Model does not support images (not a VLM). Use a vision model like google/gemma-3-4b-it."
        case let .imageProcessingFailed(msg):
            "Image processing failed: \(msg)"
        }
    }
}

// MARK: - Convenience

/// Quick generation without managing engine lifecycle
public func quickGenerate(
    modelId: String,
    prompt: String,
    maxTokens: Int = 256
) async throws -> String {
    let engine = LLMEngine()
    try await engine.loadModel(modelId: modelId)
    let result = try engine.generate(prompt: prompt, maxTokens: maxTokens)
    engine.unload()
    return result.text
}
