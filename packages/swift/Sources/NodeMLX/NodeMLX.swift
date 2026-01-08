import Foundation
import NodeMLXCore

// MARK: - Engine Manager (keeps engines in memory)

actor EngineManager {
    static let shared = EngineManager()

    private var engines: [Int: LLMEngine] = [:]
    private var nextId = 1

    func loadModel(id: String) async throws -> Int {
        let engine = LLMEngine()
        try await engine.loadModel(modelId: id)

        let engineId = nextId
        nextId += 1
        engines[engineId] = engine

        return engineId
    }

    func unloadModel(id: Int) {
        if let engine = engines.removeValue(forKey: id) {
            engine.unload()
        }
    }

    func getEngine(id: Int) -> LLMEngine? {
        return engines[id]
    }

    func generate(
        engineId: Int,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        onToken: @escaping (String) -> Bool
    ) throws -> NodeMLXCore.GenerationResult {
        guard let engine = engines[engineId] else {
            throw NodeMLXError.modelNotFound
        }

        return try engine.generateStream(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            onToken: onToken
        )
    }
}

// MARK: - Helper Types

enum NodeMLXError: Error, LocalizedError {
    case modelNotFound
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        }
    }
}

// MARK: - JSON Response Types

struct JSONGenerationResult: Codable {
    let success: Bool
    let text: String?
    let tokenCount: Int?
    let tokensPerSecond: Float?
    let error: String?
}

// MARK: - C-Exported Functions

/// Load a model and return its handle (ID)
/// Returns model ID on success, -1 on error
@_cdecl("node_mlx_load_model")
public func loadModel(modelId: UnsafePointer<CChar>?) -> Int32 {
    guard let modelId = modelId else { return -1 }
    let modelIdString = String(cString: modelId)

    var result: Int32 = -1
    let semaphore = DispatchSemaphore(value: 0)

    Task {
        do {
            let id = try await EngineManager.shared.loadModel(id: modelIdString)
            result = Int32(id)
        } catch {
            print("Error loading model: \(error)")
            result = -1
        }
        semaphore.signal()
    }

    semaphore.wait()
    return result
}

/// Unload a model from memory
@_cdecl("node_mlx_unload_model")
public func unloadModel(handle: Int32) {
    Task {
        await EngineManager.shared.unloadModel(id: Int(handle))
    }
}

/// Generate text from a prompt
/// Returns JSON string - caller must free with node_mlx_free_string
@_cdecl("node_mlx_generate")
public func generate(
    handle: Int32,
    prompt: UnsafePointer<CChar>?,
    maxTokens: Int32,
    temperature: Float,
    topP: Float
) -> UnsafeMutablePointer<CChar>? {
    guard let prompt = prompt else {
        return makeJSONError("Invalid prompt")
    }

    let promptString = String(cString: prompt)
    var jsonResult: UnsafeMutablePointer<CChar>?
    let semaphore = DispatchSemaphore(value: 0)

    Task {
        do {
            let result = try await EngineManager.shared.generate(
                engineId: Int(handle),
                prompt: promptString,
                maxTokens: Int(maxTokens),
                temperature: temperature,
                topP: topP
            ) { _ in true }  // Continue generating

            let response = JSONGenerationResult(
                success: true,
                text: result.text,
                tokenCount: result.tokenCount,
                tokensPerSecond: result.tokensPerSecond,
                error: nil
            )
            jsonResult = encodeJSON(response)
        } catch NodeMLXError.modelNotFound {
            jsonResult = makeJSONError("Model not found")
        } catch {
            jsonResult = makeJSONError("Generation failed: \(error.localizedDescription)")
        }
        semaphore.signal()
    }

    semaphore.wait()
    return jsonResult
}

/// Free a string allocated by this library
@_cdecl("node_mlx_free_string")
public func freeString(str: UnsafeMutablePointer<CChar>?) {
    if let str = str {
        free(str)
    }
}

/// Check if MLX is available on this system
@_cdecl("node_mlx_is_available")
public func isAvailable() -> Bool {
    #if arch(arm64) && os(macOS)
    return true
    #else
    return false
    #endif
}

/// Get version string - caller must free with node_mlx_free_string
@_cdecl("node_mlx_version")
public func getVersion() -> UnsafeMutablePointer<CChar>? {
    return strdup("1.0.0")  // New version without mlx-swift-lm
}

// MARK: - Private Helpers

private func makeJSONError(_ message: String) -> UnsafeMutablePointer<CChar>? {
    let response = JSONGenerationResult(
        success: false,
        text: nil,
        tokenCount: nil,
        tokensPerSecond: nil,
        error: message
    )
    return encodeJSON(response)
}

private func encodeJSON<T: Encodable>(_ value: T) -> UnsafeMutablePointer<CChar>? {
    let encoder = JSONEncoder()
    guard let data = try? encoder.encode(value),
          let jsonString = String(data: data, encoding: .utf8)
    else {
        return strdup("{\"success\":false,\"error\":\"JSON encoding failed\"}")
    }
    return strdup(jsonString)
}
