import Foundation
import NodeMLXCore
import MLX
import Cmlx

// MARK: - Metal Library Bundle Loading

/// Track if we've already loaded the metallib bundle
private var metallibBundleLoaded = false

/// Explicitly set the metallib path for MLX
@_cdecl("node_mlx_set_metallib_path")
public func setMetallibPath(_ pathPtr: UnsafePointer<CChar>) -> Bool {
    let path = String(cString: pathPtr)

    // Check if it's a bundle or direct metallib path
    if path.hasSuffix(".bundle") {
        // It's a bundle - find the metallib inside
        let metallibPath = URL(fileURLWithPath: path)
            .appendingPathComponent("Contents/Resources/default.metallib")
        if FileManager.default.fileExists(atPath: metallibPath.path) {
            setenv("MLX_METALLIB_PATH", metallibPath.path, 1)
            // Also load the bundle
            if let bundle = Bundle(url: URL(fileURLWithPath: path)) {
                bundle.load()
            }
            metallibBundleLoaded = true
            return true
        }
    } else if path.hasSuffix(".metallib") {
        // Direct metallib path
        if FileManager.default.fileExists(atPath: path) {
            setenv("MLX_METALLIB_PATH", path, 1)
            metallibBundleLoaded = true
            return true
        }
    }

    return false
}

/// Explicitly load the mlx-swift_Cmlx bundle to help MLX find the metallib
/// This needs to be called before any MLX operations
private func ensureMetalLibBundle() {
    guard !metallibBundleLoaded else { return }

    // Build list of paths to search for the bundle
    var searchPaths: [URL] = []

    // 1. Try process env for explicit path
    if let envPath = ProcessInfo.processInfo.environment["MLX_BUNDLE_PATH"] {
        searchPaths.insert(URL(fileURLWithPath: envPath), at: 0)
    }

    // 2. Try relative to the LLMEngine class (our dylib)
    let bundleURL = Bundle(for: LLMEngine.self).bundleURL.deletingLastPathComponent()
    searchPaths.append(bundleURL.appendingPathComponent("mlx-swift_Cmlx.bundle"))

    // 3. Try current working directory
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    searchPaths.append(cwd.appendingPathComponent("swift/mlx-swift_Cmlx.bundle"))
    searchPaths.append(cwd.appendingPathComponent("mlx-swift_Cmlx.bundle"))

    // 4. Try relative to packages/node-mlx (development)
    searchPaths.append(cwd.appendingPathComponent("packages/node-mlx/swift/mlx-swift_Cmlx.bundle"))

    for bundleURL in searchPaths {
        if FileManager.default.fileExists(atPath: bundleURL.path) {
            // Set the metallib path environment variable
            let metallibPath = bundleURL.appendingPathComponent("Contents/Resources/default.metallib")
            if FileManager.default.fileExists(atPath: metallibPath.path) {
                setenv("MLX_METALLIB_PATH", metallibPath.path, 1)

                // Also load the bundle for good measure
                if let bundle = Bundle(url: bundleURL) {
                    bundle.load()
                }
                metallibBundleLoaded = true
                return
            }
        }
    }
}

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

    func generateWithImage(
        engineId: Int,
        prompt: String,
        imagePath: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        onToken: @escaping (String) -> Bool
    ) throws -> NodeMLXCore.GenerationResult {
        guard let engine = engines[engineId] else {
            throw NodeMLXError.modelNotFound
        }

        guard engine.isVLM else {
            throw NodeMLXError.notAVLM
        }

        return try engine.generateStreamWithImage(
            prompt: prompt,
            imagePath: imagePath,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            onToken: onToken
        )
    }

    func isVLM(engineId: Int) -> Bool {
        guard let engine = engines[engineId] else {
            return false
        }
        return engine.isVLM
    }
}

// MARK: - Helper Types

enum NodeMLXError: Error, LocalizedError {
    case modelNotFound
    case generationFailed(String)
    case notAVLM
    case imageLoadFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model not found"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        case .notAVLM:
            return "Model does not support images (not a VLM)"
        case .imageLoadFailed(let msg):
            return "Failed to load image: \(msg)"
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

struct JSONModelInfo: Codable {
    let isVLM: Bool
    let architecture: String
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

/// Generate text from a prompt (non-streaming)
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

/// Generate text with streaming - writes tokens to stdout as they're generated
/// Returns JSON string with stats when complete - caller must free with node_mlx_free_string
@_cdecl("node_mlx_generate_streaming")
public func generateStreaming(
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
            ) { token in
                // Write token directly to stdout (unbuffered)
                if let data = token.data(using: .utf8) {
                    FileHandle.standardOutput.write(data)
                }
                return true  // Continue generating
            }

            // Return stats as JSON (text already streamed)
            let response = JSONGenerationResult(
                success: true,
                text: nil,  // Already streamed
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

/// Generate text with image input (VLM) - writes tokens to stdout as they're generated
/// Returns JSON string with stats when complete - caller must free with node_mlx_free_string
@_cdecl("node_mlx_generate_with_image")
public func generateWithImage(
    handle: Int32,
    prompt: UnsafePointer<CChar>?,
    imagePath: UnsafePointer<CChar>?,
    maxTokens: Int32,
    temperature: Float,
    topP: Float
) -> UnsafeMutablePointer<CChar>? {
    guard let prompt = prompt else {
        return makeJSONError("Invalid prompt")
    }
    guard let imagePath = imagePath else {
        return makeJSONError("Invalid image path")
    }

    let promptString = String(cString: prompt)
    let imagePathString = String(cString: imagePath)
    var jsonResult: UnsafeMutablePointer<CChar>?
    let semaphore = DispatchSemaphore(value: 0)

    Task {
        do {
            let result = try await EngineManager.shared.generateWithImage(
                engineId: Int(handle),
                prompt: promptString,
                imagePath: imagePathString,
                maxTokens: Int(maxTokens),
                temperature: temperature,
                topP: topP
            ) { token in
                // Write token directly to stdout (unbuffered)
                if let data = token.data(using: .utf8) {
                    FileHandle.standardOutput.write(data)
                }
                return true  // Continue generating
            }

            // Return stats as JSON (text already streamed)
            let response = JSONGenerationResult(
                success: true,
                text: nil,  // Already streamed
                tokenCount: result.tokenCount,
                tokensPerSecond: result.tokensPerSecond,
                error: nil
            )
            jsonResult = encodeJSON(response)
        } catch NodeMLXError.modelNotFound {
            jsonResult = makeJSONError("Model not found")
        } catch NodeMLXError.notAVLM {
            jsonResult = makeJSONError("Model does not support images (not a VLM)")
        } catch {
            jsonResult = makeJSONError("Generation failed: \(error.localizedDescription)")
        }
        semaphore.signal()
    }

    semaphore.wait()
    return jsonResult
}

/// Check if a loaded model is a VLM (Vision-Language Model)
@_cdecl("node_mlx_is_vlm")
public func isVLM(handle: Int32) -> Bool {
    var result = false
    let semaphore = DispatchSemaphore(value: 0)

    Task {
        result = await EngineManager.shared.isVLM(engineId: Int(handle))
        semaphore.signal()
    }

    semaphore.wait()
    return result
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
    ensureMetalLibBundle()
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
