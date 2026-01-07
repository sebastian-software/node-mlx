import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Model Manager (keeps models in memory)

actor ModelManager {
  static let shared = ModelManager()

  private var models: [Int: ModelContainer] = [:]
  private var nextId = 1

  func loadModel(id: String) async throws -> Int {
    let configuration = ModelConfiguration(id: id)
    let container = try await LLMModelFactory.shared.loadContainer(configuration: configuration)

    let modelId = nextId
    nextId += 1
    models[modelId] = container

    return modelId
  }

  func unloadModel(id: Int) {
    models.removeValue(forKey: id)
  }

  func getModel(id: Int) -> ModelContainer? {
    return models[id]
  }

  func generate(
    modelId: Int,
    prompt: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    onToken: @escaping (String) -> Bool
  ) async throws -> GenerationResult {
    guard let container = models[modelId] else {
      throw NodeMLXError.modelNotFound
    }

    let counter = TokenCounter()
    let params = GenerateParameters(temperature: temperature, topP: topP)

    let result = try await container.perform { context in
      let input = try await context.processor.prepare(input: .init(prompt: prompt))
      return try MLXLMCommon.generate(
        input: input,
        parameters: params,
        context: context
      ) { (token: Int) -> GenerateDisposition in
        let tokenString = context.tokenizer.decode(tokens: [token])
        counter.append(tokenString)
        counter.increment()

        let shouldContinue = onToken(tokenString)
        if !shouldContinue || counter.count >= maxTokens {
          return .stop
        }
        return .more
      }
    }

    return GenerationResult(
      text: counter.tokens.joined(),
      tokenCount: counter.count,
      tokensPerSecond: result.tokensPerSecond
    )
  }
}

// MARK: - Helper Types

final class TokenCounter: @unchecked Sendable {
  private var _count = 0
  private var _tokens: [String] = []

  var count: Int { _count }
  var tokens: [String] { _tokens }

  func increment() { _count += 1 }
  func append(_ token: String) { _tokens.append(token) }
}

enum NodeMLXError: Error {
  case modelNotFound
  case generationFailed(String)
}

struct GenerationResult {
  let text: String
  let tokenCount: Int
  let tokensPerSecond: Double
}

// MARK: - JSON Response Types

struct JSONGenerationResult: Codable {
  let success: Bool
  let text: String?
  let tokenCount: Int?
  let tokensPerSecond: Double?
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
      let id = try await ModelManager.shared.loadModel(id: modelIdString)
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
    await ModelManager.shared.unloadModel(id: Int(handle))
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
      let result = try await ModelManager.shared.generate(
        modelId: Int(handle),
        prompt: promptString,
        maxTokens: Int(maxTokens),
        temperature: temperature,
        topP: topP
      ) { _ in true }  // No streaming callback for now

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
  return strdup("0.1.0")
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
