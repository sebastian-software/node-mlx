import XCTest
import MLX
import Hub
@testable import NodeMLXCore

final class ModelLoaderTests: XCTestCase {

    let loader = ModelLoader()

    // MARK: - Download Tests

    func testDownloadSmallModel() async throws {
        // Use a small model to keep test fast
        let modelId = "mlx-community/SmolLM-135M-4bit"

        print("Downloading \(modelId)...")
        let modelDir = try await loader.download(modelId: modelId) { progress in
            print("  Progress: \(Int(progress.fractionCompleted * 100))%")
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: modelDir.path))
        print("✓ Downloaded to: \(modelDir.path)")
    }

    // MARK: - Config Loading Tests

    func testLoadConfig() async throws {
        let modelId = "mlx-community/SmolLM-135M-4bit"
        let modelDir = try await loader.download(modelId: modelId)

        let config = try loader.loadConfig(from: modelDir)

        print("Model config:")
        print("  model_type: \(config.modelType ?? "unknown")")
        print("  hidden_size: \(config.hiddenSize ?? 0)")
        print("  num_layers: \(config.numHiddenLayers ?? 0)")
        print("  vocab_size: \(config.vocabSize ?? 0)")

        XCTAssertNotNil(config.modelType)
        XCTAssertNotNil(config.hiddenSize)
    }

    func testGetModelType() async throws {
        let modelId = "mlx-community/SmolLM-135M-4bit"
        let modelDir = try await loader.download(modelId: modelId)

        let modelType = try loader.getModelType(from: modelDir)
        print("✓ Model type: \(modelType)")

        XCTAssertFalse(modelType.isEmpty)
    }

    // MARK: - Weight Loading Tests

    func testLoadWeights() async throws {
        let modelId = "mlx-community/SmolLM-135M-4bit"
        let modelDir = try await loader.download(modelId: modelId)

        let weights = try loader.loadWeights(from: modelDir)

        print("Loaded \(weights.count) weight tensors:")
        for (key, value) in weights.prefix(5) {
            print("  \(key): \(value.shape)")
        }

        XCTAssertFalse(weights.isEmpty)
        print("✓ Successfully loaded \(weights.count) tensors")
    }

    // MARK: - Weight Sanitization Tests

    func testSanitizeWeights() async throws {
        let modelId = "mlx-community/SmolLM-135M-4bit"
        let modelDir = try await loader.download(modelId: modelId)

        let rawWeights = try loader.loadWeights(from: modelDir)
        let sanitized = sanitizeWeights(rawWeights, prefix: "model.")

        // Check if prefixes were removed
        var prefixRemoved = false
        for key in rawWeights.keys {
            if key.hasPrefix("model.") {
                let newKey = String(key.dropFirst("model.".count))
                if sanitized[newKey] != nil {
                    prefixRemoved = true
                    break
                }
            }
        }

        print("✓ Weight sanitization completed")
        print("  Original keys: \(rawWeights.count)")
        print("  Sanitized keys: \(sanitized.count)")
    }
}

