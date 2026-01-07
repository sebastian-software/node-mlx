//
//  ModelLoader.swift
//  NodeMLXCore
//
//  Load MLX models from HuggingFace Hub or local path.
//

import Foundation
import MLX
import MLXNN

// MARK: - HuggingFace Hub Client

/// Simple HuggingFace Hub client for downloading models
public struct HFHub {
    public static let cacheDir: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub")
    }()

    /// Get the local cache path for a HuggingFace model
    public static func modelPath(for modelId: String) -> URL {
        let sanitized = modelId.replacingOccurrences(of: "/", with: "--")
        return cacheDir
            .appendingPathComponent("models--\(sanitized)")
            .appendingPathComponent("snapshots")
    }

    /// Check if a model is cached locally
    public static func isCached(_ modelId: String) -> Bool {
        let path = modelPath(for: modelId)
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path.path, isDirectory: &isDir) && isDir.boolValue
    }

    /// Get the latest snapshot directory for a cached model
    public static func latestSnapshot(for modelId: String) -> URL? {
        let snapshotsDir = modelPath(for: modelId)

        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        // Return the most recent snapshot
        return contents.sorted { a, b in
            let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return aDate > bDate
        }.first
    }
}

// MARK: - SafeTensors Loading

/// Load weights from SafeTensors files
public struct SafeTensorsLoader {

    /// Load all weights from .safetensors files in a directory
    public static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]

        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        for file in files {
            let fileWeights = try loadSafeTensors(from: file)
            weights.merge(fileWeights) { _, new in new }
        }

        return weights
    }

    /// Load a single .safetensors file
    public static func loadSafeTensors(from url: URL) throws -> [String: MLXArray] {
        // Use MLX's built-in safetensors loading
        return try MLX.loadArrays(url: url)
    }
}

// MARK: - Config Loading

/// Load model configuration from config.json
public struct ConfigLoader {

    public static func loadConfig<T: Decodable>(
        from directory: URL,
        as type: T.Type
    ) throws -> T {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: data)
    }
}

// MARK: - Model Container

/// Container for a loaded model with its configuration
public class ModelContainer<Config: Codable, Model: Module> {
    public let config: Config
    public let model: Model
    public let modelPath: URL

    public init(config: Config, model: Model, modelPath: URL) {
        self.config = config
        self.model = model
        self.modelPath = modelPath
    }

    /// Load weights into the model
    public func loadWeights() throws {
        let weights = try SafeTensorsLoader.loadWeights(from: modelPath)

        // Apply weights to model
        // Note: This requires the model to implement weight loading
        // model.update(parameters: weights)

        // For now, we use MLX's built-in parameter update
        try model.update(parameters: ModuleParameters(weights), verify: .noUnusedKeys)
    }
}

// MARK: - Model Factory Protocol

/// Protocol for model factories that can create models from configs
public protocol ModelFactory {
    associatedtype Config: Codable
    associatedtype Model: Module

    static func create(config: Config) -> Model
}

// MARK: - Load Model Convenience

/// Load a model from HuggingFace Hub or local path
public func loadModel<Factory: ModelFactory>(
    modelId: String,
    factory: Factory.Type
) throws -> ModelContainer<Factory.Config, Factory.Model> {
    // Determine model path
    let modelPath: URL

    if modelId.contains("/") {
        // HuggingFace model ID
        guard let snapshot = HFHub.latestSnapshot(for: modelId) else {
            throw ModelLoadError.modelNotCached(modelId)
        }
        modelPath = snapshot
    } else {
        // Local path
        modelPath = URL(fileURLWithPath: modelId)
    }

    // Load config
    let config = try ConfigLoader.loadConfig(
        from: modelPath,
        as: Factory.Config.self
    )

    // Create model
    let model = Factory.create(config: config)

    // Create container
    let container = ModelContainer(
        config: config,
        model: model,
        modelPath: modelPath
    )

    // Load weights
    try container.loadWeights()

    return container
}

// MARK: - Errors

public enum ModelLoadError: Error, LocalizedError {
    case modelNotCached(String)
    case configNotFound(URL)
    case weightsNotFound(URL)
    case invalidModelType(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotCached(let id):
            return "Model '\(id)' is not cached. Download it first using huggingface-cli or mlx_lm.convert"
        case .configNotFound(let url):
            return "Config file not found at \(url.path)"
        case .weightsNotFound(let url):
            return "Weight files not found in \(url.path)"
        case .invalidModelType(let type):
            return "Unsupported model type: \(type)"
        }
    }
}

