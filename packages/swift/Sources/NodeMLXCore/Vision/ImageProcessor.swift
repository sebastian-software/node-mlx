//
//  ImageProcessor.swift
//  NodeMLXCore
//
//  Image preprocessing for vision models.
//  Handles loading, resizing, and normalizing images.
//

import Foundation
import MLX
import CoreGraphics
import ImageIO

// MARK: - Image Processor Configuration

public struct ImageProcessorConfig: Sendable {
    /// Target image size (square)
    public var imageSize: Int

    /// Whether to rescale pixel values from [0, 255] to [0, 1]
    public var doRescale: Bool

    /// Rescale factor (typically 1/255)
    public var rescaleFactor: Float

    /// Whether to normalize with mean/std
    public var doNormalize: Bool

    /// Mean values for normalization (per channel RGB)
    public var imageMean: [Float]

    /// Std values for normalization (per channel RGB)
    public var imageStd: [Float]

    /// Default config for SigLIP/Gemma 3
    public static let siglip = ImageProcessorConfig(
        imageSize: 896,
        doRescale: true,
        rescaleFactor: 1.0 / 255.0,
        doNormalize: true,
        // SigLIP uses ImageNet normalization
        imageMean: [0.5, 0.5, 0.5],
        imageStd: [0.5, 0.5, 0.5]
    )

    /// No normalization (just resize)
    public static func resizeOnly(size: Int) -> ImageProcessorConfig {
        ImageProcessorConfig(
            imageSize: size,
            doRescale: true,
            rescaleFactor: 1.0 / 255.0,
            doNormalize: false,
            imageMean: [0, 0, 0],
            imageStd: [1, 1, 1]
        )
    }

    public init(
        imageSize: Int = 896,
        doRescale: Bool = true,
        rescaleFactor: Float = 1.0 / 255.0,
        doNormalize: Bool = true,
        imageMean: [Float] = [0.5, 0.5, 0.5],
        imageStd: [Float] = [0.5, 0.5, 0.5]
    ) {
        self.imageSize = imageSize
        self.doRescale = doRescale
        self.rescaleFactor = rescaleFactor
        self.doNormalize = doNormalize
        self.imageMean = imageMean
        self.imageStd = imageStd
    }
}

// MARK: - Image Processor

/// Preprocesses images for vision models
public struct ImageProcessor: Sendable {
    public let config: ImageProcessorConfig

    public init(config: ImageProcessorConfig = .siglip) {
        self.config = config
    }

    /// Load and preprocess an image from a file path
    /// - Parameter path: Path to image file
    /// - Returns: Preprocessed image tensor [1, C, H, W]
    public func loadAndPreprocess(path: String) throws -> MLXArray {
        let url = URL(fileURLWithPath: path)
        return try loadAndPreprocess(url: url)
    }

    /// Load and preprocess an image from a URL
    /// - Parameter url: URL to image file
    /// - Returns: Preprocessed image tensor [1, C, H, W]
    public func loadAndPreprocess(url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        return try preprocess(imageData: data)
    }

    /// Preprocess raw image data
    /// - Parameter imageData: Raw image bytes (JPEG, PNG, etc.)
    /// - Returns: Preprocessed image tensor [1, C, H, W]
    public func preprocess(imageData: Data) throws -> MLXArray {
        // Decode image using CoreGraphics
        guard let provider = CGDataProvider(data: imageData as CFData),
              let cgImage = CGImage(
                  jpegDataProviderSource: provider,
                  decode: nil,
                  shouldInterpolate: true,
                  intent: .defaultIntent
              ) ?? CGImage(
                  pngDataProviderSource: provider,
                  decode: nil,
                  shouldInterpolate: true,
                  intent: .defaultIntent
              )
        else {
            throw ImageProcessorError.decodeFailed
        }

        return preprocess(cgImage: cgImage)
    }

    /// Preprocess a CGImage
    /// - Parameter cgImage: Core Graphics image
    /// - Returns: Preprocessed image tensor [1, C, H, W]
    public func preprocess(cgImage: CGImage) -> MLXArray {
        // Resize image to target size
        let resized = resize(cgImage, to: config.imageSize)

        // Convert to MLXArray [H, W, C]
        var pixelValues = cgImageToMLXArray(resized)

        // Rescale from [0, 255] to [0, 1]
        if config.doRescale {
            pixelValues = pixelValues * config.rescaleFactor
        }

        // Normalize with mean/std
        if config.doNormalize {
            let mean = MLXArray(config.imageMean).reshaped([1, 1, 3])
            let std = MLXArray(config.imageStd).reshaped([1, 1, 3])
            pixelValues = (pixelValues - mean) / std
        }

        // Convert from [H, W, C] to [1, C, H, W] (NCHW format)
        pixelValues = pixelValues.transposed(2, 0, 1)  // [C, H, W]
        pixelValues = pixelValues.expandedDimensions(axis: 0)  // [1, C, H, W]

        return pixelValues.asType(.float32)
    }

    /// Preprocess multiple images
    /// - Parameter cgImages: Array of Core Graphics images
    /// - Returns: Batched preprocessed tensor [B, C, H, W]
    public func preprocess(cgImages: [CGImage]) -> MLXArray {
        let processed = cgImages.map { preprocess(cgImage: $0) }
        return concatenated(processed, axis: 0)
    }
}

// MARK: - Errors

public enum ImageProcessorError: Error, LocalizedError {
    case decodeFailed
    case resizeFailed
    case invalidFormat

    public var errorDescription: String? {
        switch self {
        case .decodeFailed:
            return "Failed to decode image data"
        case .resizeFailed:
            return "Failed to resize image"
        case .invalidFormat:
            return "Invalid image format"
        }
    }
}

// MARK: - Helper Functions

/// Resize a CGImage to target size (square, center crop)
private func resize(_ image: CGImage, to size: Int) -> CGImage {
    let width = image.width
    let height = image.height

    // Determine crop region (center crop to square)
    let minDim = min(width, height)
    let cropX = (width - minDim) / 2
    let cropY = (height - minDim) / 2
    let cropRect = CGRect(x: cropX, y: cropY, width: minDim, height: minDim)

    // Crop to square
    guard let croppedImage = image.cropping(to: cropRect) else {
        return image
    }

    // Create context for resized image
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: size * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        return croppedImage
    }

    // Draw resized image
    context.interpolationQuality = .high
    context.draw(croppedImage, in: CGRect(x: 0, y: 0, width: size, height: size))

    return context.makeImage() ?? croppedImage
}

/// Convert CGImage to MLXArray [H, W, C]
private func cgImageToMLXArray(_ image: CGImage) -> MLXArray {
    let width = image.width
    let height = image.height

    // Create RGBA context
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var pixelData = [UInt8](repeating: 0, count: width * height * 4)

    guard let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
        return MLXArray.zeros([height, width, 3])
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Extract RGB channels (skip alpha)
    var rgbData = [Float](repeating: 0, count: width * height * 3)
    for i in 0 ..< (width * height) {
        rgbData[i * 3 + 0] = Float(pixelData[i * 4 + 0])  // R
        rgbData[i * 3 + 1] = Float(pixelData[i * 4 + 1])  // G
        rgbData[i * 3 + 2] = Float(pixelData[i * 4 + 2])  // B
    }

    return MLXArray(rgbData).reshaped([height, width, 3])
}
