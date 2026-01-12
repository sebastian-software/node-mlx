// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Helper type for decoding JSON values that can be either string or number.

import Foundation

/// A type that can decode either a string or a number from JSON.
///
/// This is commonly needed for HuggingFace model configs where some
/// fields may be specified as either strings or numbers (e.g., rope_scaling).
public enum StringOrNumber: Codable, Hashable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let intValue = try? container.decode(Int.self) {
            self = .int(intValue)
        } else if let doubleValue = try? container.decode(Double.self) {
            self = .double(doubleValue)
        } else if let stringValue = try? container.decode(String.self) {
            self = .string(stringValue)
        } else {
            throw DecodingError.typeMismatch(
                StringOrNumber.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String, Int, or Double"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value):
            try container.encode(value)
        case let .int(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        }
    }

    /// Returns the value as a String, converting numbers if necessary.
    public var stringValue: String {
        switch self {
        case let .string(value): value
        case let .int(value): String(value)
        case let .double(value): String(value)
        }
    }

    /// Returns the value as an Int if possible.
    public var intValue: Int? {
        switch self {
        case .string: nil
        case let .int(value): value
        case let .double(value): Int(value)
        }
    }

    /// Returns the value as a Double if possible.
    public var doubleValue: Double? {
        switch self {
        case .string: nil
        case let .int(value): Double(value)
        case let .double(value): value
        }
    }

    /// Returns the value as a Float if possible.
    public var floatValue: Float? {
        switch self {
        case .string: nil
        case let .int(value): Float(value)
        case let .double(value): Float(value)
        }
    }
}

// MARK: - Dictionary Convenience

public extension [String: StringOrNumber] {
    /// Converts the dictionary to a standard [String: Any] dictionary.
    var asAnyDict: [String: Any] {
        var result: [String: Any] = [:]
        for (key, value) in self {
            switch value {
            case let .string(s):
                result[key] = s
            case let .int(i):
                result[key] = i
            case let .double(d):
                result[key] = d
            }
        }
        return result
    }
}
