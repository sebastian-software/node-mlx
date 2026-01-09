// Copyright © 2024 Apple Inc.
// Adapted for NodeMLXCore

import Foundation

/// Representation of a heterogenous type in a JSON configuration file.
///
/// This can be: a string, a numeric value or an array of numeric values.
/// There are methods to do unwrapping, see e.g. ``asFloat()`` and
/// ``asFloats()`` or callers can switch on the enum.
public enum StringOrNumber: Codable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case float(Float)
    case ints([Int])
    case floats([Float])
    case bool(Bool)

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer()

        if let v = try? values.decode(Int.self) {
            self = .int(v)
        } else if let v = try? values.decode(Float.self) {
            self = .float(v)
        } else if let v = try? values.decode([Int].self) {
            self = .ints(v)
        } else if let v = try? values.decode([Float].self) {
            self = .floats(v)
        } else if let v = try? values.decode(Bool.self) {
            self = .bool(v)
        } else {
            let v = try values.decode(String.self)
            self = .string(v)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(v): try container.encode(v)
        case let .int(v): try container.encode(v)
        case let .float(v): try container.encode(v)
        case let .ints(v): try container.encode(v)
        case let .floats(v): try container.encode(v)
        case let .bool(v): try container.encode(v)
        }
    }

    /// Return the value as an optional array of integers.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInts() -> [Int]? {
        switch self {
        case .string: nil
        case let .int(v): [v]
        case .float: nil
        case let .ints(array): array
        case .floats: nil
        case .bool: nil
        }
    }

    /// Return the value as an optional integer.
    ///
    /// This will not coerce `Float` or `String` to `Int`.
    public func asInt() -> Int? {
        switch self {
        case .string: nil
        case let .int(v): v
        case .float: nil
        case let .ints(array): array.count == 1 ? array[0] : nil
        case .floats: nil
        case let .bool(bool): bool ? 1 : 0
        }
    }

    /// Return the value as an optional array of floats.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloats() -> [Float]? {
        switch self {
        case .string: nil
        case let .int(v): [Float(v)]
        case let .float(float): [float]
        case let .ints(array): array.map { Float($0) }
        case let .floats(array): array
        case let .bool(bool): [bool ? 1.0 : 0.0]
        }
    }

    /// Return the value as an optional float.
    ///
    /// This will not coerce `Int` or `String` to `Float`.
    public func asFloat() -> Float? {
        switch self {
        case .string: nil
        case let .int(v): Float(v)
        case let .float(float): float
        case let .ints(array): array.count == 1 ? Float(array[0]) : nil
        case let .floats(array): array.count == 1 ? array[0] : nil
        case let .bool(bool): bool ? 1.0 : 0.0
        }
    }
}
