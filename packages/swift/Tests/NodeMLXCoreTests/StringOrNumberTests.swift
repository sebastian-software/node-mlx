// Copyright © 2026 Sebastian Software GmbH.
// Tests adapted from mlx-swift-lm patterns (MIT License, Apple Inc.)

import XCTest
import Foundation
@testable import NodeMLXCore

final class StringOrNumberTests: XCTestCase {

    // MARK: - Decoding Tests

    func testDecodeString() throws {
        let json = "\"hello\""
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .string(let s) = value {
            XCTAssertEqual(s, "hello")
        } else {
            XCTFail("Expected string")
        }
    }

    func testDecodeInt() throws {
        let json = "42"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .int(let i) = value {
            XCTAssertEqual(i, 42)
        } else {
            XCTFail("Expected int")
        }
    }

    func testDecodeFloat() throws {
        let json = "3.14"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .float(let f) = value {
            XCTAssertEqual(f, 3.14, accuracy: 0.001)
        } else {
            XCTFail("Expected float")
        }
    }

    func testDecodeBool() throws {
        let json = "true"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .bool(let b) = value {
            XCTAssertTrue(b)
        } else {
            XCTFail("Expected bool")
        }
    }

    func testDecodeIntArray() throws {
        let json = "[1, 2, 3]"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .ints(let arr) = value {
            XCTAssertEqual(arr, [1, 2, 3])
        } else {
            XCTFail("Expected int array")
        }
    }

    func testDecodeFloatArray() throws {
        let json = "[1.1, 2.2, 3.3]"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case .floats(let arr) = value {
            XCTAssertEqual(arr.count, 3)
            XCTAssertEqual(arr[0], 1.1, accuracy: 0.001)
        } else {
            XCTFail("Expected float array")
        }
    }

    // MARK: - Conversion Tests

    func testAsFloat() throws {
        XCTAssertEqual(StringOrNumber.int(42).asFloat(), 42.0)
        XCTAssertEqual(StringOrNumber.float(3.14).asFloat(), 3.14)
        XCTAssertNil(StringOrNumber.string("hello").asFloat())
        XCTAssertEqual(StringOrNumber.bool(true).asFloat(), 1.0)
        XCTAssertEqual(StringOrNumber.bool(false).asFloat(), 0.0)
    }

    func testAsInt() throws {
        XCTAssertEqual(StringOrNumber.int(42).asInt(), 42)
        XCTAssertNil(StringOrNumber.float(3.14).asInt())
        XCTAssertNil(StringOrNumber.string("hello").asInt())
        XCTAssertEqual(StringOrNumber.bool(true).asInt(), 1)
        XCTAssertEqual(StringOrNumber.bool(false).asInt(), 0)
    }

    func testAsFloats() throws {
        XCTAssertEqual(StringOrNumber.floats([1.1, 2.2]).asFloats(), [1.1, 2.2])
        XCTAssertEqual(StringOrNumber.ints([1, 2, 3]).asFloats(), [1.0, 2.0, 3.0])
        XCTAssertEqual(StringOrNumber.int(42).asFloats(), [42.0])
        XCTAssertNil(StringOrNumber.string("hello").asFloats())
    }

    func testAsInts() throws {
        XCTAssertEqual(StringOrNumber.ints([1, 2, 3]).asInts(), [1, 2, 3])
        XCTAssertEqual(StringOrNumber.int(42).asInts(), [42])
        XCTAssertNil(StringOrNumber.floats([1.1]).asInts())
        XCTAssertNil(StringOrNumber.string("hello").asInts())
    }

    // MARK: - Config Parsing Tests

    func testRopeScalingConfig() throws {
        let json = """
        {
            "type": "linear",
            "factor": 2.0
        }
        """

        let config = try JSONDecoder().decode(
            [String: StringOrNumber].self,
            from: json.data(using: .utf8)!
        )

        if case .string(let type) = config["type"] {
            XCTAssertEqual(type, "linear")
        } else {
            XCTFail("Expected string type")
        }

        XCTAssertEqual(config["factor"]?.asFloat(), 2.0)
    }

    func testQuantizationConfig() throws {
        // Typical quantization config
        let json = """
        {
            "group_size": 64,
            "bits": 4
        }
        """

        let config = try JSONDecoder().decode(
            [String: StringOrNumber].self,
            from: json.data(using: .utf8)!
        )

        XCTAssertEqual(config["group_size"]?.asInt(), 64)
        XCTAssertEqual(config["bits"]?.asInt(), 4)
    }

    // MARK: - Encoding Tests

    func testEncode() throws {
        let encoder = JSONEncoder()

        let stringData = try encoder.encode(StringOrNumber.string("test"))
        XCTAssertEqual(String(data: stringData, encoding: .utf8), "\"test\"")

        let intData = try encoder.encode(StringOrNumber.int(42))
        XCTAssertEqual(String(data: intData, encoding: .utf8), "42")

        let boolData = try encoder.encode(StringOrNumber.bool(true))
        XCTAssertEqual(String(data: boolData, encoding: .utf8), "true")
    }

    // MARK: - Equality Tests

    func testEquality() throws {
        XCTAssertEqual(StringOrNumber.int(42), StringOrNumber.int(42))
        XCTAssertNotEqual(StringOrNumber.int(42), StringOrNumber.int(43))
        XCTAssertNotEqual(StringOrNumber.int(42), StringOrNumber.float(42.0))
        XCTAssertEqual(StringOrNumber.string("hello"), StringOrNumber.string("hello"))
    }
}

