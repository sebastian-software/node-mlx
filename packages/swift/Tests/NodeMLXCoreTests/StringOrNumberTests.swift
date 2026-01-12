// Copyright © 2024 Sebastian Software GmbH. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Tests for StringOrNumber JSON type handling.

import Foundation
@testable import NodeMLXCore
import XCTest

final class StringOrNumberTests: XCTestCase {
    // MARK: - Decoding Tests

    func testDecodeString() throws {
        let json = "\"hello\""
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case let .string(s) = value {
            XCTAssertEqual(s, "hello")
        } else {
            XCTFail("Expected string")
        }
    }

    func testDecodeInt() throws {
        let json = "42"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case let .int(i) = value {
            XCTAssertEqual(i, 42)
        } else {
            XCTFail("Expected int")
        }
    }

    func testDecodeDouble() throws {
        let json = "3.14"
        let value = try JSONDecoder().decode(StringOrNumber.self, from: json.data(using: .utf8)!)

        if case let .double(d) = value {
            XCTAssertEqual(d, 3.14, accuracy: 0.001)
        } else {
            XCTFail("Expected double")
        }
    }

    // MARK: - Value Accessor Tests

    func testStringValue() throws {
        XCTAssertEqual(StringOrNumber.string("hello").stringValue, "hello")
        XCTAssertEqual(StringOrNumber.int(42).stringValue, "42")
        XCTAssertEqual(StringOrNumber.double(3.14).stringValue, "3.14")
    }

    func testIntValue() throws {
        XCTAssertEqual(StringOrNumber.int(42).intValue, 42)
        XCTAssertEqual(StringOrNumber.double(3.0).intValue, 3) // Converts
        XCTAssertNil(StringOrNumber.string("hello").intValue)
    }

    func testDoubleValue() throws {
        XCTAssertEqual(StringOrNumber.double(3.14).doubleValue, 3.14)
        XCTAssertEqual(StringOrNumber.int(42).doubleValue, 42.0)
        XCTAssertNil(StringOrNumber.string("hello").doubleValue)
    }

    func testFloatValue() throws {
        XCTAssertEqual(StringOrNumber.double(3.14).floatValue!, 3.14, accuracy: 0.001)
        XCTAssertEqual(StringOrNumber.int(42).floatValue!, 42.0, accuracy: 0.001)
        XCTAssertNil(StringOrNumber.string("hello").floatValue)
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

        if case let .string(type) = config["type"] {
            XCTAssertEqual(type, "linear")
        } else {
            XCTFail("Expected string type")
        }

        XCTAssertEqual(config["factor"]?.floatValue, 2.0)
    }

    func testQuantizationConfig() throws {
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

        XCTAssertEqual(config["group_size"]?.intValue, 64)
        XCTAssertEqual(config["bits"]?.intValue, 4)
    }

    // MARK: - Encoding Tests

    func testEncode() throws {
        let encoder = JSONEncoder()

        let stringData = try encoder.encode(StringOrNumber.string("test"))
        XCTAssertEqual(String(data: stringData, encoding: .utf8), "\"test\"")

        let intData = try encoder.encode(StringOrNumber.int(42))
        XCTAssertEqual(String(data: intData, encoding: .utf8), "42")

        let doubleData = try encoder.encode(StringOrNumber.double(3.14))
        XCTAssertTrue(String(data: doubleData, encoding: .utf8)?.contains("3.14") == true)
    }

    // MARK: - Equality Tests

    func testEquality() throws {
        XCTAssertEqual(StringOrNumber.int(42), StringOrNumber.int(42))
        XCTAssertNotEqual(StringOrNumber.int(42), StringOrNumber.int(43))
        XCTAssertNotEqual(StringOrNumber.int(42), StringOrNumber.double(42.0))
        XCTAssertEqual(StringOrNumber.string("hello"), StringOrNumber.string("hello"))
    }

    // MARK: - Dictionary Extension Tests

    func testAsAnyDict() throws {
        let dict: [String: StringOrNumber] = [
            "name": .string("test"),
            "count": .int(42),
            "ratio": .double(3.14),
        ]

        let anyDict = dict.asAnyDict
        XCTAssertEqual(anyDict["name"] as? String, "test")
        XCTAssertEqual(anyDict["count"] as? Int, 42)
        XCTAssertEqual(anyDict["ratio"] as? Double, 3.14)
    }
}
