import XCTest
@testable import TSUUIDKit

final class PackingTests: XCTestCase {
    // UUID v8 version/variant bits overwrite bytes 6 and 8,
    // corrupting trits at dimensions 30-34 and 40-44.
    // Same behavior as the Python implementation.
    static let corruptedDims: Set<Int> = Set(30...34).union(Set(40...44))

    func testRoundTripZero() {
        let original = TritVector()
        let uuid = Packing.packToUUID(original)
        let restored = Packing.unpackFromUUID(uuid)

        for i in 0..<81 where !Self.corruptedDims.contains(i) {
            XCTAssertEqual(original.trits[i], restored.trits[i],
                           "Mismatch at dimension \(i)")
        }
    }

    func testRoundTripMixed() {
        var original = TritVector()
        original.trits[0] = 1
        original.trits[5] = -1
        original.trits[20] = 1
        original.trits[60] = -1

        let uuid = Packing.packToUUID(original)
        let restored = Packing.unpackFromUUID(uuid)

        for i in 0..<81 where !Self.corruptedDims.contains(i) {
            XCTAssertEqual(original.trits[i], restored.trits[i],
                           "Mismatch at dimension \(i)")
        }
    }

    func testUUIDVersion8() {
        let tv = TritVector()
        let uuid = Packing.packToUUID(tv)
        // Version nibble is byte 6 high nibble
        XCTAssertEqual(uuid.uuid.6 >> 4, 8)
    }

    func testUUIDVariantRFC4122() {
        let tv = TritVector()
        let uuid = Packing.packToUUID(tv)
        // Variant is top 2 bits of byte 8
        XCTAssertEqual(uuid.uuid.8 >> 6, 2)
    }

    func testEncodeTritGroup() {
        let val = Packing.encodeTritGroup([0, 0, 0, 0, 0])
        XCTAssertEqual(val, 121)
    }

    func testDecodeTritGroup() {
        let trits = Packing.decodeTritGroup(121, count: 5)
        XCTAssertEqual(trits, [0, 0, 0, 0, 0])
    }

    func testTritGroupRoundTrip() {
        let original: [Int8] = [1, -1, 0, 1, -1]
        let encoded = Packing.encodeTritGroup(original)
        let decoded = Packing.decodeTritGroup(encoded, count: 5)
        XCTAssertEqual(original, decoded)
    }
}
