import XCTest
@testable import TSUUIDKit

final class Vector768Tests: XCTestCase {
    func testRoundTripBytes() {
        var vec = Vector768()
        vec.storage[0] = Float16(1.0)
        vec.storage[767] = Float16(-0.5)

        let data = vec.toBytes()
        XCTAssertEqual(data.count, 1536)

        let restored = Vector768.fromBytes(data)
        XCTAssertEqual(restored.storage[0], Float16(1.0))
        XCTAssertEqual(restored.storage[767], Float16(-0.5))
    }

    func testRoundTripBase64() {
        var vec = Vector768()
        vec.storage[100] = Float16(0.75)

        let b64 = vec.toBase64()
        let restored = Vector768.fromBase64(b64)
        XCTAssertEqual(restored.storage[100], Float16(0.75))
    }

    func testCosineSimilarityIdentical() {
        var vec = Vector768()
        vec.storage[0] = Float16(1.0)
        vec.storage[1] = Float16(0.5)

        let sim = vec.cosineSimilarity(to: vec)
        XCTAssertEqual(sim, 1.0, accuracy: 0.001)
    }

    func testCosineSimilarityOrthogonal() {
        var a = Vector768()
        a.storage[0] = Float16(1.0)

        var b = Vector768()
        b.storage[1] = Float16(1.0)

        let sim = a.cosineSimilarity(to: b)
        XCTAssertEqual(sim, 0.0, accuracy: 0.001)
    }

    func testCosineDistance() {
        var vec = Vector768()
        vec.storage[0] = Float16(1.0)

        let dist = vec.cosineDistance(to: vec)
        XCTAssertEqual(dist, 0.0, accuracy: 0.001)
    }
}
