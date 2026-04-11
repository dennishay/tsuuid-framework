import XCTest
@testable import TSUUIDKit

final class ComposeTests: XCTestCase {
    func testSemanticDistanceIdentical() {
        var tv = TritVector()
        tv.trits[0] = 1
        let a = Packing.packToUUID(tv)
        let dist = Compose.semanticDistance(a, a, metric: .hamming)
        XCTAssertEqual(dist, 0)
    }

    func testSemanticDistanceDifferent() {
        var tvA = TritVector()
        tvA.trits[0] = 1

        var tvB = TritVector()
        tvB.trits[0] = -1

        let a = Packing.packToUUID(tvA)
        let b = Packing.packToUUID(tvB)

        let dist = Compose.semanticDistance(a, b, metric: .hamming)
        XCTAssertGreaterThan(dist, 0)
    }

    func testComposeCommutative() {
        var tvA = TritVector()
        tvA.trits[0] = 1

        var tvB = TritVector()
        tvB.trits[1] = -1

        let a = Packing.packToUUID(tvA)
        let b = Packing.packToUUID(tvB)

        let ab = Compose.compose([a, b])
        let ba = Compose.compose([b, a])
        XCTAssertEqual(ab.trits, ba.trits)
    }

    func testSharedDimensions() {
        var tvA = TritVector()
        tvA.trits[0] = 1
        tvA.trits[5] = 1

        var tvB = TritVector()
        tvB.trits[0] = 1
        tvB.trits[5] = -1

        let a = Packing.packToUUID(tvA)
        let b = Packing.packToUUID(tvB)

        let shared = Compose.sharedDimensions(a, b)
        XCTAssertTrue(shared.contains(1))
        XCTAssertFalse(shared.contains(6))
    }

    func testDiffUUIDs() {
        var tvA = TritVector()
        tvA.trits[0] = 1

        var tvB = TritVector()
        tvB.trits[0] = -1

        let a = Packing.packToUUID(tvA)
        let b = Packing.packToUUID(tvB)

        let d = Compose.diff(a, b)
        XCTAssertEqual(d.trits[0], -1)
    }
}
