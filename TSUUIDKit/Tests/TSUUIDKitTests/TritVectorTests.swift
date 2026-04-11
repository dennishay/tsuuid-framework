import XCTest
@testable import TSUUIDKit

final class TritVectorTests: XCTestCase {
    func testInit() {
        let tv = TritVector()
        XCTAssertEqual(tv.trits.count, 81)
        XCTAssertTrue(tv.trits.allSatisfy { $0 == 0 })
    }

    func testDisplay() {
        var tv = TritVector()
        tv.trits[0] = 1
        tv.trits[1] = 0
        tv.trits[2] = -1
        let display = tv.display()
        XCTAssertTrue(display.hasPrefix("+·-"))
    }

    func testHammingDistance() {
        var a = TritVector()
        a.trits[0] = 1
        a.trits[1] = -1

        var b = TritVector()
        b.trits[0] = 1
        b.trits[1] = 1

        XCTAssertEqual(a.hammingDistance(to: b), 1)
    }

    func testL1Distance() {
        var a = TritVector()
        a.trits[0] = 1
        a.trits[1] = -1

        var b = TritVector()
        b.trits[0] = -1
        b.trits[1] = 1

        XCTAssertEqual(a.l1Distance(to: b), 4)
    }
}
