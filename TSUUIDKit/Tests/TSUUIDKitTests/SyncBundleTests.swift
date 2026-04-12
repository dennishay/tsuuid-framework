import XCTest
@testable import TSUUIDKit

final class SyncBundleTests: XCTestCase {
    private func fixedVec(_ seed: UInt8 = 7) -> Vector768 {
        var bytes = [UInt8](repeating: seed, count: Vector768.byteCount)
        for i in stride(from: 0, to: bytes.count, by: 4) { bytes[i] = UInt8(i & 0xff) }
        return Vector768.fromBytes(Data(bytes))
    }

    func testSingleRowRoundtrip() throws {
        let row = SyncRow(path: "emails/2026/04/abc.md",
                          title: "Test email",
                          domain: "cpc",
                          vec: fixedVec(),
                          version: 3)
        let text = try SyncBundle.encode([row])
        let decoded = try SyncBundle.decode(text)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded[0], row)
    }

    func testMultipleRowsPreserveOrder() throws {
        let rows = (0..<5).map {
            SyncRow(path: "p/\($0)", title: "t\($0)", domain: "d",
                    vec: fixedVec(UInt8($0)), version: $0 + 1)
        }
        let roundtrip = try SyncBundle.decode(SyncBundle.encode(rows))
        XCTAssertEqual(roundtrip, rows)
    }

    func testEmptyBundle() throws {
        XCTAssertEqual(try SyncBundle.decode(""), [])
        XCTAssertEqual(try SyncBundle.decode("\n\n  \n"), [])
    }

    func testUtf8Roundtrip() throws {
        let row = SyncRow(path: "données/café.md",
                          title: "naïve résumé ☕",
                          domain: "fr",
                          vec: fixedVec(),
                          version: 1)
        let decoded = try SyncBundle.decode(SyncBundle.encode([row]))
        XCTAssertEqual(decoded[0].path, row.path)
        XCTAssertEqual(decoded[0].title, row.title)
    }

    func testJsonlShapeIsOneObjectPerLine() throws {
        let rows = [
            SyncRow(path: "a", title: "", domain: "d", vec: fixedVec(1)),
            SyncRow(path: "b", title: "", domain: "d", vec: fixedVec(2)),
        ]
        let text = try SyncBundle.encode(rows)
        let lines = text.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 2)
        for line in lines {
            let data = line.data(using: .utf8)!
            let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            XCTAssertNotNil(obj)
            XCTAssertEqual(Set(obj!.keys),
                           Set(["path", "title", "domain", "vec_b64", "version", "encoded_at"]))
        }
    }

    func testWriteThenReadFile() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("synctest-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let rows = [SyncRow(path: "x", title: "", domain: "d", vec: fixedVec())]
        try SyncBundle.write(rows: rows, to: tmp)
        XCTAssertEqual(try SyncBundle.read(from: tmp), rows)
    }
}
