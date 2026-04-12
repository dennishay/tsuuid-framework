import XCTest
@testable import TSUUIDKit

/// Cross-language wire parity — ensures the Python emitter
/// (`tsuuid.sync_bundle.write_sync_bundle` / `sync_emitter_768.SyncEmitter`)
/// produces files that Swift ``SyncBundle.decode`` parses correctly.
///
/// The fixture is a real Python-written bundle checked in at
/// ``TSUUIDKit/Tests/TSUUIDKitTests/Fixtures/python_bundle.jsonl``.
final class CrossParityTests: XCTestCase {
    private func fixturesDir() -> URL? {
        // Walk up from this source file to find the Fixtures directory.
        // SPM's `Bundle.module` is the cleanest path when resources are
        // declared, but we avoid adding a resource declaration for a
        // single test fixture — locate it via #filePath instead.
        let here = URL(fileURLWithPath: #filePath)
        let fixtures = here.deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
        return FileManager.default.fileExists(atPath: fixtures.path) ? fixtures : nil
    }

    func testPythonEmittedBundleParsesInSwift() throws {
        guard let fixtures = fixturesDir() else {
            throw XCTSkip("Fixtures directory missing — run sync_emitter on Mac to seed")
        }
        let bundle = fixtures.appendingPathComponent("python_bundle.jsonl")
        guard FileManager.default.fileExists(atPath: bundle.path) else {
            throw XCTSkip("python_bundle.jsonl fixture missing")
        }

        let rows = try SyncBundle.read(from: bundle)
        XCTAssertFalse(rows.isEmpty, "fixture should contain ≥ 1 row")

        for row in rows {
            // Every row must carry the full SyncRow contract.
            XCTAssertFalse(row.path.isEmpty)
            XCTAssertFalse(row.vec_b64.isEmpty)
            // Vec must decode to the expected byte count for Vector768 (float16 LE).
            let vec = row.toVector()
            XCTAssertEqual(vec.storage.count, Vector768.dimensions)
            // Path format emitted by /learn hook.
            XCTAssertTrue(row.path.hasPrefix("tsuuid:") || row.path.hasPrefix("session:"))
        }
    }
}
