import Foundation
import SQLite3

/// SQLite persistence for vectors.
/// Schema matches Mac's claude_home.db tsuuid_768 table exactly.
public final class VectorDatabase: @unchecked Sendable {
    private var db: OpaquePointer?
    private let lock = NSLock()

    public init(path: String) throws {
        var dbPtr: OpaquePointer?
        let rc = sqlite3_open_v2(path, &dbPtr,
                                  SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                                  nil)
        guard rc == SQLITE_OK, let opened = dbPtr else {
            throw DatabaseError.openFailed(rc)
        }
        self.db = opened
        // Enable WAL mode
        sqlite3_exec(db, "PRAGMA journal_mode=WAL", nil, nil, nil)
        try createTables()
    }

    deinit {
        if let db = db { sqlite3_close(db) }
    }

    private func createTables() throws {
        let sql = """
        CREATE TABLE IF NOT EXISTS tsuuid_768 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            title TEXT,
            vec BLOB,
            vec_b64 TEXT,
            domain TEXT,
            version INTEGER DEFAULT 1,
            encoded_at TEXT
        );
        CREATE TABLE IF NOT EXISTS tsuuid_768_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            version INTEGER NOT NULL,
            delta BLOB NOT NULL,
            delta_b64 TEXT,
            residual BLOB,
            applied_at TEXT NOT NULL,
            UNIQUE(path, version)
        );
        """
        lock.lock()
        defer { lock.unlock() }
        let rc = sqlite3_exec(db, sql, nil, nil, nil)
        guard rc == SQLITE_OK else { throw DatabaseError.execFailed(rc) }
    }

    public func store(path: String, title: String, vec: Vector768,
                      domain: String = "general") throws {
        let sql = """
        INSERT OR REPLACE INTO tsuuid_768 (path, title, vec, vec_b64, domain, encoded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        let vecBytes = vec.toBytes()
        let vecB64 = vec.toBase64()
        let now = ISO8601DateFormatter().string(from: Date())

        lock.lock()
        defer { lock.unlock() }

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DatabaseError.prepareFailed
        }
        defer { sqlite3_finalize(stmt) }

        sqlite3_bind_text(stmt, 1, (path as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 2, (title as NSString).utf8String, -1, nil)
        vecBytes.withUnsafeBytes {
            sqlite3_bind_blob(stmt, 3, $0.baseAddress, Int32(vecBytes.count), nil)
        }
        sqlite3_bind_text(stmt, 4, (vecB64 as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 5, (domain as NSString).utf8String, -1, nil)
        sqlite3_bind_text(stmt, 6, (now as NSString).utf8String, -1, nil)

        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw DatabaseError.stepFailed
        }
    }

    /// Load all vectors into a VectorStore.
    public func loadAll(into store: VectorStore) throws -> Int {
        let sql = "SELECT path, title, vec, domain, encoded_at FROM tsuuid_768"

        lock.lock()
        defer { lock.unlock() }

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw DatabaseError.prepareFailed
        }
        defer { sqlite3_finalize(stmt) }

        var count = 0
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let pathPtr = sqlite3_column_text(stmt, 0) else { continue }
            let path = String(cString: pathPtr)
            let blobPtr = sqlite3_column_blob(stmt, 2)
            let blobLen = sqlite3_column_bytes(stmt, 2)
            let domainPtr = sqlite3_column_text(stmt, 3)
            let domain = domainPtr != nil ? String(cString: domainPtr!) : "general"

            guard blobLen == Vector768.byteCount, let ptr = blobPtr else { continue }

            let data = Data(bytes: ptr, count: Int(blobLen))
            let vec = Vector768.fromBytes(data)
            let meta = VectorMeta(uuid: UUID(), source: path,
                                  domain: domain, encodedAt: Date())
            store.insert(vec, meta: meta)
            count += 1
        }
        return count
    }

    public func count() throws -> Int {
        lock.lock()
        defer { lock.unlock() }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM tsuuid_768",
                                 -1, &stmt, nil) == SQLITE_OK else {
            throw DatabaseError.prepareFailed
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_step(stmt)
        return Int(sqlite3_column_int(stmt, 0))
    }
}

public enum DatabaseError: Error, Sendable {
    case openFailed(Int32)
    case execFailed(Int32)
    case prepareFailed
    case stepFailed
}
