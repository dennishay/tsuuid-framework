import Foundation

/// Cross-device row sync bundle format.
///
/// Each bundle file is JSON Lines (one row per line). Each row carries the
/// full metadata needed to upsert a vector on the receiving side. This is
/// deliberately separate from the raw `SparseDelta` bundle in ``SyncEngine``:
/// SparseDelta has no `path` field (it was designed for intra-document
/// version diffing), so cross-device row sync needs its own envelope.
///
/// Wire format parity with Python ``tsuuid.sync_bundle`` module.
public struct SyncRow: Codable, Sendable, Equatable {
    public let path: String
    public let title: String
    public let domain: String
    public let vec_b64: String   // base64 of Vector768 float16 bytes (1536 B)
    public let version: Int
    public let encoded_at: String?

    public init(path: String, title: String, domain: String,
                vec_b64: String, version: Int = 1,
                encoded_at: String? = nil) {
        self.path = path
        self.title = title
        self.domain = domain
        self.vec_b64 = vec_b64
        self.version = version
        self.encoded_at = encoded_at
    }

    public init(path: String, title: String, domain: String,
                vec: Vector768, version: Int = 1,
                encodedAt: Date = Date()) {
        self.path = path
        self.title = title
        self.domain = domain
        self.vec_b64 = vec.toBase64()
        self.version = version
        let fmt = ISO8601DateFormatter()
        self.encoded_at = fmt.string(from: encodedAt)
    }

    public func toVector() -> Vector768 {
        Vector768.fromBase64(vec_b64)
    }
}

public enum SyncBundle {
    /// Serialize rows as JSONL text. UTF-8 safe. One trailing newline.
    public static func encode(_ rows: [SyncRow]) throws -> String {
        let encoder = JSONEncoder()
        // JSONL: one object per line. Ensure stable key order for diffability.
        encoder.outputFormatting = [.sortedKeys]
        var lines: [String] = []
        lines.reserveCapacity(rows.count)
        for row in rows {
            let data = try encoder.encode(row)
            guard let line = String(data: data, encoding: .utf8) else {
                throw SyncBundleError.encodingFailed
            }
            lines.append(line)
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Parse JSONL text into rows. Blank lines skipped.
    public static func decode(_ text: String) throws -> [SyncRow] {
        let decoder = JSONDecoder()
        var out: [SyncRow] = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard let data = line.data(using: .utf8) else {
                throw SyncBundleError.decodingFailed
            }
            out.append(try decoder.decode(SyncRow.self, from: data))
        }
        return out
    }

    /// Write a bundle file atomically.
    public static func write(rows: [SyncRow], to url: URL) throws {
        let text = try encode(rows)
        try text.data(using: .utf8)?.write(to: url, options: .atomic)
    }

    /// Read a bundle file.
    public static func read(from url: URL) throws -> [SyncRow] {
        let data = try Data(contentsOf: url)
        guard let text = String(data: data, encoding: .utf8) else {
            throw SyncBundleError.decodingFailed
        }
        return try decode(text)
    }
}

public enum SyncBundleError: Error, Sendable {
    case encodingFailed
    case decodingFailed
}
