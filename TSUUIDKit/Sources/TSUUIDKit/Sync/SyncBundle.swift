import Foundation

/// Wire schema version. Bumped when ``SyncRow`` field layout changes in a
/// way older readers cannot safely ignore. Readers MUST refuse rows with
/// ``schema_version > syncSchemaVersion`` (forward-compat drift guard).
public let syncSchemaVersion: Int = 1

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
    /// Wire schema version for this row. Defaults to ``syncSchemaVersion``.
    /// Optional on decode so pre-schema_version bundles continue to parse
    /// (treated as schema_version = 1).
    public let schema_version: Int

    public init(path: String, title: String, domain: String,
                vec_b64: String, version: Int = 1,
                encoded_at: String? = nil,
                schema_version: Int = syncSchemaVersion) {
        self.path = path
        self.title = title
        self.domain = domain
        self.vec_b64 = vec_b64
        self.version = version
        self.encoded_at = encoded_at
        self.schema_version = schema_version
    }

    public init(path: String, title: String, domain: String,
                vec: Vector768, version: Int = 1,
                encodedAt: Date = Date(),
                schema_version: Int = syncSchemaVersion) {
        self.path = path
        self.title = title
        self.domain = domain
        self.vec_b64 = vec.toBase64()
        self.version = version
        let fmt = ISO8601DateFormatter()
        self.encoded_at = fmt.string(from: encodedAt)
        self.schema_version = schema_version
    }

    // Custom Decodable to tolerate legacy bundles without schema_version —
    // they decode as schema_version = 1 (the original wire version).
    private enum CodingKeys: String, CodingKey {
        case path, title, domain, vec_b64, version, encoded_at, schema_version
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.path = try c.decode(String.self, forKey: .path)
        self.title = (try? c.decode(String.self, forKey: .title)) ?? ""
        self.domain = (try? c.decode(String.self, forKey: .domain)) ?? "general"
        self.vec_b64 = try c.decode(String.self, forKey: .vec_b64)
        self.version = (try? c.decode(Int.self, forKey: .version)) ?? 1
        self.encoded_at = try? c.decode(String.self, forKey: .encoded_at)
        self.schema_version = (try? c.decode(Int.self, forKey: .schema_version)) ?? 1
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
