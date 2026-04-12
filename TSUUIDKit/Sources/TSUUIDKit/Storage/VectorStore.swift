import Foundation
import Accelerate

/// Metadata for a stored vector.
public struct VectorMeta: Sendable {
    public let uuid: UUID
    public let source: String
    public let domain: String
    public let encodedAt: Date
    public var flags: UInt8

    public init(uuid: UUID, source: String, domain: String,
                encodedAt: Date, flags: UInt8 = 0) {
        self.uuid = uuid
        self.source = source
        self.domain = domain
        self.encodedAt = encodedAt
        self.flags = flags
    }
}

/// Search result with metadata.
public struct SearchResult: Sendable {
    public let source: String
    public let domain: String
    public let uuid: UUID
    public let similarity: Float
    public let index: Int
}

/// In-memory vector store with brute-force cosine search via Accelerate.
public final class VectorStore: @unchecked Sendable {
    private var vectors: [[Float]]
    private var metadata: [VectorMeta]
    private let lock = NSLock()

    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return metadata.count
    }

    public init() {
        vectors = []
        metadata = []
    }

    public func insert(_ vec: Vector768, meta: VectorMeta) {
        let f32 = vec.toFloat32()
        lock.lock()
        vectors.append(f32)
        metadata.append(meta)
        lock.unlock()
    }

    /// Reset the store — used before reloading from a fresh checkpoint.
    public func clear() {
        lock.lock()
        vectors.removeAll(keepingCapacity: false)
        metadata.removeAll(keepingCapacity: false)
        lock.unlock()
    }

    public func search(_ query: Vector768, domain: String? = nil,
                       limit: Int = 10) -> [SearchResult] {
        let queryF32 = query.toFloat32()
        var queryNormSq: Float = 0
        vDSP_dotpr(queryF32, 1, queryF32, 1, &queryNormSq,
                   vDSP_Length(Vector768.dimensions))
        let queryNorm = sqrt(queryNormSq)
        guard queryNorm > 1e-8 else { return [] }

        lock.lock()
        let vecs = vectors
        let metas = metadata
        lock.unlock()

        var results: [SearchResult] = []

        for (i, storedF32) in vecs.enumerated() {
            if let domain = domain, metas[i].domain != domain { continue }

            var dot: Float = 0
            var storedNormSq: Float = 0
            vDSP_dotpr(queryF32, 1, storedF32, 1, &dot,
                       vDSP_Length(Vector768.dimensions))
            vDSP_dotpr(storedF32, 1, storedF32, 1, &storedNormSq,
                       vDSP_Length(Vector768.dimensions))
            let storedNorm = sqrt(storedNormSq)
            guard storedNorm > 1e-8 else { continue }

            results.append(SearchResult(
                source: metas[i].source,
                domain: metas[i].domain,
                uuid: metas[i].uuid,
                similarity: dot / (queryNorm * storedNorm),
                index: i
            ))
        }

        results.sort { $0.similarity > $1.similarity }
        return Array(results.prefix(limit))
    }

    public func vector(at index: Int) -> Vector768? {
        lock.lock()
        defer { lock.unlock() }
        guard index < vectors.count else { return nil }
        return Vector768(float32: vectors[index])
    }

    public func meta(at index: Int) -> VectorMeta? {
        lock.lock()
        defer { lock.unlock() }
        guard index < metadata.count else { return nil }
        return metadata[index]
    }

    public func stats() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        var counts: [String: Int] = [:]
        for m in metadata { counts[m.domain, default: 0] += 1 }
        return counts
    }
}
