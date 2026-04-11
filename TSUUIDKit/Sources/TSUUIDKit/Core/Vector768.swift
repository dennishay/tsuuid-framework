import Foundation
import Accelerate

/// 768-dimensional semantic vector. The meaning.
/// Storage is Float16 (1536 bytes), identical to Python's f16 format.
public struct Vector768: Sendable, Equatable {
    public static let dimensions = 768
    public static let byteCount = 768 * MemoryLayout<Float16>.size  // 1536

    public var storage: [Float16]

    public init() {
        storage = [Float16](repeating: 0, count: Self.dimensions)
    }

    public init(storage: [Float16]) {
        precondition(storage.count == Self.dimensions,
                     "Expected \(Self.dimensions) dimensions, got \(storage.count)")
        self.storage = storage
    }

    /// Initialize from Float32 array (cast down to Float16).
    public init(float32: [Float]) {
        precondition(float32.count == Self.dimensions)
        self.storage = float32.map { Float16($0) }
    }

    /// Lossless serialization: 768 × Float16 → 1536 bytes, little-endian.
    /// Byte-identical to Python's vec_to_f16_bytes().
    public func toBytes() -> Data {
        storage.withUnsafeBytes { Data($0) }
    }

    /// Deserialize from 1536 bytes.
    public static func fromBytes(_ data: Data) -> Vector768 {
        precondition(data.count == byteCount,
                     "Expected \(byteCount) bytes, got \(data.count)")
        let values = data.withUnsafeBytes {
            Array($0.bindMemory(to: Float16.self))
        }
        return Vector768(storage: values)
    }

    /// Base64 encoding of the Float16 bytes. ~2048 chars.
    public func toBase64() -> String {
        toBytes().base64EncodedString()
    }

    /// Decode from base64.
    public static func fromBase64(_ b64: String) -> Vector768 {
        guard let data = Data(base64Encoded: b64) else {
            fatalError("Invalid base64 string")
        }
        return fromBytes(data)
    }

    /// Convert to Float32 array for Accelerate operations.
    public func toFloat32() -> [Float] {
        storage.map { Float($0) }
    }

    /// Cosine similarity using Accelerate. 1.0 = identical, -1.0 = opposite.
    public func cosineSimilarity(to other: Vector768) -> Float {
        var a = toFloat32()
        var b = other.toFloat32()
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        vDSP_dotpr(&a, 1, &b, 1, &dot, vDSP_Length(Self.dimensions))
        vDSP_dotpr(&a, 1, &a, 1, &normA, vDSP_Length(Self.dimensions))
        vDSP_dotpr(&b, 1, &b, 1, &normB, vDSP_Length(Self.dimensions))

        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 1e-8 else { return 0 }
        return dot / denom
    }

    /// Cosine distance. 0 = identical, 2 = opposite.
    public func cosineDistance(to other: Vector768) -> Float {
        1.0 - cosineSimilarity(to: other)
    }
}
