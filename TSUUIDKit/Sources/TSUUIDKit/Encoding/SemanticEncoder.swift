import Foundation

/// Protocol for semantic encoders. Same contract as Python Encoder768.
public protocol SemanticEncoder: Sendable {
    func encode(_ text: String) async throws -> Vector768
    func encodeBatch(_ texts: [String]) async throws -> [Vector768]
    var isLoaded: Bool { get }
    func load() async throws
    func unload()
}

/// Shared projection logic: 768 → 81 trits.
/// Port of Python labse_backend.py projection + absmean quantization.
public enum Projection {
    /// Build the 81×768 projection matrix from axis definitions.
    /// direction[i] = normalize(embed(positive_text) - embed(negative_text))
    public static func buildProjectionMatrix(
        encode: ([String]) -> [[Float]]
    ) -> [[Float]] {
        let posTexts = SemanticDimensions.allAxes.map { "\($0.name): \($0.positive)" }
        let negTexts = SemanticDimensions.allAxes.map { "\($0.name): \($0.negative)" }

        let posEmbs = encode(posTexts)
        let negEmbs = encode(negTexts)

        var matrix: [[Float]] = []
        for i in 0..<TritVector.dimensions {
            var direction = zip(posEmbs[i], negEmbs[i]).map { $0 - $1 }
            let norm = sqrt(direction.reduce(0) { $0 + $1 * $1 })
            if norm > 1e-8 {
                direction = direction.map { $0 / norm }
            }
            matrix.append(direction)
        }
        return matrix
    }

    /// Project 768 → 81 floats using the projection matrix.
    public static func project(_ vec768: [Float], matrix: [[Float]]) -> [Float] {
        matrix.map { row in
            zip(row, vec768).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        }
    }

    /// BitNet b1.58 absmean ternary quantization.
    public static func quantizeAbsmean(_ values: [Float]) -> TritVector {
        let absValues = values.map { abs($0) }
        let threshold = max(absValues.reduce(0, +) / Float(absValues.count), 1e-4)

        let trits: [Int8] = values.map { v in
            if v > threshold { return 1 }
            if v < -threshold { return -1 }
            return 0
        }
        return TritVector(trits: trits)
    }

    /// Full pipeline: 768 vector → 81 trits → UUID v8.
    public static func vectorToTSUUID(_ vec768: Vector768,
                                       matrix: [[Float]]) -> (TritVector, UUID) {
        let projected = project(vec768.toFloat32(), matrix: matrix)
        let trits = quantizeAbsmean(projected)
        let uuid = Packing.packToUUID(trits)
        return (trits, uuid)
    }
}
