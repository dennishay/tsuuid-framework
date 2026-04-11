import Foundation

/// 81 ternary values {-1, 0, +1} — the compressed semantic identity.
/// Each trit is an independent axis of meaning. The trit pattern IS the meaning.
public struct TritVector: Sendable, Equatable {
    public static let dimensions = 81

    /// 81 values, each -1, 0, or +1.
    public var trits: [Int8]

    public init() {
        trits = [Int8](repeating: 0, count: Self.dimensions)
    }

    public init(trits: [Int8]) {
        precondition(trits.count == Self.dimensions,
                     "Expected \(Self.dimensions) trits, got \(trits.count)")
        precondition(trits.allSatisfy { $0 >= -1 && $0 <= 1 },
                     "All trits must be in {-1, 0, +1}")
        self.trits = trits
    }

    /// Human-readable: +1→"+", -1→"-", 0→"·"
    public func display() -> String {
        trits.map { t -> String in
            switch t {
            case  1: return "+"
            case -1: return "-"
            default: return "·"
            }
        }.joined()
    }

    /// Number of dimensions that differ.
    public func hammingDistance(to other: TritVector) -> Int {
        zip(trits, other.trits).reduce(0) { $0 + ($1.0 != $1.1 ? 1 : 0) }
    }

    /// Manhattan distance in ternary space.
    public func l1Distance(to other: TritVector) -> Int {
        zip(trits, other.trits).reduce(0) { $0 + Int(abs(Int16($1.0) - Int16($1.1))) }
    }

    /// Cosine distance in trit space. 0 = identical, 2 = opposite.
    public func cosineDistance(to other: TritVector) -> Float {
        let a = trits.map { Float($0) }
        let b = other.trits.map { Float($0) }
        let dot = zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(Float(0)) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(Float(0)) { $0 + $1 * $1 })
        guard normA > 1e-8 && normB > 1e-8 else { return 1.0 }
        return 1.0 - dot / (normA * normB)
    }

    /// Ternary difference: what changed between self and other.
    /// Clamped to {-1, 0, +1}.
    public func diff(to other: TritVector) -> TritVector {
        let diffTrits = zip(other.trits, trits).map { b, a -> Int8 in
            let d = Int16(b) - Int16(a)
            return Int8(max(-1, min(1, d)))
        }
        return TritVector(trits: diffTrits)
    }
}
