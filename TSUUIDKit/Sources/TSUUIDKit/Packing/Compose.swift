import Foundation

/// UUID composition by ternary addition.
/// Port of Python tsuuid.compose — same operations.
public enum Compose {
    public enum DistanceMetric: Sendable {
        case hamming
        case l1
        case cosine
    }

    /// Compose multiple UUIDs by ternary vector addition.
    /// Result is the sign of the sum per dimension, clamped to {-1, 0, +1}.
    /// Commutative and associative.
    public static func compose(_ uuids: [UUID]) -> TritVector {
        guard !uuids.isEmpty else { return TritVector() }

        var total = [Float](repeating: 0, count: TritVector.dimensions)
        for uuid in uuids {
            let tv = Packing.unpackFromUUID(uuid)
            for i in 0..<TritVector.dimensions {
                total[i] += Float(tv.trits[i])
            }
        }

        let composed = total.map { v -> Int8 in
            if v > 0 { return 1 }
            if v < 0 { return -1 }
            return 0
        }
        return TritVector(trits: composed)
    }

    /// Semantic distance between two UUIDs.
    public static func semanticDistance(_ a: UUID, _ b: UUID,
                                        metric: DistanceMetric = .hamming) -> Float {
        let tvA = Packing.unpackFromUUID(a)
        let tvB = Packing.unpackFromUUID(b)

        switch metric {
        case .hamming:
            return Float(tvA.hammingDistance(to: tvB))
        case .l1:
            return Float(tvA.l1Distance(to: tvB))
        case .cosine:
            return tvA.cosineDistance(to: tvB)
        }
    }

    /// Find dimensions where both UUIDs have the same non-zero trit.
    /// Returns 1-indexed dimension numbers.
    public static func sharedDimensions(_ a: UUID, _ b: UUID) -> [Int] {
        let tvA = Packing.unpackFromUUID(a)
        let tvB = Packing.unpackFromUUID(b)

        var shared: [Int] = []
        for i in 0..<TritVector.dimensions {
            if tvA.trits[i] != 0 && tvA.trits[i] == tvB.trits[i] {
                shared.append(i + 1)
            }
        }
        return shared
    }

    /// Ternary difference: what changed from a to b.
    public static func diff(_ a: UUID, _ b: UUID) -> TritVector {
        let tvA = Packing.unpackFromUUID(a)
        let tvB = Packing.unpackFromUUID(b)
        return tvA.diff(to: tvB)
    }
}
