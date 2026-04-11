import Foundation

/// Computes and encodes vector deltas for efficient transmission.
/// Port of Python tsuuid.delta.DeltaEncoder — same algorithm.
public final class DeltaEncoder: @unchecked Sendable {
    public let epsilon: Float
    public let autoCheckpointThreshold: Int
    private let lock = NSLock()
    private var residuals: [String: [Float]] = [:]

    public init(epsilon: Float = 0.001, autoCheckpointThreshold: Int = 700) {
        self.epsilon = epsilon
        self.autoCheckpointThreshold = autoCheckpointThreshold
    }

    public func computeDelta(old: Vector768, new: Vector768) -> [Float] {
        let oldF = old.toFloat32()
        let newF = new.toFloat32()
        return zip(newF, oldF).map { $0 - $1 }
    }

    public func sparsify(_ delta: [Float], docId: String? = nil,
                         epsilon: Float? = nil, version: UInt16 = 0) -> SparseDelta {
        let eps = epsilon ?? self.epsilon
        var adjusted = delta

        if let docId = docId {
            lock.lock()
            if let prev = residuals[docId] {
                for i in 0..<768 { adjusted[i] += prev[i] }
            }
            lock.unlock()
        }

        var nChanged = 0
        for i in 0..<768 where abs(adjusted[i]) > eps { nChanged += 1 }

        if nChanged > autoCheckpointThreshold {
            let indices = (0..<768).map { UInt16($0) }
            let values = adjusted.map { Float16($0) }
            if let docId = docId {
                lock.lock()
                residuals[docId] = [Float](repeating: 0, count: 768)
                lock.unlock()
            }
            return SparseDelta(indices: indices, values: values,
                             version: version, isCheckpoint: false)
        }

        var indices: [UInt16] = []
        var values: [Float16] = []
        for i in 0..<768 where abs(adjusted[i]) > eps {
            indices.append(UInt16(i))
            values.append(Float16(adjusted[i]))
        }

        if let docId = docId {
            var sparseFull = [Float](repeating: 0, count: 768)
            for (idx, val) in zip(indices, values) {
                sparseFull[Int(idx)] = Float(val)
            }
            let newResidual = zip(adjusted, sparseFull).map { $0 - $1 }
            lock.lock()
            residuals[docId] = newResidual
            lock.unlock()
        }

        return SparseDelta(indices: indices, values: values,
                         version: version, isCheckpoint: false)
    }

    public func applyDelta(stored: Vector768, delta: SparseDelta) -> Vector768 {
        if delta.isCheckpoint {
            var result = [Float](repeating: 0, count: 768)
            for (idx, val) in zip(delta.indices, delta.values) {
                result[Int(idx)] = Float(val)
            }
            return Vector768(float32: result)
        }

        var result = stored.toFloat32()
        for (idx, val) in zip(delta.indices, delta.values) {
            result[Int(idx)] += Float(val)
        }
        return Vector768(float32: result)
    }

    public func makeCheckpoint(_ vec: Vector768, version: UInt16 = 0) -> SparseDelta {
        SparseDelta(
            indices: (0..<768).map { UInt16($0) },
            values: vec.storage,
            version: version,
            isCheckpoint: true
        )
    }
}
