import Foundation

/// Sparse representation of a vector delta for transmission.
/// Wire format matches Python tsuuid.delta.SparseDelta.to_bytes() exactly.
///
/// Header (8 bytes): magic "TD" (2B) + version (uint16 LE) + n_changed (uint16 LE) + flags (uint8) + reserved (1B)
/// Body: N × (uint16 index LE) then N × (float16 value LE) = N × 4 bytes total
public struct SparseDelta: Sendable {
    public static let magic: (UInt8, UInt8) = (0x54, 0x44)  // "TD"
    public static let headerSize = 8
    private static let flagCheckpoint: UInt8 = 0x01
    public static let fullVecBytes = 768 * 4  // 3072

    public let indices: [UInt16]
    public let values: [Float16]
    public let version: UInt16
    public let isCheckpoint: Bool

    public init(indices: [UInt16], values: [Float16], version: UInt16, isCheckpoint: Bool) {
        precondition(indices.count == values.count)
        self.indices = indices
        self.values = values
        self.version = version
        self.isCheckpoint = isCheckpoint
    }

    public var nChanged: Int { indices.count }
    public var wireSize: Int { Self.headerSize + 4 * nChanged }

    public var compressionRatio: Float {
        guard wireSize > 0 else { return .infinity }
        return Float(Self.fullVecBytes) / Float(wireSize)
    }

    /// Serialize to compact binary wire format.
    /// Byte-identical to Python SparseDelta.to_bytes().
    public func toBytes() -> Data {
        var data = Data(capacity: wireSize)

        // Header
        data.append(Self.magic.0)
        data.append(Self.magic.1)
        var ver = version.littleEndian
        data.append(Data(bytes: &ver, count: 2))
        var nc = UInt16(nChanged).littleEndian
        data.append(Data(bytes: &nc, count: 2))
        data.append(isCheckpoint ? Self.flagCheckpoint : 0)
        data.append(0)

        // Indices
        for idx in indices {
            var le = idx.littleEndian
            data.append(Data(bytes: &le, count: 2))
        }
        // Values
        for val in values {
            var bits = val.bitPattern.littleEndian
            data.append(Data(bytes: &bits, count: 2))
        }

        return data
    }

    public static func fromBytes(_ data: Data) -> SparseDelta {
        precondition(data.count >= headerSize, "Data too short")
        precondition(data[0] == magic.0 && data[1] == magic.1, "Invalid magic")

        let version = data.withUnsafeBytes {
            UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
        }
        let nChanged = Int(data.withUnsafeBytes {
            UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: 4, as: UInt16.self))
        })
        let flags = data[6]

        var indices: [UInt16] = []
        var values: [Float16] = []

        let bodyStart = headerSize
        for i in 0..<nChanged {
            let idx = data.withUnsafeBytes {
                UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: bodyStart + i * 2, as: UInt16.self))
            }
            indices.append(idx)
        }

        let valStart = bodyStart + nChanged * 2
        for i in 0..<nChanged {
            let bits = data.withUnsafeBytes {
                UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: valStart + i * 2, as: UInt16.self))
            }
            values.append(Float16(bitPattern: bits))
        }

        return SparseDelta(indices: indices, values: values,
                          version: version, isCheckpoint: (flags & flagCheckpoint) != 0)
    }

    public func toBase64() -> String { toBytes().base64EncodedString() }

    public static func fromBase64(_ b64: String) -> SparseDelta {
        guard let data = Data(base64Encoded: b64) else { fatalError("Invalid base64") }
        return fromBytes(data)
    }
}
