import Foundation

/// Ternary ↔ UUID v8 bit packing.
/// Port of Python tsuuid.packing — same algorithm, same output.
///
/// Packing strategy:
///   Groups of 5 trits → 1 byte (3^5 = 243 < 256)
///   81 trits = 16 groups of 5 + 1 remainder = 17 bytes
///   122 custom bits in UUID v8 carry the payload
///   Version = 8, Variant = RFC 4122
public enum Packing {
    /// Map trit {-1, 0, +1} to unsigned {0, 1, 2}
    private static let tritToUnsigned: [Int8: UInt8] = [-1: 0, 0: 1, 1: 2]
    private static let unsignedToTrit: [UInt8: Int8] = [0: -1, 1: 0, 2: 1]

    /// Encode up to 5 trits into a single byte (0–242).
    /// val = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4  (base-3)
    public static func encodeTritGroup(_ trits: [Int8]) -> UInt8 {
        var val: UInt16 = 0
        var multiplier: UInt16 = 1
        for t in trits {
            let unsigned = tritToUnsigned[t] ?? 1
            val += UInt16(unsigned) * multiplier
            multiplier *= 3
        }
        return UInt8(val)
    }

    /// Decode a byte (0–242) back to trits.
    public static func decodeTritGroup(_ val: UInt8, count: Int = 5) -> [Int8] {
        var v = val
        var trits: [Int8] = []
        for _ in 0..<count {
            let remainder = v % 3
            trits.append(unsignedToTrit[remainder] ?? 0)
            v /= 3
        }
        return trits
    }

    /// Pack 81 trits into a UUID v8.
    /// Mirrors Python's pack_trits_to_uuid() — same algorithm.
    public static func packToUUID(_ tritVec: TritVector) -> UUID {
        let trits = tritVec.trits

        // Encode groups of 5 trits into bytes
        var encodedBytes: [UInt8] = []
        var i = 0
        while i < TritVector.dimensions {
            let end = min(i + 5, TritVector.dimensions)
            var group = Array(trits[i..<end])
            while group.count < 5 { group.append(0) }
            encodedBytes.append(encodeTritGroup(group))
            i += 5
        }

        // Build 128-bit value from first 16 bytes (big-endian packing)
        var hi: UInt64 = 0
        var lo: UInt64 = 0
        for j in 0..<min(encodedBytes.count, 8) {
            hi = (hi << 8) | UInt64(encodedBytes[j])
        }
        for j in 8..<min(encodedBytes.count, 16) {
            lo = (lo << 8) | UInt64(encodedBytes[j])
        }

        // Embed 17th byte in lower 8 bits if present
        if encodedBytes.count > 16 {
            lo = (lo & ~0xFF) | UInt64(encodedBytes[16])
        }

        // Set version = 8 (bits 48–51 of the 128-bit value)
        // In the hi word, version is at bits 12-15 (from LSB of hi)
        hi &= ~(UInt64(0xF) << 12)
        hi |= (UInt64(0x8) << 12)

        // Set variant = 0b10 (bits 64-65 of the 128-bit value)
        // In the lo word, variant is at bits 62-63
        lo &= ~(UInt64(0x3) << 62)
        lo |= (UInt64(0x2) << 62)

        // Convert to uuid_t (16 bytes, big-endian)
        var bytes = uuid_t(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        withUnsafeMutableBytes(of: &bytes) { buf in
            for k in 0..<8 {
                buf[7 - k] = UInt8(hi & 0xFF)
                hi >>= 8
            }
            for k in 0..<8 {
                buf[15 - k] = UInt8(lo & 0xFF)
                lo >>= 8
            }
        }

        return UUID(uuid: bytes)
    }

    /// Unpack a UUID v8 back to 81 trits.
    /// Mirrors Python's unpack_uuid_to_trits().
    public static func unpackFromUUID(_ uuid: UUID) -> TritVector {
        let bytes = uuid.uuid

        // Extract 16 raw bytes
        let rawBytes: [UInt8] = withUnsafeBytes(of: bytes) { Array($0) }

        // Decode trit groups from each byte
        var trits: [Int8] = []
        for byte in rawBytes.prefix(16) {
            let clamped = min(byte, 242)
            let group = decodeTritGroup(clamped, count: 5)
            trits.append(contentsOf: group)
        }

        // Truncate to 81
        while trits.count < TritVector.dimensions { trits.append(0) }
        trits = Array(trits.prefix(TritVector.dimensions))

        return TritVector(trits: trits)
    }
}
