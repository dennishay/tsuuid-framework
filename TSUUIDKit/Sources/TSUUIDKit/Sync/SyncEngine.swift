import Foundation

/// Bidirectional delta sync via Dropbox folder.
/// Watches __768_sync/ for incoming deltas, writes outgoing deltas.
public final class SyncEngine: @unchecked Sendable {
    public enum Peer: String, Sendable {
        case mac
        case iphone
    }

    private let peer: Peer
    private let syncRoot: URL
    private let deltaEncoder: DeltaEncoder
    private var pendingDeltas: [SparseDelta] = []
    private let lock = NSLock()

    public var outgoingDir: URL {
        syncRoot.appendingPathComponent("deltas/\(peer.rawValue)")
    }

    public var incomingDir: URL {
        let other: Peer = (peer == .mac) ? .iphone : .mac
        return syncRoot.appendingPathComponent("deltas/\(other.rawValue)")
    }

    public var ackFile: URL {
        syncRoot.appendingPathComponent("ack/\(peer.rawValue).ack")
    }

    public init(peer: Peer, syncRoot: URL,
                deltaEncoder: DeltaEncoder = DeltaEncoder()) {
        self.peer = peer
        self.syncRoot = syncRoot
        self.deltaEncoder = deltaEncoder
    }

    public func queueDelta(_ delta: SparseDelta) {
        lock.lock()
        pendingDeltas.append(delta)
        lock.unlock()
    }

    /// Write all pending deltas as a timestamped bundle file.
    public func flush() throws {
        lock.lock()
        let deltas = pendingDeltas
        pendingDeltas = []
        lock.unlock()

        guard !deltas.isEmpty else { return }

        try FileManager.default.createDirectory(at: outgoingDir,
                                                 withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMddHHmm"
        let filename = "\(formatter.string(from: Date())).delta"
        let fileURL = outgoingDir.appendingPathComponent(filename)

        // Bundle: count(u32) + N × (length(u32) + delta bytes)
        var bundle = Data()
        var count = UInt32(deltas.count).littleEndian
        bundle.append(Data(bytes: &count, count: 4))

        for delta in deltas {
            let bytes = delta.toBytes()
            var length = UInt32(bytes.count).littleEndian
            bundle.append(Data(bytes: &length, count: 4))
            bundle.append(bytes)
        }

        try bundle.write(to: fileURL)
    }

    /// Read and apply incoming delta bundles. Returns count applied.
    public func pullIncoming(apply: (SparseDelta) throws -> Void) throws -> Int {
        let fm = FileManager.default
        guard fm.fileExists(atPath: incomingDir.path) else { return 0 }

        let lastAck = readAck()

        let files = try fm.contentsOfDirectory(at: incomingDir,
                                               includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "delta" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var applied = 0
        for file in files {
            let filename = file.deletingPathExtension().lastPathComponent
            guard filename > lastAck else { continue }

            let bundle = try Data(contentsOf: file)
            let deltas = parseDeltaBundle(bundle)

            for delta in deltas {
                try apply(delta)
                applied += 1
            }
            try writeAck(filename)
        }
        return applied
    }

    private func parseDeltaBundle(_ data: Data) -> [SparseDelta] {
        guard data.count >= 4 else { return [] }
        let count = data.withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self))
        }
        var offset = 4
        var deltas: [SparseDelta] = []

        for _ in 0..<count {
            guard offset + 4 <= data.count else { break }
            let length = Int(data.withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
            })
            offset += 4
            guard offset + length <= data.count else { break }
            let deltaData = Data(data[offset..<offset + length])
            deltas.append(SparseDelta.fromBytes(deltaData))
            offset += length
        }
        return deltas
    }

    private func readAck() -> String {
        (try? String(contentsOf: ackFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private func writeAck(_ timestamp: String) throws {
        try FileManager.default.createDirectory(
            at: ackFile.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try timestamp.write(to: ackFile, atomically: true, encoding: .utf8)
    }
}
