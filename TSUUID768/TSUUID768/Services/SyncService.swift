import Foundation
import TSUUIDKit

@MainActor
class SyncService: ObservableObject {
    enum ConnectionState: String {
        case disconnected, connecting, connected, syncing
    }

    @Published var state: ConnectionState = .disconnected
    @Published var lastSync: Date?
    @Published var pendingOutgoing: Int = 0
    @Published var pendingIncoming: Int = 0

    private var syncEngine: SyncEngine?

    /// On iOS, sync folder lives in the App Group container.
    /// Delta files are exchanged via Dropbox API (SwiftyDropbox),
    /// downloaded to this local staging directory.
    private var syncRoot: URL? {
        guard let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        ) else { return nil }
        let syncDir = container.appendingPathComponent("__768_sync")
        try? FileManager.default.createDirectory(at: syncDir,
                                                  withIntermediateDirectories: true)
        return syncDir
    }

    func connect() {
        guard let root = syncRoot else {
            state = .disconnected
            return
        }
        syncEngine = SyncEngine(peer: .iphone, syncRoot: root)
        state = .connected
    }

    func syncNow(knowledge: KnowledgeService) async {
        guard let engine = syncEngine else { return }
        state = .syncing

        // Pull incoming
        do {
            let applied = try engine.pullIncoming { delta in
                let encoder = DeltaEncoder()
                let stored = Vector768()  // placeholder — real impl looks up by path
                let vec = encoder.applyDelta(stored: stored, delta: delta)
                knowledge.insert(vec, path: "sync-\(UUID().uuidString.prefix(8))",
                               title: "Synced", domain: "sync")
            }
            pendingIncoming = 0
            if applied > 0 {
                lastSync = Date()
            }
        } catch {
            print("Sync pull failed: \(error)")
        }

        // Flush outgoing
        do {
            try engine.flush()
            pendingOutgoing = 0
            lastSync = Date()
        } catch {
            print("Sync flush failed: \(error)")
        }

        state = .connected
    }

    func queueDelta(_ delta: SparseDelta) {
        syncEngine?.queueDelta(delta)
        pendingOutgoing += 1
    }
}
