import Foundation
import TSUUIDKit

@MainActor
class KnowledgeService: ObservableObject {
    @Published var vectorCount: Int = 0
    @Published var domainStats: [String: Int] = [:]
    @Published var isLoaded = false
    @Published var statusMessage: String = "Loading..."

    let store = VectorStore()
    private var database: VectorDatabase?

    /// On first launch, copy bundled checkpoint from app bundle to Documents.
    /// Subsequent launches read from Documents (which persists across app updates).
    private func bootstrapIfNeeded() {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let destPath = docs.appendingPathComponent("vectors.db")

        // Already exists — skip
        if fm.fileExists(atPath: destPath.path) { return }

        // Copy bundled checkpoint
        if let bundled = Bundle.main.url(forResource: "checkpoint-10k", withExtension: "db") {
            do {
                try fm.copyItem(at: bundled, to: destPath)
                statusMessage = "Bootstrapped from bundled checkpoint"
            } catch {
                statusMessage = "Bootstrap failed: \(error.localizedDescription)"
            }
        }
    }

    private func findDatabase() -> String? {
        let fm = FileManager.default

        // Documents (primary — writable, persists)
        if let d = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let path = d.appendingPathComponent("vectors.db").path
            if fm.fileExists(atPath: path) {
                statusMessage = "DB: Documents/vectors.db"
                return path
            }
        }

        // App Group container
        if let c = fm.containerURL(forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768") {
            let path = c.appendingPathComponent("vectors.db").path
            if fm.fileExists(atPath: path) {
                statusMessage = "DB: AppGroup/vectors.db"
                return path
            }
        }

        statusMessage = "No vector database found"
        return nil
    }

    func load() async {
        bootstrapIfNeeded()

        guard let dbPath = findDatabase() else { return }

        statusMessage = "Loading vectors..."
        store.clear()
        do {
            database = try VectorDatabase(path: dbPath)
            let count = try database!.loadAll(into: store)
            vectorCount = count
            domainStats = store.stats()
            isLoaded = true
            statusMessage = "Loaded \(count) vectors"
        } catch {
            statusMessage = "Load failed: \(error.localizedDescription)"
        }
    }

    /// Force reload — call after downloading a new checkpoint from Dropbox.
    func reload() async {
        database = nil
        await load()
    }

    func search(_ query: Vector768, domain: String? = nil,
                limit: Int = 10) -> [SearchResult] {
        store.search(query, domain: domain, limit: limit)
    }

    func insert(_ vec: Vector768, path: String, title: String,
                domain: String) {
        let meta = VectorMeta(uuid: UUID(), source: path,
                              domain: domain, encodedAt: Date())
        store.insert(vec, meta: meta)
        try? database?.store(path: path, title: title, vec: vec, domain: domain)
        vectorCount = store.count
        domainStats = store.stats()
    }
}
