import Foundation
import TSUUIDKit

@MainActor
class KnowledgeService: ObservableObject {
    @Published var vectorCount: Int = 0
    @Published var domainStats: [String: Int] = [:]
    @Published var isLoaded = false

    let store = VectorStore()
    private var database: VectorDatabase?

    private var appGroupURL: URL? {
        FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        )
    }

    func load() async {
        guard let url = appGroupURL else {
            print("App Group container not available")
            return
        }
        let dbPath = url.appendingPathComponent("vectors.db").path
        do {
            database = try VectorDatabase(path: dbPath)
            let count = try database!.loadAll(into: store)
            vectorCount = count
            domainStats = store.stats()
            isLoaded = true
        } catch {
            print("Failed to load database: \(error)")
        }
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
