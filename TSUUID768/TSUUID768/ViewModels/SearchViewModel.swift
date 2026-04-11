import Foundation
import TSUUIDKit

@MainActor
class SearchViewModel: ObservableObject {
    @Published var query = ""
    @Published var results: [SearchResult] = []
    @Published var isSearching = false
    @Published var selectedDomain: String?

    private var debounceTask: Task<Void, Never>?

    func search(using knowledge: KnowledgeService, model: ModelService) {
        debounceTask?.cancel()
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            results = []
            return
        }

        debounceTask = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }

            isSearching = true
            do {
                let queryVec = try await model.encodeText(q)
                results = knowledge.search(queryVec, domain: selectedDomain)
            } catch {
                results = []
            }
            isSearching = false
        }
    }
}
