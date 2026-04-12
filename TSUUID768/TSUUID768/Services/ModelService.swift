import Foundation
import TSUUIDKit
#if canImport(CoreML)
import CoreML
#endif

@MainActor
class ModelService: ObservableObject {
    enum ModelState: String {
        case unloaded, loading, ready, error, notFound
    }

    @Published var labseState: ModelState = .unloaded
    @Published var clipState: ModelState = .unloaded
    @Published var statusMessage: String = ""

    #if canImport(CoreML)
    private let labse = LaBSEEncoder()
    private let clip = CLIPEncoder()
    #endif

    private func findModel(named name: String) -> URL? {
        let fm = FileManager.default

        if let bundled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return bundled
        }

        if let container = fm.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        ) {
            let path = container.appendingPathComponent("models/\(name).mlmodelc")
            if fm.fileExists(atPath: path.path) { return path }
        }

        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let path = docs.appendingPathComponent("models/\(name).mlmodelc")
            if fm.fileExists(atPath: path.path) { return path }
        }

        return nil
    }

    /// Don't call at startup. Models load on first encode to save memory.
    func loadModels() async {
        // Skip auto-loading — models are loaded lazily on first encodeText() call.
        // This saves ~1.1GB RAM at launch.
        statusMessage = "Models load on first search"
    }

    /// Lazy-load LaBSE on demand
    private func ensureLaBSELoaded() async throws {
        #if canImport(CoreML)
        guard labseState != .ready else { return }

        labseState = .loading
        statusMessage = "Loading LaBSE..."

        guard let url = findModel(named: "LaBSE-full") else {
            labseState = .notFound
            statusMessage = "LaBSE not found — download via Dropbox sync"
            throw ModelError.notReady("notFound")
        }

        do {
            try await labse.load(modelURL: url)
            labseState = .ready
            statusMessage = "LaBSE ready"
        } catch {
            labseState = .error
            statusMessage = "LaBSE error: \(error.localizedDescription)"
            throw error
        }
        #endif
    }

    func handleMemoryWarning() {
        #if canImport(CoreML)
        labse.unload()
        clip.unload()
        labseState = .unloaded
        clipState = .unloaded
        statusMessage = "Models unloaded (memory pressure)"
        #endif
    }

    func encodeText(_ text: String) async throws -> Vector768 {
        #if canImport(CoreML)
        try await ensureLaBSELoaded()
        return try await labse.encode(text)
        #else
        throw ModelError.notAvailable
        #endif
    }
}

enum ModelError: Error, LocalizedError {
    case notReady(String)
    case notAvailable

    var errorDescription: String? {
        switch self {
        case .notReady(let state): return "LaBSE model is \(state)"
        case .notAvailable: return "Core ML not available on this platform"
        }
    }
}
