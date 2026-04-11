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

    /// Search order: app bundle → App Group container → Documents
    private func findModel(named name: String) -> URL? {
        // 1. App bundle (compiled .mlmodelc)
        if let bundled = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return bundled
        }

        // 2. App Group container
        if let container = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        ) {
            let groupPath = container.appendingPathComponent("models/\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: groupPath.path) {
                return groupPath
            }
            // Also check for .mlpackage (Core ML compiles on first load)
            let pkgPath = container.appendingPathComponent("models/\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkgPath.path) {
                return pkgPath
            }
        }

        // 3. Documents directory (for manual sideloading via Files app)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        if let docsPath = docs?.appendingPathComponent("models/\(name).mlmodelc"),
           FileManager.default.fileExists(atPath: docsPath.path) {
            return docsPath
        }

        return nil
    }

    func loadModels() async {
        #if canImport(CoreML)
        // LaBSE
        labseState = .loading
        if let labseURL = findModel(named: "LaBSE-full") {
            do {
                try await labse.load(modelURL: labseURL)
                labseState = .ready
                statusMessage = "LaBSE loaded"
            } catch {
                labseState = .error
                statusMessage = "LaBSE error: \(error.localizedDescription)"
            }
        } else {
            labseState = .notFound
            statusMessage = "LaBSE model not found — see Sync tab for setup"
        }

        // CLIP
        clipState = .loading
        if let clipURL = findModel(named: "CLIP-full") {
            do {
                try await clip.load(modelURL: clipURL)
                clipState = .ready
            } catch {
                clipState = .error
            }
        } else {
            clipState = .notFound
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
        guard labseState == .ready else {
            throw ModelError.notReady(labseState.rawValue)
        }
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
