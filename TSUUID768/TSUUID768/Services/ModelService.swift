import Foundation
import TSUUIDKit
#if canImport(CoreML)
import CoreML
#endif

@MainActor
class ModelService: ObservableObject {
    enum ModelState: String {
        case unloaded, loading, ready, error
    }

    @Published var labseState: ModelState = .unloaded
    @Published var clipState: ModelState = .unloaded

    #if canImport(CoreML)
    private let labse = LaBSEEncoder()
    private let clip = CLIPEncoder()
    #endif

    private var appGroupURL: URL? {
        FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        )
    }

    func loadModels() async {
        #if canImport(CoreML)
        guard let url = appGroupURL else { return }

        labseState = .loading
        do {
            try await labse.load(modelURL: url.appendingPathComponent("models/LaBSE-full.mlmodelc"))
            labseState = .ready
        } catch {
            labseState = .error
        }

        clipState = .loading
        do {
            try await clip.load(modelURL: url.appendingPathComponent("models/CLIP-full.mlmodelc"))
            clipState = .ready
        } catch {
            clipState = .error
        }
        #endif
    }

    func handleMemoryWarning() {
        #if canImport(CoreML)
        labse.unload()
        clip.unload()
        labseState = .unloaded
        clipState = .unloaded
        #endif
    }

    func encodeText(_ text: String) async throws -> Vector768 {
        #if canImport(CoreML)
        return try await labse.encode(text)
        #else
        fatalError("Core ML not available on this platform")
        #endif
    }
}
