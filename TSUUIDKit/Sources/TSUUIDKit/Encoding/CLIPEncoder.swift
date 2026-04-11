#if canImport(CoreML) && canImport(CoreImage)
import Foundation
import CoreML
import CoreImage

/// Core ML wrapper for CLIP visual encoding.
/// Loads a converted CLIP-ViT-B/32.mlmodelc, encodes images → Vector768.
public final class CLIPEncoder: @unchecked Sendable {
    private var model: MLModel?

    public var isLoaded: Bool { model != nil }

    public func load(modelURL: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)
    }

    /// Encode an image → Vector768.
    /// Image preprocessing wired during model conversion (Task 13).
    public func encode(_ image: CGImage) async throws -> Vector768 {
        guard model != nil else { throw EncoderError.modelNotLoaded }
        throw EncoderError.predictionFailed("Image preprocessing not yet wired")
    }

    public func unload() { model = nil }
}
#endif
