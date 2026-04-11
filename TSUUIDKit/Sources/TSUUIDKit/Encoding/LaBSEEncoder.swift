#if canImport(CoreML)
import Foundation
import CoreML

/// Core ML wrapper for LaBSE text encoding.
/// Loads a converted LaBSE.mlmodelc, runs inference, returns Vector768.
public final class LaBSEEncoder: @unchecked Sendable {
    private var model: MLModel?
    private var projectionMatrix: [[Float]]?

    public var isLoaded: Bool { model != nil }

    public func load(modelURL: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)
    }

    public func setProjectionMatrix(_ matrix: [[Float]]) {
        projectionMatrix = matrix
    }

    /// Encode text → Vector768.
    /// Tokenization wired during model conversion (Task 13).
    public func encode(_ text: String) async throws -> Vector768 {
        guard model != nil else { throw EncoderError.modelNotLoaded }
        // Tokenization + inference stub — requires converted model
        throw EncoderError.tokenizationFailed
    }

    /// Encode text → TritVector (via projection + quantization).
    public func encodeToTrits(_ text: String) async throws -> (TritVector, UUID) {
        let vec = try await encode(text)
        guard let matrix = projectionMatrix else {
            throw EncoderError.projectionMatrixNotSet
        }
        return Projection.vectorToTSUUID(vec, matrix: matrix)
    }

    public func unload() { model = nil }
}

public enum EncoderError: Error, Sendable {
    case modelNotLoaded
    case projectionMatrixNotSet
    case tokenizationFailed
    case predictionFailed(String)
}
#endif
