#if canImport(CoreML)
import Foundation
import CoreML
import Tokenizers

/// Core ML wrapper for LaBSE text encoding.
/// Tokenizes with HuggingFace Tokenizers, runs Core ML inference, returns Vector768.
public final class LaBSEEncoder: @unchecked Sendable {
    private var model: MLModel?
    private var tokenizer: Tokenizer?
    private var projectionMatrix: [[Float]]?
    private let maxLength = 128

    public init() {}

    public var isLoaded: Bool { model != nil && tokenizer != nil }

    /// Load Core ML model and download tokenizer from HuggingFace Hub.
    public func load(modelURL: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)

        // Load the LaBSE tokenizer (WordPiece, same as Python)
        tokenizer = try await AutoTokenizer.from(pretrained: "sentence-transformers/LaBSE")
    }

    /// Load with a pre-loaded tokenizer (for offline / bundled use).
    public func load(modelURL: URL, tokenizer: Tokenizer) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try await MLModel.load(contentsOf: modelURL, configuration: config)
        self.tokenizer = tokenizer
    }

    public func setProjectionMatrix(_ matrix: [[Float]]) {
        projectionMatrix = matrix
    }

    /// Encode text → Vector768.
    public func encode(_ text: String) async throws -> Vector768 {
        guard let model = model else { throw EncoderError.modelNotLoaded }
        guard let tokenizer = tokenizer else { throw EncoderError.tokenizationFailed }

        // Tokenize
        let tokenIds = tokenizer.encode(text: text)
        var inputIds = tokenIds.map { Int32($0) }
        var attentionMask = [Int32](repeating: 1, count: inputIds.count)

        // Pad or truncate to maxLength
        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength))
            attentionMask = Array(attentionMask.prefix(maxLength))
        } else {
            let padding = maxLength - inputIds.count
            inputIds.append(contentsOf: [Int32](repeating: 0, count: padding))
            attentionMask.append(contentsOf: [Int32](repeating: 0, count: padding))
        }

        // Create MLMultiArray inputs
        let idsArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)

        for i in 0..<maxLength {
            idsArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: inputIds[i])
            maskArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: attentionMask[i])
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])

        // Run inference
        let output = try await model.prediction(from: provider)

        // Extract the 768-dim embedding from output
        // The Core ML model output name depends on the conversion — try common names
        guard let embedding = output.featureValue(for: "embedding")?.multiArrayValue
                ?? output.featureValue(for: "pooler_output")?.multiArrayValue
                ?? output.featureValue(for: "output")?.multiArrayValue else {
            // Try first available output
            guard let firstName = output.featureNames.first,
                  let firstArray = output.featureValue(for: firstName)?.multiArrayValue else {
                throw EncoderError.predictionFailed("No embedding output found")
            }
            return multiArrayToVector768(firstArray)
        }

        return multiArrayToVector768(embedding)
    }

    /// Encode text → TritVector (via projection + quantization).
    public func encodeToTrits(_ text: String) async throws -> (TritVector, UUID) {
        let vec = try await encode(text)
        guard let matrix = projectionMatrix else {
            throw EncoderError.projectionMatrixNotSet
        }
        return Projection.vectorToTSUUID(vec, matrix: matrix)
    }

    public func unload() {
        model = nil
        tokenizer = nil
    }

    // MARK: - Private

    private func multiArrayToVector768(_ array: MLMultiArray) -> Vector768 {
        // The output may be (1, 768) or (1, seq, 768) — take the pooled output
        let totalCount = array.count
        var floats = [Float](repeating: 0, count: 768)

        if totalCount == 768 {
            // Direct (768,) or (1, 768)
            for i in 0..<768 {
                floats[i] = array[i].floatValue
            }
        } else if totalCount > 768 {
            // (1, seq, 768) — take first token [CLS] as pooled
            for i in 0..<768 {
                floats[i] = array[i].floatValue
            }
        }

        return Vector768(float32: floats)
    }
}

public enum EncoderError: Error, Sendable {
    case modelNotLoaded
    case projectionMatrixNotSet
    case tokenizationFailed
    case predictionFailed(String)
}
#endif
