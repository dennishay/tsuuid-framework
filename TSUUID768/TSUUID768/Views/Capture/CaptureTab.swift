import SwiftUI
import TSUUIDKit
import Vision

struct CaptureTab: View {
    enum CaptureMode: String, CaseIterable {
        case text = "Text"
        case document = "Scan"
        case photo = "Photo"

        var icon: String {
            switch self {
            case .text: return "text.cursor"
            case .document: return "doc.viewfinder"
            case .photo: return "camera"
            }
        }
    }

    @EnvironmentObject var knowledge: KnowledgeService
    @EnvironmentObject var model: ModelService
    @State private var mode: CaptureMode = .text
    @State private var inputText = ""
    @State private var title = ""
    @State private var domain = "general"
    @State private var isEncoding = false
    @State private var showConfirmation = false
    @State private var showScanner = false
    @State private var showCamera = false
    @State private var confirmMessage = ""

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Picker("Mode", selection: $mode) {
                    ForEach(CaptureMode.allCases, id: \.self) { m in
                        Label(m.rawValue, systemImage: m.icon).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                switch mode {
                case .text:
                    textInputView
                case .document:
                    documentModeView
                case .photo:
                    photoModeView
                }

                Spacer()
            }
            .navigationTitle("Capture")
            .alert("Encoded", isPresented: $showConfirmation) {
                Button("OK") { }
            } message: {
                Text(confirmMessage)
            }
            .sheet(isPresented: $showScanner) {
                DocumentScannerView(
                    onFinish: { text in
                        inputText = text
                        showScanner = false
                        Task { await encodeScanText() }
                    },
                    onCancel: { showScanner = false }
                )
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showCamera) {
                CameraView(
                    sourceType: .camera,
                    onFinish: { image in
                        showCamera = false
                        Task { await encodePhoto(image) }
                    },
                    onCancel: { showCamera = false }
                )
                .ignoresSafeArea()
            }
        }
    }

    // MARK: - Text mode

    private var textInputView: some View {
        VStack(spacing: 12) {
            TextField("Title", text: $title)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            TextField("Domain (e.g. cpc, tssa)", text: $domain)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            TextEditor(text: $inputText)
                .frame(minHeight: 150)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.3))
                )
                .padding(.horizontal)

            Button {
                Task { await encodeText() }
            } label: {
                HStack {
                    if isEncoding {
                        ProgressView().tint(.white)
                    }
                    Text(isEncoding ? "Encoding..." : "Encode")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(inputText.isEmpty || isEncoding)
            .padding(.horizontal)
        }
    }

    // MARK: - Document mode

    private var documentModeView: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.viewfinder")
                .font(.system(size: 80))
                .foregroundStyle(.tint)
                .padding(.top, 40)

            Text("Scan Documents")
                .font(.title2)
                .fontWeight(.semibold)

            Text("VisionKit scans + Live Text OCR + LaBSE encoding")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            TextField("Domain", text: $domain)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            Button {
                showScanner = true
            } label: {
                HStack {
                    Image(systemName: "doc.viewfinder")
                    Text("Open Scanner")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)

            if isEncoding {
                ProgressView("Encoding scanned text...")
            }
        }
    }

    // MARK: - Photo mode

    private var photoModeView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera")
                .font(.system(size: 80))
                .foregroundStyle(.tint)
                .padding(.top, 40)

            Text("Photo Capture")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Snap a photo — CLIP encodes the visual content")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            TextField("Domain", text: $domain)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            Button {
                showCamera = true
            } label: {
                HStack {
                    Image(systemName: "camera")
                    Text("Open Camera")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal)

            if isEncoding {
                ProgressView("Encoding photo...")
            }

            Text("Note: CLIP wiring is in progress. Photos are saved with Live Text OCR fallback.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }

    // MARK: - Encoding

    private func encodeText() async {
        isEncoding = true
        do {
            let vec = try await model.encodeText(inputText)
            let name = title.isEmpty ? String(inputText.prefix(40)) : title
            knowledge.insert(vec, path: "capture/\(UUID().uuidString.prefix(8))",
                           title: name, domain: domain)
            confirmMessage = "Added to graph (\(knowledge.vectorCount) vectors)"
            inputText = ""
            title = ""
            showConfirmation = true
        } catch {
            confirmMessage = "Failed: \(error.localizedDescription)"
            showConfirmation = true
        }
        isEncoding = false
    }

    private func encodeScanText() async {
        guard !inputText.isEmpty else { return }
        isEncoding = true
        do {
            let vec = try await model.encodeText(inputText)
            let preview = String(inputText.prefix(60))
            knowledge.insert(vec, path: "scan/\(UUID().uuidString.prefix(8))",
                           title: preview, domain: domain)
            confirmMessage = "Scanned + encoded (\(knowledge.vectorCount) vectors)"
            inputText = ""
            showConfirmation = true
        } catch {
            confirmMessage = "Encoding failed: \(error.localizedDescription)"
            showConfirmation = true
        }
        isEncoding = false
    }

    /// Photo: upload to Dropbox → tinyurl → OCR → encode with link as source.
    /// The tinyurl ID becomes the retrieval key — tap a search result to open the image.
    private func encodePhoto(_ image: UIImage) async {
        isEncoding = true
        guard let cgImage = image.cgImage else {
            confirmMessage = "Couldn't read image"
            showConfirmation = true
            isEncoding = false
            return
        }

        // 1. Upload to Dropbox → tinyurl
        let uploader = PhotoUploadService()
        var tinyurlId: String? = nil
        do {
            let result = try await uploader.upload(image)
            tinyurlId = result.tinyurlId
        } catch {
            confirmMessage = "Upload failed: \(error.localizedDescription)"
            showConfirmation = true
            isEncoding = false
            return
        }

        // 2. OCR
        let extractedText = await extractText(from: cgImage)

        // 3. Encode — use OCR text if available, else a placeholder
        let textToEncode = extractedText.isEmpty
            ? "photo captured at \(Date().formatted())"
            : extractedText

        do {
            let vec = try await model.encodeText(textToEncode)
            let preview = extractedText.isEmpty
                ? "Photo (\(tinyurlId ?? "?"))"
                : String(extractedText.prefix(60))
            let source = tinyurlId.map { "tinyurl:\($0)" } ?? "photo/\(UUID().uuidString.prefix(8))"
            knowledge.insert(vec, path: source, title: preview, domain: domain)
            confirmMessage = "Photo → tinyurl/\(tinyurlId ?? "?") (\(knowledge.vectorCount) vectors)"
            showConfirmation = true
        } catch {
            confirmMessage = "Encoding failed: \(error.localizedDescription)"
            showConfirmation = true
        }
        isEncoding = false
    }

    private func extractText(from cgImage: CGImage) async -> String {
        await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { req, _ in
                let text = (req.results as? [VNRecognizedTextObservation])?
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n") ?? ""
                continuation.resume(returning: text)
            }
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            try? handler.perform([request])
        }
    }
}
