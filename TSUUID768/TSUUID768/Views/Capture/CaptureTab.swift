import SwiftUI
import TSUUIDKit

struct CaptureTab: View {
    enum CaptureMode: String, CaseIterable {
        case text = "Text"
        case document = "Document"
        case photo = "Photo"
        case voice = "Voice"

        var icon: String {
            switch self {
            case .text: return "text.cursor"
            case .document: return "doc.viewfinder"
            case .photo: return "camera"
            case .voice: return "mic"
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

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Mode picker
                Picker("Mode", selection: $mode) {
                    ForEach(CaptureMode.allCases, id: \.self) { m in
                        Label(m.rawValue, systemImage: m.icon).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                // Input area
                switch mode {
                case .text:
                    textInputView
                case .document:
                    placeholderView("Document Scanner", icon: "doc.viewfinder",
                                   detail: "VisionKit — coming soon")
                case .photo:
                    placeholderView("Photo Capture", icon: "camera",
                                   detail: "CLIP encoding — coming soon")
                case .voice:
                    placeholderView("Voice Capture", icon: "mic",
                                   detail: "Speech framework — coming soon")
                }

                Spacer()
            }
            .navigationTitle("Capture")
            .alert("Encoded", isPresented: $showConfirmation) {
                Button("OK") { }
            } message: {
                Text("Added to knowledge graph (\(knowledge.vectorCount) vectors)")
            }
        }
    }

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
                        ProgressView()
                            .tint(.white)
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

    private func placeholderView(_ title: String, icon: String, detail: String) -> some View {
        ContentUnavailableView(title, systemImage: icon,
                               description: Text(detail))
    }

    private func encodeText() async {
        isEncoding = true
        do {
            let vec = try await model.encodeText(inputText)
            let name = title.isEmpty ? String(inputText.prefix(40)) : title
            knowledge.insert(vec, path: "capture/\(UUID().uuidString.prefix(8))",
                           title: name, domain: domain)
            inputText = ""
            title = ""
            showConfirmation = true
        } catch {
            print("Encode failed: \(error)")
        }
        isEncoding = false
    }
}
