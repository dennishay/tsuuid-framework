import SwiftUI
import TSUUIDKit

struct SearchDetailView: View {
    let result: SearchResult
    @State private var imageLoadState: ImageLoadState = .idle

    enum ImageLoadState {
        case idle, loading, loaded(UIImage), failed
    }

    var body: some View {
        List {
            Section("Match") {
                LabeledContent("Similarity",
                    value: String(format: "%.4f", result.similarity))
                LabeledContent("Distance",
                    value: String(format: "%.4f", 1 - result.similarity))
            }

            Section("Source") {
                if let tinyurlId = tinyurlId {
                    // Tappable tinyurl link
                    if let url = URL(string: "https://tinyurl.com/\(tinyurlId)") {
                        Link(destination: url) {
                            HStack {
                                Image(systemName: "link")
                                Text("tinyurl/\(tinyurlId)")
                                Spacer()
                                Image(systemName: "arrow.up.right.square")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    // Inline image preview
                    imagePreview
                } else {
                    LabeledContent("Path", value: result.source)
                }
                LabeledContent("Domain", value: result.domain)
            }

            Section("Identity") {
                Text(result.uuid.uuidString)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
        }
        .navigationTitle("Detail")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            await loadImageIfNeeded()
        }
    }

    private var tinyurlId: String? {
        guard result.source.hasPrefix("tinyurl:") else { return nil }
        return String(result.source.dropFirst("tinyurl:".count))
    }

    @ViewBuilder
    private var imagePreview: some View {
        switch imageLoadState {
        case .idle, .loading:
            HStack {
                ProgressView()
                Text("Loading image...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .loaded(let img):
            Image(uiImage: img)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 400)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        case .failed:
            Text("Image unavailable")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func loadImageIfNeeded() async {
        guard let id = tinyurlId else { return }
        guard case .idle = imageLoadState else { return }

        imageLoadState = .loading

        guard let url = URL(string: "https://tinyurl.com/\(id)") else {
            imageLoadState = .failed
            return
        }

        do {
            // Follow redirect to final Dropbox URL
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            let (data, _) = try await URLSession.shared.data(for: request)
            if let img = UIImage(data: data) {
                imageLoadState = .loaded(img)
            } else {
                imageLoadState = .failed
            }
        } catch {
            imageLoadState = .failed
        }
    }
}
