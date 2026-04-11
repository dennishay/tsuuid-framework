import SwiftUI
import TSUUIDKit

struct SearchDetailView: View {
    let result: SearchResult

    var body: some View {
        List {
            Section("Match") {
                LabeledContent("Similarity",
                    value: String(format: "%.4f", result.similarity))
                LabeledContent("Distance",
                    value: String(format: "%.4f", 1 - result.similarity))
            }

            Section("Source") {
                LabeledContent("Path", value: result.source)
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
    }
}
