import SwiftUI
import TSUUIDKit

struct SearchResultRow: View {
    let result: SearchResult

    var body: some View {
        HStack(spacing: 12) {
            Text(String(format: "%.2f", result.similarity))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(similarityColor)
                .frame(width: 36)

            VStack(alignment: .leading, spacing: 2) {
                Text(filename)
                    .font(.body)
                    .lineLimit(1)

                Text(result.domain)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.2))
                    .clipShape(Capsule())
            }

            Spacer()
        }
    }

    private var filename: String {
        result.source.components(separatedBy: "/").last ?? result.source
    }

    private var similarityColor: Color {
        if result.similarity > 0.9 { return .green }
        if result.similarity > 0.7 { return .yellow }
        return .secondary
    }
}
