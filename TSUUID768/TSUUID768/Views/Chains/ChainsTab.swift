import SwiftUI
import TSUUIDKit

struct ChainsTab: View {
    @EnvironmentObject var knowledge: KnowledgeService

    var body: some View {
        NavigationStack {
            if knowledge.domainStats.isEmpty {
                ContentUnavailableView("No chains yet",
                    systemImage: "link",
                    description: Text("Chains appear as you encode knowledge"))
            } else {
                List {
                    ForEach(knowledge.domainStats.sorted(by: { $0.value > $1.value }),
                            id: \.key) { domain, count in
                        NavigationLink {
                            ChainDetailView(domain: domain)
                        } label: {
                            HStack {
                                Image(systemName: "link")
                                    .foregroundStyle(.tint)
                                VStack(alignment: .leading) {
                                    Text(domain)
                                        .font(.headline)
                                    Text("\(count) vectors")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Chains")
    }
}

struct ChainDetailView: View {
    let domain: String

    var body: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 4) {
                ForEach(0..<10, id: \.self) { i in
                    ChainBeadView(index: i)
                }
            }
            .padding()
        }
        .navigationTitle(domain)
    }
}

struct ChainBeadView: View {
    let index: Int

    var body: some View {
        Circle()
            .fill(Color.accentColor.opacity(0.7))
            .frame(width: 32, height: 32)
            .overlay {
                Text("\(index + 1)")
                    .font(.caption2)
                    .foregroundStyle(.white)
            }
    }
}
