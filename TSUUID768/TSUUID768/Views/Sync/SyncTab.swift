import SwiftUI
import TSUUIDKit

struct SyncTab: View {
    @EnvironmentObject var knowledge: KnowledgeService
    @EnvironmentObject var models: ModelService
    @EnvironmentObject var sync: SyncService

    var body: some View {
        NavigationStack {
            List {
                Section("Connection") {
                    LabeledContent("Status", value: sync.state.rawValue.capitalized)
                    if let last = sync.lastSync {
                        LabeledContent("Last sync", value: last.formatted())
                    }
                    Button("Connect") {
                        sync.connect()
                    }
                    .disabled(sync.state == .connected || sync.state == .syncing)
                }

                Section("Sync") {
                    LabeledContent("Pending outgoing", value: "\(sync.pendingOutgoing)")
                    LabeledContent("Pending incoming", value: "\(sync.pendingIncoming)")
                    Button {
                        Task { await sync.syncNow(knowledge: knowledge) }
                    } label: {
                        HStack {
                            if sync.state == .syncing {
                                ProgressView()
                            }
                            Text(sync.state == .syncing ? "Syncing..." : "Sync Now")
                        }
                    }
                    .disabled(sync.state != .connected)
                }

                Section("Knowledge Graph") {
                    LabeledContent("Total vectors", value: "\(knowledge.vectorCount)")
                    ForEach(knowledge.domainStats.sorted(by: { $0.value > $1.value }),
                            id: \.key) { domain, count in
                        LabeledContent(domain, value: "\(count)")
                    }
                }

                Section("Models") {
                    LabeledContent("LaBSE", value: models.labseState.rawValue)
                    LabeledContent("CLIP", value: models.clipState.rawValue)
                }

                Section("Debug") {
                    Text(knowledge.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(models.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Sync")
        }
    }
}
