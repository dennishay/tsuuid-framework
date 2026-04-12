import SwiftUI
import TSUUIDKit

struct SyncTab: View {
    @EnvironmentObject var knowledge: KnowledgeService
    @EnvironmentObject var models: ModelService
    @EnvironmentObject var sync: SyncService

    var body: some View {
        NavigationStack {
            List {
                Section("Dropbox") {
                    LabeledContent("Status", value: sync.state.rawValue.capitalized)
                    if let last = sync.lastSync {
                        LabeledContent("Last sync", value: last.formatted())
                    }

                    if !sync.isAuthorized {
                        Button("Sign in to Dropbox") {
                            // Get the root view controller for OAuth
                            if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                               let vc = scene.windows.first?.rootViewController {
                                sync.authorize(from: vc)
                            }
                        }
                    } else if sync.state == .disconnected {
                        Button("Connect") { sync.connect() }
                    }
                }

                Section("Sync") {
                    LabeledContent("Pending outgoing", value: "\(sync.pendingOutgoing)")
                    LabeledContent("Pending incoming", value: "\(sync.pendingIncoming)")

                    Button {
                        Task {
                            let dest = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                                .appendingPathComponent("vectors.db")
                            await sync.downloadCheckpoint(to: dest)
                            await knowledge.load()
                        }
                    } label: {
                        HStack {
                            if sync.state == .syncing { ProgressView() }
                            Text(sync.state == .syncing ? "Syncing..." : "Download Checkpoint")
                        }
                    }
                    .disabled(sync.state != .connected)

                    Button {
                        Task { await sync.pullDeltas(knowledge: knowledge) }
                    } label: {
                        Text("Pull Deltas")
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
                    Text(sync.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Sync")
            .onAppear {
                sync.connect()
            }
        }
    }
}
