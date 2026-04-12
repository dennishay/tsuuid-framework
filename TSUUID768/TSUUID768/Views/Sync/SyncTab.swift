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
                            signInToDropbox()
                        }
                    } else {
                        Button {
                            Task {
                                let dest = FileManager.default
                                    .urls(for: .documentDirectory, in: .userDomainMask).first!
                                    .appendingPathComponent("vectors.db")
                                await sync.downloadCheckpoint(to: dest)
                                await knowledge.load()
                            }
                        } label: {
                            HStack {
                                if sync.state == .syncing { ProgressView() }
                                Text(sync.state == .syncing ? "Downloading..." : "Download Checkpoint")
                            }
                        }
                        .disabled(sync.state == .syncing)

                        Button {
                            Task { await sync.pullDeltas(knowledge: knowledge) }
                        } label: {
                            Text("Pull Deltas")
                        }
                        .disabled(sync.state == .syncing)
                    }
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
        }
    }

    private func signInToDropbox() {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let vc = scene.keyWindow?.rootViewController else {
            return
        }
        sync.authorize(from: vc)
    }
}
