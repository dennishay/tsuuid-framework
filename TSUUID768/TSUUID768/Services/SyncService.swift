import Foundation
import TSUUIDKit
import SwiftyDropbox

@MainActor
class SyncService: ObservableObject {
    enum ConnectionState: String {
        case disconnected, connecting, connected, syncing, error
    }

    // Replace with your Dropbox App Key from developers.dropbox.com
    static let dropboxAppKey = "REPLACE_WITH_APP_KEY"
    static let syncPath = "/__768_sync"

    @Published var state: ConnectionState = .disconnected
    @Published var lastSync: Date?
    @Published var pendingOutgoing: Int = 0
    @Published var pendingIncoming: Int = 0
    @Published var statusMessage: String = ""

    private var client: DropboxClient?

    /// Local staging directory for downloaded deltas
    private var localSyncDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("__768_sync")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Initialize Dropbox SDK — call once at app launch
    func setup() {
        DropboxClientsManager.setupWithAppKey(Self.dropboxAppKey)
    }

    /// Check if we have a saved Dropbox auth token
    var isAuthorized: Bool {
        DropboxClientsManager.authorizedClient != nil
    }

    /// Start OAuth flow — returns the auth URL for ASWebAuthenticationSession
    func authorize(from viewController: UIViewController) {
        let scopeRequest = ScopeRequest(
            scopeType: .user,
            scopes: ["files.metadata.read", "files.content.read", "files.content.write"],
            includeGrantedScopes: false
        )
        DropboxClientsManager.authorizeFromControllerV2(
            UIApplication.shared,
            controller: viewController,
            loadingStatusDelegate: nil,
            openURL: { UIApplication.shared.open($0) },
            scopeRequest: scopeRequest
        )
    }

    /// Call after OAuth redirect completes
    func handleAuthRedirect(_ url: URL) -> Bool {
        let result = DropboxClientsManager.handleRedirectURL(url)
        if case .success = result {
            connect()
            return true
        }
        return false
    }

    /// Connect using existing auth
    func connect() {
        if let authorized = DropboxClientsManager.authorizedClient {
            client = authorized
            state = .connected
            statusMessage = "Connected to Dropbox"
        } else {
            state = .disconnected
            statusMessage = "Not authorized — tap Connect to sign in"
        }
    }

    /// Download the latest checkpoint from Dropbox
    func downloadCheckpoint(to destination: URL) async {
        guard let client = client else { return }
        state = .syncing
        statusMessage = "Downloading checkpoint..."

        let checkpointPath = "\(Self.syncPath)/checkpoints/latest.db"

        do {
            let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Files.FileMetadata, Error>) in
                client.files.download(path: checkpointPath, overwrite: true, destination: { _, _ in destination })
                    .response { result, error in
                        if let result = result {
                            continuation.resume(returning: result.0)
                        } else if let error = error {
                            continuation.resume(throwing: NSError(domain: "Dropbox", code: -1,
                                userInfo: [NSLocalizedDescriptionKey: "\(error)"]))
                        }
                    }
            }
            let size = result.size / 1_000_000
            statusMessage = "Checkpoint downloaded (\(size) MB)"
            lastSync = Date()
        } catch {
            statusMessage = "Download failed: \(error.localizedDescription)"
        }
        state = .connected
    }

    /// Check for new delta files from Mac
    func pullDeltas(knowledge: KnowledgeService) async {
        guard let client = client else { return }
        state = .syncing
        statusMessage = "Checking for deltas..."

        let deltasPath = "\(Self.syncPath)/deltas/mac"

        do {
            let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Files.ListFolderResult, Error>) in
                client.files.listFolder(path: deltasPath).response { result, error in
                    if let result = result {
                        continuation.resume(returning: result)
                    } else if let error = error {
                        continuation.resume(throwing: NSError(domain: "Dropbox", code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "\(error)"]))
                    }
                }
            }

            let deltaFiles = result.entries
                .compactMap { $0 as? Files.FileMetadata }
                .filter { $0.name.hasSuffix(".delta") }
                .sorted { $0.name < $1.name }

            pendingIncoming = deltaFiles.count
            statusMessage = "\(deltaFiles.count) delta files found"

            // Download and apply each delta
            for file in deltaFiles {
                let localPath = localSyncDir.appendingPathComponent(file.name)
                _ = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Files.FileMetadata, Error>) in
                    client.files.download(path: file.pathLower!, overwrite: true, destination: { _, _ in localPath })
                        .response { result, error in
                            if let result = result {
                                continuation.resume(returning: result.0)
                            } else if let error = error {
                                continuation.resume(throwing: NSError(domain: "Dropbox", code: -1,
                                    userInfo: [NSLocalizedDescriptionKey: "\(error)"]))
                            }
                        }
                }

                // Apply the delta bundle
                if let data = try? Data(contentsOf: localPath) {
                    let syncEngine = SyncEngine(peer: .iphone, syncRoot: localSyncDir)
                    _ = try? syncEngine.pullIncoming { delta in
                        let encoder = DeltaEncoder()
                        let stored = Vector768()
                        let vec = encoder.applyDelta(stored: stored, delta: delta)
                        knowledge.insert(vec, path: "sync-\(UUID().uuidString.prefix(8))",
                                       title: file.name, domain: "sync")
                    }
                }
                pendingIncoming -= 1
            }

            lastSync = Date()
            statusMessage = "Sync complete"
        } catch {
            statusMessage = "Sync error: \(error.localizedDescription)"
        }
        state = .connected
    }

    /// Upload outgoing deltas to Dropbox
    func pushDeltas() async {
        guard let client = client else { return }
        let outDir = localSyncDir.appendingPathComponent("deltas/iphone")
        guard FileManager.default.fileExists(atPath: outDir.path) else { return }

        let files = (try? FileManager.default.contentsOfDirectory(at: outDir,
            includingPropertiesForKeys: nil))?.filter { $0.pathExtension == "delta" } ?? []

        for file in files {
            if let data = try? Data(contentsOf: file) {
                let remotePath = "\(Self.syncPath)/deltas/iphone/\(file.lastPathComponent)"
                client.files.upload(path: remotePath, mode: .overwrite, input: data)
                    .response { _, _ in }
            }
        }
        pendingOutgoing = 0
    }
}
