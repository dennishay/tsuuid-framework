import Foundation
import UIKit
import TSUUIDKit
import SwiftyDropbox

@MainActor
class SyncService: ObservableObject {
    enum ConnectionState: String {
        case disconnected, connecting, connected, syncing, error
    }

    static let dropboxAppKey = "vq19kmto50rkj6v"
    static let syncPath = "/__768_sync"

    @Published var state: ConnectionState = .disconnected
    @Published var lastSync: Date?
    @Published var pendingOutgoing: Int = 0
    @Published var pendingIncoming: Int = 0
    @Published var statusMessage: String = ""

    private var client: DropboxClient?

    private var localSyncDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("__768_sync")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func setup() {
        DropboxClientsManager.setupWithAppKey(Self.dropboxAppKey)
    }

    var isAuthorized: Bool {
        DropboxClientsManager.authorizedClient != nil
    }

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

    func handleAuthRedirect(_ url: URL) {
        DropboxClientsManager.handleRedirectURL(url, includeBackgroundClient: false) { [weak self] result in
            Task { @MainActor in
                if let result, case .success = result {
                    self?.connect()
                }
            }
        }
    }

    func connect() {
        if let authorized = DropboxClientsManager.authorizedClient {
            client = authorized
            state = .connected
            statusMessage = "Connected to Dropbox"
        } else {
            state = .disconnected
            statusMessage = "Not authorized"
        }
    }

    /// Download the latest checkpoint from Dropbox
    func downloadCheckpoint(to destination: URL) async {
        guard let client = client else { return }
        state = .syncing
        statusMessage = "Downloading checkpoint..."

        let checkpointPath = "\(Self.syncPath)/checkpoints/latest.db"

        do {
            let result = try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Files.FileMetadata, Error>) in
                client.files.download(path: checkpointPath, overwrite: true, destination: destination)
                    .response { result, error in
                        if let (metadata, _) = result {
                            continuation.resume(returning: metadata)
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
            let result = try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Files.ListFolderResult, Error>) in
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

            for file in deltaFiles {
                let localPath = localSyncDir.appendingPathComponent(file.name)
                _ = try await withCheckedThrowingContinuation {
                    (continuation: CheckedContinuation<Files.FileMetadata, Error>) in
                    client.files.download(path: file.pathLower!, overwrite: true, destination: localPath)
                        .response { result, error in
                            if let (metadata, _) = result {
                                continuation.resume(returning: metadata)
                            } else if let error = error {
                                continuation.resume(throwing: NSError(domain: "Dropbox", code: -1,
                                    userInfo: [NSLocalizedDescriptionKey: "\(error)"]))
                            }
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
