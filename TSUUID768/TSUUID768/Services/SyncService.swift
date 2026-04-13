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
    private var localInsertObserver: NSObjectProtocol?
    private let pendingQueueLock = NSLock()

    private var localSyncDir: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("__768_sync")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Append-only JSONL file of locally-inserted rows waiting to be pushed.
    /// Survives app restarts.
    private var pendingOutboundFile: URL {
        let dir = localSyncDir.appendingPathComponent("deltas/iphone")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("pending.jsonl")
    }

    func setup() {
        DropboxClientsManager.setupWithAppKey(Self.dropboxAppKey)
        observeLocalInserts()
        refreshPendingCount()
    }

    private func observeLocalInserts() {
        if let existing = localInsertObserver {
            NotificationCenter.default.removeObserver(existing)
        }
        localInsertObserver = NotificationCenter.default.addObserver(
            forName: KnowledgeService.localInsertNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self = self,
                  let row = note.userInfo?["row"] as? SyncRow else { return }
            self.appendToOutboundQueue(row)
        }
    }

    private func appendToOutboundQueue(_ row: SyncRow) {
        pendingQueueLock.lock()
        defer { pendingQueueLock.unlock() }
        do {
            let line = try JSONEncoder.sortedKeys.encode(row)
            var data = line
            data.append(0x0a)  // newline
            let url = pendingOutboundFile
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                try handle.write(contentsOf: data)
            } else {
                try data.write(to: url, options: .atomic)
            }
            pendingOutgoing += 1
        } catch {
            statusMessage = "Queue failed: \(error.localizedDescription)"
        }
    }

    private func refreshPendingCount() {
        pendingQueueLock.lock()
        defer { pendingQueueLock.unlock() }
        let url = pendingOutboundFile
        guard FileManager.default.fileExists(atPath: url.path),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            pendingOutgoing = 0
            return
        }
        pendingOutgoing = text.split(separator: "\n",
                                      omittingEmptySubsequences: true).count
    }

    /// Move `pending.jsonl` to a timestamped bundle ready for upload.
    /// Returns the bundle URL if there was anything to rotate, nil otherwise.
    private func rotatePendingBundle() -> URL? {
        pendingQueueLock.lock()
        defer { pendingQueueLock.unlock() }

        let pending = pendingOutboundFile
        guard FileManager.default.fileExists(atPath: pending.path) else { return nil }

        let attrs = try? FileManager.default.attributesOfItem(atPath: pending.path)
        if let size = attrs?[.size] as? NSNumber, size.intValue == 0 { return nil }

        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMddHHmm"
        let stamp = fmt.string(from: Date())
        let bundle = pending.deletingLastPathComponent()
            .appendingPathComponent("\(stamp).jsonl")

        do {
            if FileManager.default.fileExists(atPath: bundle.path) {
                // Same-minute rotation — append.
                let existing = (try? Data(contentsOf: bundle)) ?? Data()
                let incoming = (try? Data(contentsOf: pending)) ?? Data()
                try (existing + incoming).write(to: bundle, options: .atomic)
            } else {
                try FileManager.default.moveItem(at: pending, to: bundle)
            }
            // Clear pending on success.
            if FileManager.default.fileExists(atPath: pending.path) {
                try FileManager.default.removeItem(at: pending)
            }
            return bundle
        } catch {
            statusMessage = "Rotate failed: \(error.localizedDescription)"
            return nil
        }
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

    /// Key under which the last-applied inbound bundle filename is persisted.
    /// Guards against re-applying the same bundle on repeated pulls.
    private static let lastAppliedKey = "TSUUID768.sync.lastAppliedMacBundle"

    /// Check for new delta files from Mac, parse JSONL, upsert each row.
    func pullDeltas(knowledge: KnowledgeService) async {
        guard let client = client else { return }
        state = .syncing
        statusMessage = "Checking for deltas..."

        let deltasPath = "\(Self.syncPath)/deltas/mac"
        let lastApplied = UserDefaults.standard.string(forKey: Self.lastAppliedKey) ?? ""

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

            // Accept both .jsonl (new format) and .delta (legacy name) as bundle extensions.
            // Filter by lastApplied — filenames are timestamped lexicographically.
            let bundleFiles = result.entries
                .compactMap { $0 as? Files.FileMetadata }
                .filter { $0.name.hasSuffix(".jsonl") || $0.name.hasSuffix(".delta") }
                .filter { $0.name > lastApplied }
                .sorted { $0.name < $1.name }

            pendingIncoming = bundleFiles.count
            statusMessage = "\(bundleFiles.count) new bundles"

            var totalApplied = 0
            var latestApplied = lastApplied

            for file in bundleFiles {
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

                do {
                    let rows = try SyncBundle.read(from: localPath)
                    var skippedFutureRows = 0
                    for row in rows {
                        // Drift guard: refuse rows written under a wire schema
                        // we don't understand. Row is skipped (logged) rather
                        // than blindly inserted — when we ship schema_version
                        // > 1 support, the next iOS build picks them up on
                        // the next pull.
                        if row.schema_version > syncSchemaVersion {
                            skippedFutureRows += 1
                            continue
                        }
                        knowledge.insert(row.toVector(),
                                         path: row.path,
                                         title: row.title,
                                         domain: row.domain,
                                         origin: .sync)
                        totalApplied += 1
                    }
                    if skippedFutureRows > 0 {
                        statusMessage = "\(file.name): skipped \(skippedFutureRows) future-schema rows"
                    }
                    // Ack advances either way: the file is fully read, and
                    // future-schema rows are a forward-compat skip, not a
                    // parse failure. Re-reading the same bundle next pull
                    // would just skip them again.
                    latestApplied = file.name
                    UserDefaults.standard.set(latestApplied, forKey: Self.lastAppliedKey)
                } catch {
                    // Skip malformed bundle but keep going — do NOT advance the ack.
                    statusMessage = "Skipped \(file.name): \(error.localizedDescription)"
                }

                pendingIncoming -= 1
            }

            lastSync = Date()
            statusMessage = "Applied \(totalApplied) rows from \(bundleFiles.count) bundles"
        } catch {
            statusMessage = "Sync error: \(error.localizedDescription)"
        }
        state = .connected
    }

    /// Rotate any pending rows into a timestamped bundle and upload all unsent
    /// bundles to Dropbox. On success, deletes local copies.
    func pushDeltas() async {
        guard let client = client else { return }
        state = .syncing

        _ = rotatePendingBundle()

        let outDir = localSyncDir.appendingPathComponent("deltas/iphone")
        guard FileManager.default.fileExists(atPath: outDir.path) else {
            state = .connected
            return
        }

        let files = ((try? FileManager.default.contentsOfDirectory(at: outDir,
            includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "jsonl" && $0.lastPathComponent != "pending.jsonl" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        statusMessage = "Pushing \(files.count) bundles..."
        var uploaded = 0

        for file in files {
            guard let data = try? Data(contentsOf: file) else { continue }
            let remotePath = "\(Self.syncPath)/deltas/iphone/\(file.lastPathComponent)"

            let success = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
                client.files.upload(path: remotePath, mode: .overwrite, input: data)
                    .response { result, error in
                        cont.resume(returning: result != nil && error == nil)
                    }
            }

            if success {
                try? FileManager.default.removeItem(at: file)
                uploaded += 1
            } else {
                // Leave on disk — will be retried next push.
                statusMessage = "Upload failed for \(file.lastPathComponent)"
            }
        }

        refreshPendingCount()
        lastSync = Date()
        statusMessage = "Pushed \(uploaded)/\(files.count) bundles"
        state = .connected
    }
}

private extension JSONEncoder {
    static let sortedKeys: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }()
}
