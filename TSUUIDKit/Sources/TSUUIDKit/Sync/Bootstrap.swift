import Foundation

/// Reusable bootstrap pipeline for sister iOS apps that share the 768
/// knowledge graph. Sister apps (Shipments-Scan, WristMeta-Companion,
/// CPC-Ops, …) each link TSUUIDKit and call
/// ``Bootstrap.bootstrapIfNeeded(…)`` on launch:
///
/// 1. **First launch:** downloads the latest checkpoint SQLite once,
///    roughly 300 MB over WiFi.
/// 2. **Every subsequent launch:** pulls only the new JSONL bundles
///    in ``__768_sync/deltas/mac/`` and upserts each ``SyncRow`` into
///    the local ``VectorDatabase``.
///
/// The concrete Dropbox / iCloud / HTTPS fetch is injected through
/// ``RemoteFetcher``, so TSUUIDKit itself doesn't hard-depend on any
/// specific sync provider.
public enum Bootstrap {

    /// Default remote paths on the 768 sync root (Dropbox __768_sync).
    public enum RemotePath {
        public static let syncRoot = "/__768_sync"
        public static let checkpoint = "/__768_sync/checkpoints/latest.db"
        public static let deltasMac = "/__768_sync/deltas/mac"
    }

    /// Persistence key for the last-applied Mac→phone bundle filename.
    /// Sister apps may share this namespace if they share the same
    /// on-device ``VectorDatabase``; otherwise each app should pass a
    /// unique key via ``bootstrapIfNeeded(lastAppliedKey:…)``.
    public static let defaultLastAppliedKey = "TSUUIDKit.sync.lastAppliedMacBundle"

    /// Inject a file listener and downloader. Consumers implement with
    /// Dropbox SDK, iCloud Drive, HTTP, etc. All three methods are async
    /// so they can bridge to SwiftyDropbox's callback API via
    /// ``withCheckedContinuation``.
    public protocol RemoteFetcher: Sendable {
        /// List children of a folder; returns remote paths in lexicographic order.
        /// Only filenames ending in ``.jsonl`` or ``.delta`` are considered by
        /// ``bootstrapIfNeeded``.
        func listFolder(_ remotePath: String) async throws -> [String]

        /// Download one remote file to ``destination``. The file is
        /// overwritten if it already exists.
        func download(_ remotePath: String, to destination: URL) async throws

        /// True iff the remote checkpoint exists. Used to decide whether
        /// first-launch bootstrap is runnable at all (vs. empty graph).
        func checkpointExists() async throws -> Bool
    }

    /// Fired for observability. Consumers can log these or bind them to a UI.
    public struct Progress: Sendable {
        public enum Phase: String, Sendable {
            case checkingCheckpoint
            case downloadingCheckpoint
            case checkpointReady
            case pullingDeltas
            case applyingBundle
            case bundleSkippedFutureSchema
            case bundleSkippedParseError
            case done
        }
        public let phase: Phase
        public let detail: String
    }

    /// One-shot bootstrap entry point.
    ///
    /// - Parameters:
    ///   - db: the local vector database to populate.
    ///   - fetcher: consumer-supplied fetch implementation.
    ///   - defaults: UserDefaults used for ack persistence. Pass a shared
    ///     suite when multiple sister apps must coordinate.
    ///   - lastAppliedKey: persistence key for the last-applied bundle name.
    ///   - localStageDir: directory to stage downloaded bundles into
    ///     (defaults to the process document dir / ``__768_sync``).
    ///   - onProgress: optional progress observer.
    /// - Returns: the number of ``SyncRow``s applied to ``db`` across
    ///   checkpoint + deltas. Checkpoint fetch alone counts as 0; the
    ///   checkpoint SQLite file is handed off intact and sister apps
    ///   typically call ``VectorDatabase.loadAll(into:)`` afterwards.
    @discardableResult
    public static func bootstrapIfNeeded(
        into db: VectorDatabase,
        fetcher: RemoteFetcher,
        defaults: UserDefaults = .standard,
        lastAppliedKey: String = Bootstrap.defaultLastAppliedKey,
        localStageDir: URL? = nil,
        onProgress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> Int {
        let stage = try resolveStageDir(localStageDir)
        let checkpointStamped = defaults.string(forKey: checkpointStampKey)

        if checkpointStamped == nil {
            onProgress?(Progress(phase: .checkingCheckpoint,
                                 detail: RemotePath.checkpoint))
            if try await fetcher.checkpointExists() {
                let dest = stage.appendingPathComponent("checkpoint-latest.db")
                onProgress?(Progress(phase: .downloadingCheckpoint,
                                     detail: RemotePath.checkpoint))
                try await fetcher.download(RemotePath.checkpoint, to: dest)
                defaults.set(Date(), forKey: checkpointStampKey)
                onProgress?(Progress(phase: .checkpointReady, detail: dest.path))
            }
            // If checkpointExists() == false we just move on to deltas.
            // A fresh remote with no checkpoint means the graph is empty
            // except for whatever the Mac has emitted as deltas.
        }

        var totalApplied = 0
        let lastApplied = defaults.string(forKey: lastAppliedKey) ?? ""
        onProgress?(Progress(phase: .pullingDeltas, detail: "after=\(lastApplied)"))

        let children = try await fetcher.listFolder(RemotePath.deltasMac)
        let bundleNames = children
            .map { ($0 as NSString).lastPathComponent }
            .filter { $0.hasSuffix(".jsonl") || $0.hasSuffix(".delta") }
            .filter { $0 > lastApplied }
            .sorted()

        var latestApplied = lastApplied
        for name in bundleNames {
            let remote = "\(RemotePath.deltasMac)/\(name)"
            let localPath = stage.appendingPathComponent(name)

            do {
                try await fetcher.download(remote, to: localPath)
                onProgress?(Progress(phase: .applyingBundle, detail: name))
                let rows = try SyncBundle.read(from: localPath)

                var skippedFutureRows = 0
                for row in rows {
                    if row.schema_version > syncSchemaVersion {
                        skippedFutureRows += 1
                        continue
                    }
                    try db.store(path: row.path,
                                 title: row.title,
                                 vec: row.toVector(),
                                 domain: row.domain)
                    totalApplied += 1
                }
                if skippedFutureRows > 0 {
                    onProgress?(Progress(phase: .bundleSkippedFutureSchema,
                                         detail: "\(name): \(skippedFutureRows)"))
                }
                latestApplied = name
                defaults.set(latestApplied, forKey: lastAppliedKey)
            } catch {
                // Parse failures: skip but don't advance the ack, so the
                // next run retries the same bundle. Future-schema rows
                // advance the ack because they are a deliberate skip, not
                // an error — re-reading wouldn't help.
                onProgress?(Progress(phase: .bundleSkippedParseError,
                                     detail: "\(name): \(error.localizedDescription)"))
            }
        }

        onProgress?(Progress(phase: .done, detail: "applied=\(totalApplied)"))
        return totalApplied
    }

    // MARK: - Internal

    /// Persistence key for "we've already downloaded the initial checkpoint".
    /// Bumping this key will force all sister apps to re-fetch the checkpoint
    /// on the next bootstrap (e.g. after an incompatible schema migration).
    private static let checkpointStampKey = "TSUUIDKit.sync.checkpointFetchedAt"

    private static func resolveStageDir(_ provided: URL?) throws -> URL {
        if let provided = provided {
            try FileManager.default.createDirectory(at: provided,
                                                   withIntermediateDirectories: true)
            return provided
        }
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("__768_sync", isDirectory: true)
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        return dir
    }
}
