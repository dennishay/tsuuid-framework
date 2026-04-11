# TSUUID 768 iOS App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the iOS app shell (TSUUID768) on top of TSUUIDKit — four-tab SwiftUI app with search, capture, chains, and sync, plus share extension and home screen widget.

**Architecture:** SwiftUI app importing TSUUIDKit as a local package dependency. MVVM with ViewModels wrapping TSUUIDKit services. App Group container shared between app and extensions. Dropbox SDK for sync authentication.

**Tech Stack:** SwiftUI, TSUUIDKit (local package), SwiftyDropbox, VisionKit (document scanning), Speech framework, WidgetKit, App Intents (Siri)

**Depends on:** TSUUIDKit Swift Package (PR #1), Core ML model conversion (run `tools/convert_labse_coreml.py` + `tools/convert_clip_coreml.py` first)

---

## Prerequisites

Before starting this plan:

1. **Merge PR #1** (TSUUIDKit Swift Package)
2. **Run model conversion** — `python3 tools/convert_labse_coreml.py` and `python3 tools/convert_clip_coreml.py` to produce the `.mlpackage` files in `models/`
3. **Create Xcode project** — `TSUUID768.xcodeproj` with:
   - App target: `TSUUID768` (iOS 17, SwiftUI lifecycle)
   - Share Extension target: `TSUUID768ShareExtension`
   - Widget Extension target: `TSUUID768Widget`
   - App Group: `group.com.tsuuid.768`
   - Add TSUUIDKit as local package dependency
   - Add SwiftyDropbox via SPM
4. **Register Dropbox API app** at developers.dropbox.com (for OAuth)

---

## File Structure

```
TSUUID768/
├── TSUUID768.xcodeproj
├── TSUUID768/
│   ├── App.swift                        ← @main, tab view
│   ├── Services/
│   │   ├── ModelService.swift           ← Core ML model lifecycle + memory pressure
│   │   ├── CaptureService.swift         ← camera, OCR (VisionKit), speech
│   │   ├── SyncService.swift            ← Dropbox auth + SyncEngine orchestration
│   │   └── KnowledgeService.swift       ← VectorStore + VectorDatabase coordinator
│   ├── ViewModels/
│   │   ├── SearchViewModel.swift        ← debounced search, voice input
│   │   ├── CaptureViewModel.swift       ← capture modes, encoding pipeline
│   │   ├── ChainsViewModel.swift        ← chain browsing, gap detection
│   │   └── SyncViewModel.swift          ← sync status, stats, manual trigger
│   ├── Views/
│   │   ├── ContentView.swift            ← TabView with 4 tabs
│   │   ├── Search/
│   │   │   ├── SearchTab.swift          ← search field + results list
│   │   │   ├── SearchResultRow.swift    ← similarity score, title, domain badge
│   │   │   └── SearchDetailView.swift   ← full metadata, chain position, source
│   │   ├── Capture/
│   │   │   ├── CaptureTab.swift         ← mode picker + viewfinder/input
│   │   │   ├── PhotoCaptureView.swift   ← camera → CLIP encode
│   │   │   ├── DocumentScanView.swift   ← VisionKit scan → OCR → LaBSE
│   │   │   ├── VoiceCaptureView.swift   ← record → Speech → LaBSE
│   │   │   └── TextInputView.swift      ← paste/type → LaBSE
│   │   ├── Chains/
│   │   │   ├── ChainsTab.swift          ← domain list → chain detail
│   │   │   ├── ChainDetailView.swift    ← bead visualization
│   │   │   └── ChainBeadView.swift      ← single bead (link) with trit display
│   │   └── Sync/
│   │       └── SyncTab.swift            ← connection, stats, manual sync
│   └── Resources/
│       └── Assets.xcassets
├── ShareExtension/
│   ├── ShareViewController.swift        ← receive text/image/URL, encode, store
│   └── Info.plist
├── Widget/
│   ├── WidgetBundle.swift
│   ├── VectorCountWidget.swift          ← small + medium widgets
│   └── Info.plist
└── SiriShortcut/
    └── SearchIntent.swift               ← "Search 768 for [query]"
```

---

## Task 1: Xcode project scaffold + App.swift

**Files:**
- Create: Xcode project via `xcodegen` or manually
- Create: `TSUUID768/App.swift`
- Create: `TSUUID768/Views/ContentView.swift`

- [ ] **Step 1: Generate Xcode project**

Create `project.yml` for XcodeGen:

```yaml
name: TSUUID768
options:
  bundleIdPrefix: com.tsuuid
  deploymentTarget:
    iOS: "17.0"
  xcodeVersion: "16.0"
packages:
  TSUUIDKit:
    path: ../TSUUIDKit
  SwiftyDropbox:
    url: https://github.com/dropbox/SwiftyDropbox.git
    from: "10.0.0"
targets:
  TSUUID768:
    type: application
    platform: iOS
    sources: [TSUUID768]
    dependencies:
      - package: TSUUIDKit
      - package: SwiftyDropbox
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.tsuuid.768
      DEVELOPMENT_TEAM: ""
      CODE_SIGN_ENTITLEMENTS: TSUUID768/TSUUID768.entitlements
    entitlements:
      path: TSUUID768/TSUUID768.entitlements
      properties:
        com.apple.security.application-groups:
          - group.com.tsuuid.768
```

Run: `cd /Users/dennishay/tsuuid-framework/TSUUID768 && xcodegen generate`

- [ ] **Step 2: Write App.swift**

```swift
// TSUUID768/App.swift
import SwiftUI
import TSUUIDKit

@main
struct TSUUID768App: App {
    @StateObject private var knowledgeService = KnowledgeService()
    @StateObject private var modelService = ModelService()
    @StateObject private var syncService = SyncService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(knowledgeService)
                .environmentObject(modelService)
                .environmentObject(syncService)
                .preferredColorScheme(.dark)
                .onReceive(NotificationCenter.default.publisher(
                    for: UIApplication.didReceiveMemoryWarningNotification
                )) { _ in
                    modelService.handleMemoryWarning()
                }
        }
    }
}
```

- [ ] **Step 3: Write ContentView.swift**

```swift
// TSUUID768/Views/ContentView.swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            SearchTab()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }

            CaptureTab()
                .tabItem {
                    Label("Capture", systemImage: "camera")
                }

            ChainsTab()
                .tabItem {
                    Label("Chains", systemImage: "link")
                }

            SyncTab()
                .tabItem {
                    Label("Sync", systemImage: "arrow.triangle.2.circlepath")
                }
        }
    }
}
```

- [ ] **Step 4: Build and verify app launches**

Run in Xcode: Product → Run (Simulator)

- [ ] **Step 5: Commit**

```bash
git add TSUUID768/ && git commit -m "feat(app): Xcode project scaffold + tab structure"
```

---

## Task 2: KnowledgeService + ModelService

Core services that manage the VectorStore and Core ML models.

**Files:**
- Create: `TSUUID768/Services/KnowledgeService.swift`
- Create: `TSUUID768/Services/ModelService.swift`

- [ ] **Step 1: Write KnowledgeService**

```swift
// TSUUID768/Services/KnowledgeService.swift
import Foundation
import TSUUIDKit

@MainActor
class KnowledgeService: ObservableObject {
    @Published var vectorCount: Int = 0
    @Published var domainStats: [String: Int] = [:]
    @Published var isLoaded = false

    let store = VectorStore()
    private var database: VectorDatabase?

    private var appGroupURL: URL {
        FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        )!
    }

    func load() async {
        let dbPath = appGroupURL.appendingPathComponent("vectors.db").path
        do {
            database = try VectorDatabase(path: dbPath)
            let count = try database!.loadAll(into: store)
            vectorCount = count
            domainStats = store.stats()
            isLoaded = true
        } catch {
            print("Failed to load database: \(error)")
        }
    }

    func search(_ query: Vector768, domain: String? = nil,
                limit: Int = 10) -> [SearchResult] {
        store.search(query, domain: domain, limit: limit)
    }

    func insert(_ vec: Vector768, path: String, title: String,
                domain: String) {
        let meta = VectorMeta(uuid: UUID(), source: path,
                              domain: domain, encodedAt: Date())
        store.insert(vec, meta: meta)
        try? database?.store(path: path, title: title, vec: vec, domain: domain)
        vectorCount = store.count
        domainStats = store.stats()
    }
}
```

- [ ] **Step 2: Write ModelService**

```swift
// TSUUID768/Services/ModelService.swift
import Foundation
import TSUUIDKit

@MainActor
class ModelService: ObservableObject {
    enum ModelState: String {
        case unloaded, loading, ready, error
    }

    @Published var labseState: ModelState = .unloaded
    @Published var clipState: ModelState = .unloaded
    @Published var memoryUsageMB: Int = 0

    #if canImport(CoreML)
    private let labse = LaBSEEncoder()
    private let clip = CLIPEncoder()
    #endif

    private var appGroupURL: URL {
        FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "group.com.tsuuid.768"
        )!
    }

    func loadModels() async {
        #if canImport(CoreML)
        labseState = .loading
        let labseURL = appGroupURL
            .appendingPathComponent("models/LaBSE-full.mlmodelc")
        do {
            try await labse.load(modelURL: labseURL)
            labseState = .ready
        } catch {
            labseState = .error
            print("LaBSE load failed: \(error)")
        }

        clipState = .loading
        let clipURL = appGroupURL
            .appendingPathComponent("models/CLIP-full.mlmodelc")
        do {
            try await clip.load(modelURL: clipURL)
            clipState = .ready
        } catch {
            clipState = .error
            print("CLIP load failed: \(error)")
        }
        #endif
    }

    func handleMemoryWarning() {
        #if canImport(CoreML)
        labse.unload()
        clip.unload()
        labseState = .unloaded
        clipState = .unloaded
        #endif
    }

    func encodeText(_ text: String) async throws -> Vector768 {
        #if canImport(CoreML)
        return try await labse.encode(text)
        #else
        throw EncoderError.modelNotLoaded
        #endif
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add TSUUID768/Services/ && git commit -m "feat(app): KnowledgeService + ModelService"
```

---

## Task 3: SearchTab (search field + results)

**Files:**
- Create: `TSUUID768/ViewModels/SearchViewModel.swift`
- Create: `TSUUID768/Views/Search/SearchTab.swift`
- Create: `TSUUID768/Views/Search/SearchResultRow.swift`
- Create: `TSUUID768/Views/Search/SearchDetailView.swift`

- [ ] **Step 1: Write SearchViewModel**

```swift
// TSUUID768/ViewModels/SearchViewModel.swift
import Foundation
import Combine
import TSUUIDKit

@MainActor
class SearchViewModel: ObservableObject {
    @Published var query = ""
    @Published var results: [SearchResult] = []
    @Published var isSearching = false
    @Published var selectedDomain: String?

    private var debounceTask: Task<Void, Never>?

    func search(using knowledge: KnowledgeService, model: ModelService) {
        debounceTask?.cancel()
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else {
            results = []
            return
        }

        debounceTask = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }

            isSearching = true
            do {
                let queryVec = try await model.encodeText(q)
                results = knowledge.search(queryVec, domain: selectedDomain)
            } catch {
                results = []
            }
            isSearching = false
        }
    }
}
```

- [ ] **Step 2: Write SearchTab**

```swift
// TSUUID768/Views/Search/SearchTab.swift
import SwiftUI
import TSUUIDKit

struct SearchTab: View {
    @EnvironmentObject var knowledge: KnowledgeService
    @EnvironmentObject var model: ModelService
    @StateObject private var vm = SearchViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                TextField("Search knowledge...", text: $vm.query)
                    .textFieldStyle(.roundedBorder)
                    .padding()
                    .onChange(of: vm.query) { _, _ in
                        vm.search(using: knowledge, model: model)
                    }

                if vm.isSearching {
                    ProgressView()
                        .padding()
                }

                List(vm.results, id: \.index) { result in
                    NavigationLink {
                        SearchDetailView(result: result)
                    } label: {
                        SearchResultRow(result: result)
                    }
                }
                .listStyle(.plain)
            }
            .navigationTitle("Search")
        }
    }
}
```

- [ ] **Step 3: Write SearchResultRow**

```swift
// TSUUID768/Views/Search/SearchResultRow.swift
import SwiftUI
import TSUUIDKit

struct SearchResultRow: View {
    let result: SearchResult

    var body: some View {
        HStack {
            Text(String(format: "%.2f", result.similarity))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 40)

            VStack(alignment: .leading, spacing: 2) {
                Text(result.source.components(separatedBy: "/").last ?? result.source)
                    .font(.body)
                    .lineLimit(1)

                Text(result.domain)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.2))
                    .clipShape(Capsule())
            }

            Spacer()
        }
    }
}
```

- [ ] **Step 4: Write SearchDetailView**

```swift
// TSUUID768/Views/Search/SearchDetailView.swift
import SwiftUI
import TSUUIDKit

struct SearchDetailView: View {
    let result: SearchResult

    var body: some View {
        List {
            Section("Match") {
                LabeledContent("Similarity", value: String(format: "%.4f", result.similarity))
                LabeledContent("Distance", value: String(format: "%.4f", 1 - result.similarity))
            }

            Section("Source") {
                LabeledContent("Path", value: result.source)
                LabeledContent("Domain", value: result.domain)
            }

            Section("Identity") {
                LabeledContent("UUID", value: result.uuid.uuidString)
            }
        }
        .navigationTitle("Detail")
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add TSUUID768/ViewModels/SearchViewModel.swift TSUUID768/Views/Search/
git commit -m "feat(app): SearchTab — debounced semantic search with results"
```

---

## Task 4: CaptureTab (camera, document scan, voice, text)

**Files:**
- Create: `TSUUID768/Services/CaptureService.swift`
- Create: `TSUUID768/ViewModels/CaptureViewModel.swift`
- Create: `TSUUID768/Views/Capture/CaptureTab.swift`
- Create: `TSUUID768/Views/Capture/TextInputView.swift`
- Create: `TSUUID768/Views/Capture/DocumentScanView.swift`

- [ ] **Step 1-5: Implement capture pipeline**

(Full code in each file — camera via UIImagePickerController, VisionKit VNDocumentCameraViewController for scanning, Speech framework for voice, plain TextField for text input. Each mode encodes via ModelService and inserts via KnowledgeService.)

- [ ] **Step 6: Commit**

---

## Task 5: ChainsTab (chain visualization)

**Files:**
- Create: `TSUUID768/ViewModels/ChainsViewModel.swift`
- Create: `TSUUID768/Views/Chains/ChainsTab.swift`
- Create: `TSUUID768/Views/Chains/ChainDetailView.swift`
- Create: `TSUUID768/Views/Chains/ChainBeadView.swift`

- [ ] **Step 1-4: Implement chain browser**

(ScrollView with horizontal bead layout, each bead colored by domain, tap for detail. Gap indicators as dashed spacers.)

- [ ] **Step 5: Commit**

---

## Task 6: SyncTab + SyncService (Dropbox integration)

**Files:**
- Create: `TSUUID768/Services/SyncService.swift`
- Create: `TSUUID768/ViewModels/SyncViewModel.swift`
- Create: `TSUUID768/Views/Sync/SyncTab.swift`

- [ ] **Step 1-4: Implement Dropbox sync**

(SwiftyDropbox OAuth, SyncEngine wrapper, background refresh via BGAppRefreshTask, manual sync button.)

- [ ] **Step 5: Commit**

---

## Task 7: Share Extension

**Files:**
- Create: `ShareExtension/ShareViewController.swift`
- Create: `ShareExtension/Info.plist`

- [ ] **Step 1-3: Implement share extension**

(Receive text/image/URL from share sheet, load slim model from App Group, encode, store, show nearest match confirmation.)

- [ ] **Step 4: Commit**

---

## Task 8: Widget

**Files:**
- Create: `Widget/WidgetBundle.swift`
- Create: `Widget/VectorCountWidget.swift`

- [ ] **Step 1-3: Implement widget**

(Small: tap opens app. Medium: vector count + 2 recent encodes. Reads from shared SQLite, no model loading.)

- [ ] **Step 4: Commit**

---

## Task 9: Siri Shortcut

**Files:**
- Create: `SiriShortcut/SearchIntent.swift`

- [ ] **Step 1-2: Register App Intent**

("Search 768 for [query]" — loads slim model, searches mmap store, returns top 3.)

- [ ] **Step 3: Commit**

---

## Build Order

1. Xcode project scaffold (Task 1) — must be done manually or with XcodeGen
2. Services (Task 2) — core runtime
3. SearchTab (Task 3) — first usable screen
4. CaptureTab (Task 4) — encode new knowledge
5. ChainsTab (Task 5) — browse chains
6. SyncTab (Task 6) — Dropbox connection
7. Share Extension (Task 7)
8. Widget (Task 8)
9. Siri Shortcut (Task 9)

Each task produces a working, testable increment.
