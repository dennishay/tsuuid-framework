# TSUUID iOS — Full-Parity 768/768 Knowledge Peer

**Date:** 2026-04-11
**Status:** Design approved, pending implementation plan
**Repo:** `tsuuid-framework` (Swift Package) + new iOS app target

## Summary

A native Swift iOS app that carries the complete 768/768 knowledge graph in memory, encodes new knowledge on-device using Core ML (LaBSE + CLIP), and syncs bidirectionally with the Mac via Dropbox delta files. The phone is a full peer — not a satellite. Same models, same vectors, same TSUUIDs, same math.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment model | Full mirror — phone is a peer | Knowledge must be available offline, anywhere |
| Sync transport | Dropbox | Existing infrastructure spine for CPC |
| Input parity | Text, camera, voice, OCR — everything Mac does | Phone is a peer, not a reader |
| Platform | Swift + Core ML | Neural Engine acceleration, native iOS, best battery |
| Architecture | App + Swift Package + Extensions | Reusable framework, deep iOS integration |

## Architecture Overview

Six components, built in layers:

```
┌─────────────────────────────────────────────┐
│  iOS App (Search, Capture, Chains, Sync)    │
├──────────┬──────────┬───────────────────────┤
│  Share   │  Widget  │  Siri Shortcut        │
│  Extension│ (WidgetKit)│                     │
├──────────┴──────────┴───────────────────────┤
│              TSUUIDKit Swift Package         │
│  ┌──────────┬──────────┬──────────────────┐ │
│  │ Encoding │ TSUUID   │ Chain            │ │
│  │ LaBSE    │ Packing  │ ChainLink        │ │
│  │ CLIP     │ Compose  │ Chain            │ │
│  │ Speech   │ Dims     │ GapReport        │ │
│  │ OCR      │          │                  │ │
│  ├──────────┼──────────┴──────────────────┤ │
│  │ Storage  │ Sync                        │ │
│  │ VectorStore │ DeltaEncoder             │ │
│  │ SQLite   │ DropboxSync                 │ │
│  │ mmap     │ ConflictResolver            │ │
│  └──────────┴─────────────────────────────┘ │
├─────────────────────────────────────────────┤
│  Core ML Models (LaBSE + CLIP, two sizes)  │
├─────────────────────────────────────────────┤
│  Shared App Group Container                 │
│  (vectors.db, vectors.mmap, models/, deltas)│
└─────────────────────────────────────────────┘
```

## Component 1: TSUUIDKit Swift Package

Pure Swift Package with zero UI dependencies. Mirrors the Python `tsuuid-framework` 1:1.

### Module structure

```
TSUUIDKit/
├── Sources/TSUUIDKit/
│   ├── Encoding/
│   │   ├── LaBSEEncoder.swift       ← Core ML wrapper, encode(String) → Vector768
│   │   ├── CLIPEncoder.swift        ← Core ML wrapper, encode(CGImage) → Vector768
│   │   ├── SpeechEncoder.swift      ← Apple Speech → transcript → LaBSE
│   │   └── OCREncoder.swift         ← Apple Vision Live Text → LaBSE
│   ├── TSUUID/
│   │   ├── Packing.swift            ← 768 → 81 trits → UUID v8
│   │   ├── Compose.swift            ← semantic_distance, diff_uuids
│   │   └── Dimensions.swift         ← axis semantic labels
│   ├── Chain/
│   │   ├── ChainLink.swift          ← the atom
│   │   ├── Chain.swift              ← self-assembling doubly-linked chain
│   │   └── GapReport.swift          ← detected knowledge gaps
│   ├── Storage/
│   │   ├── VectorStore.swift        ← in-memory float16 + SQLite persistence
│   │   ├── Serialization.swift      ← vec↔bytes, vec↔b64 (identical to Python)
│   │   └── Migration.swift          ← import Mac SQLite checkpoint
│   ├── Sync/
│   │   ├── DeltaEncoder.swift       ← sparse diffs, same wire format as Python
│   │   ├── DropboxSync.swift        ← watch/read/write __768_sync/ folder
│   │   └── ConflictResolver.swift   ← last-writer-wins, loser kept in history
│   └── Models/
│       ├── ModelManager.swift        ← load/unload Core ML on memory pressure
│       └── QuantizedFallback.swift   ← int8 models for extensions
├── Tests/TSUUIDKitTests/
└── Package.swift
```

### Key types

```swift
/// 768-dimensional semantic vector. The meaning.
struct Vector768: Sendable {
    var storage: [Float16]  // 1536 bytes, identical to Python f16 format
    
    func toBytes() -> Data        // 1536 bytes, lossless
    func toBase64() -> String     // 2048 chars, transmittable
    static func fromBytes(_ data: Data) -> Vector768
    static func fromBase64(_ b64: String) -> Vector768
}

/// 81 ternary values {-1, 0, +1} projected from 768 dims
struct TritVector: Sendable {
    var trits: [Int8]  // 81 values, each -1/0/+1
    
    func toUUID() -> UUID         // pack into UUID v8
    func display() -> String      // "+·-+·+·--..." human-readable
    static func fromUUID(_ uuid: UUID) -> TritVector
}

/// The encoder interface — same contract as Python Encoder768
protocol SemanticEncoder {
    func encode(_ text: String) async -> Vector768
    func encodeBatch(_ texts: [String]) async -> [Vector768]
    var isLoaded: Bool { get }
    func load() async throws
    func unload()
}
```

### Wire format compatibility

The Swift `DeltaEncoder` produces byte-identical output to the Python `SparseDelta.to_bytes()`. This is a hard requirement — Mac and phone must exchange deltas without any conversion layer. The format:

```
SparseDelta binary layout (same on both sides):
  [4 bytes] version (uint32, little-endian)
  [1 byte]  is_checkpoint flag
  [4 bytes] n_changed (uint32)
  [n_changed × 6 bytes] entries:
    [2 bytes] dimension index (uint16)
    [4 bytes] delta value (float32)
```

## Component 2: Core ML Model Pipeline

### Model conversion (one-time, on Mac)

```python
# LaBSE → Core ML conversion
import coremltools as ct
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("sentence-transformers/LaBSE")
dummy = model.tokenize(["hello world"])
traced = torch.jit.trace(model.model, (dummy["input_ids"], dummy["attention_mask"]))
onnx_path = "labse.onnx"
torch.onnx.export(traced, (dummy["input_ids"], dummy["attention_mask"]), onnx_path)

# Full precision (float16) — main app, Neural Engine
mlmodel_full = ct.converters.onnx.convert(onnx_path)
mlmodel_full = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel_full, nbits=16)
mlmodel_full.save("LaBSE-full.mlpackage")  # ~235MB

# Quantized (int8) — extensions, CPU fallback
mlmodel_slim = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel_full, nbits=8)
mlmodel_slim.save("LaBSE-slim.mlpackage")  # ~120MB
```

```python
# CLIP → Core ML conversion
import coremltools as ct
import clip, torch

model, preprocess = clip.load("ViT-B/32")
dummy_image = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model.visual, dummy_image)
onnx_path = "clip.onnx"
torch.onnx.export(traced, dummy_image, onnx_path)

mlmodel_full = ct.converters.onnx.convert(onnx_path)
mlmodel_full = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel_full, nbits=16)
mlmodel_full.save("CLIP-full.mlpackage")  # ~170MB

mlmodel_slim = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel_full, nbits=8)
mlmodel_slim.save("CLIP-slim.mlpackage")  # ~85MB
```

### Model variants

| Model | Precision | Size | Target | Inference | Hardware |
|-------|-----------|------|--------|-----------|----------|
| LaBSE-full | float16 | ~235MB | Main app | ~50ms | Neural Engine |
| LaBSE-slim | int8 | ~120MB | Extensions | ~150ms | CPU |
| CLIP-full | float16 | ~170MB | Main app | ~40ms | Neural Engine |
| CLIP-slim | int8 | ~85MB | Extensions | ~120ms | CPU |

### Memory management

```swift
class ModelManager {
    enum State { case unloaded, loading, ready }
    
    // App foreground: both models loaded (~405MB)
    // App background: models unloaded, vectors stay (~245MB)
    // Extension: slim model on demand (<120MB total)
    
    func handleMemoryWarning() {
        // Priority: unload models first (reload in ~2s)
        // Last resort: evict vectors (reload from mmap)
    }
}
```

Models stored in shared App Group container. Downloaded on first launch (not bundled in app binary). Core ML caches compiled models after first load.

## Component 3: Vector Store + Search

### In-memory layout

```swift
class VectorStore {
    /// Contiguous float16 buffer — one allocation, mmap-backed
    private var vectors: UnsafeMutableBufferPointer<Float16>  // N × 768
    
    /// Parallel metadata array
    private var metadata: [VectorMeta]  // N entries
    
    /// Current count
    private(set) var count: Int
}

struct VectorMeta {
    let uuid: UUID           // 16 bytes — the TSUUID
    let source: String       // path or URL pointer
    let domain: String       // tssa, ieso, turbine, cpc:emails...
    let encodedAt: Date
    var flags: UInt8         // dirty, synced, deleted
}
```

### Memory budget at current scale

| Component | Size |
|-----------|------|
| 147K vectors × 1536 bytes (f16) | 225 MB |
| 147K metadata entries | ~15 MB |
| Working overhead | ~5 MB |
| **Total vector store** | **~245 MB** |

iPhone 15/16 Pro has 8GB RAM. After iOS overhead (~2-3GB), this leaves ample room for the app, models, and vector store simultaneously.

### Search: brute force cosine via Accelerate

```swift
func search(_ query: Vector768, domain: String? = nil, limit: Int = 10) -> [SearchResult] {
    // vDSP.dotProduct across contiguous buffer
    // Apple Accelerate framework, SIMD-optimized
    // 147K comparisons: ~15ms on A17 Pro
    // Domain filter: skip non-matching entries in metadata array
}
```

No ANN index (FAISS, HNSW) needed at this scale. Brute force with vDSP SIMD is fast enough and uses zero additional memory. Revisit if graph exceeds ~500K vectors.

### Persistence

```
group.com.tsuuid.768/
├── vectors.db          ← SQLite, WAL mode
│   ├── tsuuid_768      ← path, title, vec (f16 blob), domain, version
│   └── tsuuid_768_deltas ← sync history
└── vectors.mmap        ← contiguous buffer, precomputed hot cache
```

- **Launch:** open mmap file directly — instant, no deserialization
- **SQLite** is source of truth — mmap is rebuilt when deltas arrive
- **Schema** matches the Mac's `claude_home.db` tsuuid_768 table exactly

## Component 4: Dropbox Sync Engine

### Folder structure

```
~/Dropbox/__768_sync/
├── manifests/
│   ├── mac.manifest.json        ← vector count, version map
│   └── iphone.manifest.json
├── deltas/
│   ├── mac/                     ← Mac writes, phone reads
│   │   └── YYYYMMDDHHMI.delta   ← timestamped bundles
│   └── iphone/                  ← Phone writes, Mac reads
│       └── YYYYMMDDHHMI.delta
├── checkpoints/
│   └── YYYYMMDD.checkpoint.db   ← full SQLite snapshot (weekly/bootstrap)
└── ack/
    ├── mac.ack                  ← "applied through this timestamp"
    └── iphone.ack
```

### Sync lifecycle

**Phone → Mac:**
1. Phone encodes new knowledge → `VectorStore.insert()`
2. `DeltaEncoder.record()` queues sparse delta
3. Batch flush (every 60s foreground, or on app background)
4. Write timestamped `.delta` file to `__768_sync/deltas/iphone/`
5. Dropbox syncs to cloud
6. Mac daemon detects new file, applies deltas, updates `mac.ack`
7. Phone sees ack, prunes old delta files

**Mac → Phone:** identical, folders swapped.

### Bootstrap (first sync)

1. Mac writes `checkpoint.db` — full SQLite with all f16 blobs (~225MB)
2. Phone downloads via Dropbox (one-time)
3. Phone builds `vectors.mmap` from checkpoint
4. From then on, only deltas flow (~1-50KB per encode)

### Conflict resolution

Two peers encoding different documents is not a conflict — both deltas apply cleanly (different paths). Same-path conflict: **last-writer-wins by timestamp**, losing version preserved in delta history table (never lost, always recoverable).

### Batching and battery

```swift
class DropboxSync {
    // Foreground: flush pending deltas every 60 seconds
    // Background: BGAppRefreshTask, check every 15 minutes
    // Manual: pull-to-refresh in Sync tab
    // Never continuous polling — event-driven + batched
}
```

## Component 5: iOS App

### Tab structure

Four tabs. Search is default.

**Tab 1 — Search**
- Text field at top, results stream as you type (300ms debounce)
- Each result: similarity score, title, domain badge, source path
- Tap result → detail view (metadata, chain position, source preview)
- Voice search button (hold to dictate, Speech → LaBSE → search)

**Tab 2 — Capture**
- Camera viewfinder with three modes:
  - **Photo:** CLIP encodes the image
  - **Document:** VisionKit scan → deskew → Live Text OCR → LaBSE encodes text
  - **Voice:** record → Apple Speech transcription → LaBSE encodes transcript
- Post-capture: preview, auto-title, domain picker, confirm to encode
- Also accepts text paste — type or paste anything, encode it
- One-tap workflow: capture → encode → store → delta queued

**Tab 3 — Chains**
- Browse chains by domain
- Visual: horizontal bead chain, each bead = ChainLink, colored by domain
- Tap bead → TSUUID, trit pattern display, source, neighbors
- Gap indicators: dashed segments (from `Chain.detect_gaps()`)
- Coherence score at top

**Tab 4 — Sync**
- Dropbox connection status, last sync time
- Pending deltas count (outgoing / incoming)
- Vector store stats: total vectors, per-domain breakdown, storage size
- Manual sync button
- First-sync progress bar (checkpoint download)
- Model status (loaded/unloaded, memory usage)

### Design principles

- **No onboarding wizard.** First launch: connect Dropbox, download checkpoint, done.
- **Dark mode default.** Plant floor — bright screens are bad.
- **No accounts, no cloud services, no telemetry.** Data lives in Dropbox, nowhere else.
- **Swipe navigation everywhere.** No hamburger menus.
- **Sub-second response.** Results from mmap — no spinners except model inference (~50ms).

## Component 6: Extensions

### Share Extension — "Encode to 768"

Appears in iOS share sheet from any app. Accepts text, images, URLs.

```
User selects text in Safari → Share → "Encode to 768"
  → Load slim int8 LaBSE (~120MB, ~1s)
  → Encode → TSUUID
  → Store in shared App Group VectorStore
  → Queue delta for Dropbox sync
  → Show confirmation with nearest existing match
  → Total time: ~1.5 seconds
```

### Home Screen Widget (WidgetKit)

Small (2x2): tap target opens app with keyboard ready.

Medium (4x2): vector count, search tap target, two most recent encodes.

Widget reads from shared SQLite — no model loading, no inference. Lightweight, always current.

### Siri Shortcut

Registered action: "Search 768 for [query]"

- "Hey Siri, search 768 for boiler inspection reports"
- Shortcut automations (e.g., arriving at plant → surface recent CPC chains)
- Loads slim model, searches mmap store, returns top 3 results

## Determinism Invariant

Any vector encoded anywhere (Mac, phone, future iPad) produces the **same TSUUID** from the same input. LaBSE weights are identical, trit projection is deterministic, UUID v8 packing is deterministic.

The UUID IS the meaning, regardless of which device computed it.

## Memory Budget (iPhone 15/16 Pro, 8GB)

| Component | RAM |
|-----------|-----|
| iOS + system | ~2.5 GB |
| Vector store (147K f16) | ~245 MB |
| LaBSE-full (loaded) | ~235 MB |
| CLIP-full (loaded) | ~170 MB |
| App + UI | ~50 MB |
| **Total** | **~3.2 GB** |
| **Headroom** | **~4.8 GB** |

Comfortable. Models unload on background (drops to ~300MB app footprint).

## Build Order

Layer by layer, each independently testable:

1. **TSUUIDKit Swift Package** — framework, no UI, full test suite
2. **Core ML model conversion** — convert LaBSE + CLIP, verify vector parity with Python
3. **Vector store + search** — mmap, brute force, verify against Mac results
4. **Dropbox sync** — delta exchange, checkpoint bootstrap
5. **iOS app** — four tabs, capture pipeline
6. **Extensions** — share, widget, Siri shortcut

## Future

- **iPad target** — same app, larger chain visualization
- **Mac Catalyst** — TSUUIDKit already has no UI deps, app can compile for Mac
- **watchOS** — slim model on Apple Watch, voice capture only
- **Multi-peer** — more than two devices sync through same Dropbox folder
