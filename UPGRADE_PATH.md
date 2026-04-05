# BitNet b1.58 Upgrade Path

## Current State (March 2026)

TSUUID has three encoding backends:

| Backend | Model | Weights | Semantic Quality | Speed |
|---------|-------|---------|-----------------|-------|
| `hash` | SHA-256 + keywords | N/A | 11.5% alignment | <1ms |
| `bitnet` | all-MiniLM-L6-v2 (80MB) | float32 | 46.2% alignment | ~50ms |
| `causal` | any HuggingFace causal LM | varies | varies | varies |

The `bitnet` backend uses sentence-transformers with BitNet-style absmean
ternary quantization on the OUTPUT. The model weights themselves are float32.

## Phase Roadmap

### Phase 1: Hash Reference (DONE)
- Keyword matching + SHA-256 deterministic trit generation
- No ML dependencies, pure Python + numpy
- Useful for testing and demonstration only

### Phase 1.5: Sentence Transformer + Absmean (DONE)
- all-MiniLM-L6-v2 produces 384-dim float embeddings
- Differential projection onto 81 semantic axes
- AbsMean quantization: threshold = mean(|values|)
- ~155MB RAM total, <100ms per encode on CPU
- 4x better semantic alignment than hash backend

### Phase 2: Causal LM Encoder (DONE - architecture)
- Mean-pool hidden states from any decoder-only LM
- Same projection + absmean pipeline as Phase 1.5
- Tested with tiny-gpt2; ready for BitNet 2B when desired
- Usage: `SemanticCodec(backend="causal", model_config={"model_name": "..."})`

### Phase 3: True BitNet Ternary-Weight Encoder (BLOCKED)
**Status: Waiting on ecosystem**

A true BitNet upgrade means the model WEIGHTS are ternary {-1, 0, +1},
not just the output quantization. This gives:
- Addition replaces multiplication in inference
- ~1.58 bits per parameter (vs 32 bits)
- Massive efficiency on edge devices

**What exists today:**
- Microsoft's BitNet 2B model (bitnet-b1.58-2B-4T) — works for text generation
- bitnet.cpp — C++ inference engine, macOS ARM64 compatible, fast
- HuggingFace Transformers — can load BitNet models

**What's missing:**
- No BitNet sentence-transformer / embedding model published
- No embedding extraction API in bitnet.cpp
- HuggingFace runs BitNet through float ops (no ternary kernel efficiency)
- No pip-installable ternary embedding model

**When this unblocks, the upgrade is:**
```python
# Just change the model_name — architecture is ready
codec = SemanticCodec(
    backend="causal",
    model_config={"model_name": "microsoft/bitnet-embedding-model"}
)
```

Or if a sentence-transformer variant ships:
```python
codec = SemanticCodec(
    backend="bitnet",
    model_config={"model_name": "some-org/bitnet-sentence-transformer"}
)
```

## Swapping Models

Any model can be used with either backend:

```python
from tsuuid import SemanticCodec

# Default (all-MiniLM-L6-v2)
codec = SemanticCodec(backend="bitnet")

# Larger sentence transformer (better quality, more RAM)
codec = SemanticCodec(
    backend="bitnet",
    model_config={"model_name": "sentence-transformers/all-mpnet-base-v2"}
)

# Causal LM (decoder-only)
codec = SemanticCodec(
    backend="causal",
    model_config={"model_name": "sshleifer/tiny-gpt2", "device": "cpu"}
)
```

### Phase 8: Delta Encoding for Efficient Vector Updates (NEW)
**Status: Active**

Full 768-dim vectors are 3,072 bytes. When a document changes slightly,
most dimensions stay the same. Delta encoding sends only what changed:

- **SparseDelta format**: 4 bytes per changed dimension + 8 byte header
- **Typical savings**: 5-15x for minor edits (50-150 dims change)
- **Error feedback**: Residual accumulator prevents float16 quantization drift
- **Periodic checkpoints**: Full vector sync resets error accumulation
- **Auto-checkpoint**: When >700 dims change, promotes to full vector (no negative compression)

Usage:
```python
from tsuuid.delta import DeltaEncoder, SparseDelta
from tsuuid.labse_768 import Encoder768

# Standalone delta math
enc = DeltaEncoder(epsilon=0.001)
delta = enc.compute_delta(old_vec, new_vec)
sparse = enc.sparsify(delta, doc_id="doc-001")
print(f"Sent {sparse.wire_size} bytes instead of 3072")

# With storage integration
enc768 = Encoder768()
sparse = enc768.update("path/to/doc.md", new_vec)
enc768.checkpoint("path/to/doc.md")  # periodic full sync
```

Origin: SVN concept (Claude conversation 187ea329, Nov 2025) +
LaBSE note 2BED (Jan 2026). Federated learning error feedback
technique (Top-K sparsification with residual accumulation).

## Monitoring the Ecosystem

Watch for:
1. **BitNet embedding model** — Microsoft or community publishes a ternary-weight encoder
2. **bitnet.cpp embedding mode** — `--embedding` flag added (like llama.cpp's `llama-embedding`)
3. **Optimized Python kernels** — ternary matmul in PyTorch or custom CUDA/Metal kernels
4. **sentence-transformers BitNet** — ternary-weight model compatible with the ST library
