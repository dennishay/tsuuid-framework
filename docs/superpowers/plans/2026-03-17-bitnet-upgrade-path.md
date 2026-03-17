# BitNet b1.58 Model Upgrade Path — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the BitNet backend model-agnostic so any HuggingFace encoder can be swapped in, add a benchmark suite to validate semantic alignment quality, and document the upgrade path for when true ternary-weight embedding models ship.

**Architecture:** Refactor BitNetEncoder to accept arbitrary model configs (not just all-MiniLM-L6-v2). Add a benchmark script that scores each backend's semantic alignment across all 81 axes. Add a causal LM encoder backend that extracts embeddings from decoder-only models via mean-pooling (for BitNet 2B when desired).

**Tech Stack:** Python 3.12, sentence-transformers, transformers, numpy, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tsuuid/bitnet_backend.py` | MODIFY | Add model config support, expose model metadata, add `from_config()` factory |
| `src/tsuuid/codec.py` | MODIFY | Pass model config through to backend |
| `src/tsuuid/causal_encoder.py` | CREATE | Mean-pool hidden states from any causal LM (for BitNet 2B) |
| `tests/test_bitnet_backend.py` | MODIFY | Add tests for model-agnostic features |
| `tests/test_causal_encoder.py` | CREATE | Tests for causal LM encoder (gated by transformers) |
| `src/examples/09_benchmark_backends.py` | CREATE | Benchmark semantic alignment across backends |
| `src/examples/10_custom_model.py` | CREATE | Demo swapping in different models |
| `UPGRADE_PATH.md` | CREATE | Document the roadmap from sentence-transformer → true BitNet |

---

## Chunk 1: Model-Agnostic Backend

### Task 1: Refactor BitNetEncoder to accept model config

**Files:**
- Modify: `src/tsuuid/bitnet_backend.py:48-56`
- Test: `tests/test_bitnet_backend.py`

- [ ] **Step 1: Write failing tests for model config**

```python
# In tests/test_bitnet_backend.py, add to TestBitNetEncoder class:

def test_model_info(self, encoder):
    """Encoder should expose model metadata."""
    info = encoder.model_info()
    assert "model_name" in info
    assert "embedding_dim" in info
    assert "device" in info
    assert info["embedding_dim"] == 384  # MiniLM default

def test_custom_model_name(self):
    """Should accept alternate model names."""
    enc = BitNetEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    info = enc.model_info()
    assert info["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_from_config(self):
    """Factory method should create encoder from config dict."""
    config = {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"}
    enc = BitNetEncoder.from_config(config)
    trits = enc.encode("test")
    assert trits.shape == (N_DIMS,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/test_bitnet_backend.py -v -k "model_info or custom_model or from_config"`
Expected: FAIL — `model_info` and `from_config` don't exist

- [ ] **Step 3: Implement model_info() and from_config()**

Add to `BitNetEncoder` in `bitnet_backend.py`:

```python
def model_info(self) -> dict:
    """Return metadata about the loaded model."""
    self._ensure_loaded()
    return {
        "model_name": self._model_name,
        "embedding_dim": self._model.get_sentence_embedding_dimension(),
        "device": str(self._model.device),
        "projection_shape": self._projection_matrix.shape,
        "cache_fingerprint": self._fingerprint(),
    }

@classmethod
def from_config(cls, config: dict) -> "BitNetEncoder":
    """Create encoder from a config dictionary.

    Args:
        config: Dict with keys: model_name (str), device (str, optional)
    """
    return cls(
        model_name=config.get("model_name"),
        device=config.get("device"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/test_bitnet_backend.py -v -k "model_info or custom_model or from_config"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd ~/tsuuid-framework && git add src/tsuuid/bitnet_backend.py tests/test_bitnet_backend.py
git commit -m "feat(#24): model-agnostic BitNetEncoder — model_info() + from_config() factory"
```

---

### Task 2: Pass model config through SemanticCodec

**Files:**
- Modify: `src/tsuuid/codec.py:67-81,95-99`
- Test: `tests/test_bitnet_backend.py`

- [ ] **Step 1: Write failing test for codec model passthrough**

```python
# In tests/test_bitnet_backend.py, add to TestBitNetCodecIntegration:

def test_codec_with_model_config(self):
    """Codec should pass model config to BitNet backend."""
    codec = SemanticCodec(
        backend="bitnet",
        model_config={"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
    )
    uid = codec.encode("test document")
    assert uid.version == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/test_bitnet_backend.py::TestBitNetCodecIntegration::test_codec_with_model_config -v`
Expected: FAIL — `__init__` doesn't accept `model_config`

- [ ] **Step 3: Modify SemanticCodec.__init__ to accept model_config**

In `codec.py`, update `__init__`:

```python
def __init__(self, backend: str = "hash", model_config: Optional[Dict] = None):
    """Initialize codec.

    Args:
        backend: Encoding backend.
            "hash" — deterministic hash-based (reference implementation)
            "bitnet" — BitNet b1.58 model (requires bitnet dependency)
        model_config: Optional config dict passed to the backend encoder.
            For bitnet: {"model_name": "...", "device": "cpu|mps|cuda"}
    """
    self.dims = SemanticDimensions()
    self.backend = backend
    self._model_config = model_config or {}
    self._bitnet_encoder = None  # Lazy initialization
    self._keyword_map = self._build_keyword_map()
```

Update the bitnet branch in `encode()`:

```python
elif self.backend == "bitnet":
    if self._bitnet_encoder is None:
        from tsuuid.bitnet_backend import BitNetEncoder
        self._bitnet_encoder = BitNetEncoder.from_config(self._model_config)
    trits = self._bitnet_encoder.encode(content, metadata)
```

- [ ] **Step 4: Run all tests**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
cd ~/tsuuid-framework && git add src/tsuuid/codec.py tests/test_bitnet_backend.py
git commit -m "feat(#24): codec passes model_config to BitNet backend"
```

---

## Chunk 2: Benchmark Suite

### Task 3: Semantic alignment benchmark

**Files:**
- Create: `src/examples/09_benchmark_backends.py`

- [ ] **Step 1: Create benchmark script**

```python
"""
Example 09: Benchmark Semantic Alignment

Tests how well each backend's encodings align with the intended
meaning of the 81 semantic axes. For each axis, encodes a
positive-pole and negative-pole sentence and checks whether
the corresponding trit has the correct sign.

Requires: pip install tsuuid[bitnet]
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from tsuuid import SemanticCodec
from tsuuid.dimensions import ALL_AXES
from tsuuid.packing import unpack_uuid_to_trits

# Test cases: (axis_index_0based, positive_text, negative_text)
# Covers a representative subset of axes across all layers
ALIGNMENT_TESTS = [
    # Protocol layer
    (0, "future projections and upcoming plans", "historical records from the past"),
    (1, "create a new document", "delete the old records"),
    (2, "confirmed and approved", "denied and rejected"),
    (4, "mandatory requirement", "forbidden action"),
    (5, "public announcement shared openly", "private confidential internal memo"),
    (7, "outbound export shipment sent", "inbound import received"),
    # Organization layer
    (20, "internal company operations", "external vendor supplier"),
    (25, "critical urgent emergency", "deferred low priority backlog"),
    # Application layer
    (35, "business financial invoice payment revenue", "technical engineering server code system"),
    (37, "expense cost payment outflow", "income revenue receipt inflow"),
    (41, "completed done finished resolved", "failed error broken incomplete"),
    (46, "excellent wonderful positive happy", "terrible horrible negative sad"),
    # Field layer
    (73, "increasing rising growing climbing", "decreasing declining falling dropping"),
]


def benchmark_backend(backend_name, codec):
    """Score alignment: +1 correct, 0 neutral, -1 wrong."""
    correct = 0
    neutral = 0
    wrong = 0
    details = []

    for axis_idx, pos_text, neg_text in ALIGNMENT_TESTS:
        axis = ALL_AXES[axis_idx]
        pos_uuid = codec.encode(pos_text)
        neg_uuid = codec.encode(neg_text)
        pos_trits = unpack_uuid_to_trits(pos_uuid)
        neg_trits = unpack_uuid_to_trits(neg_uuid)

        pos_val = int(pos_trits[axis_idx])
        neg_val = int(neg_trits[axis_idx])

        # Positive text should get +1, negative text should get -1
        pos_ok = pos_val == 1
        neg_ok = neg_val == -1

        if pos_ok and neg_ok:
            status = "CORRECT"
            correct += 1
        elif pos_val == 0 and neg_val == 0:
            status = "NEUTRAL"
            neutral += 1
        elif pos_val == neg_val:
            status = "SAME"
            wrong += 1
        else:
            # Partial credit: at least one is right
            if pos_ok or neg_ok:
                status = "PARTIAL"
                correct += 0.5
                neutral += 0.5
            else:
                status = "WRONG"
                wrong += 1

        details.append((axis.name, axis_idx + 1, pos_val, neg_val, status))

    return correct, neutral, wrong, details


def main():
    print("=" * 70)
    print("TSUUID Semantic Alignment Benchmark")
    print("=" * 70)

    backends = [("hash", SemanticCodec(backend="hash"))]
    try:
        backends.append(("bitnet", SemanticCodec(backend="bitnet")))
    except Exception as e:
        print(f"BitNet backend unavailable: {e}")

    for name, codec in backends:
        correct, neutral, wrong, details = benchmark_backend(name, codec)
        total = len(ALIGNMENT_TESTS)
        score = correct / total * 100

        print(f"\n--- {name.upper()} Backend ---")
        print(f"Score: {score:.0f}% ({correct:.0f}/{total} correct, "
              f"{neutral:.0f} neutral, {wrong:.0f} wrong)")
        print()
        print(f"  {'Axis':<20} {'Dim':>4} {'Pos':>4} {'Neg':>4} {'Result':<8}")
        print(f"  {'-'*20} {'-'*4} {'-'*4} {'-'*4} {'-'*8}")
        for axis_name, dim, pos, neg, status in details:
            print(f"  {axis_name:<20} {dim:>4} {pos:>+4} {neg:>+4} {status:<8}")

    if len(backends) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        h_score = benchmark_backend("hash", backends[0][1])[0]
        b_score = benchmark_backend("bitnet", backends[1][1])[0]
        total = len(ALIGNMENT_TESTS)
        print(f"  Hash:   {h_score/total*100:.0f}%")
        print(f"  BitNet: {b_score/total*100:.0f}%")
        diff = b_score - h_score
        if diff > 0:
            print(f"  BitNet is {diff:.0f} axes better")
        elif diff < 0:
            print(f"  Hash is {-diff:.0f} axes better")
        else:
            print(f"  Tied")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `cd ~/tsuuid-framework && .venv/bin/python src/examples/09_benchmark_backends.py`
Expected: Output showing alignment scores for both backends

- [ ] **Step 3: Commit**

```bash
cd ~/tsuuid-framework && git add src/examples/09_benchmark_backends.py
git commit -m "feat(#24): semantic alignment benchmark — hash vs bitnet comparison"
```

---

## Chunk 3: Causal LM Encoder + Upgrade Path Doc

### Task 4: Causal LM encoder backend

**Files:**
- Create: `src/tsuuid/causal_encoder.py`
- Create: `tests/test_causal_encoder.py`

- [ ] **Step 1: Write failing tests**

```python
"""
Tests for CausalLMEncoder — extracts embeddings from decoder-only models.

Gated by transformers import so existing tests are unaffected.
Uses a tiny model for testing speed.
"""

import sys
import pytest

sys.path.insert(0, "src")

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

import numpy as np
from tsuuid.causal_encoder import CausalLMEncoder
from tsuuid.packing import N_DIMS


class TestCausalLMEncoder:
    """Tests for hidden-state mean-pool encoder."""

    @pytest.fixture(scope="class")
    def encoder(self):
        # Use the smallest available model for testing
        return CausalLMEncoder(
            model_name="sshleifer/tiny-gpt2",
            device="cpu",
        )

    def test_output_shape(self, encoder):
        trits = encoder.encode("test document")
        assert trits.shape == (N_DIMS,)

    def test_output_dtype(self, encoder):
        trits = encoder.encode("test document")
        assert trits.dtype == np.int8

    def test_output_values(self, encoder):
        trits = encoder.encode("test document about finances")
        assert all(t in (-1, 0, 1) for t in trits)

    def test_determinism(self, encoder):
        t1 = encoder.encode("test input")
        t2 = encoder.encode("test input")
        assert np.array_equal(t1, t2)

    def test_different_inputs(self, encoder):
        t1 = encoder.encode("financial invoice payment")
        t2 = encoder.encode("server error crash failure")
        assert not np.array_equal(t1, t2)

    def test_model_info(self, encoder):
        info = encoder.model_info()
        assert "model_name" in info
        assert "embedding_dim" in info
        assert "pooling" in info
        assert info["pooling"] == "mean"
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/test_causal_encoder.py -v`
Expected: FAIL — `causal_encoder` module doesn't exist

- [ ] **Step 3: Implement CausalLMEncoder**

```python
"""
tsuuid.causal_encoder — Embedding extraction from causal (decoder-only) LMs

Extracts embeddings from any HuggingFace causal language model by
mean-pooling the last hidden states. This enables using models like
Microsoft's BitNet b1.58 2B as sentence encoders.

The extracted embeddings are projected onto the 81 semantic axes
using the same differential projection as BitNetEncoder, then
quantized to ternary via absmean.

This is the bridge to true BitNet: when a ternary-weight embedding
model ships, swap the model_name and get native efficiency.

Requires: pip install transformers torch
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tsuuid.dimensions import ALL_AXES
from tsuuid.packing import N_DIMS

_CACHE_DIR = Path.home() / ".cache" / "tsuuid"


class CausalLMEncoder:
    """Extract embeddings from decoder-only LMs via mean-pooling.

    Architecture:
        text → Tokenize → CausalLM forward → mean(hidden_states[-1]) → Project → Absmean → 81 trits
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize with a HuggingFace causal LM model ID.

        Args:
            model_name: HuggingFace model ID (e.g., "microsoft/bitnet-b1.58-2B-4T")
            device: Torch device. Auto-detected if None.
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._projection_matrix = None
        self._embedding_dim = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "CausalLMEncoder requires transformers and torch. "
                "Install with: pip install transformers torch"
            )

        device = self._device
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, output_hidden_states=True
        ).to(device).eval()

        # Detect embedding dimension from a probe forward pass
        with torch.no_grad():
            probe = self._tokenizer("probe", return_tensors="pt").to(device)
            out = self._model(**probe)
            self._embedding_dim = out.hidden_states[-1].shape[-1]

        self._projection_matrix = self._load_or_build_projection()

    def _fingerprint(self) -> str:
        parts = [self._model_name, f"dim={self._embedding_dim}"]
        for ax in ALL_AXES:
            parts.append(f"{ax.id}:{ax.name}:{ax.negative}:{ax.positive}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    def _load_or_build_projection(self) -> np.ndarray:
        cache_path = _CACHE_DIR / f"causal_proj_{self._fingerprint()}.npy"
        if cache_path.exists():
            matrix = np.load(cache_path)
            if matrix.shape == (N_DIMS, self._embedding_dim):
                return matrix

        matrix = self._build_projection_matrix()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        return matrix

    def _embed_text(self, text: str) -> np.ndarray:
        """Extract mean-pooled embedding from last hidden state."""
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean-pool last hidden state (exclude padding)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    def _build_projection_matrix(self) -> np.ndarray:
        pos_texts = [f"{ax.name}: {ax.positive}" for ax in ALL_AXES]
        neg_texts = [f"{ax.name}: {ax.negative}" for ax in ALL_AXES]

        pos_embs = np.array([self._embed_text(t) for t in pos_texts])
        neg_embs = np.array([self._embed_text(t) for t in neg_texts])

        directions = pos_embs - neg_embs
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        directions = directions / norms

        return directions.astype(np.float32)

    def encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode text to 81-dimensional ternary vector."""
        self._ensure_loaded()
        embedding = self._embed_text(content)
        projections = self._projection_matrix @ embedding
        return self._quantize_absmean(projections)

    def _quantize_absmean(self, values: np.ndarray) -> np.ndarray:
        threshold = max(np.mean(np.abs(values)), 1e-4)
        trits = np.zeros(len(values), dtype=np.int8)
        trits[values > threshold] = 1
        trits[values < -threshold] = -1
        return trits

    def model_info(self) -> dict:
        self._ensure_loaded()
        return {
            "model_name": self._model_name,
            "embedding_dim": self._embedding_dim,
            "device": str(self._device),
            "pooling": "mean",
            "projection_shape": self._projection_matrix.shape,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CausalLMEncoder":
        return cls(
            model_name=config["model_name"],
            device=config.get("device"),
        )
```

- [ ] **Step 4: Run tests**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/test_causal_encoder.py -v`
Expected: All PASS

- [ ] **Step 5: Wire causal backend into codec**

In `codec.py`, add `"causal"` backend option:

```python
elif self.backend == "causal":
    if self._causal_encoder is None:
        from tsuuid.causal_encoder import CausalLMEncoder
        self._causal_encoder = CausalLMEncoder.from_config(self._model_config)
    trits = self._causal_encoder.encode(content, metadata)
```

Add `self._causal_encoder = None` in `__init__`.

- [ ] **Step 6: Run all tests**

Run: `cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
cd ~/tsuuid-framework && git add src/tsuuid/causal_encoder.py tests/test_causal_encoder.py src/tsuuid/codec.py
git commit -m "feat(#24): causal LM encoder — mean-pool hidden states from any decoder-only model"
```

---

### Task 5: Upgrade path documentation

**Files:**
- Create: `UPGRADE_PATH.md`

- [ ] **Step 1: Write upgrade path doc**

Document the three phases, what's done, what's blocked, and how to swap in a new model when one becomes available.

- [ ] **Step 2: Commit**

```bash
cd ~/tsuuid-framework && git add UPGRADE_PATH.md
git commit -m "docs(#24): BitNet upgrade path — current state and roadmap"
```

---

### Task 6: Custom model example

**Files:**
- Create: `src/examples/10_custom_model.py`

- [ ] **Step 1: Write example showing model swap**

Demo showing how to use the codec with different models via `model_config`.

- [ ] **Step 2: Commit**

```bash
cd ~/tsuuid-framework && git add src/examples/10_custom_model.py
git commit -m "feat(#24): custom model example — swap sentence-transformer models"
```

---

## Verification

After all tasks:

```bash
# Full test suite
cd ~/tsuuid-framework && .venv/bin/python -m pytest tests/ -v

# Benchmark
.venv/bin/python src/examples/09_benchmark_backends.py

# Smoke test causal encoder (if tiny model available)
.venv/bin/python -c "
from tsuuid.causal_encoder import CausalLMEncoder
enc = CausalLMEncoder('sshleifer/tiny-gpt2', device='cpu')
trits = enc.encode('test')
print(f'Trits: {trits.shape}, non-zero: {sum(t != 0 for t in trits)}')
"
```
