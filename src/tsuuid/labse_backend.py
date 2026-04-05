"""
tsuuid.labse_backend — Language-agnostic semantic encoder using Google LaBSE

LaBSE (Language-agnostic BERT Sentence Embeddings) provides a universal
multilingual embedding space across 109 languages. This means a document
in French, Japanese, or Arabic gets the same semantic UUID as its English
equivalent — enabling cross-lingual semantic search and composition.

Uses the same differential anchoring + absmean quantization pipeline as
the bitnet backend, just with LaBSE's 768-dim embeddings instead of
MiniLM's 384-dim.

Requires: pip install sentence-transformers
Model: sentence-transformers/LaBSE (~1.9GB, auto-downloaded)

Reference: TSUUID Framework paper, Section 4.2 (Hay, 2026)
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tsuuid.dimensions import ALL_AXES
from tsuuid.packing import N_DIMS


_CACHE_DIR = Path.home() / ".cache" / "tsuuid"


class LaBSEEncoder:
    """Semantic encoder using Google LaBSE + axis-aligned projection.

    Architecture:
        text → LaBSE (768-dim, 109 languages) → Projection (81-dim) → Absmean (81 trits)

    The key advantage over BitNetEncoder (MiniLM-L6-v2, 384-dim) is true
    multilingual support. The same document encoded in any of 109 languages
    produces the same (or very close) semantic UUID.
    """

    DEFAULT_MODEL = "sentence-transformers/LaBSE"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._model = None
        self._projection_matrix = None

    def _ensure_loaded(self):
        """Lazy-load model and projection matrix on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "LaBSE backend requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        device = self._device
        if device is None:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"

        self._model = SentenceTransformer(self._model_name, device=device)
        self._projection_matrix = self._load_or_build_projection()

    def _load_or_build_projection(self) -> np.ndarray:
        """Load cached projection matrix or build from axis definitions."""
        cache_path = _CACHE_DIR / f"projection_labse_{self._fingerprint()}.npy"

        if cache_path.exists():
            matrix = np.load(cache_path)
            if matrix.shape == (N_DIMS, self._model.get_sentence_embedding_dimension()):
                return matrix

        matrix = self._build_projection_matrix()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
        return matrix

    def _fingerprint(self) -> str:
        """Cache key from model name + axis definitions."""
        parts = [self._model_name]
        for ax in ALL_AXES:
            parts.append(f"{ax.id}:{ax.name}:{ax.negative}:{ax.positive}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    def _build_projection_matrix(self) -> np.ndarray:
        """Build 81xD projection matrix from axis definitions.

        Same differential anchoring as BitNetEncoder:
            direction[i] = normalize(embed(positive_text) - embed(negative_text))

        LaBSE's multilingual training means these directions are
        consistent across languages — the "future" direction in English
        aligns with the "future" direction in French, Japanese, etc.
        """
        pos_texts = [f"{ax.name}: {ax.positive}" for ax in ALL_AXES]
        neg_texts = [f"{ax.name}: {ax.negative}" for ax in ALL_AXES]

        pos_embs = self._model.encode(pos_texts, show_progress_bar=False)
        neg_embs = self._model.encode(neg_texts, show_progress_bar=False)

        directions = pos_embs - neg_embs
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        directions = directions / norms

        return directions.astype(np.float32)

    def encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode text to 81-dimensional ternary vector.

        Works identically across 109 languages — the core LaBSE advantage.
        """
        self._ensure_loaded()
        embedding = self._model.encode(content, show_progress_bar=False)
        projections = self._projection_matrix @ embedding
        trits = self._quantize_absmean(projections)
        return trits

    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        self._ensure_loaded()
        return {
            "model_name": self._model_name,
            "embedding_dim": self._model.get_sentence_embedding_dimension(),
            "device": str(self._model.device),
            "projection_shape": self._projection_matrix.shape,
            "cache_fingerprint": self._fingerprint(),
            "multilingual": True,
            "languages": 109,
        }

    @classmethod
    def from_config(cls, config: dict) -> "LaBSEEncoder":
        """Create encoder from a config dictionary."""
        return cls(
            model_name=config.get("model_name"),
            device=config.get("device"),
        )

    def _quantize_absmean(self, values: np.ndarray) -> np.ndarray:
        """BitNet b1.58 absmean ternary quantization."""
        threshold = max(np.mean(np.abs(values)), 1e-4)
        trits = np.zeros(len(values), dtype=np.int8)
        trits[values > threshold] = 1
        trits[values < -threshold] = -1
        return trits
