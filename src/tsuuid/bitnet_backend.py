"""
tsuuid.bitnet_backend — Semantic encoder using sentence transformers

Implements the BitNet backend for SemanticCodec. Uses a lightweight
sentence transformer to produce embeddings, projects them onto the
81 semantic axes defined in dimensions.py via differential anchoring,
and quantizes to ternary {-1, 0, +1} using BitNet b1.58-style
absmean thresholding.

The projection is axis-aligned by construction: for each axis, the
direction vector is embed(positive_pole) - embed(negative_pole).
This guarantees semantic alignment without training data.

Requires: pip install tsuuid[bitnet]

Reference: TSUUID Framework paper, Section 4.2 (Hay, 2026)
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from tsuuid.dimensions import ALL_AXES
from tsuuid.packing import N_DIMS


# Cache directory for projection matrix
_CACHE_DIR = Path.home() / ".cache" / "tsuuid"


class BitNetEncoder:
    """Semantic encoder using sentence transformers + axis-aligned projection.

    Architecture:
        text → SentenceTransformer (384-dim) → Projection (81-dim) → Absmean (81 trits)

    The projection matrix is built from the 81 axis definitions in dimensions.py.
    Each row is the normalized direction from the negative pole to the positive pole
    in embedding space. The dot product of a document embedding with this direction
    tells you which pole the document aligns with on that axis.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the BitNet encoder.

        Args:
            model_name: HuggingFace model ID. Defaults to all-MiniLM-L6-v2 (~80MB).
            device: Torch device ('cpu', 'mps', 'cuda'). Auto-detected if None.
        """
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
                "BitNet backend requires sentence-transformers. "
                "Install with: pip install tsuuid[bitnet]"
            )

        # Auto-detect device
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
        cache_path = _CACHE_DIR / f"projection_{self._fingerprint()}.npy"

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

        For each axis, computes the direction from negative pole to positive
        pole in embedding space:
            direction[i] = normalize(embed(positive_text) - embed(negative_text))

        This is differential anchoring: the projection is axis-aligned
        by construction.
        """
        pos_texts = [f"{ax.name}: {ax.positive}" for ax in ALL_AXES]
        neg_texts = [f"{ax.name}: {ax.negative}" for ax in ALL_AXES]

        # Batch encode for efficiency
        pos_embs = self._model.encode(pos_texts, show_progress_bar=False)
        neg_embs = self._model.encode(neg_texts, show_progress_bar=False)

        # Direction = positive - negative, then normalize
        directions = pos_embs - neg_embs
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        directions = directions / norms

        return directions.astype(np.float32)

    def encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode text to 81-dimensional ternary vector.

        Args:
            content: Text to encode
            metadata: Optional metadata (reserved for future use)

        Returns:
            np.ndarray shape (81,), dtype int8, values in {-1, 0, +1}
        """
        self._ensure_loaded()

        # Embed document
        embedding = self._model.encode(content, show_progress_bar=False)

        # Project onto 81 semantic axes
        projections = self._projection_matrix @ embedding

        # Quantize to ternary via absmean (BitNet b1.58 style)
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
        }

    @classmethod
    def from_config(cls, config: dict) -> "BitNetEncoder":
        """Create encoder from a config dictionary.

        Args:
            config: Dict with optional keys: model_name (str), device (str).
                    If model_name is omitted, uses DEFAULT_MODEL.
        """
        return cls(
            model_name=config.get("model_name"),
            device=config.get("device"),
        )

    def _quantize_absmean(self, values: np.ndarray) -> np.ndarray:
        """BitNet b1.58 absmean ternary quantization.

        threshold = mean(|values|)
        +1 where value > threshold
        -1 where value < -threshold
        0 otherwise (neutral on this axis)
        """
        threshold = max(np.mean(np.abs(values)), 1e-4)
        trits = np.zeros(len(values), dtype=np.int8)
        trits[values > threshold] = 1
        trits[values < -threshold] = -1
        return trits
