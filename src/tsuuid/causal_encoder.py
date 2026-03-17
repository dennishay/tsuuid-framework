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
        text -> Tokenize -> CausalLM forward -> mean(hidden_states[-1]) -> Project -> Absmean -> 81 trits
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

        # Get embedding dimension from model config (avoids probe forward pass)
        self._embedding_dim = self._model.config.hidden_size

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
        """Create encoder from config dict. model_name is required."""
        if "model_name" not in config:
            raise ValueError("CausalLMEncoder requires 'model_name' in config")
        return cls(
            model_name=config["model_name"],
            device=config.get("device"),
        )
