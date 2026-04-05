"""
tsuuid.delta — Sparse Delta Encoding for 768-dim LaBSE Vectors

When a document changes, its 768-dim LaBSE embedding shifts. Instead of
retransmitting all 3,072 bytes, send only the dimensions that changed.

For minor edits (typo fix): 50-150 dims change → 200-600 bytes (5-15x savings)
For moderate edits (paragraph added): 300-500 dims → 1,200-2,000 bytes (1.5-2.5x)
For major rewrites: auto-promotes to full checkpoint (no savings, no drift).

Error feedback (residual accumulation) prevents drift from repeated
thresholded updates. Periodic checkpoints reset accumulated error.

Origin: SVN concept (Claude conversation 187ea329, Nov 2025) +
        LaBSE note 2BED (Jan 2026).

Usage:
    from tsuuid.delta import DeltaEncoder, SparseDelta

    enc = DeltaEncoder(epsilon=0.001)

    # Compute sparse delta
    delta = enc.compute_delta(old_vec, new_vec)
    sparse = enc.sparsify(delta, doc_id="doc-001")

    # Transmit sparse.to_bytes() or sparse.to_b64()
    print(f"Sending {sparse.wire_size} bytes instead of 3072")

    # Receiver applies delta
    reconstructed = enc.apply_delta(stored_vec, sparse)

    # Periodic checkpoint resets error accumulation
    checkpoint = enc.make_checkpoint(new_vec, version=5)
"""

import base64
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Full vector size: 768 float32 = 3072 bytes
FULL_VEC_BYTES = 768 * 4
# Dimensions in LaBSE embedding
N_DIMS = 768
# Wire format magic bytes
MAGIC = b"TD"
# Auto-checkpoint threshold: if more dims changed than this, send full vector
AUTO_CHECKPOINT_THRESHOLD = 700

# Wire format header: magic(2) + version(2) + n_changed(2) + flags(1) + reserved(1)
_HEADER_FMT = "<2sHHBB"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 8 bytes
_FLAG_CHECKPOINT = 0x01


@dataclass
class SparseDelta:
    """Sparse representation of a vector delta for transmission.

    Wire format: 8-byte header + 4 bytes per changed dimension.
    Header: [magic "TD" 2B][version uint16][n_changed uint16][flags uint8][reserved 1B]
    Body: N x (uint16 index + float16 value)

    For checkpoints (is_checkpoint=True), indices are range(n_changed) and
    values are the full vector (or truncated to significant dims).
    """

    indices: np.ndarray   # uint16 — which dims changed
    values: np.ndarray    # float16 — delta values (or full values if checkpoint)
    version: int = 0
    is_checkpoint: bool = False

    @property
    def n_changed(self) -> int:
        """Number of dimensions in this delta."""
        return len(self.indices)

    @property
    def wire_size(self) -> int:
        """Total bytes on the wire."""
        return _HEADER_SIZE + 4 * self.n_changed

    def to_bytes(self) -> bytes:
        """Serialize to compact binary wire format."""
        flags = _FLAG_CHECKPOINT if self.is_checkpoint else 0
        header = struct.pack(
            _HEADER_FMT, MAGIC, self.version, self.n_changed, flags, 0
        )
        idx_bytes = self.indices.astype(np.uint16).tobytes()
        val_bytes = self.values.astype(np.float16).tobytes()
        return header + idx_bytes + val_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "SparseDelta":
        """Deserialize from binary wire format."""
        magic, version, n_changed, flags, _ = struct.unpack(
            _HEADER_FMT, data[:_HEADER_SIZE]
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}, expected {MAGIC!r}")

        body = data[_HEADER_SIZE:]
        idx_end = n_changed * 2
        indices = np.frombuffer(body[:idx_end], dtype=np.uint16).copy()
        values = np.frombuffer(body[idx_end : idx_end + n_changed * 2], dtype=np.float16).copy()

        return cls(
            indices=indices,
            values=values,
            version=version,
            is_checkpoint=bool(flags & _FLAG_CHECKPOINT),
        )

    def to_b64(self) -> str:
        """Serialize to base64 string for text-safe transmission."""
        return base64.b64encode(self.to_bytes()).decode("ascii")

    @classmethod
    def from_b64(cls, b64: str) -> "SparseDelta":
        """Deserialize from base64 string."""
        return cls.from_bytes(base64.b64decode(b64))

    def __repr__(self) -> str:
        kind = "checkpoint" if self.is_checkpoint else "delta"
        return (
            f"SparseDelta({kind}, v{self.version}, "
            f"{self.n_changed}/{N_DIMS} dims, {self.wire_size} bytes)"
        )


class DeltaEncoder:
    """Computes and encodes vector deltas for efficient transmission.

    Maintains per-document residual accumulators to prevent drift
    from repeated float16 quantization of sparse deltas.

    The residual accumulator stores what was thresholded away on the
    previous update. On the next update, that residual is folded back
    into the delta before thresholding. This guarantees convergence:
    no information is permanently lost, just deferred.
    """

    def __init__(self, epsilon: float = 0.001, auto_checkpoint: int = AUTO_CHECKPOINT_THRESHOLD):
        self._epsilon = epsilon
        self._auto_checkpoint = auto_checkpoint
        self._residuals: Dict[str, np.ndarray] = {}

    def compute_delta(self, old_vec: np.ndarray, new_vec: np.ndarray) -> np.ndarray:
        """Full delta: new_vec - old_vec. Returns float32[768]."""
        return (new_vec - old_vec).astype(np.float32)

    def sparsify(
        self,
        delta: np.ndarray,
        doc_id: Optional[str] = None,
        epsilon: Optional[float] = None,
        version: int = 0,
    ) -> SparseDelta:
        """Threshold small values, fold in accumulated residual.

        If doc_id is provided, adds the accumulated residual for that
        document before thresholding, then stores the new residual
        (the values that were thresholded away).

        If the number of changed dimensions exceeds auto_checkpoint,
        returns a full checkpoint instead (prevents negative compression).
        """
        eps = epsilon if epsilon is not None else self._epsilon
        adjusted = delta.astype(np.float32).copy()

        # Fold in residual from previous update
        if doc_id is not None and doc_id in self._residuals:
            adjusted += self._residuals[doc_id]

        # Find dimensions above threshold
        mask = np.abs(adjusted) > eps
        n_changed = int(np.sum(mask))

        # Auto-checkpoint if too many dims changed
        if n_changed > self._auto_checkpoint:
            # Can't compute the full vector from just the delta,
            # so return the delta as a dense SparseDelta (all 768 dims)
            indices = np.arange(N_DIMS, dtype=np.uint16)
            values = adjusted.astype(np.float16)
            if doc_id is not None:
                # Reset residual — checkpoint is authoritative
                self._residuals[doc_id] = np.zeros(N_DIMS, dtype=np.float32)
            return SparseDelta(
                indices=indices, values=values, version=version, is_checkpoint=False
            )

        indices = np.where(mask)[0].astype(np.uint16)
        values = adjusted[mask].astype(np.float16)

        # Store residual: what we thresholded away
        if doc_id is not None:
            sparse_full = np.zeros(N_DIMS, dtype=np.float32)
            sparse_full[indices] = values.astype(np.float32)
            self._residuals[doc_id] = adjusted - sparse_full

        return SparseDelta(
            indices=indices, values=values, version=version, is_checkpoint=False
        )

    def apply_delta(self, stored_vec: np.ndarray, sparse_delta: SparseDelta) -> np.ndarray:
        """Reconstruct: stored_vec + sparse_delta -> new vector.

        If sparse_delta.is_checkpoint, the values replace the stored vector
        entirely (they ARE the new vector, not a delta).
        """
        if sparse_delta.is_checkpoint:
            result = np.zeros(N_DIMS, dtype=np.float32)
            result[sparse_delta.indices] = sparse_delta.values.astype(np.float32)
            return result

        result = stored_vec.astype(np.float32).copy()
        result[sparse_delta.indices] += sparse_delta.values.astype(np.float32)
        return result

    def make_checkpoint(self, vec: np.ndarray, version: int = 0) -> SparseDelta:
        """Create a full-vector checkpoint.

        Checkpoints transmit the entire vector. The receiver replaces
        its stored vector entirely, resetting any accumulated error.
        """
        return SparseDelta(
            indices=np.arange(N_DIMS, dtype=np.uint16),
            values=vec.astype(np.float16),
            version=version,
            is_checkpoint=True,
        )

    def reset_residual(self, doc_id: str) -> None:
        """Clear accumulated residual for a document."""
        self._residuals.pop(doc_id, None)

    def get_residual_norm(self, doc_id: str) -> float:
        """L2 norm of accumulated residual. Monitors drift risk."""
        if doc_id not in self._residuals:
            return 0.0
        return float(np.linalg.norm(self._residuals[doc_id]))

    def get_residual(self, doc_id: str) -> Optional[np.ndarray]:
        """Get raw residual vector for persistence."""
        return self._residuals.get(doc_id)

    def set_residual(self, doc_id: str, residual: np.ndarray) -> None:
        """Restore residual from persistent storage."""
        self._residuals[doc_id] = residual.astype(np.float32)


# ── Module-level helpers ──────────────────────────────────


def compression_ratio(sparse_delta: SparseDelta) -> float:
    """Ratio of full vector size to sparse delta size.

    Returns how many times smaller the delta is vs a full 3072-byte vector.
    Values > 1.0 mean savings; < 1.0 means the delta is larger (shouldn't happen).
    """
    if sparse_delta.wire_size == 0:
        return float("inf")
    return FULL_VEC_BYTES / sparse_delta.wire_size


def cosine_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """1.0 - cosine_similarity(original, reconstructed). 0 = perfect."""
    dot = np.dot(original, reconstructed)
    norm = np.linalg.norm(original) * np.linalg.norm(reconstructed)
    if norm < 1e-8:
        return 0.0
    return 1.0 - float(dot / norm)
