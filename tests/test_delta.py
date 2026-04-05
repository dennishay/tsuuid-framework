"""Tests for tsuuid.delta — SparseDelta encoding and DeltaEncoder."""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from tsuuid.delta import (
    AUTO_CHECKPOINT_THRESHOLD,
    FULL_VEC_BYTES,
    N_DIMS,
    DeltaEncoder,
    SparseDelta,
    compression_ratio,
    cosine_error,
)


def _random_vec(seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(N_DIMS).astype(np.float32)


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / max(norm, 1e-8))


# ── SparseDelta ──────────────────────────────────────────


class TestSparseDelta:
    def test_to_bytes_from_bytes_roundtrip(self):
        sd = SparseDelta(
            indices=np.array([0, 10, 500], dtype=np.uint16),
            values=np.array([0.5, -0.3, 0.1], dtype=np.float16),
            version=7,
        )
        rebuilt = SparseDelta.from_bytes(sd.to_bytes())
        assert rebuilt.version == 7
        assert rebuilt.n_changed == 3
        assert not rebuilt.is_checkpoint
        np.testing.assert_array_equal(rebuilt.indices, sd.indices)
        np.testing.assert_array_equal(rebuilt.values, sd.values)

    def test_to_b64_from_b64_roundtrip(self):
        sd = SparseDelta(
            indices=np.array([100, 200], dtype=np.uint16),
            values=np.array([0.25, -0.75], dtype=np.float16),
            version=3,
        )
        rebuilt = SparseDelta.from_b64(sd.to_b64())
        assert rebuilt.version == 3
        assert rebuilt.n_changed == 2
        np.testing.assert_array_equal(rebuilt.indices, sd.indices)
        np.testing.assert_array_equal(rebuilt.values, sd.values)

    def test_wire_size_calculation(self):
        sd = SparseDelta(
            indices=np.array([1, 2, 3], dtype=np.uint16),
            values=np.array([0.1, 0.2, 0.3], dtype=np.float16),
        )
        # 8 header + 3 * 4 = 20
        assert sd.wire_size == 20
        assert len(sd.to_bytes()) == 20

    def test_checkpoint_flag_preserved(self):
        sd = SparseDelta(
            indices=np.arange(768, dtype=np.uint16),
            values=np.zeros(768, dtype=np.float16),
            version=1,
            is_checkpoint=True,
        )
        rebuilt = SparseDelta.from_bytes(sd.to_bytes())
        assert rebuilt.is_checkpoint is True

    def test_empty_delta(self):
        sd = SparseDelta(
            indices=np.array([], dtype=np.uint16),
            values=np.array([], dtype=np.float16),
        )
        assert sd.n_changed == 0
        assert sd.wire_size == 8  # header only
        rebuilt = SparseDelta.from_bytes(sd.to_bytes())
        assert rebuilt.n_changed == 0

    def test_full_delta(self):
        sd = SparseDelta(
            indices=np.arange(768, dtype=np.uint16),
            values=np.ones(768, dtype=np.float16),
        )
        assert sd.n_changed == 768
        assert sd.wire_size == 8 + 768 * 4

    def test_repr(self):
        sd = SparseDelta(
            indices=np.array([1], dtype=np.uint16),
            values=np.array([0.5], dtype=np.float16),
            version=3,
        )
        r = repr(sd)
        assert "delta" in r
        assert "v3" in r
        assert "1/768" in r

    def test_invalid_magic_raises(self):
        data = b"XX" + b"\x00" * 6  # wrong magic
        with pytest.raises(ValueError, match="Invalid magic"):
            SparseDelta.from_bytes(data)


# ── DeltaEncoder ─────────────────────────────────────────


class TestDeltaEncoder:
    def test_compute_delta_exact(self):
        enc = DeltaEncoder()
        a = np.ones(N_DIMS, dtype=np.float32)
        b = np.ones(N_DIMS, dtype=np.float32) * 2
        delta = enc.compute_delta(a, b)
        np.testing.assert_allclose(delta, np.ones(N_DIMS))

    def test_zero_delta_for_identical_vectors(self):
        enc = DeltaEncoder()
        v = _random_vec()
        delta = enc.compute_delta(v, v)
        assert np.allclose(delta, 0)

    def test_sparsify_removes_small_values(self):
        enc = DeltaEncoder(epsilon=0.05)
        delta = np.zeros(N_DIMS, dtype=np.float32)
        delta[0] = 0.1    # above threshold
        delta[1] = 0.01   # below threshold
        delta[2] = -0.2   # above threshold

        sparse = enc.sparsify(delta)
        assert sparse.n_changed == 2
        assert 0 in sparse.indices
        assert 2 in sparse.indices
        assert 1 not in sparse.indices

    def test_sparsify_preserves_large_values(self):
        enc = DeltaEncoder(epsilon=0.001)
        delta = np.zeros(N_DIMS, dtype=np.float32)
        delta[100] = 0.5
        delta[200] = -0.3

        sparse = enc.sparsify(delta)
        assert 100 in sparse.indices
        assert 200 in sparse.indices
        idx_100 = np.where(sparse.indices == 100)[0][0]
        assert abs(float(sparse.values[idx_100]) - 0.5) < 0.01  # float16 precision

    def test_apply_delta_reconstructs_vector(self):
        enc = DeltaEncoder(epsilon=0.01)
        old = _random_vec(seed=1)
        new = old.copy()
        new[:100] += np.random.RandomState(2).randn(100) * 0.1

        delta = enc.compute_delta(old, new)
        sparse = enc.sparsify(delta)
        reconstructed = enc.apply_delta(old, sparse)

        sim = _cosine_sim(new, reconstructed)
        assert sim > 0.999, f"Cosine similarity {sim} too low"

    def test_lossless_roundtrip_no_threshold(self):
        enc = DeltaEncoder(epsilon=0.0)  # no thresholding
        old = _random_vec(seed=10)
        new = _random_vec(seed=11)

        delta = enc.compute_delta(old, new)
        sparse = enc.sparsify(delta)
        # All 768 dims should be present (no thresholding)
        assert sparse.n_changed == N_DIMS

        reconstructed = enc.apply_delta(old, sparse)
        # float16 precision limits: ~0.001 relative error
        sim = _cosine_sim(new, reconstructed)
        assert sim > 0.999

    def test_residual_accumulation_prevents_drift(self):
        """Apply 100 sequential small perturbations with error feedback.
        Cosine similarity to true cumulative vector must stay > 0.999."""
        enc = DeltaEncoder(epsilon=0.01)
        rng = np.random.RandomState(42)

        true_vec = rng.randn(N_DIMS).astype(np.float32)
        stored_vec = true_vec.copy()
        doc_id = "drift-test"

        for i in range(100):
            # Small perturbation to random 20 dims
            new_vec = true_vec.copy()
            dims = rng.choice(N_DIMS, 20, replace=False)
            new_vec[dims] += rng.randn(20) * 0.02

            delta = enc.compute_delta(true_vec, new_vec)
            sparse = enc.sparsify(delta, doc_id=doc_id, version=i + 1)
            stored_vec = enc.apply_delta(stored_vec, sparse)
            true_vec = new_vec

        sim = _cosine_sim(true_vec, stored_vec)
        assert sim > 0.999, (
            f"After 100 updates, cosine similarity dropped to {sim:.6f}. "
            f"Residual norm: {enc.get_residual_norm(doc_id):.6f}"
        )

    def test_checkpoint_resets_residual(self):
        enc = DeltaEncoder(epsilon=0.1)
        delta = np.random.randn(N_DIMS).astype(np.float32) * 0.05
        enc.sparsify(delta, doc_id="doc-cp")
        assert enc.get_residual_norm("doc-cp") > 0

        vec = np.random.randn(N_DIMS).astype(np.float32)
        cp = enc.make_checkpoint(vec, version=5)
        enc.reset_residual("doc-cp")
        assert enc.get_residual_norm("doc-cp") == 0.0
        assert cp.is_checkpoint

    def test_apply_checkpoint_replaces_vector(self):
        enc = DeltaEncoder()
        old = np.ones(N_DIMS, dtype=np.float32)
        new = np.ones(N_DIMS, dtype=np.float32) * 3

        cp = enc.make_checkpoint(new, version=1)
        result = enc.apply_delta(old, cp)
        # Checkpoint replaces, not adds — result should be close to new
        sim = _cosine_sim(new, result)
        assert sim > 0.999

    def test_auto_checkpoint_when_many_dims(self):
        enc = DeltaEncoder(epsilon=0.001, auto_checkpoint=100)
        # Delta that changes all 768 dims
        delta = np.random.randn(N_DIMS).astype(np.float32)
        sparse = enc.sparsify(delta)
        # Should auto-promote: all 768 dims included
        assert sparse.n_changed == N_DIMS

    def test_cosine_bound(self):
        """For L2-normalized vectors, verify cos(v, v') >= 1 - ||error||^2 / 2."""
        enc = DeltaEncoder(epsilon=0.01)
        rng = np.random.RandomState(99)

        v_true = rng.randn(N_DIMS).astype(np.float32)
        v_true /= np.linalg.norm(v_true)  # L2 normalize

        perturbation = rng.randn(N_DIMS).astype(np.float32) * 0.05
        v_new = v_true + perturbation
        v_new /= np.linalg.norm(v_new)

        delta = enc.compute_delta(v_true, v_new)
        sparse = enc.sparsify(delta)
        v_recon = enc.apply_delta(v_true, sparse)

        error = v_new - v_recon
        error_norm_sq = float(np.dot(error, error))
        cos_sim = _cosine_sim(v_new, v_recon)
        bound = 1.0 - error_norm_sq / 2.0

        assert cos_sim >= bound - 1e-6, (
            f"Cosine {cos_sim:.6f} violated bound {bound:.6f}"
        )

    def test_get_set_residual(self):
        enc = DeltaEncoder()
        r = np.random.randn(N_DIMS).astype(np.float32)
        enc.set_residual("doc-x", r)
        got = enc.get_residual("doc-x")
        np.testing.assert_array_equal(got, r)

    def test_no_residual_returns_none(self):
        enc = DeltaEncoder()
        assert enc.get_residual("nonexistent") is None
        assert enc.get_residual_norm("nonexistent") == 0.0


# ── Compression Stats ────────────────────────────────────


class TestCompressionStats:
    def test_compression_ratio_minor_edit(self):
        # 50 dims changed → 208 bytes → ~14.8x
        sd = SparseDelta(
            indices=np.arange(50, dtype=np.uint16),
            values=np.ones(50, dtype=np.float16),
        )
        ratio = compression_ratio(sd)
        assert ratio > 10

    def test_compression_ratio_major_edit(self):
        # 700 dims changed → 2808 bytes → ~1.1x
        sd = SparseDelta(
            indices=np.arange(700, dtype=np.uint16),
            values=np.ones(700, dtype=np.float16),
        )
        ratio = compression_ratio(sd)
        assert 1.0 < ratio < 1.5

    def test_compression_ratio_full_delta(self):
        # 768 dims → 3080 bytes → ~1.0x
        sd = SparseDelta(
            indices=np.arange(768, dtype=np.uint16),
            values=np.ones(768, dtype=np.float16),
        )
        ratio = compression_ratio(sd)
        assert ratio < 1.1

    def test_cosine_error_zero_for_identical(self):
        v = _random_vec()
        assert cosine_error(v, v) < 1e-6

    def test_cosine_error_positive_for_different(self):
        a = _random_vec(seed=1)
        b = _random_vec(seed=2)
        assert cosine_error(a, b) > 0
