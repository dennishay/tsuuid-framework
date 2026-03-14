"""
Tests for BitNet semantic backend.

All tests are skipped if sentence-transformers is not installed,
so the original 9/9 hash-backend tests remain unaffected.
"""

import sys
import pytest

sys.path.insert(0, "src")

st = pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")

import numpy as np
from tsuuid.codec import SemanticCodec
from tsuuid.bitnet_backend import BitNetEncoder
from tsuuid.packing import N_DIMS


class TestBitNetEncoder:
    """Unit tests for BitNetEncoder."""

    @pytest.fixture(scope="class")
    def encoder(self):
        return BitNetEncoder()

    def test_output_shape(self, encoder):
        trits = encoder.encode("test document")
        assert trits.shape == (N_DIMS,)

    def test_output_dtype(self, encoder):
        trits = encoder.encode("test document")
        assert trits.dtype == np.int8

    def test_output_values(self, encoder):
        trits = encoder.encode("A complex financial invoice for server maintenance")
        assert all(t in (-1, 0, 1) for t in trits)

    def test_determinism(self, encoder):
        t1 = encoder.encode("invoice from Acme Corp")
        t2 = encoder.encode("invoice from Acme Corp")
        assert np.array_equal(t1, t2)

    def test_different_inputs_different_outputs(self, encoder):
        t1 = encoder.encode("financial invoice payment approved")
        t2 = encoder.encode("critical server error system failure")
        assert not np.array_equal(t1, t2)

    def test_not_all_zeros(self, encoder):
        trits = encoder.encode("important financial document requiring approval")
        assert np.any(trits != 0)

    def test_sparsity_reasonable(self, encoder):
        """Absmean should produce a reasonable number of zeros."""
        trits = encoder.encode("A detailed technical specification document")
        n_zero = np.sum(trits == 0)
        # Should be roughly 1/3 zero (wide tolerance for model variation)
        assert 5 < n_zero < 65, f"Expected ~27 zeros, got {n_zero}"

    def test_semantic_alignment_domain(self, encoder):
        """Axis 36 (dim index 35) should distinguish technical from business."""
        t_tech = encoder.encode("server error code database connection refused crash")
        t_biz = encoder.encode("quarterly revenue financial report profit margin sales")
        # At minimum they should not both be the same non-zero value
        if t_tech[35] != 0 and t_biz[35] != 0:
            assert t_tech[35] != t_biz[35]

    def test_semantic_alignment_sentiment(self, encoder):
        """Axis 47 (dim index 46) should distinguish negative from positive sentiment."""
        t_neg = encoder.encode("terrible failure disaster loss broken error")
        t_pos = encoder.encode("excellent success achievement profit growth winning")
        if t_neg[46] != 0 and t_pos[46] != 0:
            assert t_neg[46] != t_pos[46]


class TestBitNetCodecIntegration:
    """Integration tests: BitNet backend through SemanticCodec."""

    def test_codec_encode_decode_roundtrip(self):
        codec = SemanticCodec(backend="bitnet")
        uid = codec.encode("test document about financial invoices")
        meaning = codec.decode(uid)
        assert meaning.uuid == uid
        assert meaning.uuid.version == 8
        assert len(meaning.trits) == N_DIMS

    def test_codec_determinism(self):
        codec = SemanticCodec(backend="bitnet")
        u1 = codec.encode("test")
        u2 = codec.encode("test")
        assert u1 == u2

    def test_codec_distance(self):
        codec = SemanticCodec(backend="bitnet")
        u1 = codec.encode("financial invoice payment")
        u2 = codec.encode("technical server error")
        d = codec.distance(u1, u2)
        assert d > 0

    def test_composition_with_bitnet(self):
        from tsuuid.compose import compose_uuids
        codec = SemanticCodec(backend="bitnet")
        u1 = codec.encode("alpha")
        u2 = codec.encode("beta")
        composed = compose_uuids([u1, u2])
        assert len(composed) == N_DIMS
        assert composed.dtype == np.int8
