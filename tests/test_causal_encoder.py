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
        assert info["model_name"] == "sshleifer/tiny-gpt2"

    def test_not_all_zeros(self, encoder):
        trits = encoder.encode("important document requiring review")
        assert np.any(trits != 0)
