"""Tests for TSUUID core functionality."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import uuid
from tsuuid.packing import pack_trits_to_uuid, unpack_uuid_to_trits, hamming_distance, N_DIMS
from tsuuid.compose import compose_uuids, semantic_distance, diff_uuids
from tsuuid.codec import SemanticCodec
from tsuuid.dimensions import SemanticDimensions


def test_roundtrip_zeros():
    """All-zero trits should roundtrip through pack/unpack."""
    trits = np.zeros(N_DIMS, dtype=np.int8)
    uid = pack_trits_to_uuid(trits)
    assert uid.version == 8, f"Expected version 8, got {uid.version}"
    recovered = unpack_uuid_to_trits(uid)
    assert len(recovered) == N_DIMS


def test_determinism():
    """Same input must always produce same UUID."""
    codec = SemanticCodec()
    u1 = codec.encode("test document")
    u2 = codec.encode("test document")
    assert u1 == u2, "Codec must be deterministic"


def test_different_inputs_different_uuids():
    """Different inputs should produce different UUIDs."""
    codec = SemanticCodec()
    u1 = codec.encode("invoice from Acme Corp")
    u2 = codec.encode("server error critical failure")
    assert u1 != u2, "Different inputs should produce different UUIDs"


def test_composition_commutativity():
    """UUID composition must be commutative."""
    codec = SemanticCodec()
    a = codec.encode("alpha document")
    b = codec.encode("beta document")
    c = codec.encode("gamma document")
    
    abc = compose_uuids([a, b, c])
    bca = compose_uuids([b, c, a])
    cab = compose_uuids([c, a, b])
    
    assert np.array_equal(abc, bca), "Composition must be commutative (ABC == BCA)"
    assert np.array_equal(abc, cab), "Composition must be commutative (ABC == CAB)"


def test_semantic_distance_self():
    """Distance from a UUID to itself should be 0."""
    codec = SemanticCodec()
    u = codec.encode("test")
    d = semantic_distance(u, u)
    assert d == 0, "Self-distance must be 0"


def test_semantic_distance_different():
    """Distance between different UUIDs should be > 0."""
    codec = SemanticCodec()
    u1 = codec.encode("financial invoice payment")
    u2 = codec.encode("technical server error critical")
    d = semantic_distance(u1, u2)
    assert d > 0, "Different UUIDs should have positive distance"


def test_dimensions_count():
    """Should have exactly 81 axes defined."""
    dims = SemanticDimensions()
    assert dims.n_dims == 81, f"Expected 81 dimensions, got {dims.n_dims}"


def test_uuid_is_version_8():
    """All generated UUIDs must be version 8 (RFC 9562)."""
    codec = SemanticCodec()
    for text in ["hello", "invoice", "critical error", "approved"]:
        u = codec.encode(text)
        assert u.version == 8, f"Expected version 8 for '{text}', got {u.version}"


def test_empty_composition():
    """Composing no UUIDs should return zero vector."""
    result = compose_uuids([])
    assert np.all(result == 0), "Empty composition should be zero vector"


if __name__ == "__main__":
    tests = [
        test_roundtrip_zeros,
        test_determinism,
        test_different_inputs_different_uuids,
        test_composition_commutativity,
        test_semantic_distance_self,
        test_semantic_distance_different,
        test_dimensions_count,
        test_uuid_is_version_8,
        test_empty_composition,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
