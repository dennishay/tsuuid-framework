"""Tests for tsuuid.bundle — delta bundle wire format parity with Swift."""

import struct
import sys

sys.path.insert(0, "src")

import numpy as np

from tsuuid.bundle import read_bundle, write_bundle
from tsuuid.delta import SparseDelta


def _make_delta(indices, values, version=1, checkpoint=False):
    return SparseDelta(
        indices=np.array(indices, dtype=np.uint16),
        values=np.array(values, dtype=np.float16),
        version=version,
        is_checkpoint=checkpoint,
    )


def test_empty_bundle():
    data = write_bundle([])
    assert data == struct.pack("<I", 0)
    assert read_bundle(data) == []


def test_single_delta_roundtrip():
    d = _make_delta([0, 10, 500], [0.5, -0.3, 0.1], version=7)
    data = write_bundle([d])

    # Header: count=1
    assert struct.unpack_from("<I", data, 0)[0] == 1
    # Length of first body
    body_len = struct.unpack_from("<I", data, 4)[0]
    assert body_len == d.wire_size
    # Offset 8 is start of delta body
    assert data[8:10] == b"TD"

    rebuilt = read_bundle(data)
    assert len(rebuilt) == 1
    assert rebuilt[0].version == 7
    np.testing.assert_array_equal(rebuilt[0].indices, d.indices)
    np.testing.assert_array_equal(rebuilt[0].values, d.values)


def test_multiple_deltas_preserve_order():
    d1 = _make_delta([1], [0.1], version=1)
    d2 = _make_delta([2, 3], [0.2, 0.3], version=2)
    d3 = _make_delta(list(range(768)), [0.0] * 768, version=3, checkpoint=True)

    data = write_bundle([d1, d2, d3])
    rebuilt = read_bundle(data)

    assert len(rebuilt) == 3
    assert rebuilt[0].version == 1
    assert rebuilt[1].version == 2
    assert rebuilt[2].version == 3
    assert rebuilt[2].is_checkpoint


def test_malformed_truncated_bundle_returns_valid_prefix():
    d1 = _make_delta([1], [0.1], version=1)
    d2 = _make_delta([2], [0.2], version=2)
    data = write_bundle([d1, d2])
    truncated = data[:-2]

    rebuilt = read_bundle(truncated)
    # First delta survives; second is dropped silently (defensive parser).
    assert len(rebuilt) == 1
    assert rebuilt[0].version == 1


def test_bundle_is_concatenation_of_length_prefixed_deltas():
    """Byte-level equivalence check — sanity against Swift's SyncEngine.flush."""
    d1 = _make_delta([5, 15], [0.25, -0.75], version=11)
    d2 = _make_delta([7], [0.5], version=12)

    bundle = write_bundle([d1, d2])
    b1 = d1.to_bytes()
    b2 = d2.to_bytes()

    expected = (
        struct.pack("<I", 2)
        + struct.pack("<I", len(b1)) + b1
        + struct.pack("<I", len(b2)) + b2
    )
    assert bundle == expected
