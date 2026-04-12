"""
tsuuid.bundle — Delta bundle wire format.

Matches Swift TSUUIDKit SyncEngine.flush / pullIncoming byte layout exactly:

    count (uint32 LE)
    N × (length (uint32 LE) + SparseDelta bytes)

Bundles group multiple SparseDelta payloads into a single file written to
``__768_sync/deltas/{peer}/{YYYYMMDDHHMM}.delta`` for Dropbox-mediated sync.
"""

from __future__ import annotations

import struct
from typing import Iterable, List

from .delta import SparseDelta


def write_bundle(deltas: Iterable[SparseDelta]) -> bytes:
    """Serialize a sequence of SparseDelta to the Swift-compatible bundle format."""
    deltas = list(deltas)
    out = bytearray()
    out += struct.pack("<I", len(deltas))
    for d in deltas:
        body = d.to_bytes()
        out += struct.pack("<I", len(body))
        out += body
    return bytes(out)


def read_bundle(data: bytes) -> List[SparseDelta]:
    """Parse a bundle produced by write_bundle or Swift SyncEngine.flush."""
    if len(data) < 4:
        return []
    (count,) = struct.unpack_from("<I", data, 0)
    offset = 4
    deltas: List[SparseDelta] = []
    for _ in range(count):
        if offset + 4 > len(data):
            break
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if offset + length > len(data):
            break
        deltas.append(SparseDelta.from_bytes(data[offset : offset + length]))
        offset += length
    return deltas
