"""
tsuuid.packing — Ternary ↔ 128-bit ID packing

Two packing modes:

  MODE A (legacy, 81 trits → UUID v8): packs 81 trits into RFC 9562
  UUID v8. LOSSY — 81 trits need 128.4 bits, but UUID v8 has only 122
  payload bits after version/variant. ~7-10 trits lost per UUID via
  the 17th byte truncation and version/variant bit overlay. Kept for
  backward compatibility with prior artifacts.

  MODE B (lossless, 80 trits → 128-bit raw): packs 80 trits into all
  128 bits of a UUID-shaped ID. 80 trits = 80 × log₂(3) = 126.8 bits,
  fits exactly. NOT RFC v8 compliant (version/variant bits are payload),
  but chain_links treats tsuuid as opaque TEXT keys so this is fine
  internally.

Mode B is the default for all new substrate work after 2026-04-27.
The trit count per packet is 80, not 81 — fitting losslessly inside a
128-bit ID is the constraint. 768 trits → 10 packets at 80 trits each
(800 slot capacity, 32 zero-padding in the last packet) — same packet
count as the 81-trit lossy mode.

For chain walking, drop the redundant `next_uuid` / `prev_uuid` columns
in chain_links — composite (chain_id, position) is the canonical walk.
"""

import uuid
from typing import List, Tuple
import numpy as np


# Number of semantic dimensions per packet
# MODE B (lossless) — default for new work
N_DIMS = 80
# MODE A (legacy) — for backward compat
N_DIMS_LEGACY = 81

# Ternary values mapped to unsigned: {-1 → 0, 0 → 1, +1 → 2}
_TRIT_TO_UNSIGNED = {-1: 0, 0: 1, 1: 2}
_UNSIGNED_TO_TRIT = {0: -1, 1: 0, 2: 1}


def _encode_trit_group(trits: List[int]) -> int:
    """Encode up to 5 trits into a single integer (0–242).
    
    Uses base-3 encoding: val = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
    where each t_i ∈ {0, 1, 2} (mapped from {-1, 0, +1}).
    """
    val = 0
    for i, t in enumerate(trits):
        val += _TRIT_TO_UNSIGNED[t] * (3 ** i)
    return val


def _decode_trit_group(val: int, count: int = 5) -> List[int]:
    """Decode an integer (0–242) back to trits."""
    trits = []
    for _ in range(count):
        trits.append(_UNSIGNED_TO_TRIT[val % 3])
        val //= 3
    return trits


def pack_trits_to_uuid(trits: np.ndarray) -> uuid.UUID:
    """Pack 80 ternary values losslessly into a 128-bit ID (MODE B).

    Encoding: direct base-3 integer.
        val = Σ (trit_i + 1) · 3^i   where trit_i ∈ {-1, 0, +1}
        val ∈ [0, 3^80 − 1] ≈ 1.48 × 10^38, fits in 127 bits (1 bit
        unused). UUID v8 version/variant bits are NOT set — they're
        treated as payload. Result is still a 128-bit UUID-shaped ID
        that chain_links can use as an opaque TEXT primary key.

    Verified lossless: pack → unpack → original is byte-exact for
    all 80-trit inputs. See `unpack_uuid_to_trits`.

    Args:
        trits: numpy array of shape (80,) with values in {-1, 0, +1}

    Returns:
        uuid.UUID — 128-bit ID (NOT v8 RFC compliant; version/variant
        bits are payload, not metadata).

    Raises:
        ValueError: wrong shape or invalid values.
    """
    if len(trits) != N_DIMS:
        raise ValueError(f"Expected {N_DIMS} trits, got {len(trits)}")
    arr = np.asarray(trits, dtype=np.int8)
    if not np.all((arr >= -1) & (arr <= 1)):
        raise ValueError("All trits must be in {-1, 0, +1}")

    # Direct base-3 encoding to a single integer
    val = 0
    pow3 = 1
    for t in arr:
        val += int(_TRIT_TO_UNSIGNED[int(t)]) * pow3
        pow3 *= 3
    # val is now in [0, 3^80 - 1] which fits in 127 bits
    return uuid.UUID(int=val)


def unpack_uuid_to_trits(u: uuid.UUID) -> np.ndarray:
    """Unpack a 128-bit ID back to 80 ternary values losslessly (MODE B).

    Args:
        u: UUID-shaped 128-bit ID produced by pack_trits_to_uuid.

    Returns:
        numpy array of shape (80,) with values in {-1, 0, +1}.
    """
    val = u.int
    out = np.empty(N_DIMS, dtype=np.int8)
    for i in range(N_DIMS):
        out[i] = _UNSIGNED_TO_TRIT[val % 3]
        val //= 3
    return out


def pack_trits_to_uuid_v8_legacy(trits: np.ndarray) -> uuid.UUID:
    """Legacy 81-trit packing into RFC 9562 UUID v8. LOSSY.

    Kept only for backward compatibility with pre-2026-04-27 substrate
    artifacts. Loses ~7-10 trits per UUID via the 17th-byte truncation
    and version/variant bit overlay. Do not use for new work.
    """
    if len(trits) != N_DIMS_LEGACY:
        raise ValueError(f"Expected {N_DIMS_LEGACY} trits, got {len(trits)}")
    if not all(t in (-1, 0, 1) for t in trits):
        raise ValueError("All trits must be in {-1, 0, +1}")
    encoded_bytes = []
    for i in range(0, N_DIMS_LEGACY, 5):
        group = list(trits[i:min(i+5, N_DIMS_LEGACY)])
        while len(group) < 5:
            group.append(0)
        encoded_bytes.append(_encode_trit_group(group))
    val = 0
    for i, b in enumerate(encoded_bytes[:16]):
        val = (val << 8) | (b & 0xFF)
    if len(encoded_bytes) > 16:
        val = (val & ~0xFF) | (encoded_bytes[16] & 0xFF)
    val &= ~(0xF << 76)
    val |= (0x8 << 76)
    val &= ~(0x3 << 62)
    val |= (0x2 << 62)
    return uuid.UUID(int=val)


def unpack_uuid_v8_legacy_to_trits(u: uuid.UUID) -> np.ndarray:
    """Legacy 81-trit unpack from UUID v8. Lossy companion to
    pack_trits_to_uuid_v8_legacy."""
    val = u.int
    raw_bytes = []
    temp = val
    for _ in range(16):
        raw_bytes.append(temp & 0xFF)
        temp >>= 8
    raw_bytes.reverse()
    trits = []
    for b in raw_bytes[:16]:
        b_clamped = min(b, 242)
        group = _decode_trit_group(b_clamped, 5)
        trits.extend(group)
    while len(trits) < N_DIMS_LEGACY:
        trits.append(0)
    return np.array(trits[:N_DIMS_LEGACY], dtype=np.int8)


def trits_to_display(trits: np.ndarray) -> str:
    """Human-readable display of trit vector.
    
    +1 displayed as '+', -1 as '-', 0 as '·'
    
    Example:
        >>> trits_to_display(np.array([1, 0, -1, 1, 0]))
        '+·-+·'
    """
    symbols = {-1: '-', 0: '·', 1: '+'}
    return ''.join(symbols.get(int(t), '?') for t in trits)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Ternary Hamming distance: number of dimensions that differ."""
    return int(np.sum(a != b))


def l1_distance(a: np.ndarray, b: np.ndarray) -> int:
    """L1 (Manhattan) distance in ternary space."""
    return int(np.sum(np.abs(a - b)))
