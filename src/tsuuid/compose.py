"""
tsuuid.compose — UUID Composition by Ternary Addition

Combines multiple semantic UUIDs to produce emergent knowledge.
Composition is commutative and associative (order doesn't matter).

This is the Lego principle: independent pieces snap together
because they share the same coordinate system.

Reference: TSUUID Framework paper, Section 6.1 (Hay, 2026)
"""

import uuid
from typing import List
import numpy as np

from tsuuid.packing import (
    pack_trits_to_uuid,
    unpack_uuid_to_trits,
    hamming_distance,
    l1_distance,
    N_DIMS,
)


def compose_uuids(uuids: List[uuid.UUID]) -> np.ndarray:
    """Compose multiple UUIDs by ternary vector addition.
    
    The result is the sum of all trit vectors, clamped to {-1, 0, +1}.
    This represents the combined semantic displacement — the emergent
    meaning that arises from combining independent knowledge.
    
    Args:
        uuids: List of semantic UUIDs to compose
        
    Returns:
        Composed trit vector (81 dimensions)
        
    Note:
        Composition is commutative: compose([A,B]) == compose([B,A])
        Composition is associative: compose([A,B,C]) order-independent
    """
    if not uuids:
        return np.zeros(N_DIMS, dtype=np.int8)
    
    # Sum all trit vectors
    total = np.zeros(N_DIMS, dtype=np.float64)
    for u in uuids:
        trits = unpack_uuid_to_trits(u)
        total += trits.astype(np.float64)
    
    # Normalize: take sign of sum (ternary majority vote per dimension)
    composed = np.sign(total).astype(np.int8)
    
    return composed


def semantic_distance(a: uuid.UUID, b: uuid.UUID, metric: str = "hamming") -> float:
    """Compute semantic distance between two UUIDs.
    
    Nearby UUIDs in the 81-dimensional space encode nearby meanings.
    
    Args:
        a, b: Semantic UUIDs to compare
        metric: "hamming" (dimensions that differ) or "l1" (Manhattan distance)
        
    Returns:
        Distance value (0 = identical semantics)
    """
    trits_a = unpack_uuid_to_trits(a)
    trits_b = unpack_uuid_to_trits(b)
    
    if metric == "hamming":
        return hamming_distance(trits_a, trits_b)
    elif metric == "l1":
        return l1_distance(trits_a, trits_b)
    elif metric == "cosine":
        dot = np.dot(trits_a.astype(float), trits_b.astype(float))
        norm_a = np.linalg.norm(trits_a.astype(float))
        norm_b = np.linalg.norm(trits_b.astype(float))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def shared_dimensions(a: uuid.UUID, b: uuid.UUID) -> List[int]:
    """Find dimensions where two UUIDs have the same non-zero displacement.
    
    These are the semantic axes they share — the basis for cross-schema
    alignment without explicit mapping.
    
    Returns:
        List of 1-indexed dimension numbers with matching non-zero trits
    """
    trits_a = unpack_uuid_to_trits(a)
    trits_b = unpack_uuid_to_trits(b)
    
    shared = []
    for i in range(N_DIMS):
        if trits_a[i] != 0 and trits_a[i] == trits_b[i]:
            shared.append(i + 1)  # 1-indexed
    
    return shared


def diff_uuids(a: uuid.UUID, b: uuid.UUID) -> np.ndarray:
    """Compute the ternary difference between two UUIDs.
    
    This is the "version diff" — what changed between two states.
    diff(v1, v2) tells you exactly which semantic dimensions changed
    and in which direction.
    
    Args:
        a: Source UUID (e.g., document v1)
        b: Target UUID (e.g., document v2)
        
    Returns:
        Ternary diff vector (clamped to {-1, 0, +1})
    """
    trits_a = unpack_uuid_to_trits(a)
    trits_b = unpack_uuid_to_trits(b)
    
    diff = trits_b.astype(np.int16) - trits_a.astype(np.int16)
    return np.clip(diff, -1, 1).astype(np.int8)
