"""
TSUUID — Ternary Semantic UUID Framework
Where Data IS Understanding

Encode semantic meaning as 81-dimensional ternary displacements 
within standard UUIDs (RFC 9562 v8).
"""

__version__ = "0.1.0"
__author__ = "Dennis Evan Hay"

from tsuuid.codec import SemanticCodec
from tsuuid.packing import pack_trits_to_uuid, unpack_uuid_to_trits
from tsuuid.dimensions import SemanticDimensions
from tsuuid.compose import compose_uuids, semantic_distance
from tsuuid.delta import DeltaEncoder, SparseDelta, compression_ratio, cosine_error

__all__ = [
    "SemanticCodec",
    "pack_trits_to_uuid",
    "unpack_uuid_to_trits",
    "SemanticDimensions",
    "compose_uuids",
    "semantic_distance",
    "DeltaEncoder",
    "SparseDelta",
    "compression_ratio",
    "cosine_error",
]
