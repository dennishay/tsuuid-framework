"""
tsuuid.codec — Core Semantic Codec

Encodes documents/data as ternary displacements from the universal
reference frame, packed into UUID v8. Decodes back.

This is the central component of the TSUUID framework.
In the full implementation, encoding uses a BitNet 1.58 model to
map content → 81-dimensional semantic coordinates. This reference
implementation provides a rule-based encoder for demonstration
and testing, with the BitNet integration as a pluggable backend.

Reference: TSUUID Framework paper, Section 4 (Hay, 2026)
"""

import uuid
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from tsuuid.packing import pack_trits_to_uuid, unpack_uuid_to_trits, trits_to_display, N_DIMS
from tsuuid.dimensions import SemanticDimensions, ALL_AXES


@dataclass
class SemanticMeaning:
    """Decoded semantic content from a UUID."""
    uuid: uuid.UUID
    trits: np.ndarray
    active_dimensions: List[str]
    layer_summary: Dict[str, Any]
    trit_display: str
    
    def __repr__(self):
        n_active = sum(1 for t in self.trits if t != 0)
        return (
            f"SemanticMeaning(uuid={self.uuid}, "
            f"active_dims={n_active}/{N_DIMS}, "
            f"trits='{self.trit_display}')"
        )


class SemanticCodec:
    """Encode and decode semantic content as UUID v8.
    
    This reference implementation uses a deterministic hash-based encoder
    for demonstration. The production implementation would use a fine-tuned
    BitNet b1.58 model as the encoding backend.
    
    Usage:
        codec = SemanticCodec()
        
        # Encode
        uid = codec.encode("Invoice #4471 from Acme Corp, $2340, due April 15")
        
        # Decode
        meaning = codec.decode(uid)
        print(meaning.active_dimensions)
        
        # Compare
        uid2 = codec.encode("Purchase order #891 from Acme Corp, $1200")
        distance = codec.distance(uid, uid2)
    """
    
    def __init__(self, backend: str = "hash"):
        """Initialize codec.
        
        Args:
            backend: Encoding backend.
                "hash" — deterministic hash-based (reference implementation)
                "bitnet" — BitNet b1.58 model (requires bitnet dependency)
        """
        self.dims = SemanticDimensions()
        self.backend = backend
        self._bitnet_encoder = None  # Lazy initialization

        # Keyword → dimension mappings for the hash-based encoder
        # This is a simplified demonstration; the BitNet backend learns these
        self._keyword_map = self._build_keyword_map()
    
    def encode(self, content: str, metadata: Optional[Dict] = None) -> uuid.UUID:
        """Encode content as a semantic UUID.
        
        Args:
            content: Text content to encode
            metadata: Optional structured metadata to inform encoding
            
        Returns:
            UUID v8 encoding the 81-dimensional semantic displacement
        """
        if self.backend == "hash":
            trits = self._hash_encode(content, metadata)
        elif self.backend == "bitnet":
            if self._bitnet_encoder is None:
                from tsuuid.bitnet_backend import BitNetEncoder
                self._bitnet_encoder = BitNetEncoder()
            trits = self._bitnet_encoder.encode(content, metadata)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return pack_trits_to_uuid(trits)
    
    def decode(self, uid: uuid.UUID) -> SemanticMeaning:
        """Decode a semantic UUID back to its meaning.
        
        Args:
            uid: UUID v8 to decode
            
        Returns:
            SemanticMeaning with decoded dimensional information
        """
        trits = unpack_uuid_to_trits(uid)
        
        return SemanticMeaning(
            uuid=uid,
            trits=trits,
            active_dimensions=self.dims.describe(trits),
            layer_summary=self.dims.layer_summary(trits),
            trit_display=trits_to_display(trits),
        )
    
    def distance(self, a: uuid.UUID, b: uuid.UUID) -> float:
        """Semantic distance between two UUIDs (0 = identical)."""
        from tsuuid.compose import semantic_distance
        return semantic_distance(a, b, metric="hamming")
    
    def _hash_encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Reference encoder: deterministic hash-based trit generation.
        
        This is NOT the production encoder. It demonstrates the interface
        and produces deterministic, reproducible UUIDs for testing.
        The production encoder uses BitNet b1.58 for true semantic encoding.
        """
        trits = np.zeros(N_DIMS, dtype=np.int8)
        content_lower = content.lower()
        
        # Phase 1: Keyword-driven dimensional activation
        for keywords, dim_idx, value in self._keyword_map:
            if any(kw in content_lower for kw in keywords):
                trits[dim_idx] = value
        
        # Phase 2: Hash-driven activation for remaining dimensions
        # Uses SHA-256 to deterministically assign trits to unset dimensions
        h = hashlib.sha256(content.encode('utf-8')).digest()
        for i in range(N_DIMS):
            if trits[i] == 0:
                byte_val = h[i % len(h)]
                # Map byte to trit: 0-84 → -1, 85-169 → 0, 170-255 → +1
                if byte_val < 85:
                    trits[i] = -1
                elif byte_val < 170:
                    trits[i] = 0
                else:
                    trits[i] = 1
        
        return trits
    
    def _build_keyword_map(self):
        """Build keyword → (dimension, trit_value) mappings.
        
        This is the reference implementation's simple heuristic encoder.
        Production would replace this with BitNet inference.
        """
        return [
            # Temporality (dim 1)
            (["history", "historical", "previous", "past", "last year", "ago"], 0, -1),
            (["future", "forecast", "projected", "upcoming", "next"], 0, 1),
            
            # Modality (dim 2)
            (["delete", "remove", "cancel", "void"], 1, -1),
            (["create", "new", "add", "insert", "generate"], 1, 1),
            
            # Certainty (dim 3)
            (["denied", "rejected", "refused", "invalid", "false"], 2, -1),
            (["confirmed", "approved", "verified", "valid", "true"], 2, 1),
            
            # Obligation (dim 5)
            (["forbidden", "prohibited", "must not", "cannot"], 4, -1),
            (["required", "mandatory", "must", "shall"], 4, 1),
            
            # Visibility (dim 6)
            (["private", "confidential", "secret", "internal"], 5, -1),
            (["public", "open", "published", "shared"], 5, 1),
            
            # Directionality (dim 8)
            (["received", "incoming", "inbound", "import"], 7, -1),
            (["sent", "outgoing", "outbound", "export"], 7, 1),
            
            # Ownership (dim 21)
            (["external", "vendor", "supplier", "client", "customer"], 20, -1),
            (["internal", "our", "company", "organization"], 20, 1),
            
            # Urgency (dim 26)
            (["deferred", "low priority", "backlog", "someday"], 25, -1),
            (["critical", "urgent", "emergency", "asap", "immediate"], 25, 1),
            
            # Domain (dim 36)
            (["technical", "engineering", "code", "system", "server"], 35, -1),
            (["business", "financial", "invoice", "payment", "revenue"], 35, 1),
            
            # Flow direction (dim 38)
            (["expense", "cost", "payment", "outflow", "debit"], 37, -1),
            (["income", "revenue", "receipt", "inflow", "credit"], 37, 1),
            
            # Value sign (dim 39)
            (["loss", "deficit", "negative", "decrease", "reduction"], 38, -1),
            (["profit", "surplus", "positive", "increase", "growth"], 38, 1),
            
            # Scale (dim 40)
            (["small", "minor", "tiny", "negligible"], 39, -1),
            (["large", "major", "significant", "massive", "enterprise"], 39, 1),
            
            # Status (dim 42)
            (["failed", "error", "broken", "incomplete"], 41, -1),
            (["complete", "done", "finished", "resolved", "closed"], 41, 1),
            
            # Time horizon (dim 51)
            (["historical", "archive", "legacy", "old"], 50, -1),
            (["forecast", "prediction", "estimate", "projection"], 50, 1),
            
            # Trend (dim 74)
            (["decreasing", "declining", "falling", "dropping"], 73, -1),
            (["increasing", "rising", "growing", "climbing"], 73, 1),
        ]
