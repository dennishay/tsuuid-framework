"""
tsuuid.chain — Self-Assembling Semantic Knowledge Chains

Knowledge as DNA. Each ChainLink is a gene: 81 ti-pairs carrying meaning,
knowing its neighbors, self-assembling by semantic proximity.

Chains are doubly-linked sequences of TSUUIDs that:
  - Self-assemble from scattered links via semantic proximity
  - Validate coherence (does each link belong between its neighbors?)
  - Detect gaps (large semantic distance = missing knowledge)
  - Grow without fixed end (append, insert, extend)

No LLM in the chain logic. Composition, distance, and gap detection
are pure vector/trit operations using the existing TSUUID primitives.

Reference: Dennis Hay, 2026-04-11 — "TSUUID DNA chains" breakthrough
"""

from __future__ import annotations

import time
import uuid as uuid_lib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tsuuid.packing import (
    pack_trits_to_uuid,
    unpack_uuid_to_trits,
    trits_to_display,
    N_DIMS,
)
from tsuuid.compose import semantic_distance, diff_uuids


@dataclass
class ChainLink:
    """A single gene in a knowledge chain.

    The TSUUID (81 trits packed into UUID v8) IS the meaning.
    The vec_768 is the full-resolution source vector (optional).
    The source is a pointer to the original document — never moves.
    """

    uuid: uuid_lib.UUID  # This link's TSUUID (81 trits packed)
    trits: np.ndarray  # 81 ternary values {-1, 0, +1}
    vec_768: Optional[np.ndarray] = None  # Full 768 vector (if available)
    prev_uuid: Optional[uuid_lib.UUID] = None  # Previous link
    next_uuid: Optional[uuid_lib.UUID] = None  # Next link
    chain_id: str = ""  # Which knowledge chain
    part: int = 0  # Which part (1-indexed)
    total: Optional[int] = None  # Total parts (None = growing)
    source: str = ""  # Original document pointer (path or URL)
    domain: str = ""  # Semantic domain (tssa, ieso, turbine, etc.)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @classmethod
    def from_text(cls, text: str, encoder, source: str = "", domain: str = "",
                  chain_id: str = "", part: int = 0) -> "ChainLink":
        """Create a ChainLink from raw text using LaBSE + trit projection.

        Args:
            text: Raw text to encode
            encoder: A LaBSEEncoder instance (from labse_backend)
            source: Pointer to original document
            domain: Semantic domain tag
            chain_id: Chain this link belongs to
            part: Position in chain (1-indexed)
        """
        trits = encoder.encode(text)
        tsuuid = pack_trits_to_uuid(trits)

        # Also get the full 768 vector if the encoder exposes it
        vec_768 = None
        if hasattr(encoder, "_model") and encoder._model is not None:
            vec_768 = encoder._model.encode(text, show_progress_bar=False)

        return cls(
            uuid=tsuuid,
            trits=trits,
            vec_768=vec_768,
            chain_id=chain_id,
            part=part,
            source=source,
            domain=domain,
        )

    @classmethod
    def from_vec_768(cls, vec: np.ndarray, encoder, source: str = "",
                     domain: str = "", chain_id: str = "", part: int = 0) -> "ChainLink":
        """Create a ChainLink from an existing 768-dim vector.

        Skips the LaBSE encoding step — projects directly to 81 trits.
        Use when the 768 vector is already available (from the graph).
        """
        encoder._ensure_loaded()
        projections = encoder._projection_matrix @ vec
        trits = encoder._quantize_absmean(projections)
        tsuuid = pack_trits_to_uuid(trits)

        return cls(
            uuid=tsuuid,
            trits=trits,
            vec_768=vec,
            chain_id=chain_id,
            part=part,
            source=source,
            domain=domain,
        )

    @classmethod
    def from_uuid(cls, tsuuid: uuid_lib.UUID, **kwargs) -> "ChainLink":
        """Create a ChainLink from an existing TSUUID.

        The trits are unpacked from the UUID. No 768 vector available
        unless provided separately.
        """
        trits = unpack_uuid_to_trits(tsuuid)
        return cls(uuid=tsuuid, trits=trits, **kwargs)

    def display(self) -> str:
        """Human-readable trit pattern: +·-+·+·--..."""
        return trits_to_display(self.trits)

    def distance_to(self, other: "ChainLink", metric: str = "cosine") -> float:
        """Semantic distance to another link."""
        return semantic_distance(self.uuid, other.uuid, metric=metric)

    def diff_to(self, other: "ChainLink") -> np.ndarray:
        """What changed between this link and another."""
        return diff_uuids(self.uuid, other.uuid)


@dataclass
class GapReport:
    """A detected gap in a knowledge chain."""

    position: int  # Between link[position-1] and link[position]
    before_uuid: uuid_lib.UUID
    after_uuid: uuid_lib.UUID
    distance: float  # Semantic distance across the gap
    expected_distance: float  # Median distance in this chain
    severity: float  # How many standard deviations above median
    inferred_trits: np.ndarray  # Midpoint trits (what should go here)


class Chain:
    """Self-assembling semantic knowledge chain.

    A doubly-linked sequence of ChainLinks that can validate its own
    coherence, detect gaps, and grow. The chain IS the knowledge —
    each link carries meaning in its 81 trits.
    """

    def __init__(self, chain_id: str = "", domain: str = ""):
        self.chain_id = chain_id or str(uuid_lib.uuid4())[:8]
        self.domain = domain
        self.links: list[ChainLink] = []

    def __len__(self) -> int:
        return len(self.links)

    def __getitem__(self, idx: int) -> ChainLink:
        return self.links[idx]

    def __iter__(self):
        return iter(self.links)

    def append(self, link: ChainLink) -> None:
        """Add a link to the end of the chain. Updates prev/next pointers."""
        link.chain_id = self.chain_id
        link.domain = self.domain or link.domain
        link.part = len(self.links) + 1

        if self.links:
            last = self.links[-1]
            last.next_uuid = link.uuid
            link.prev_uuid = last.uuid

        self.links.append(link)

    def insert(self, position: int, link: ChainLink) -> None:
        """Insert a link at a position. Updates all prev/next pointers."""
        link.chain_id = self.chain_id
        link.domain = self.domain or link.domain

        self.links.insert(position, link)
        self._rebuild_pointers()

    def _rebuild_pointers(self) -> None:
        """Rebuild all prev/next pointers and part numbers."""
        for i, link in enumerate(self.links):
            link.part = i + 1
            link.total = len(self.links)
            link.prev_uuid = self.links[i - 1].uuid if i > 0 else None
            link.next_uuid = self.links[i + 1].uuid if i < len(self.links) - 1 else None

    def distances(self, metric: str = "cosine") -> list[float]:
        """Compute semantic distance between each adjacent pair.

        Returns list of N-1 distances for N links.
        """
        if len(self.links) < 2:
            return []
        return [
            self.links[i].distance_to(self.links[i + 1], metric=metric)
            for i in range(len(self.links) - 1)
        ]

    def validate_coherence(self, metric: str = "cosine") -> list[tuple[int, float, str]]:
        """Check semantic coherence between adjacent links.

        Returns list of (position, distance, status) where status is
        'ok', 'weak', or 'break' based on deviation from chain median.
        """
        dists = self.distances(metric)
        if not dists:
            return []

        median = float(np.median(dists))
        std = float(np.std(dists)) if len(dists) > 2 else median * 0.3

        results = []
        for i, d in enumerate(dists):
            if std < 1e-6:
                status = "ok"
            elif d > median + 2 * std:
                status = "break"
            elif d > median + std:
                status = "weak"
            else:
                status = "ok"
            results.append((i + 1, d, status))

        return results

    def detect_gaps(self, metric: str = "cosine") -> list[GapReport]:
        """Find positions where semantic distance suggests missing links.

        A gap is where the distance between adjacent links is significantly
        larger than the chain's typical distance — something is missing.
        """
        dists = self.distances(metric)
        if len(dists) < 3:
            return []

        median = float(np.median(dists))
        std = float(np.std(dists))
        threshold = median + 1.5 * std

        gaps = []
        for i, d in enumerate(dists):
            if d > threshold and std > 1e-6:
                # Infer what should go in the gap: midpoint of neighbor trits
                before_trits = self.links[i].trits.astype(np.float64)
                after_trits = self.links[i + 1].trits.astype(np.float64)
                midpoint = np.sign(before_trits + after_trits).astype(np.int8)

                gaps.append(GapReport(
                    position=i + 1,
                    before_uuid=self.links[i].uuid,
                    after_uuid=self.links[i + 1].uuid,
                    distance=d,
                    expected_distance=median,
                    severity=(d - median) / std if std > 0 else 0,
                    inferred_trits=midpoint,
                ))

        return gaps

    def to_tsuuids(self) -> list[uuid_lib.UUID]:
        """Export chain as ordered list of TSUUIDs."""
        return [link.uuid for link in self.links]

    def to_dict(self) -> dict:
        """Serialize chain to dict for persistence."""
        return {
            "chain_id": self.chain_id,
            "domain": self.domain,
            "length": len(self.links),
            "links": [
                {
                    "uuid": str(link.uuid),
                    "trits": trits_to_display(link.trits),
                    "prev": str(link.prev_uuid) if link.prev_uuid else None,
                    "next": str(link.next_uuid) if link.next_uuid else None,
                    "part": link.part,
                    "source": link.source,
                    "domain": link.domain,
                    "created_at": link.created_at,
                }
                for link in self.links
            ],
        }

    @classmethod
    def from_links(cls, links: list[ChainLink], chain_id: str = "",
                   domain: str = "") -> "Chain":
        """Build a chain from an existing list of links."""
        chain = cls(chain_id=chain_id, domain=domain)
        for link in links:
            chain.append(link)
        return chain

    @classmethod
    def discover(cls, seed_uuid: uuid_lib.UUID, all_links: list[ChainLink],
                 max_distance: float = 0.8, domain: str = "") -> "Chain":
        """Self-assemble a chain from scattered links.

        Given one seed TSUUID and a pool of unlinked ChainLinks,
        discover which links belong to this chain by semantic proximity.
        Sorts them into coherent order by minimizing total chain distance.

        Args:
            seed_uuid: Starting TSUUID
            all_links: Pool of candidate links
            max_distance: Maximum cosine distance to include a link
            domain: Filter candidates by domain (empty = all)

        Returns:
            A Chain assembled from the most semantically coherent subset.
        """
        seed_trits = unpack_uuid_to_trits(seed_uuid)

        # Filter candidates by domain if specified
        candidates = all_links
        if domain:
            candidates = [l for l in candidates if l.domain == domain or not l.domain]

        # Find links within max_distance of the seed
        nearby = []
        for link in candidates:
            d = semantic_distance(seed_uuid, link.uuid, metric="cosine")
            if d <= max_distance:
                nearby.append((d, link))

        if not nearby:
            # Just the seed
            seed_link = ChainLink.from_uuid(seed_uuid, domain=domain)
            return cls.from_links([seed_link], domain=domain)

        # Sort by distance from seed (greedy nearest-neighbor ordering)
        nearby.sort(key=lambda x: x[0])
        ordered = [link for _, link in nearby]

        # Build chain with greedy nearest-neighbor walk for better ordering
        if len(ordered) > 2:
            ordered = _nearest_neighbor_order(ordered)

        return cls.from_links(ordered, domain=domain)

    def summary(self) -> str:
        """One-line summary of the chain."""
        gaps = self.detect_gaps()
        coherence = self.validate_coherence()
        breaks = sum(1 for _, _, s in coherence if s == "break")
        return (
            f"Chain[{self.chain_id}] domain={self.domain} "
            f"links={len(self.links)} gaps={len(gaps)} breaks={breaks}"
        )


def _nearest_neighbor_order(links: list[ChainLink]) -> list[ChainLink]:
    """Order links by greedy nearest-neighbor walk.

    Starts from the first link, repeatedly picks the nearest unvisited
    link. This produces a semantically smooth ordering — DNA-like.
    """
    if len(links) <= 2:
        return links

    remaining = list(range(1, len(links)))
    order = [0]

    while remaining:
        current = links[order[-1]]
        best_idx = None
        best_dist = float("inf")

        for idx in remaining:
            d = current.distance_to(links[idx])
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx is not None:
            order.append(best_idx)
            remaining.remove(best_idx)

    return [links[i] for i in order]
