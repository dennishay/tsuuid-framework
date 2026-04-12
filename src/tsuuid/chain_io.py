"""
tsuuid.chain_io — SQLite persistence for Chain topology

Translates Chain objects to/from the chain_links + knowledge_chains tables
defined by the TSUUID Hot-Swap migration (migrate_tsuuid_chains_home.sql).

Split from chain.py so the core data structures stay DB-agnostic and testable
without a SQLite connection. Chain.persist_to_sqlite / Chain.from_sqlite are
thin wrappers over these functions.

Storage format:
  trits    → 81 int8 bytes (.tobytes(), lossless)
  vec_768  → 1536 float16 bytes (vec_to_f16_bytes, matches other vector tables)
  uuids    → str(uuid.UUID), hex-dashed form
  NULL prev/next at chain endpoints
"""
from __future__ import annotations

import sqlite3
import time
import uuid as uuid_lib
from typing import List, Optional

import numpy as np

from tsuuid.chain import Chain, ChainLink
from tsuuid.packing import unpack_uuid_to_trits


def _trits_to_blob(trits: np.ndarray) -> bytes:
    """81 int8 trits → 81-byte BLOB. Lossless."""
    arr = np.asarray(trits, dtype=np.int8)
    if arr.shape != (81,):
        raise ValueError(f"trits must be shape (81,), got {arr.shape}")
    return arr.tobytes()


def _blob_to_trits(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(81, dtype=np.int8)
    return np.frombuffer(data, dtype=np.int8).copy()


def _vec_to_f16_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float16).tobytes()


def _f16_blob_to_vec(data: Optional[bytes]) -> Optional[np.ndarray]:
    if not data:
        return None
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()


# ── Persistence ───────────────────────────────────────────


def upsert_knowledge_chain(
    conn: sqlite3.Connection,
    chain_id: str,
    domain: str,
    link_count: int,
    coherence_score: Optional[float] = None,
) -> None:
    """Create or update the chain header row in knowledge_chains."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO knowledge_chains
            (chain_id, domain, created_at, last_modified, link_count, coherence_score)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(chain_id) DO UPDATE SET
            domain = excluded.domain,
            last_modified = excluded.last_modified,
            link_count = excluded.link_count,
            coherence_score = excluded.coherence_score
        """,
        (chain_id, domain, now, now, link_count, coherence_score),
    )


def persist_link(conn: sqlite3.Connection, link: ChainLink,
                 coherence_distance: Optional[float] = None,
                 source_db: Optional[str] = None,
                 source_table: Optional[str] = None,
                 source_id: Optional[str] = None) -> None:
    """Upsert a single ChainLink into chain_links."""
    conn.execute(
        """
        INSERT INTO chain_links
            (tsuuid, chain_id, position, trits, vec_768,
             source_db, source_table, source_id, source_path, domain,
             prev_uuid, next_uuid, created_at, coherence_distance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tsuuid) DO UPDATE SET
            chain_id = excluded.chain_id,
            position = excluded.position,
            trits = excluded.trits,
            vec_768 = excluded.vec_768,
            source_db = COALESCE(excluded.source_db, chain_links.source_db),
            source_table = COALESCE(excluded.source_table, chain_links.source_table),
            source_id = COALESCE(excluded.source_id, chain_links.source_id),
            source_path = COALESCE(excluded.source_path, chain_links.source_path),
            domain = excluded.domain,
            prev_uuid = excluded.prev_uuid,
            next_uuid = excluded.next_uuid,
            coherence_distance = excluded.coherence_distance
        """,
        (
            str(link.uuid),
            link.chain_id,
            link.part,
            _trits_to_blob(link.trits),
            _vec_to_f16_blob(link.vec_768),
            source_db,
            source_table,
            source_id,
            link.source or None,
            link.domain or None,
            str(link.prev_uuid) if link.prev_uuid else None,
            str(link.next_uuid) if link.next_uuid else None,
            link.created_at,
            coherence_distance,
        ),
    )


def persist_chain(conn: sqlite3.Connection, chain: Chain) -> int:
    """Persist a whole chain atomically. Returns number of links written.

    Computes pairwise coherence distances for each link (distance to prev)
    and records mean as the chain's coherence_score.
    """
    if not chain.links:
        upsert_knowledge_chain(conn, chain.chain_id, chain.domain, 0)
        return 0

    # Compute adjacent distances once — used for link-level
    # coherence_distance AND chain-level coherence_score.
    distances = chain.distances(metric="cosine")
    coherence_score = float(np.mean(distances)) if distances else None

    for i, link in enumerate(chain.links):
        dist = distances[i - 1] if i > 0 else None
        persist_link(conn, link, coherence_distance=dist)

    upsert_knowledge_chain(
        conn,
        chain.chain_id,
        chain.domain,
        link_count=len(chain.links),
        coherence_score=coherence_score,
    )
    return len(chain.links)


# ── Loading ───────────────────────────────────────────────


def _row_to_link(row: sqlite3.Row) -> ChainLink:
    """Reconstruct a ChainLink from a chain_links row."""
    trits = _blob_to_trits(row["trits"])
    vec = _f16_blob_to_vec(row["vec_768"])

    prev_u = uuid_lib.UUID(row["prev_uuid"]) if row["prev_uuid"] else None
    next_u = uuid_lib.UUID(row["next_uuid"]) if row["next_uuid"] else None

    return ChainLink(
        uuid=uuid_lib.UUID(row["tsuuid"]),
        trits=trits,
        vec_768=vec,
        prev_uuid=prev_u,
        next_uuid=next_u,
        chain_id=row["chain_id"] or "",
        part=row["position"] or 0,
        source=row["source_path"] or "",
        domain=row["domain"] or "",
        created_at=row["created_at"] or time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def load_chain(conn: sqlite3.Connection, chain_id: str) -> Optional[Chain]:
    """Load a Chain by chain_id. Returns None if not found."""
    conn.row_factory = sqlite3.Row
    header = conn.execute(
        "SELECT chain_id, domain FROM knowledge_chains WHERE chain_id = ?",
        (chain_id,),
    ).fetchone()
    if header is None:
        return None

    rows = conn.execute(
        """
        SELECT tsuuid, chain_id, position, trits, vec_768,
               source_db, source_table, source_id, source_path, domain,
               prev_uuid, next_uuid, created_at, coherence_distance
          FROM chain_links
         WHERE chain_id = ?
         ORDER BY position
        """,
        (chain_id,),
    ).fetchall()
    if not rows:
        return None

    links = [_row_to_link(r) for r in rows]
    chain = Chain(chain_id=header["chain_id"], domain=header["domain"] or "")
    chain.links = links  # preserve persisted order and pointers verbatim
    return chain


def load_all_chains(conn: sqlite3.Connection,
                    domains: Optional[List[str]] = None) -> List[Chain]:
    """Load every chain from the DB, optionally filtering by domain."""
    conn.row_factory = sqlite3.Row
    if domains:
        placeholders = ",".join("?" * len(domains))
        header_rows = conn.execute(
            f"SELECT chain_id FROM knowledge_chains WHERE domain IN ({placeholders})",
            domains,
        ).fetchall()
    else:
        header_rows = conn.execute("SELECT chain_id FROM knowledge_chains").fetchall()

    chains = []
    for h in header_rows:
        c = load_chain(conn, h["chain_id"])
        if c is not None:
            chains.append(c)
    return chains


# ── Chain class extension ─────────────────────────────────


def _persist_to_sqlite(self, conn: sqlite3.Connection) -> int:
    """Bound to Chain.persist_to_sqlite — see persist_chain."""
    return persist_chain(conn, self)


def _from_sqlite(cls, conn: sqlite3.Connection, chain_id: str) -> Optional[Chain]:
    """Bound to Chain.from_sqlite — see load_chain."""
    return load_chain(conn, chain_id)


# Monkey-patch so callers can use Chain.persist_to_sqlite / Chain.from_sqlite
# without importing chain_io directly. Keeps chain.py DB-free.
Chain.persist_to_sqlite = _persist_to_sqlite  # type: ignore[assignment]
Chain.from_sqlite = classmethod(_from_sqlite)  # type: ignore[assignment]
