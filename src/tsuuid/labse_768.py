#!/usr/bin/env python3
"""
tsuuid.labse_768 — The Encoder. Document → LaBSE → 768 floats.

That IS the meaning. 768 float values from LaBSE = the document's
true semantic position in universal multilingual space (109 languages).

The 81 trits and UUIDs are compressed FORMATS of these 768 values —
like JPEG is a compressed format of raw pixels. Same picture, smaller file.

Encode → Store → Search → Compare. Everything else is packaging.

Usage:
    from tsuuid.labse_768 import Encoder768

    enc = Encoder768()

    # Encode
    vec = enc.encode("Invoice from Acme for boiler repair")

    # Store
    enc.store("path/to/doc.md", "Acme Invoice", vec)

    # Search
    results = enc.search("boiler tube maintenance")

    # Compare
    sim = enc.similarity(vec_a, vec_b)

CLI:
    python3 -m tsuuid.labse_768 encode "document text here"
    python3 -m tsuuid.labse_768 encode-file /path/to/document.md
    python3 -m tsuuid.labse_768 search "query text"
    python3 -m tsuuid.labse_768 compare "text A" "text B"
    python3 -m tsuuid.labse_768 stats
"""

import base64
import json
import os
import sqlite3
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

DB_PATH = os.path.expanduser("~/.claude/claude_home.db")


# ── Serialization ──────────────────────────────────────

def vec_to_bytes(vec: np.ndarray) -> bytes:
    """768 float32 → 3072 bytes. Lossless."""
    return vec.astype(np.float32).tobytes()


def bytes_to_vec(data: bytes) -> np.ndarray:
    """3072 bytes → 768 float32. Lossless."""
    return np.frombuffer(data, dtype=np.float32).copy()


def vec_to_f16_bytes(vec: np.ndarray) -> bytes:
    """768 float32 → 1536 bytes (float16). Preserves all rankings."""
    return vec.astype(np.float16).tobytes()


def f16_bytes_to_vec(data: bytes) -> np.ndarray:
    """1536 bytes → 768 float32 (from float16)."""
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()


def vec_to_b64(vec: np.ndarray) -> str:
    """768 float16 → base64 string. 2048 chars, transmittable anywhere."""
    return base64.b64encode(vec_to_f16_bytes(vec)).decode("ascii")


def b64_to_vec(b64: str) -> np.ndarray:
    """Base64 string → 768 float32."""
    return f16_bytes_to_vec(base64.b64decode(b64))


# ── Encoder ────────────────────────────────────────────

class Encoder768:
    """LaBSE 768-dim encoder. Document → meaning."""

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        device = self._device
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    def encode(self, text: str) -> np.ndarray:
        """Encode text → 768 float32 vector. This IS the meaning."""
        self._ensure_loaded()
        return self._model.encode(text, show_progress_bar=False)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts → (N, 768) array."""
        self._ensure_loaded()
        return self._model.encode(texts, show_progress_bar=False)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors. 1.0=identical, -1.0=opposite."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / max(norm, 1e-8))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance. 0=identical, 2=opposite."""
        return 1.0 - self.similarity(a, b)

    # ── Storage ────────────────────────────────────────

    def store(self, path: str, title: str, vec: np.ndarray, domain: str = "general"):
        """Store encoding in knowledge base."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tsuuid_768 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                title TEXT,
                vec BLOB,
                vec_b64 TEXT,
                domain TEXT,
                encoded_at TEXT,
                UNIQUE(path)
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO tsuuid_768 (path, title, vec, vec_b64, domain, encoded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (path, title, vec_to_f16_bytes(vec), vec_to_b64(vec), domain,
              time.strftime("%Y-%m-%dT%H:%M:%S")))
        conn.commit()
        conn.close()

    def encode_and_store(self, path: str, title: str, text: str, domain: str = "general") -> np.ndarray:
        """Encode + store in one call."""
        vec = self.encode(text)
        self.store(path, title, vec, domain)
        return vec

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search stored encodings by meaning. Returns (path, title, similarity)."""
        query_vec = self.encode(query)
        return self.search_vec(query_vec, limit)

    def search_vec(self, query_vec: np.ndarray, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search stored encodings by vector similarity."""
        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute("SELECT path, title, vec FROM tsuuid_768").fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

        results = []
        for path, title, vec_blob in rows:
            stored_vec = f16_bytes_to_vec(vec_blob)
            sim = self.similarity(query_vec, stored_vec)
            results.append((path, title, sim))

        results.sort(key=lambda x: -x[2])
        return results[:limit]

    def stats(self) -> dict:
        """Knowledge base statistics."""
        conn = sqlite3.connect(DB_PATH)
        try:
            count = conn.execute("SELECT COUNT(*) FROM tsuuid_768").fetchone()[0]
            domains = conn.execute("SELECT domain, COUNT(*) FROM tsuuid_768 GROUP BY domain").fetchall()
            recent = conn.execute("SELECT title, encoded_at FROM tsuuid_768 ORDER BY encoded_at DESC LIMIT 5").fetchall()
        except sqlite3.OperationalError:
            return {"count": 0, "domains": {}, "recent": []}
        finally:
            conn.close()

        return {
            "count": count,
            "domains": dict(domains),
            "recent": [(t, d) for t, d in recent],
            "bytes_per_doc": 1536,
            "format": "float16 (768 dims × 2 bytes)",
        }


# ── CLI ────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    enc = Encoder768()

    if cmd == "encode" and len(sys.argv) >= 3:
        text = " ".join(sys.argv[2:])
        vec = enc.encode(text)
        b64 = vec_to_b64(vec)
        print(f"Dims:   768")
        print(f"Bytes:  1536 (float16)")
        print(f"Base64: {len(b64)} chars")
        print(f"B64:    {b64[:80]}...")
        print(f"Norm:   {np.linalg.norm(vec):.4f}")
        top = np.argsort(np.abs(vec))[-5:][::-1]
        print(f"Top 5:  {', '.join(f'd{i}={vec[i]:+.4f}' for i in top)}")

    elif cmd == "encode-file" and len(sys.argv) >= 3:
        path = sys.argv[2]
        with open(path) as f:
            text = f.read()
        title = os.path.basename(path)
        vec = enc.encode_and_store(path, title, text)
        print(f"Encoded: {title}")
        print(f"Stored:  {path}")
        print(f"B64:     {vec_to_b64(vec)[:60]}...")

    elif cmd == "search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        results = enc.search(query)
        if not results:
            print("No documents in knowledge base. Use encode-file to add some.")
        for path, title, sim in results:
            print(f"  {sim:.3f}  {title:<40}  {path}")

    elif cmd == "compare" and len(sys.argv) >= 4:
        a = enc.encode(sys.argv[2])
        b = enc.encode(sys.argv[3])
        sim = enc.similarity(a, b)
        print(f"Similarity: {sim:.4f}")
        print(f"Distance:   {1-sim:.4f}")

    elif cmd == "stats":
        s = enc.stats()
        print(f"Documents: {s['count']}")
        print(f"Format:    {s['format']}")
        if s['domains']:
            for d, c in s['domains'].items():
                print(f"  {d}: {c}")
        if s['recent']:
            print("Recent:")
            for t, d in s['recent']:
                print(f"  {d}  {t}")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
