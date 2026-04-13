"""
tsuuid.sync_bundle — Cross-device row sync bundle format.

Each bundle file is JSON Lines (one row per line). Each row carries the
full metadata needed to upsert a vector on the receiving side:

    {"path": "...", "title": "...", "domain": "...",
     "vec_b64": "<base64 of float32 768 bytes>", "version": 1,
     "encoded_at": "2026-04-12T12:00:00Z"}

Why JSONL (not the raw SparseDelta bundle from tsuuid.bundle):
  - SparseDelta wire format has no path field; it's designed for intra-document
    version diffing with path implicit from context.
  - Cross-device sync routes new rows by path, so path must travel in-band.
  - JSONL is trivially cross-language (Swift Codable, Python stdlib) and
    remains debuggable with `cat`/`jq`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional


# Wire schema version. Bumped when SyncRow field layout changes in a way
# that older readers cannot safely ignore. Readers MUST refuse rows with
# schema_version > SYNC_SCHEMA_VERSION (forward-compat drift guard).
SYNC_SCHEMA_VERSION = 1


@dataclass
class SyncRow:
    path: str
    title: str
    domain: str
    vec_b64: str
    version: int = 1
    encoded_at: Optional[str] = None
    schema_version: int = SYNC_SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "SyncRow":
        obj = json.loads(line)
        return cls(
            path=obj["path"],
            title=obj.get("title", ""),
            domain=obj.get("domain", "general"),
            vec_b64=obj["vec_b64"],
            version=int(obj.get("version", 1)),
            encoded_at=obj.get("encoded_at"),
            schema_version=int(obj.get("schema_version", 1)),
        )

    def vec_bytes(self) -> bytes:
        return base64.b64decode(self.vec_b64)


def write_sync_bundle(rows: Iterable[SyncRow]) -> str:
    """Serialize rows as JSONL. One row per line, UTF-8 safe."""
    return "\n".join(r.to_json() for r in rows) + "\n"


def read_sync_bundle(text: str) -> List[SyncRow]:
    """Parse JSONL. Skips blank lines. Raises on malformed JSON."""
    rows: List[SyncRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(SyncRow.from_json(line))
    return rows
