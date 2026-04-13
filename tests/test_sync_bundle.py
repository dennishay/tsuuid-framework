"""Tests for tsuuid.sync_bundle — JSONL row sync format."""

import base64
import json
import sys

sys.path.insert(0, "src")

from tsuuid.sync_bundle import SyncRow, read_sync_bundle, write_sync_bundle


def _row(path="p/1", title="t", domain="d", version=1):
    # Vector768 wire format is 768 × float16 LE = 1536 bytes (Vector768.byteCount)
    vec = bytes(range(256)) * 6  # 1536 bytes
    return SyncRow(
        path=path,
        title=title,
        domain=domain,
        vec_b64=base64.b64encode(vec).decode("ascii"),
        version=version,
        encoded_at="2026-04-12T12:00:00Z",
    )


def test_single_row_roundtrip():
    r = _row()
    text = write_sync_bundle([r])
    rebuilt = read_sync_bundle(text)
    assert len(rebuilt) == 1
    assert rebuilt[0] == r


def test_multiple_rows_preserve_order():
    rows = [_row(path=f"p/{i}", title=f"t{i}") for i in range(5)]
    text = write_sync_bundle(rows)
    rebuilt = read_sync_bundle(text)
    assert rebuilt == rows


def test_empty_bundle():
    assert read_sync_bundle("") == []
    assert read_sync_bundle("\n\n  \n") == []


def test_jsonl_shape():
    rows = [_row(path="a"), _row(path="b")]
    text = write_sync_bundle(rows)
    lines = [l for l in text.split("\n") if l]
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert set(obj.keys()) == {"path", "title", "domain", "vec_b64", "version",
                                    "encoded_at", "schema_version"}
        assert obj["schema_version"] == 1


def test_schema_version_default_on_missing_field():
    # Backward-compat: old bundles without schema_version default to 1.
    legacy = ('{"path":"p","title":"t","domain":"d",'
              '"vec_b64":"' + base64.b64encode(b"\x00" * 1536).decode("ascii") + '",'
              '"version":1,"encoded_at":"2026-04-12T12:00:00Z"}')
    rows = read_sync_bundle(legacy)
    assert len(rows) == 1
    assert rows[0].schema_version == 1


def test_vec_bytes_roundtrip():
    r = _row()
    recovered = read_sync_bundle(write_sync_bundle([r]))[0]
    assert len(recovered.vec_bytes()) == 1536


def test_utf8_path_and_title():
    r = SyncRow(
        path="données/café.md",
        title="naïve résumé ☕",
        domain="fr",
        vec_b64=base64.b64encode(b"\x00" * 1536).decode("ascii"),
        version=1,
    )
    rebuilt = read_sync_bundle(write_sync_bundle([r]))[0]
    assert rebuilt.path == r.path
    assert rebuilt.title == r.title
