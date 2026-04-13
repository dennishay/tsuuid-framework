"""End-to-end verification of the Phase 2 producer unification.

Exercises the three retrofitted producer paths:

  1. /encode — shared encode primitive (deals query encoding, any "just
     give me a vec" caller)
  2. /learn with project='advocate' domain='deals' (deals new-deal flow)
  3. /learn with project='cpc' domain='shipments' (receiving agent hook)

Plus the schema_version drift guard on /learn-prefetched added in Phase 1
and reaffirmed by the Phase 3 iOS-side sentinel.

Skipped when the 768 server isn't running locally — these tests are not
a CI gate, they're an integration smoke for Dennis's dev box.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import pytest

SERVER_768 = "http://127.0.0.1:7680"


def _get_json(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(f"{SERVER_768}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _post_json(path: str, body: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        f"{SERVER_768}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _server_is_up() -> bool:
    try:
        h = _get_json("/health", timeout=2.0)
        return bool(h.get("vectors", 0) > 0)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_is_up(),
    reason="768 server not running on :7680; start com.claude-home.search-768-server",
)


def test_encode_returns_vec_and_tsuuid():
    code, body = _post_json("/encode", {"text": "Nike Air Max sneakers"})
    assert code == 200
    assert body["status"] == "ok"
    assert len(body["vec_b64"]) > 100  # base64 of 1536 bytes is ~2048 chars
    assert len(body["vec"]) == 768
    assert body["tsuuid"].count("-") == 4  # UUID format


def test_learn_accepts_deals_producer():
    """Deals encoding.py emits project=advocate domain=deals."""
    code, body = _post_json("/learn", {
        "text": "test deal: ASICS Gel-Kayano running shoes size 10",
        "session": "test-producer-deals",
        "project": "advocate",
        "domain": "deals",
        "source_type": "advocate_deal",
    })
    assert code == 200
    assert body["status"] == "learned"
    assert body["tsuuid"].count("-") == 4
    assert body["chain_id"] == "test-producer-deals"


def test_learn_accepts_shipments_producer():
    """receiving_agent.py._learn_shipment emits project=cpc domain=shipments."""
    code, body = _post_json("/learn", {
        "text": "test shipment: Rexroth VT5005 servo valve, ProMach, SO-12345",
        "session": "test-producer-shipments",
        "project": "cpc",
        "domain": "shipments",
        "source_type": "shipment_capture",
    })
    assert code == 200
    assert body["status"] == "learned"


def test_learn_prefetched_rejects_future_schema_version():
    """Phase 1 drift guard — must reject schema_version > SYNC_SCHEMA_VERSION."""
    dummy_vec = base64.b64encode(bytes(1536)).decode("ascii")
    code, body = _post_json("/learn-prefetched", {
        "vec_b64": dummy_vec,
        "title": "test-future-schema",
        "schema_version": 99,
    })
    assert code == 400
    assert "schema_version" in body.get("error", "").lower()
    assert body["incoming"] == 99
    assert body["supported"] == 1


def test_learn_prefetched_accepts_current_schema_version():
    """Current schema_version passes — uses /encode to get a real vec."""
    enc_code, enc = _post_json("/encode", {"text": "schema_version=1 acceptance test"})
    assert enc_code == 200

    code, body = _post_json("/learn-prefetched", {
        "vec_b64": enc["vec_b64"],
        "title": "schema_version=1 acceptance test",
        "schema_version": 1,
        "session": "test-schema-accept",
        "domain": "test",
    })
    # Accept either fresh learn (200 status=learned) or duplicate (200 status=duplicate)
    assert code == 200
    assert body["status"] in {"learned", "duplicate"}


def test_learn_prefetched_defaults_missing_schema_version():
    """Legacy rows without schema_version field should default to 1 and succeed."""
    enc_code, enc = _post_json("/encode", {"text": "legacy-no-schema-version"})
    assert enc_code == 200

    code, body = _post_json("/learn-prefetched", {
        "vec_b64": enc["vec_b64"],
        "title": "legacy-no-schema-version",
        "session": "test-legacy",
        "domain": "test",
        # schema_version deliberately omitted
    })
    assert code == 200
    assert body["status"] in {"learned", "duplicate"}
