#!/usr/bin/env python3
"""
Example 01: Basic Encode/Decode
================================
This is the "hello world" of TSUUID. We encode a document as a semantic
UUID and decode it back to see the 81-dimensional displacement.

What to notice:
- The UUID is a standard UUID v8 (RFC 9562 compliant)
- The trit display shows which dimensions are displaced
- Only non-zero dimensions carry meaning (sparse representation)
- The same input always produces the same UUID (deterministic)
"""

import sys
sys.path.insert(0, 'src')

from tsuuid import SemanticCodec

codec = SemanticCodec()

# ── Encode a business document ──
print("=" * 70)
print("ENCODING A BUSINESS DOCUMENT")
print("=" * 70)

doc = "Invoice #4471: $2,340 from Acme Corp, payment due 2026-04-15, approved"
uuid1 = codec.encode(doc)
meaning1 = codec.decode(uuid1)

print(f"\nDocument:  {doc}")
print(f"UUID:      {uuid1}")
print(f"Version:   {uuid1.version}")
print(f"Trits:     {meaning1.trit_display}")
print(f"\nActive dimensions ({len(meaning1.active_dimensions)} non-zero):")
for desc in meaning1.active_dimensions[:15]:  # Show first 15
    print(f"  • {desc}")
if len(meaning1.active_dimensions) > 15:
    print(f"  ... and {len(meaning1.active_dimensions) - 15} more")

# ── Encode a technical document ──
print("\n" + "=" * 70)
print("ENCODING A TECHNICAL DOCUMENT")
print("=" * 70)

doc2 = "Server error: database connection failed, critical, requires immediate fix"
uuid2 = codec.encode(doc2)
meaning2 = codec.decode(uuid2)

print(f"\nDocument:  {doc2}")
print(f"UUID:      {uuid2}")
print(f"Trits:     {meaning2.trit_display}")
print(f"\nActive dimensions ({len(meaning2.active_dimensions)} non-zero):")
for desc in meaning2.active_dimensions[:15]:
    print(f"  • {desc}")

# ── Semantic distance ──
print("\n" + "=" * 70)
print("SEMANTIC DISTANCE")
print("=" * 70)

distance = codec.distance(uuid1, uuid2)
print(f"\nDistance between business doc and technical doc: {distance}/81 dimensions differ")
print(f"Similarity: {(81 - distance) / 81 * 100:.1f}%")

# ── Determinism proof ──
print("\n" + "=" * 70)
print("DETERMINISM PROOF")
print("=" * 70)

uuid1_again = codec.encode(doc)
print(f"\nOriginal UUID:  {uuid1}")
print(f"Re-encoded:     {uuid1_again}")
print(f"Identical:      {uuid1 == uuid1_again}")

# ── UUID is only 16 bytes ──
print("\n" + "=" * 70)
print("COMPRESSION RATIO")
print("=" * 70)

doc_bytes = len(doc.encode('utf-8'))
uuid_bytes = 16

print(f"\nDocument size:   {doc_bytes} bytes")
print(f"UUID size:       {uuid_bytes} bytes")
print(f"Ratio:           {doc_bytes / uuid_bytes:.1f}:1")
print(f"\n(In production with BitNet backend, the UUID encodes the full")
print(f" semantic meaning — not just a hash — and reconstructs via")
print(f" addition-only forward pass against the universal model.)")
