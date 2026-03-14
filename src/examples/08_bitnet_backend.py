"""
Example 08: BitNet Semantic Backend

Demonstrates the BitNet backend which uses a sentence transformer
to produce true semantic encodings. Compares hash vs bitnet backends
and shows semantic alignment quality.

Requires: pip install tsuuid[bitnet]
"""

import sys
sys.path.insert(0, "src")

from tsuuid import SemanticCodec
from tsuuid.packing import unpack_uuid_to_trits, trits_to_display

# Test documents spanning different domains
DOCS = [
    "Financial invoice from Acme Corp for $2,340 due April 15",
    "Critical server error: database connection refused on prod-db-01",
    "Quarterly revenue report shows 15% growth over last year",
    "Delete all archived backup files older than 90 days",
    "New employee onboarding checklist approved by HR",
]


def main():
    print("=" * 70)
    print("TSUUID Example 08: BitNet Semantic Backend")
    print("=" * 70)

    hash_codec = SemanticCodec(backend="hash")
    bitnet_codec = SemanticCodec(backend="bitnet")

    for doc in DOCS:
        print(f"\nDocument: {doc[:60]}...")

        h_uuid = hash_codec.encode(doc)
        b_uuid = bitnet_codec.encode(doc)

        h_trits = unpack_uuid_to_trits(h_uuid)
        b_trits = unpack_uuid_to_trits(b_uuid)

        print(f"  Hash UUID:   {h_uuid}")
        print(f"  BitNet UUID: {b_uuid}")
        print(f"  Hash trits:   {trits_to_display(h_trits)}")
        print(f"  BitNet trits: {trits_to_display(b_trits)}")

        # Show active dimensions from bitnet
        meaning = bitnet_codec.decode(b_uuid)
        print(f"  Active dims: {sum(1 for t in meaning.trits if t != 0)}/81")
        for desc in meaning.active_dimensions[:5]:
            print(f"    {desc}")
        if len(meaning.active_dimensions) > 5:
            print(f"    ... and {len(meaning.active_dimensions) - 5} more")

    # Semantic distance comparison
    print("\n" + "=" * 70)
    print("Semantic Distance Matrix (BitNet)")
    print("=" * 70)

    uuids = [bitnet_codec.encode(doc) for doc in DOCS]
    labels = ["Invoice", "Server err", "Revenue", "Delete", "Onboarding"]

    print(f"\n{'':>12}", end="")
    for l in labels:
        print(f"{l:>12}", end="")
    print()

    for i, li in enumerate(labels):
        print(f"{li:>12}", end="")
        for j in range(len(labels)):
            d = bitnet_codec.distance(uuids[i], uuids[j])
            print(f"{d:>12.0f}", end="")
        print()


if __name__ == "__main__":
    main()
