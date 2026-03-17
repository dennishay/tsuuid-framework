"""
Example 09: Benchmark Semantic Alignment

Tests how well each backend's encodings align with the intended
meaning of the 81 semantic axes. For each axis, encodes a
positive-pole and negative-pole sentence and checks whether
the corresponding trit has the correct sign.

Test sentences use contextual meaning, NOT the axis pole labels,
to avoid conflating keyword matching with semantic understanding.

Requires: pip install tsuuid[bitnet]
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from tsuuid import SemanticCodec
from tsuuid.dimensions import ALL_AXES
from tsuuid.packing import unpack_uuid_to_trits

# Test cases: (axis_index_0based, positive_text, negative_text)
# Sentences convey meaning through context, not by echoing pole labels.
ALIGNMENT_TESTS = [
    # Protocol layer
    (0, "next quarter's forecast and upcoming milestones",
        "ten years ago the factory was built during the war"),
    (1, "spin up a brand-new microservice from scratch",
        "tear down and decommission the legacy cluster"),
    (2, "the board unanimously ratified the proposal",
        "the application was turned down on all counts"),
    (4, "every employee shall complete safety training",
        "staff are explicitly barred from the restricted area"),
    (5, "the report was released to the general public today",
        "the memo is eyes-only for senior leadership"),
    (7, "we shipped the package via overnight courier",
        "the delivery arrived at the loading dock this morning"),
    # Organization layer
    (20, "our engineering team designed it in-house",
        "the third-party contractor submitted the bid"),
    (25, "the reactor tripped and we need hands on deck now",
        "we can revisit that improvement sometime next year"),
    # Application layer
    (35, "quarterly earnings beat expectations with strong margins",
        "the kernel panicked after a null pointer dereference"),
    (37, "we paid the vendor for the maintenance contract",
        "the customer wired their quarterly subscription fee"),
    (41, "all punch-list items resolved and certificate issued",
        "the build crashed with seventeen unresolved errors"),
    (46, "the team celebrated after winning the championship",
        "the building collapsed and people were trapped"),
    # Field layer
    (73, "production output has climbed steadily each month",
        "demand has been sliding downward all quarter"),
]


def benchmark_backend(backend_name, codec):
    """Score alignment: correct, partial, neutral, or wrong per axis."""
    correct = 0.0
    neutral = 0.0
    wrong = 0.0
    details = []

    for axis_idx, pos_text, neg_text in ALIGNMENT_TESTS:
        axis = ALL_AXES[axis_idx]
        pos_uuid = codec.encode(pos_text)
        neg_uuid = codec.encode(neg_text)
        pos_trits = unpack_uuid_to_trits(pos_uuid)
        neg_trits = unpack_uuid_to_trits(neg_uuid)

        pos_val = int(pos_trits[axis_idx])
        neg_val = int(neg_trits[axis_idx])

        # Positive text should get +1, negative text should get -1
        pos_ok = pos_val == 1
        neg_ok = neg_val == -1

        if pos_ok and neg_ok:
            status = "CORRECT"
            correct += 1
        elif pos_val == 0 and neg_val == 0:
            status = "NEUTRAL"
            neutral += 1
        elif pos_val == neg_val:
            status = "SAME"
            wrong += 1
        elif pos_ok or neg_ok:
            status = "PARTIAL"
            correct += 0.5
            neutral += 0.5
        else:
            status = "WRONG"
            wrong += 1

        details.append((axis.name, axis_idx + 1, pos_val, neg_val, status))

    return correct, neutral, wrong, details


def main():
    print("=" * 70)
    print("TSUUID Semantic Alignment Benchmark")
    print("=" * 70)

    backends = [("hash", SemanticCodec(backend="hash"))]
    try:
        backends.append(("bitnet", SemanticCodec(backend="bitnet")))
    except Exception as e:
        print(f"BitNet backend unavailable: {e}")

    scores = {}
    for name, codec in backends:
        correct, neutral, wrong, details = benchmark_backend(name, codec)
        total = len(ALIGNMENT_TESTS)
        score = correct / total * 100
        scores[name] = score

        print(f"\n--- {name.upper()} Backend ---")
        print(f"Score: {score:.1f}% ({correct:.1f}/{total} correct, "
              f"{neutral:.1f} neutral, {wrong:.1f} wrong)")
        print()
        print(f"  {'Axis':<20} {'Dim':>4} {'Pos':>4} {'Neg':>4} {'Result':<8}")
        print(f"  {'-'*20} {'-'*4} {'-'*4} {'-'*4} {'-'*8}")
        for axis_name, dim, pos, neg, status in details:
            print(f"  {axis_name:<20} {dim:>4} {pos:>+4} {neg:>+4} {status:<8}")

    if len(scores) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        for name, score in scores.items():
            print(f"  {name.capitalize():>8}: {score:.1f}%")
        diff = scores.get("bitnet", 0) - scores.get("hash", 0)
        if diff > 0:
            print(f"  BitNet is {diff:.1f}% better")
        elif diff < 0:
            print(f"  Hash is {-diff:.1f}% better")
        else:
            print(f"  Tied")


if __name__ == "__main__":
    main()
