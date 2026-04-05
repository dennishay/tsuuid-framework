"""
Example 11: Delta Encoding for Efficient Vector Updates

Demonstrates sending only the changed dimensions when a document's
LaBSE 768-dim embedding changes, instead of the full 3,072-byte vector.

Uses synthetic vectors (numpy random) — no sentence-transformers needed.
For real LaBSE encoding, use Encoder768 from tsuuid.labse_768.

Origin: SVN concept (Claude 187ea329, Nov 2025) + LaBSE note 2BED.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from tsuuid.delta import DeltaEncoder, SparseDelta, compression_ratio, cosine_error


def simulate_edit(base_vec, n_dims_changed, magnitude=0.05, seed=None):
    """Simulate a document edit by perturbing N dimensions."""
    rng = np.random.RandomState(seed)
    new_vec = base_vec.copy()
    dims = rng.choice(768, n_dims_changed, replace=False)
    new_vec[dims] += rng.randn(n_dims_changed) * magnitude
    return new_vec


def main():
    rng = np.random.RandomState(42)
    enc = DeltaEncoder(epsilon=0.005)

    # Base document vector (simulating LaBSE output)
    doc = rng.randn(768).astype(np.float32)
    doc /= np.linalg.norm(doc)  # L2 normalize like LaBSE

    print("=" * 60)
    print("TSUUID Delta Encoding — Compression Demo")
    print("=" * 60)
    print(f"Full vector: 768 dims x 4 bytes = 3,072 bytes")
    print()

    # ── Scenario 1: Minor edit (typo fix) ──
    print("--- Scenario 1: Minor Edit (50 dims) ---")
    edited = simulate_edit(doc, 50, magnitude=0.02, seed=1)
    delta = enc.compute_delta(doc, edited)
    sparse = enc.sparsify(delta, doc_id="minor", version=1)
    recon = enc.apply_delta(doc, sparse)

    print(f"  Changed:     {sparse.n_changed}/768 dims")
    print(f"  Wire size:   {sparse.wire_size} bytes")
    print(f"  Compression: {compression_ratio(sparse):.1f}x")
    print(f"  Cosine err:  {cosine_error(edited, recon):.8f}")
    print()

    # ── Scenario 2: Moderate edit (paragraph added) ──
    print("--- Scenario 2: Moderate Edit (300 dims) ---")
    edited2 = simulate_edit(doc, 300, magnitude=0.05, seed=2)
    delta2 = enc.compute_delta(doc, edited2)
    sparse2 = enc.sparsify(delta2, doc_id="moderate", version=1)
    recon2 = enc.apply_delta(doc, sparse2)

    print(f"  Changed:     {sparse2.n_changed}/768 dims")
    print(f"  Wire size:   {sparse2.wire_size} bytes")
    print(f"  Compression: {compression_ratio(sparse2):.1f}x")
    print(f"  Cosine err:  {cosine_error(edited2, recon2):.8f}")
    print()

    # ── Scenario 3: Major rewrite (auto-checkpoint) ──
    print("--- Scenario 3: Major Rewrite (all dims) ---")
    rewrite = rng.randn(768).astype(np.float32)
    rewrite /= np.linalg.norm(rewrite)
    delta3 = enc.compute_delta(doc, rewrite)
    sparse3 = enc.sparsify(delta3, doc_id="major", version=1)

    print(f"  Changed:     {sparse3.n_changed}/768 dims (auto-promoted)")
    print(f"  Wire size:   {sparse3.wire_size} bytes")
    print(f"  Compression: {compression_ratio(sparse3):.1f}x (no savings — use checkpoint)")
    print()

    # ── Scenario 4: Drift prevention over 50 updates ──
    print("--- Scenario 4: 50 Sequential Updates (drift test) ---")
    drift_enc = DeltaEncoder(epsilon=0.01)
    true_vec = doc.copy()
    stored_vec = doc.copy()

    for i in range(50):
        new_vec = simulate_edit(true_vec, 20, magnitude=0.02, seed=100 + i)
        d = drift_enc.compute_delta(true_vec, new_vec)
        s = drift_enc.sparsify(d, doc_id="drift", version=i + 1)
        stored_vec = drift_enc.apply_delta(stored_vec, s)
        true_vec = new_vec

    sim = 1.0 - cosine_error(true_vec, stored_vec)
    resid = drift_enc.get_residual_norm("drift")
    print(f"  After 50 updates (no checkpoint):")
    print(f"  Cosine sim:  {sim:.6f}")
    print(f"  Residual L2: {resid:.6f}")

    # Demonstrate checkpoint recovery
    cp = drift_enc.make_checkpoint(true_vec, version=51)
    stored_vec = drift_enc.apply_delta(np.zeros(768, dtype=np.float32), cp)
    drift_enc.reset_residual("drift")
    sim_after = 1.0 - cosine_error(true_vec, stored_vec)
    print(f"  After checkpoint: {sim_after:.6f} (drift reset)")
    print(f"  Tip: checkpoint every ~25 updates for sim > 0.999")
    print()

    # ── Scenario 5: Serialization roundtrip ──
    print("--- Scenario 5: Serialization ---")
    wire_bytes = sparse.to_bytes()
    wire_b64 = sparse.to_b64()
    rebuilt = SparseDelta.from_bytes(wire_bytes)
    rebuilt_b64 = SparseDelta.from_b64(wire_b64)

    print(f"  Binary:  {len(wire_bytes)} bytes → {rebuilt}")
    print(f"  Base64:  {len(wire_b64)} chars → {rebuilt_b64}")
    print(f"  Match:   {np.array_equal(rebuilt.indices, sparse.indices)}")
    print()

    # ── Summary table ──
    print("=" * 60)
    print(f"{'Scenario':<25} {'Dims':>6} {'Bytes':>7} {'Ratio':>7} {'Error':>10}")
    print("-" * 60)
    for name, sp, err in [
        ("Minor edit", sparse, cosine_error(simulate_edit(doc, 50, 0.02, 1), enc.apply_delta(doc, sparse))),
        ("Moderate edit", sparse2, cosine_error(edited2, recon2)),
        ("Major rewrite", sparse3, 0.0),
    ]:
        print(f"  {name:<23} {sp.n_changed:>5} {sp.wire_size:>7} {compression_ratio(sp):>6.1f}x {err:>10.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
