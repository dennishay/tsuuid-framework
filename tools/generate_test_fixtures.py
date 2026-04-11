#!/usr/bin/env python3
"""Generate golden test fixtures from Python for Swift cross-platform tests.

Run: python3 tools/generate_test_fixtures.py
Output: TSUUIDKit/Resources/test_fixtures/golden_fixtures.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tsuuid.packing import pack_trits_to_uuid, unpack_uuid_to_trits, N_DIMS
from tsuuid.delta import DeltaEncoder

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "TSUUIDKit", "Resources", "test_fixtures")


def generate_packing_fixtures():
    cases = []

    # All zeros
    trits = np.zeros(N_DIMS, dtype=np.int8)
    uuid_val = pack_trits_to_uuid(trits)
    cases.append({"name": "all_zeros", "trits": trits.tolist(), "uuid": str(uuid_val)})

    # Alternating
    trits = np.array([(1 if i % 2 == 0 else -1) for i in range(N_DIMS)], dtype=np.int8)
    uuid_val = pack_trits_to_uuid(trits)
    cases.append({"name": "alternating", "trits": trits.tolist(), "uuid": str(uuid_val)})

    # All positive
    trits = np.ones(N_DIMS, dtype=np.int8)
    uuid_val = pack_trits_to_uuid(trits)
    cases.append({"name": "all_positive", "trits": trits.tolist(), "uuid": str(uuid_val)})

    # Sparse mixed
    trits = np.zeros(N_DIMS, dtype=np.int8)
    trits[0] = 1
    trits[5] = -1
    trits[20] = 1
    trits[60] = -1
    uuid_val = pack_trits_to_uuid(trits)
    cases.append({"name": "sparse_mixed", "trits": trits.tolist(), "uuid": str(uuid_val)})

    return cases


def generate_delta_fixtures():
    enc = DeltaEncoder(epsilon=0.001)

    old_vec = np.zeros(768, dtype=np.float32)
    old_vec[0] = 1.0
    new_vec = old_vec.copy()
    new_vec[0] = 1.5
    new_vec[100] = 0.3

    delta = enc.compute_delta(old_vec, new_vec)
    sparse = enc.sparsify(delta, version=1)
    wire_bytes = sparse.to_bytes()

    checkpoint = enc.make_checkpoint(new_vec, version=2)
    checkpoint_bytes = checkpoint.to_bytes()

    return {
        "delta": {
            "wire_hex": wire_bytes.hex(),
            "version": int(sparse.version),
            "n_changed": int(sparse.n_changed),
            "is_checkpoint": bool(sparse.is_checkpoint),
            "indices": [int(i) for i in sparse.indices],
        },
        "checkpoint": {
            "wire_hex": checkpoint_bytes.hex(),
            "version": int(checkpoint.version),
            "n_changed": int(checkpoint.n_changed),
            "is_checkpoint": bool(checkpoint.is_checkpoint),
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fixtures = {
        "packing": generate_packing_fixtures(),
        "delta": generate_delta_fixtures(),
        "generated_by": "tools/generate_test_fixtures.py",
        "python_version": sys.version,
    }

    path = os.path.join(OUT_DIR, "golden_fixtures.json")
    with open(path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
