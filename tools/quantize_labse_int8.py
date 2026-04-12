#!/usr/bin/env python3
"""Quantize LaBSE-full.mlpackage to int8 for a smaller app binary.

Requires: Python 3.11 + coremltools 7.2
Input:  models/LaBSE-full.mlpackage (942 MB float16)
Output: models/LaBSE-slim.mlpackage (~120 MB int8)

Usage:
    /tmp/coreml-py311/bin/python3 tools/quantize_labse_int8.py
"""
import os
import warnings
warnings.filterwarnings("ignore")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    input_path = os.path.join(out_dir, "LaBSE-full.mlpackage")
    output_path = os.path.join(out_dir, "LaBSE-slim.mlpackage")

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found. Run convert_labse_coreml.py first.")
        return

    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    print(f"Loading {input_path}...")
    mlmodel = ct.models.MLModel(input_path)

    print("Quantizing to int8...")
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=np.int8,
        )
    )
    quantized = cto.linear_quantize_weights(mlmodel, config=config)

    print(f"Saving {output_path}...")
    quantized.save(output_path)

    size_mb = sum(
        f.stat().st_size for f in __import__("pathlib").Path(output_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Saved: {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
