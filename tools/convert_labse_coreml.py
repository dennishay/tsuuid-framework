#!/usr/bin/env python3
"""Convert LaBSE from sentence-transformers to Core ML.

Requirements:
    pip install sentence-transformers coremltools torch onnx onnxruntime

Output:
    models/LaBSE-full.mlpackage  (float16, ~235MB)
    models/LaBSE-slim.mlpackage  (int8, ~120MB)

Usage:
    python3 tools/convert_labse_coreml.py
"""
import os
import sys
import numpy as np


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading LaBSE model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")

    print("Tracing model...")
    import torch

    transformer = model[0].auto_model
    dummy_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)

    transformer.eval()
    traced = torch.jit.trace(transformer, (dummy_ids, dummy_mask))

    onnx_path = os.path.join(out_dir, "labse_temp.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        traced,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    import coremltools as ct
    print("Converting to Core ML (float16)...")
    mlmodel = ct.convert(
        onnx_path,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    full_path = os.path.join(out_dir, "LaBSE-full.mlpackage")
    mlmodel.save(full_path)
    print(f"Saved: {full_path}")

    print("Quantizing to int8...")
    from coremltools.models.neural_network.quantization_utils import quantize_weights
    mlmodel_slim = quantize_weights(mlmodel, nbits=8)

    slim_path = os.path.join(out_dir, "LaBSE-slim.mlpackage")
    mlmodel_slim.save(slim_path)
    print(f"Saved: {slim_path}")

    # Verify output matches Python
    print("\nVerification:")
    test_text = "Invoice from Acme for boiler repair"
    py_vec = model.encode(test_text, show_progress_bar=False)
    print(f"  Python norm: {np.linalg.norm(py_vec):.4f}")
    print(f"  Top 3 dims: {np.argsort(np.abs(py_vec))[-3:][::-1]}")
    print("  (Compare against Swift output after integration)")

    os.remove(onnx_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
