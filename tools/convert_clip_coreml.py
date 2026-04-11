#!/usr/bin/env python3
"""Convert CLIP ViT-B/32 visual encoder to Core ML.

Requirements:
    pip install clip-by-openai coremltools torch onnx

Output:
    models/CLIP-full.mlpackage  (float16, ~170MB)
    models/CLIP-slim.mlpackage  (int8, ~85MB)

Usage:
    python3 tools/convert_clip_coreml.py
"""
import os
import sys


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading CLIP ViT-B/32...")
    import torch
    import clip

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    visual = model.visual
    visual.eval()

    dummy = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(visual, dummy)

    onnx_path = os.path.join(out_dir, "clip_temp.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        traced,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["embedding"],
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

    full_path = os.path.join(out_dir, "CLIP-full.mlpackage")
    mlmodel.save(full_path)
    print(f"Saved: {full_path}")

    print("Quantizing to int8...")
    from coremltools.models.neural_network.quantization_utils import quantize_weights
    mlmodel_slim = quantize_weights(mlmodel, nbits=8)

    slim_path = os.path.join(out_dir, "CLIP-slim.mlpackage")
    mlmodel_slim.save(slim_path)
    print(f"Saved: {slim_path}")

    os.remove(onnx_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
