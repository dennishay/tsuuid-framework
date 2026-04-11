#!/usr/bin/env python3
"""Convert CLIP ViT-B/32 visual encoder to Core ML.

Uses same recipe as LaBSE: torch.jit.trace + coremltools PyTorch converter.
Requires: Python 3.11 + torch 2.2 + coremltools 7.2

Output:
    models/CLIP-full.mlpackage

Usage:
    /tmp/coreml-py311/bin/python3 tools/convert_clip_coreml.py

Setup (if venv doesn't exist):
    python3.11 -m venv /tmp/coreml-py311
    /tmp/coreml-py311/bin/pip install torch==2.2.0 coremltools==7.2 \
        ftfy regex numpy<2 setuptools
    /tmp/coreml-py311/bin/pip install git+https://github.com/openai/CLIP.git
"""
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    import coremltools as ct
    print(f"torch {torch.__version__}, coremltools {ct.__version__}")

    print("Loading CLIP ViT-B/32 on CPU...")
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    visual = model.visual
    visual.eval()
    visual = visual.float()  # ensure float32 for tracing

    print("Tracing visual encoder...")
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(visual, dummy, strict=False)
        test_out = traced(dummy)
        print(f"  Traced output shape: {test_out.shape}")  # (1, 512) for ViT-B/32

    print("Converting to Core ML (float16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(name="image", shape=(1, 3, 224, 224),
                         scale=1.0/255.0, bias=[0, 0, 0]),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    full_path = os.path.join(out_dir, "CLIP-full.mlpackage")
    mlmodel.save(full_path)
    size_mb = sum(
        f.stat().st_size for f in __import__("pathlib").Path(full_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Saved: {full_path} ({size_mb:.0f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
