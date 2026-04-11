#!/usr/bin/env python3
"""Convert LaBSE from sentence-transformers to Core ML.

Uses torch.export.export (modern API) + coremltools.

Usage:
    .venv/bin/python3 tools/convert_labse_coreml.py
"""
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(out_dir, exist_ok=True)

    import torch
    import coremltools as ct
    print(f"torch {torch.__version__}, coremltools {ct.__version__}")

    print("Loading LaBSE model on CPU...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")

    transformer = model[0].auto_model
    transformer.eval()
    transformer.to("cpu")

    class PoolerWrapper(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return out.pooler_output

    wrapper = PoolerWrapper(transformer)
    wrapper.eval()

    dummy_ids = torch.ones(1, 128, dtype=torch.long, device="cpu")
    dummy_mask = torch.ones(1, 128, dtype=torch.long, device="cpu")

    print("Tracing with torch.jit.trace (strict=False)...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask), strict=False)
        test_out = traced(dummy_ids, dummy_mask)
        print(f"  Traced output shape: {test_out.shape}")

    print("Converting to Core ML (float16)...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 128), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    full_path = os.path.join(out_dir, "LaBSE-full.mlpackage")
    mlmodel.save(full_path)
    size_mb = sum(
        f.stat().st_size for f in __import__("pathlib").Path(full_path).rglob("*") if f.is_file()
    ) / 1e6
    print(f"  Saved: {full_path} ({size_mb:.0f} MB)")

    print("\nVerification:")
    test_text = "Invoice from Acme for boiler repair"
    py_vec = model.encode(test_text, show_progress_bar=False)
    print(f"  Python LaBSE norm: {np.linalg.norm(py_vec):.4f}")
    print(f"  Dims: {py_vec.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
