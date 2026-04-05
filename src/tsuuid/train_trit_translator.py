#!/usr/bin/env python3
"""
tsuuid.train_trit_translator — Train the 768→N trit encoder-decoder.

Language 110: the trit language. Same as French or Japanese — just another
representation that translates through the 768 universal hub.

Architecture:
    Encoder: 768 float → Linear → N float → Ternary quantize → N trits
    Decoder: N trits → Linear → 768 float (reconstruct)

Training signal: the N-trit encoding must preserve cosine similarity
rankings from the 768 gold standard. If 768 says doc A is closer to
doc B than to doc C, the N-trit encoding must agree.

Uses BitNet-style straight-through estimator (STE) for ternary quantization
during training — forward pass quantizes, backward pass passes gradients through.

Usage:
    python3 -m tsuuid.train_trit_translator --n-trits 256 --epochs 100
    python3 -m tsuuid.train_trit_translator --n-trits 128 --epochs 100
    python3 -m tsuuid.train_trit_translator --n-trits 81 --epochs 200
    python3 -m tsuuid.train_trit_translator --sweep   # try multiple N values
    python3 -m tsuuid.train_trit_translator --test     # test saved model
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DB_PATH = os.path.expanduser("~/.claude/claude_home.db")
CPC_DB = os.path.expanduser("~/ClaudeProjects/cpc/_automation/state.db")
MODEL_DIR = Path.home() / ".cache" / "tsuuid" / "translators"


def load_training_vectors():
    """Load all 768-dim vectors from the knowledge base as training data."""
    vectors = []
    sources = []

    # From tsuuid_768 table
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT path, title, vec FROM tsuuid_768 WHERE vec IS NOT NULL").fetchall()
        conn.close()
        for path, title, vec_blob in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float16).astype(np.float32).copy()
            if len(vec) == 768:
                vectors.append(vec)
                sources.append(f"tsuuid_768:{title}")
    except Exception as e:
        print(f"  tsuuid_768: {e}")

    # From CPC emails
    try:
        conn = sqlite3.connect(CPC_DB)
        rows = conn.execute("SELECT subject, vec_768 FROM emails WHERE vec_768 IS NOT NULL").fetchall()
        conn.close()
        for subject, vec_blob in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float16).astype(np.float32).copy()
            if len(vec) == 768:
                vectors.append(vec)
                sources.append(f"email:{subject}")
    except Exception as e:
        print(f"  emails: {e}")

    return np.array(vectors, dtype=np.float32), sources


class TernaryQuantize(torch.autograd.Function):
    """BitNet-style ternary quantization with straight-through estimator.

    Forward: quantize to {-1, 0, +1} via absmean threshold
    Backward: pass gradients straight through (STE)
    """
    @staticmethod
    def forward(ctx, x):
        threshold = x.abs().mean()
        out = torch.zeros_like(x)
        out[x > threshold] = 1.0
        out[x < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through


class TritTranslator(nn.Module):
    """Encoder-decoder for 768→N trits→768.

    Encoder: Linear(768, N) → TernaryQuantize → N trits
    Decoder: Linear(N, 768) → reconstruct 768 floats
    """
    def __init__(self, n_trits=81):
        super().__init__()
        self.n_trits = n_trits
        self.encoder = nn.Linear(768, n_trits, bias=False)
        self.decoder = nn.Linear(n_trits, 768, bias=False)

    def encode(self, x):
        """768 floats → N trits."""
        projected = self.encoder(x)
        trits = TernaryQuantize.apply(projected)
        return trits

    def decode(self, trits):
        """N trits → 768 floats (approximate reconstruction)."""
        return self.decoder(trits)

    def forward(self, x):
        trits = self.encode(x)
        reconstructed = self.decode(trits)
        return reconstructed, trits


def ranking_loss(original_vecs, reconstructed_vecs, margin=0.1):
    """Loss that preserves cosine similarity RANKINGS.

    For triplets (anchor, positive, negative): the reconstructed vectors
    must maintain the same relative ordering as the originals.
    """
    batch_size = original_vecs.shape[0]
    if batch_size < 3:
        return torch.tensor(0.0)

    # Compute original similarities
    orig_norm = torch.nn.functional.normalize(original_vecs, dim=1)
    orig_sim = torch.mm(orig_norm, orig_norm.t())

    # Compute reconstructed similarities
    recon_norm = torch.nn.functional.normalize(reconstructed_vecs, dim=1)
    recon_sim = torch.mm(recon_norm, recon_norm.t())

    # Loss: MSE between similarity matrices (preserve relative distances)
    loss = torch.nn.functional.mse_loss(recon_sim, orig_sim)
    return loss


def reconstruction_loss(original, reconstructed):
    """MSE reconstruction loss."""
    return torch.nn.functional.mse_loss(reconstructed, original)


def train(n_trits=81, epochs=100, lr=0.001, device="cpu"):
    """Train the trit translator."""
    print(f"Training 768→{n_trits} trit translator")
    print(f"Device: {device}")

    # Load data
    vectors, sources = load_training_vectors()
    print(f"Training data: {len(vectors)} vectors")

    if len(vectors) < 10:
        print("ERROR: Need at least 10 vectors for training")
        return None

    # Convert to torch
    data = torch.tensor(vectors, dtype=torch.float32).to(device)

    # Normalize (LaBSE vectors should already be normalized, but ensure it)
    data = torch.nn.functional.normalize(data, dim=1)

    # Model
    model = TritTranslator(n_trits=n_trits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_loss = float("inf")
    best_ranking_match = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstructed, trits = model(data)

        # Combined loss: reconstruction + ranking preservation
        r_loss = reconstruction_loss(data, reconstructed)
        k_loss = ranking_loss(data, reconstructed)
        loss = r_loss + 2.0 * k_loss  # weight ranking preservation higher

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Evaluate ranking preservation
            with torch.no_grad():
                recon, t = model(data)
                match = evaluate_rankings(data.cpu().numpy(), recon.cpu().numpy())
                active = (t.abs() > 0).float().mean().item() * n_trits

            print(f"  Epoch {epoch+1:>4}: loss={loss.item():.6f} recon={r_loss.item():.6f} rank={k_loss.item():.6f} ranking_match={match}/10 active_trits={active:.0f}/{n_trits}")

            if match > best_ranking_match:
                best_ranking_match = match
                save_model(model, n_trits, match, len(vectors))

    return model


def evaluate_rankings(original_vecs, reconstructed_vecs, top_k=10):
    """Compare top-K similarity rankings between original and reconstructed."""
    n = len(original_vecs)
    if n < 5:
        return 0

    # Pick 5 reference documents
    refs = [0, n // 4, n // 2, 3 * n // 4, min(n - 1, n - 2)]

    # Compute original similarities for each reference
    orig_norms = original_vecs / (np.linalg.norm(original_vecs, axis=1, keepdims=True) + 1e-8)
    recon_norms = reconstructed_vecs / (np.linalg.norm(reconstructed_vecs, axis=1, keepdims=True) + 1e-8)

    matches = 0
    total = 0

    for ref in refs:
        orig_sims = orig_norms @ orig_norms[ref]
        recon_sims = recon_norms @ recon_norms[ref]

        orig_top = np.argsort(orig_sims)[-top_k:][::-1]
        recon_top = np.argsort(recon_sims)[-top_k:][::-1]

        # Check if top-1 and top-3 match
        if orig_top[1] == recon_top[1]:  # skip self (index 0)
            matches += 1
        total += 1

        overlap = len(set(orig_top[:5]) & set(recon_top[:5]))
        if overlap >= 4:
            matches += 1
        total += 1

    return matches


def save_model(model, n_trits, ranking_match, n_docs):
    """Save trained model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"translator_{n_trits}trits.pt"
    torch.save({
        "n_trits": n_trits,
        "state_dict": model.state_dict(),
        "ranking_match": ranking_match,
        "n_docs": n_docs,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, path)
    print(f"  Saved: {path}")


def load_model(n_trits=81, device="cpu"):
    """Load a trained translator model."""
    path = MODEL_DIR / f"translator_{n_trits}trits.pt"
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model = TritTranslator(n_trits=checkpoint["n_trits"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def sweep():
    """Try multiple N values, find the minimum that preserves rankings."""
    device = "mps" if HAS_TORCH and torch.backends.mps.is_available() else "cpu"

    for n in [768, 512, 384, 256, 192, 128, 96, 81, 64, 48, 32]:
        print(f"\n{'='*60}")
        print(f"N = {n} trits ({n * 1.585:.0f} bits, {n * 1.585 / 8:.0f} bytes)")
        print(f"{'='*60}")
        train(n_trits=n, epochs=100, device=device)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("ERROR: PyTorch required. Install in the OCR venv:")
        print("  ~/.claude/ocr/.venv/bin/pip install torch")
        sys.exit(1)

    args = sys.argv[1:]

    if "--sweep" in args:
        sweep()
    elif "--test" in args:
        n = 81
        for i, a in enumerate(args):
            if a == "--n-trits" and i + 1 < len(args):
                n = int(args[i + 1])
        model = load_model(n)
        if model:
            print(f"Loaded {n}-trit translator")
        else:
            print(f"No saved model for {n} trits")
    else:
        n = 81
        epochs = 100
        for i, a in enumerate(args):
            if a == "--n-trits" and i + 1 < len(args):
                n = int(args[i + 1])
            if a == "--epochs" and i + 1 < len(args):
                epochs = int(args[i + 1])

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        train(n_trits=n, epochs=epochs, device=device)
