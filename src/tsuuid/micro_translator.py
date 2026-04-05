#!/usr/bin/env python3
"""
tsuuid.micro_translator — MicroGPT-style transformer for 768→N trit translation.

A small transformer (not a linear layer) that learns to compress 768 LaBSE
floats into N ternary values while preserving semantic similarity rankings.

Architecture (inspired by Karpathy's nanoGPT, adapted for embedding translation):
    Encoder: 768 → MLP → SelfAttention × L layers → Linear → N → TernaryQuantize
    Decoder: N trits → Linear → SelfAttention × L layers → MLP → 768

The attention mechanism lets the model learn WHICH dimensions of the 768 interact
with each other — something a single linear layer can't capture.

Designed to train on Apple Silicon M1 (8GB) via MPS backend.

Usage:
    python3 -m tsuuid.micro_translator train --n-trits 128 --epochs 500
    python3 -m tsuuid.micro_translator train --n-trits 81 --epochs 500
    python3 -m tsuuid.micro_translator sweep
    python3 -m tsuuid.micro_translator test --n-trits 128
    python3 -m tsuuid.micro_translator encode "text to encode"
"""

import json
import math
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
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DB_PATH = os.path.expanduser("~/.claude/claude_home.db")
CPC_DB = os.path.expanduser("~/ClaudeProjects/cpc/_automation/state.db")
MODEL_DIR = Path.home() / ".cache" / "tsuuid" / "micro_translators"


# ── Data Loading ───────────────────────────────────────

def load_training_vectors():
    """Load all 768-dim vectors as training data."""
    vectors = []

    for db_path, query in [
        (DB_PATH, "SELECT vec FROM tsuuid_768 WHERE vec IS NOT NULL"),
        (CPC_DB, "SELECT vec_768 FROM emails WHERE vec_768 IS NOT NULL"),
    ]:
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(query).fetchall()
            conn.close()
            for (blob,) in rows:
                vec = np.frombuffer(blob, dtype=np.float16).astype(np.float32).copy()
                if len(vec) == 768:
                    vectors.append(vec)
        except Exception as e:
            print(f"  Loading from {os.path.basename(db_path)}: {e}")

    arr = np.array(vectors, dtype=np.float32)
    print(f"Loaded {len(arr)} training vectors")
    return arr


# ── Ternary Quantization (BitNet STE) ─────────────────

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        threshold = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        out = torch.zeros_like(x)
        out[x > threshold] = 1.0
        out[x < -threshold] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ── Micro Transformer Block ───────────────────────────

class MicroAttention(nn.Module):
    """Multi-head self-attention for embedding dimensions (not sequence)."""

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, D = x.shape
        # Reshape to (B, n_heads, head_dim) for attention
        x_reshaped = x.view(B, self.n_heads, self.head_dim)

        qkv = self.qkv(x).view(B, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention across heads
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.reshape(B, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""

    def __init__(self, dim, n_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MicroAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ── MicroGPT Translator ──────────────────────────────

class MicroTranslator(nn.Module):
    """MicroGPT-style 768→N trit translator.

    Encoder: 768 → transformer blocks → project to N → ternary quantize
    Decoder: N → project to hidden → transformer blocks → 768
    """

    def __init__(self, n_trits=81, n_layers=3, n_heads=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.n_trits = n_trits

        # Encoder
        self.enc_proj = nn.Linear(768, hidden_dim)
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.enc_out = nn.Linear(hidden_dim, n_trits)

        # Decoder
        self.dec_proj = nn.Linear(n_trits, hidden_dim)
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.dec_out = nn.Linear(hidden_dim, 768)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        """768 floats → N trits."""
        h = self.enc_proj(x)
        for block in self.enc_blocks:
            h = block(h)
        projected = self.enc_out(h)
        return TernaryQuantize.apply(projected)

    def decode(self, trits):
        """N trits → 768 floats."""
        h = self.dec_proj(trits)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out(h)

    def forward(self, x):
        trits = self.encode(x)
        reconstructed = self.decode(trits)
        return reconstructed, trits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Loss Functions ────────────────────────────────────

def similarity_preservation_loss(original, reconstructed):
    """Preserve the FULL similarity matrix, not just reconstruction."""
    orig_norm = F.normalize(original, dim=1)
    recon_norm = F.normalize(reconstructed, dim=1)

    orig_sim = orig_norm @ orig_norm.t()
    recon_sim = recon_norm @ recon_norm.t()

    return F.mse_loss(recon_sim, orig_sim)


def triplet_ranking_loss(original, reconstructed, margin=0.05):
    """Sample triplets and enforce correct ranking order."""
    B = original.shape[0]
    if B < 3:
        return torch.tensor(0.0, device=original.device)

    orig_norm = F.normalize(original, dim=1)
    recon_norm = F.normalize(reconstructed, dim=1)

    # Sample random triplets
    n_triplets = min(B * 2, 200)
    idx = torch.randint(0, B, (n_triplets, 3), device=original.device)
    a, p, n = idx[:, 0], idx[:, 1], idx[:, 2]

    # Original distances determine which is positive vs negative
    orig_d_ap = 1 - (orig_norm[a] * orig_norm[p]).sum(dim=1)
    orig_d_an = 1 - (orig_norm[a] * orig_norm[n]).sum(dim=1)

    # Swap so p is always closer than n in original space
    swap = orig_d_ap > orig_d_an
    p_final = torch.where(swap, n, p)
    n_final = torch.where(swap, p, n)

    # Reconstructed distances should preserve the same ordering
    recon_d_ap = 1 - (recon_norm[a] * recon_norm[p_final]).sum(dim=1)
    recon_d_an = 1 - (recon_norm[a] * recon_norm[n_final]).sum(dim=1)

    # Triplet loss: positive should be closer than negative by margin
    loss = F.relu(recon_d_ap - recon_d_an + margin).mean()
    return loss


# ── Evaluation ────────────────────────────────────────

def evaluate_ranking_preservation(model, data, n_refs=10):
    """Measure how well trit encoding preserves similarity rankings."""
    model.eval()
    with torch.no_grad():
        reconstructed, trits = model(data)

    orig = F.normalize(data, dim=1).cpu().numpy()
    recon = F.normalize(reconstructed, dim=1).cpu().numpy()

    n = len(orig)
    refs = np.linspace(0, n - 1, min(n_refs, n), dtype=int)

    top1_matches = 0
    top5_overlaps = []

    for ref in refs:
        orig_sims = orig @ orig[ref]
        recon_sims = recon @ recon[ref]

        orig_top = np.argsort(orig_sims)[-6:][::-1]  # top 6 (including self)
        recon_top = np.argsort(recon_sims)[-6:][::-1]

        if orig_top[1] == recon_top[1]:  # top-1 match (excluding self)
            top1_matches += 1

        overlap = len(set(orig_top[1:6]) & set(recon_top[1:6]))
        top5_overlaps.append(overlap / 5.0)

    active = (trits.abs() > 0).float().mean().item() * model.n_trits

    return {
        "top1_accuracy": top1_matches / len(refs),
        "top5_overlap": np.mean(top5_overlaps),
        "active_trits": active,
    }


# ── Training ──────────────────────────────────────────

def train(n_trits=81, n_layers=3, hidden_dim=256, epochs=500, lr=0.0005, device="cpu"):
    """Train the micro translator."""
    print(f"MicroTranslator: 768→{n_trits} trits")
    print(f"Architecture: {n_layers} layers, {hidden_dim} hidden, 4 heads")
    print(f"Device: {device}")

    vectors = load_training_vectors()
    if len(vectors) < 20:
        print("ERROR: Need at least 20 vectors")
        return None

    data = torch.tensor(vectors, dtype=torch.float32).to(device)
    data = F.normalize(data, dim=1)

    model = MicroTranslator(n_trits=n_trits, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_top5 = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        reconstructed, trits = model(data)

        # Combined loss
        recon_loss = F.mse_loss(reconstructed, data)
        sim_loss = similarity_preservation_loss(data, reconstructed)
        triplet_loss = triplet_ranking_loss(data, reconstructed)

        loss = recon_loss + 2.0 * sim_loss + 1.0 * triplet_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            metrics = evaluate_ranking_preservation(model, data)
            print(f"  Epoch {epoch+1:>4}: loss={loss.item():.5f} recon={recon_loss.item():.5f} "
                  f"sim={sim_loss.item():.5f} trip={triplet_loss.item():.5f} | "
                  f"top1={metrics['top1_accuracy']:.0%} top5={metrics['top5_overlap']:.0%} "
                  f"active={metrics['active_trits']:.0f}/{n_trits}")

            if metrics['top5_overlap'] > best_top5:
                best_top5 = metrics['top5_overlap']
                best_epoch = epoch + 1
                save_model(model, n_trits, n_layers, hidden_dim, metrics, len(vectors))

    print(f"\nBest top5 overlap: {best_top5:.0%} at epoch {best_epoch}")
    return model


def save_model(model, n_trits, n_layers, hidden_dim, metrics, n_docs):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"micro_{n_trits}t_{n_layers}l.pt"
    torch.save({
        "n_trits": n_trits,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "n_docs": n_docs,
        "params": model.count_params(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, path)
    print(f"  Saved: {path}")


def load_model(n_trits=81, n_layers=3, hidden_dim=256, device="cpu"):
    path = MODEL_DIR / f"micro_{n_trits}t_{n_layers}l.pt"
    if not path.exists():
        return None
    cp = torch.load(path, map_location=device, weights_only=False)
    model = MicroTranslator(
        n_trits=cp["n_trits"], n_layers=cp["n_layers"], hidden_dim=cp["hidden_dim"]
    ).to(device)
    model.load_state_dict(cp["state_dict"])
    return model, cp


def sweep():
    """Test multiple configurations."""
    device = "mps" if HAS_TORCH and torch.backends.mps.is_available() else "cpu"

    configs = [
        # (n_trits, n_layers, hidden_dim, epochs)
        (256, 3, 256, 300),
        (128, 3, 256, 300),
        (81,  3, 256, 500),
        (81,  4, 384, 500),
        (64,  4, 384, 500),
    ]

    for n_trits, n_layers, hidden_dim, epochs in configs:
        print(f"\n{'='*70}")
        bytes_per_doc = int(n_trits * 1.585 / 8)
        print(f"N={n_trits} trits ({bytes_per_doc} bytes) | {n_layers} layers | {hidden_dim} hidden | {epochs} epochs")
        print(f"{'='*70}")
        train(n_trits=n_trits, n_layers=n_layers, hidden_dim=hidden_dim, epochs=epochs, device=device)


if __name__ == "__main__":
    if not HAS_TORCH:
        print("ERROR: PyTorch required")
        sys.exit(1)

    args = sys.argv[1:]

    if not args or args[0] == "sweep":
        sweep()
    elif args[0] == "train":
        n, layers, hidden, epochs = 81, 3, 256, 500
        for i, a in enumerate(args):
            if a == "--n-trits" and i + 1 < len(args): n = int(args[i + 1])
            if a == "--layers" and i + 1 < len(args): layers = int(args[i + 1])
            if a == "--hidden" and i + 1 < len(args): hidden = int(args[i + 1])
            if a == "--epochs" and i + 1 < len(args): epochs = int(args[i + 1])
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        train(n_trits=n, n_layers=layers, hidden_dim=hidden, epochs=epochs, device=device)
    elif args[0] == "test":
        n = 81
        for i, a in enumerate(args):
            if a == "--n-trits" and i + 1 < len(args): n = int(args[i + 1])
        result = load_model(n)
        if result:
            model, cp = result
            print(f"Model: {n} trits, {cp['n_layers']} layers, {cp['params']:,} params")
            print(f"Metrics: {json.dumps(cp['metrics'], indent=2)}")
        else:
            print(f"No saved model for {n} trits")
    else:
        print(__doc__)
