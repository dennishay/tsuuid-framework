#!/usr/bin/env python3
"""
tsuuid.clip_768 — The Visual Encoder. Image → CLIP ViT-L/14 → 768 floats.

The visual counterpart to labse_768. LaBSE encodes what something MEANS
(text → 768). CLIP encodes what something LOOKS LIKE (image → 768).

Same dimensionality. Different spaces. Complementary information.

CLIP also encodes text into the SAME space as images, enabling cross-modal
search: type "dark gothic horror" and find images that match.

Usage:
    from tsuuid.clip_768 import VisualEncoder768

    enc = VisualEncoder768()

    # Encode an image
    vec = enc.encode_image("/path/to/image.jpg")

    # Encode a batch
    vecs = enc.encode_images(["/path/a.jpg", "/path/b.jpg"])

    # Cross-modal: text → visual space
    vec = enc.encode_text("dark gothic horror creature")

    # Compare
    sim = enc.similarity(vec_a, vec_b)

    # Store
    enc.store("path/to/image.jpg", "Card Art", vec)

    # Search stored images by text
    results = enc.search("fiery dragon")

    # Search stored images by image
    results = enc.search_by_image("/path/to/query.jpg")

CLI:
    python3 -m tsuuid.clip_768 encode-image /path/to/image.jpg
    python3 -m tsuuid.clip_768 encode-text "dark gothic horror"
    python3 -m tsuuid.clip_768 search "fiery dragon"
    python3 -m tsuuid.clip_768 search-image /path/to/query.jpg
    python3 -m tsuuid.clip_768 compare-images /path/a.jpg /path/b.jpg
    python3 -m tsuuid.clip_768 stats
"""

import base64
import os
import sqlite3
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

DB_PATH = os.path.expanduser("~/.claude/claude_home.db")


# ── Serialization (shared with labse_768) ─────────────────

def vec_to_bytes(vec: np.ndarray) -> bytes:
    """768 float32 → 3072 bytes. Lossless."""
    return vec.astype(np.float32).tobytes()

def bytes_to_vec(data: bytes) -> np.ndarray:
    """3072 bytes → 768 float32. Lossless."""
    return np.frombuffer(data, dtype=np.float32).copy()

def vec_to_f16_bytes(vec: np.ndarray) -> bytes:
    """768 float32 → 1536 bytes (float16). Preserves rankings."""
    return vec.astype(np.float16).tobytes()

def f16_bytes_to_vec(data: bytes) -> np.ndarray:
    """1536 bytes → 768 float32 (from float16)."""
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()

def vec_to_b64(vec: np.ndarray) -> str:
    """768 float16 → base64 string."""
    return base64.b64encode(vec_to_f16_bytes(vec)).decode("ascii")

def b64_to_vec(b64: str) -> np.ndarray:
    """Base64 string → 768 float32."""
    return f16_bytes_to_vec(base64.b64decode(b64))


# ── Encoder ───────────────────────────────────────────────

class VisualEncoder768:
    """CLIP ViT-L/14 768-dim encoder. Image → appearance."""

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._processor = None
        self._device = device

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = self._device
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._device = device

        self._model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self._model.to(self._device)
        self._model.eval()

    def _to_numpy(self, features) -> np.ndarray:
        """Handle both tensor and BaseModelOutput returns."""
        if hasattr(features, 'cpu'):
            return features.cpu().numpy()
        if hasattr(features, 'pooler_output'):
            return features.pooler_output.cpu().numpy()
        if hasattr(features, 'last_hidden_state'):
            return features.last_hidden_state[:, 0, :].cpu().numpy()
        raise ValueError(f"Unknown features type: {type(features)}")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector or batch."""
        if vec.ndim == 1:
            return vec / (np.linalg.norm(vec) + 1e-8)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / (norms + 1e-8)

    # ── Image encoding ────────────────────────────────

    def encode_image(self, image_path: str) -> np.ndarray:
        """Single image → 768 float32 vector. L2-normalized."""
        from PIL import Image
        import torch

        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(pixel_values=pixel_values)
        vec = self._to_numpy(features).flatten()
        return self._normalize(vec).astype(np.float32)

    def encode_images(self, image_paths: List[str]) -> Tuple[np.ndarray, List[int]]:
        """Batch of images → (N, 768) array + list of valid indices."""
        from PIL import Image
        import torch

        self._ensure_loaded()
        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception:
                pass

        if not images:
            return np.zeros((0, 768), dtype=np.float32), []

        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(pixel_values=pixel_values)
        vecs = self._to_numpy(features)
        return self._normalize(vecs).astype(np.float32), valid_indices

    # ── Text encoding (cross-modal) ──────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """Text → 768 vector in CLIP visual space. For cross-modal search."""
        import torch

        self._ensure_loaded()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        with torch.no_grad():
            features = self._model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        vec = self._to_numpy(features).flatten()
        return self._normalize(vec).astype(np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch of texts → (N, 768) in CLIP visual space."""
        import torch

        self._ensure_loaded()
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        with torch.no_grad():
            features = self._model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        vecs = self._to_numpy(features)
        return self._normalize(vecs).astype(np.float32)

    # ── Comparison ────────────────────────────────────

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Works for image-image, image-text, text-text."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / max(norm, 1e-8))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance. 0=identical, 2=opposite."""
        return 1.0 - self.similarity(a, b)

    # ── Storage (in claude_home.db) ───────────────────

    def store(self, path: str, title: str, vec: np.ndarray, domain: str = "visual"):
        """Store visual encoding in knowledge base."""
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tsuuid_768_visual (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                title TEXT,
                vec BLOB,
                vec_b64 TEXT,
                domain TEXT,
                encoded_at TEXT,
                UNIQUE(path)
            )
        """)
        conn.execute("""
            INSERT OR REPLACE INTO tsuuid_768_visual (path, title, vec, vec_b64, domain, encoded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (path, title, vec_to_f16_bytes(vec), vec_to_b64(vec), domain,
              time.strftime("%Y-%m-%dT%H:%M:%S")))
        conn.commit()
        conn.close()

    def encode_and_store(self, path: str, title: str, domain: str = "visual") -> np.ndarray:
        """Encode image + store in one call."""
        vec = self.encode_image(path)
        self.store(path, title, vec, domain)
        return vec

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search stored visual encodings by text query (cross-modal)."""
        query_vec = self.encode_text(query)
        return self.search_vec(query_vec, limit)

    def search_by_image(self, image_path: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search stored visual encodings by image similarity."""
        query_vec = self.encode_image(image_path)
        return self.search_vec(query_vec, limit)

    def search_vec(self, query_vec: np.ndarray, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Search stored visual encodings by vector."""
        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute("SELECT path, title, vec FROM tsuuid_768_visual").fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

        results = []
        for path, title, vec_blob in rows:
            stored_vec = f16_bytes_to_vec(vec_blob)
            sim = self.similarity(query_vec, stored_vec)
            results.append((path, title, sim))

        results.sort(key=lambda x: -x[2])
        return results[:limit]

    def stats(self) -> dict:
        """Visual knowledge base statistics."""
        conn = sqlite3.connect(DB_PATH)
        try:
            count = conn.execute("SELECT COUNT(*) FROM tsuuid_768_visual").fetchone()[0]
            domains = conn.execute(
                "SELECT domain, COUNT(*) FROM tsuuid_768_visual GROUP BY domain"
            ).fetchall()
            recent = conn.execute(
                "SELECT title, encoded_at FROM tsuuid_768_visual ORDER BY encoded_at DESC LIMIT 5"
            ).fetchall()
        except sqlite3.OperationalError:
            return {"count": 0, "domains": {}, "recent": []}
        finally:
            conn.close()

        return {
            "count": count,
            "domains": dict(domains),
            "recent": [(t, e) for t, e in recent],
        }


# ── CLI ───────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    enc = VisualEncoder768()

    if cmd == "encode-image":
        if len(sys.argv) < 3:
            print("Usage: python3 -m tsuuid.clip_768 encode-image /path/to/image.jpg")
            return
        path = sys.argv[2]
        vec = enc.encode_image(path)
        print(f"Encoded: {path}")
        print(f"Vector: {vec[:8]}... (768 dims)")
        print(f"Norm: {np.linalg.norm(vec):.4f}")
        # Optionally store
        if "--store" in sys.argv:
            title = os.path.basename(path)
            enc.store(path, title, vec)
            print(f"Stored in {DB_PATH}")

    elif cmd == "encode-text":
        if len(sys.argv) < 3:
            print("Usage: python3 -m tsuuid.clip_768 encode-text \"description\"")
            return
        text = " ".join(sys.argv[2:])
        vec = enc.encode_text(text)
        print(f"Text → CLIP visual space: \"{text}\"")
        print(f"Vector: {vec[:8]}... (768 dims)")

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python3 -m tsuuid.clip_768 search \"query\"")
            return
        query = " ".join(sys.argv[2:])
        results = enc.search(query)
        print(f"\nVisual search: \"{query}\"\n")
        for path, title, sim in results:
            print(f"  {sim:.4f}  {title}")
            print(f"          {path}")

    elif cmd == "search-image":
        if len(sys.argv) < 3:
            print("Usage: python3 -m tsuuid.clip_768 search-image /path/to/query.jpg")
            return
        results = enc.search_by_image(sys.argv[2])
        print(f"\nImage similarity search: {sys.argv[2]}\n")
        for path, title, sim in results:
            print(f"  {sim:.4f}  {title}")
            print(f"          {path}")

    elif cmd == "compare-images":
        if len(sys.argv) < 4:
            print("Usage: python3 -m tsuuid.clip_768 compare-images /a.jpg /b.jpg")
            return
        vec_a = enc.encode_image(sys.argv[2])
        vec_b = enc.encode_image(sys.argv[3])
        sim = enc.similarity(vec_a, vec_b)
        print(f"Similarity: {sim:.4f}")
        print(f"  {sys.argv[2]}")
        print(f"  {sys.argv[3]}")

    elif cmd == "stats":
        s = enc.stats()
        print(f"\nVisual Knowledge Base (CLIP 768)")
        print(f"  Entries: {s['count']}")
        if s['domains']:
            print(f"  Domains:")
            for d, c in s['domains'].items():
                print(f"    {d}: {c}")
        if s['recent']:
            print(f"  Recent:")
            for t, e in s['recent']:
                print(f"    {t} ({e})")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
