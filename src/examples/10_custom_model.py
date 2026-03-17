"""
Example 10: Custom Model Configuration

Demonstrates how to swap different models into the TSUUID codec.
The architecture is model-agnostic — any sentence transformer or
causal LM can be used as the encoding backend.

Requires: pip install tsuuid[bitnet]
"""

import sys
sys.path.insert(0, "src")

from tsuuid import SemanticCodec
from tsuuid.packing import unpack_uuid_to_trits, trits_to_display


def demo_backend(name, codec, text):
    """Encode text and show results."""
    uid = codec.encode(text)
    meaning = codec.decode(uid)
    n_active = sum(1 for t in meaning.trits if t != 0)
    print(f"  [{name}] UUID: {uid}")
    print(f"           Active: {n_active}/81")
    print(f"           Trits:  {meaning.trit_display[:40]}...")
    for desc in meaning.active_dimensions[:3]:
        print(f"             {desc}")
    print()


def main():
    print("=" * 70)
    print("TSUUID Example 10: Custom Model Configuration")
    print("=" * 70)

    text = "Quarterly financial report shows 12% revenue growth"

    # Backend 1: Hash (no dependencies)
    print("\n1. Hash Backend (reference)")
    demo_backend("hash", SemanticCodec(backend="hash"), text)

    # Backend 2: Default BitNet (all-MiniLM-L6-v2)
    print("2. BitNet Backend (default model)")
    demo_backend("bitnet", SemanticCodec(backend="bitnet"), text)

    # Backend 3: BitNet with explicit model config
    print("3. BitNet Backend (explicit config, CPU forced)")
    codec = SemanticCodec(
        backend="bitnet",
        model_config={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        },
    )
    demo_backend("bitnet-cpu", codec, text)

    # Backend 4: Causal LM (decoder-only model)
    print("4. Causal LM Backend (tiny-gpt2)")
    try:
        codec = SemanticCodec(
            backend="causal",
            model_config={
                "model_name": "sshleifer/tiny-gpt2",
                "device": "cpu",
            },
        )
        demo_backend("causal", codec, text)
    except Exception as e:
        print(f"  Causal backend unavailable: {e}\n")

    # Show model info
    print("=" * 70)
    print("Model Metadata")
    print("=" * 70)

    from tsuuid.bitnet_backend import BitNetEncoder
    enc = BitNetEncoder()
    info = enc.model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
