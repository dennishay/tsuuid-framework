"""
tsuuid.gold_standard — Multi-LLM Gold Standard Encoding Pipeline

Uses the smartest available LLMs (Claude 4.6, GPT 5.4, Grok 4.2) to
produce the highest-fidelity semantic encoding. Multiple passes, council
reconciliation, chairman synthesis.

The gold standard encoding IS the document's semantic identity. Every
other encoding method is measured against this.

Usage:
    from tsuuid.gold_standard import GoldStandardEncoder
    encoder = GoldStandardEncoder()
    trits, report = encoder.encode_with_report("document text here")
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tsuuid.dimensions import ALL_AXES
from tsuuid.llm_encoder import LLMEncoder, _parse_trits, ENCODING_SYSTEM_PROMPT, ENCODING_USER_PROMPT, AXIS_BLOCK
from tsuuid.packing import N_DIMS, pack_trits_to_uuid


@dataclass
class EncodingPass:
    """Result from a single LLM encoding pass."""
    model: str
    provider: str
    trits: np.ndarray
    active_count: int
    elapsed_s: float


@dataclass
class GoldReport:
    """Full report from a gold standard encoding."""
    passes: List[EncodingPass]
    agreement_matrix: Dict[str, Dict[str, int]]  # model → model → Hamming distance
    disagreed_axes: List[int]  # axis IDs where models disagreed
    chairman_model: str
    final_trits: np.ndarray
    final_uuid: str
    active_count: int
    encoding_time_s: float
    confidence: str  # HIGH (unanimous), MEDIUM (minor disagreements), LOW (major disagreements)


class GoldStandardEncoder:
    """Multi-LLM gold standard encoding pipeline.

    Dispatches to multiple top-tier LLMs, compares their encodings,
    reconciles disagreements via chairman synthesis.
    """

    def __init__(self, encoders: Optional[List[LLMEncoder]] = None,
                 chairman: Optional[LLMEncoder] = None):
        """
        Args:
            encoders: List of LLMEncoders to use as council members.
                      Default: builds from available providers.
            chairman: The encoder used for final reconciliation.
                      Default: first encoder in the list (presumably the smartest).
        """
        if encoders is None:
            encoders = self._build_default_encoders()
        self.encoders = encoders
        self.chairman = chairman or (encoders[0] if encoders else None)

    def _build_default_encoders(self) -> List[LLMEncoder]:
        """Build encoders from available providers.

        Tries Claude API first, falls back to local models.
        The order matters — first encoder is the chairman default.
        """
        encoders = []

        # Claude API (if key available)
        if os.environ.get("ANTHROPIC_API_KEY"):
            encoders.append(LLMEncoder(
                model="claude-sonnet-4-20250514",
                provider="claude",
                multi_pass=False,  # gold standard handles multi-pass at this level
            ))

        # Local models — architecturally diverse
        local_models = [
            "gemma4:e2b",                           # Google
            "huihui_ai/qwen3.5-abliterated:4B",    # Alibaba (unrestricted)
            "huihui_ai/deepseek-r1-abliterated:8b", # DeepSeek (reasoning)
        ]

        for model in local_models:
            encoders.append(LLMEncoder(
                model=model,
                provider="ollama",
                multi_pass=False,
            ))

        if not encoders:
            raise RuntimeError("No encoders available — need either ANTHROPIC_API_KEY or Ollama running")

        return encoders

    def encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode using gold standard pipeline. Returns 81 trits."""
        _, report = self.encode_with_report(content, metadata)
        return report.final_trits

    def encode_with_report(self, content: str, metadata: Optional[Dict] = None) -> Tuple[np.ndarray, GoldReport]:
        """Encode with full report showing all passes and reconciliation."""
        start_time = time.time()
        doc_text = content[:8000]

        user_prompt = ENCODING_USER_PROMPT.format(
            document=doc_text,
            axis_block=AXIS_BLOCK,
        )

        # Run all encoders
        passes = []
        for enc in self.encoders:
            try:
                t0 = time.time()
                trits = enc.encode(doc_text)
                elapsed = time.time() - t0
                passes.append(EncodingPass(
                    model=enc.model,
                    provider=enc.provider,
                    trits=trits,
                    active_count=int(np.sum(trits != 0)),
                    elapsed_s=round(elapsed, 1),
                ))
            except Exception as e:
                print(f"  [gold] {enc.model} failed: {e}")

        if not passes:
            raise RuntimeError("All encoders failed")

        # Single encoder — just return it
        if len(passes) == 1:
            final = passes[0].trits
            uid = pack_trits_to_uuid(final)
            return final, GoldReport(
                passes=passes,
                agreement_matrix={},
                disagreed_axes=[],
                chairman_model=passes[0].model,
                final_trits=final,
                final_uuid=str(uid),
                active_count=int(np.sum(final != 0)),
                encoding_time_s=round(time.time() - start_time, 1),
                confidence="MEDIUM",
            )

        # Compare all pairs
        agreement = {}
        for i, p1 in enumerate(passes):
            for j, p2 in enumerate(passes):
                if i != j:
                    dist = int(np.sum(p1.trits != p2.trits))
                    agreement.setdefault(p1.model, {})[p2.model] = dist

        # Find disagreed axes
        all_trits = np.array([p.trits for p in passes])
        disagreed = []
        for axis_idx in range(N_DIMS):
            values = all_trits[:, axis_idx]
            if not np.all(values == values[0]):
                disagreed.append(axis_idx + 1)  # 1-indexed

        # Majority vote for each axis
        final = np.zeros(N_DIMS, dtype=np.int8)
        for axis_idx in range(N_DIMS):
            values = all_trits[:, axis_idx]
            # Majority vote: sum and sign
            total = int(np.sum(values))
            if total > 0:
                final[axis_idx] = 1
            elif total < 0:
                final[axis_idx] = -1
            else:
                final[axis_idx] = 0

        # If significant disagreements, chairman reconciles
        if len(disagreed) > 10 and self.chairman:
            final = self._chairman_reconcile(doc_text, passes, disagreed, final)

        # Confidence assessment
        if len(disagreed) == 0:
            confidence = "HIGH"
        elif len(disagreed) <= 10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        uid = pack_trits_to_uuid(final)
        total_time = round(time.time() - start_time, 1)

        return final, GoldReport(
            passes=passes,
            agreement_matrix=agreement,
            disagreed_axes=disagreed,
            chairman_model=self.chairman.model if self.chairman else "majority_vote",
            final_trits=final,
            final_uuid=str(uid),
            active_count=int(np.sum(final != 0)),
            encoding_time_s=total_time,
            confidence=confidence,
        )

    def _chairman_reconcile(self, doc_text: str, passes: List[EncodingPass],
                            disagreed: List[int], majority: np.ndarray) -> np.ndarray:
        """Chairman reviews disagreements and makes final call."""
        result = majority.copy()

        # Build context showing each model's position on disagreed axes
        context_lines = []
        for axis_id in disagreed:
            ax = ALL_AXES[axis_id - 1]
            positions = []
            for p in passes:
                val = int(p.trits[axis_id - 1])
                label = {-1: ax.negative, 0: ax.neutral, 1: ax.positive}.get(val, "?")
                positions.append(f"{p.model.split('/')[-1]}={val}({label})")
            context_lines.append(
                f"Axis {ax.id} ({ax.name}): {', '.join(positions)}. "
                f"Options: -1={ax.negative}, 0={ax.neutral}, +1={ax.positive}"
            )

        prompt = (
            f"You are the chairman resolving {len(disagreed)} encoding disagreements.\n\n"
            f"Document:\n\"\"\"\n{doc_text[:4000]}\n\"\"\"\n\n"
            f"Disagreements:\n" + "\n".join(context_lines) + "\n\n"
            f"For each axis, give the correct value. Consider the document carefully.\n"
            f"Output ONLY a JSON object: {{\"axis_id\": value, ...}}"
        )

        try:
            response = self.chairman._single_pass.__func__(
                self.chairman,
                prompt  # just reuse _single_pass logic but with custom prompt
            )
            # Actually, let's call the model directly
            from tsuuid.llm_encoder import _call_ollama, _call_claude

            if self.chairman.provider == "ollama":
                resp = _call_ollama(self.chairman.model, [{"role": "user", "content": prompt}], temperature=0.1)
            else:
                resp = _call_claude(prompt, model=self.chairman.model)

            text = resp.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                fixes = json.loads(text[start:end + 1])
                for axis_id_str, value in fixes.items():
                    idx = int(axis_id_str) - 1
                    if 0 <= idx < N_DIMS:
                        result[idx] = max(-1, min(1, int(value)))
        except Exception as e:
            print(f"  [gold] Chairman reconciliation failed: {e}, using majority vote")

        return result


def print_report(report: GoldReport):
    """Pretty-print a gold standard encoding report."""
    print(f"=== Gold Standard Encoding Report ===")
    print(f"UUID: {report.final_uuid}")
    print(f"Active dims: {report.active_count}/81")
    print(f"Confidence: {report.confidence}")
    print(f"Time: {report.encoding_time_s}s")
    print(f"Chairman: {report.chairman_model}")
    print()

    print(f"Passes ({len(report.passes)}):")
    for p in report.passes:
        print(f"  {p.model:<45} active={p.active_count:>2}  time={p.elapsed_s}s")

    if report.agreement_matrix:
        print(f"\nAgreement (Hamming distance, lower=better):")
        for m1, dists in report.agreement_matrix.items():
            for m2, d in dists.items():
                print(f"  {m1.split('/')[-1]:<20} vs {m2.split('/')[-1]:<20} = {d} differences")

    if report.disagreed_axes:
        print(f"\nDisagreed axes ({len(report.disagreed_axes)}):")
        for axis_id in report.disagreed_axes[:20]:
            ax = ALL_AXES[axis_id - 1]
            val = int(report.final_trits[axis_id - 1])
            label = {-1: ax.negative, 0: ax.neutral, 1: ax.positive}.get(val, "?")
            print(f"  {ax.id:>2}. {ax.name:<20} → {val:+d} ({label})")
        if len(report.disagreed_axes) > 20:
            print(f"  ... and {len(report.disagreed_axes) - 20} more")

    # Show final encoding by layer
    print(f"\nFinal encoding by layer:")
    from tsuuid.dimensions import SemanticDimensions
    dims = SemanticDimensions()
    summary = dims.layer_summary(report.final_trits)
    for layer, info in summary.items():
        print(f"  {layer:<15} {info['active_dims']}/{info['total_dims']} active")
        for desc in info['descriptions']:
            print(f"    {desc}")
