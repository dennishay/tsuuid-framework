"""
tsuuid.llm_encoder — LLM Comprehension-Based Semantic Encoder

Instead of mathematical projection (embedding → dot product → quantize),
this encoder uses LLM UNDERSTANDING to assign each trit. The LLM reads
the document, reads the 81 axis definitions, and assigns each value
based on genuine semantic comprehension.

Why LLMs: Humans can't read 81 trits — it's like reading encryption.
But LLMs exist IN the semantic matrix. They have deeper command of
semantic meaning than any dot product. The smartest LLM produces the
most faithful encoding.

Architecture:
    document → LLM reads doc + 81 axis definitions → assigns each trit → verify

Gold standard uses multi-pass with council of top-tier LLMs.
Cascade testing validates cheaper models against gold standard.

Reference: TSUUID Framework paper (Hay, 2026), Phase 7 plan.
"""

import json
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

from tsuuid.dimensions import ALL_AXES, Axis
from tsuuid.packing import N_DIMS


OLLAMA_URL = "http://localhost:11434"


def _build_axis_prompt_block() -> str:
    """Build the axis definitions block for the encoding prompt."""
    lines = []
    for ax in ALL_AXES:
        lines.append(f"{ax.id:>2}. {ax.name}: -1={ax.negative}, 0={ax.neutral}, +1={ax.positive}")
    return "\n".join(lines)


AXIS_BLOCK = _build_axis_prompt_block()

ENCODING_SYSTEM_PROMPT = """You are a semantic encoder. Your task is to read a document and encode its meaning into 81 dimensions.

For EACH dimension, assign exactly one value:
  -1 = document clearly expresses the NEGATIVE pole
   0 = dimension is not relevant to this document, OR document is neutral on this axis
  +1 = document clearly expresses the POSITIVE pole

Rules:
- Read the ENTIRE document before assigning any values.
- Consider every sentence. Do not rush.
- 0 means "this axis doesn't apply to this document" — it is NOT "I'm unsure."
- If the document has elements of BOTH poles, assign 0 (neutral/mixed).
- Be precise. A financial invoice IS about business (axis 36 = +1). A code snippet IS technical (axis 36 = -1).
- Every non-zero trit is a claim about the document's meaning. Make each one count.

Output ONLY a JSON array of exactly 81 integers (-1, 0, or +1). No explanation, no markdown, no text."""


ENCODING_USER_PROMPT = """Encode this document into 81 semantic dimensions.

Document:
\"\"\"
{document}
\"\"\"

Dimensions:
{axis_block}

Output a JSON array of 81 integers:"""


INFERENCE_PROMPT = """You are reading a semantic barcode — 81 ternary values that encode a document's meaning.

Each non-zero value tells you something about the original document:

Active dimensions:
{active_dims}

Reconstruct what this document was about. Describe its content, purpose, domain, and key characteristics based ONLY on what the semantic dimensions tell you. Be specific."""


def _call_ollama(model: str, messages: list, temperature: float = 0.2, max_tokens: int = 500) -> str:
    """Call Ollama chat API. Disables thinking mode for models that support it."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")


def _call_claude(prompt: str, system: str = "", model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API directly for gold standard encoding."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — required for gold standard encoding")

    payload = json.dumps({
        "model": model,
        "max_tokens": 500,
        "temperature": 0.1,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("content", [{}])[0].get("text", "")


def _parse_trits(response: str) -> Optional[np.ndarray]:
    """Parse JSON array of 81 integers from LLM response."""
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None

    try:
        values = json.loads(text[start:end + 1])
        if not isinstance(values, list) or len(values) != N_DIMS:
            return None
        # Clamp to valid range
        trits = np.array([max(-1, min(1, int(v))) for v in values], dtype=np.int8)
        return trits
    except (json.JSONDecodeError, ValueError):
        return None


class LLMEncoder:
    """Semantic encoder using LLM comprehension.

    The LLM reads the document and the 81 axis definitions,
    then assigns each trit based on genuine understanding.
    """

    def __init__(self, model: str = "gemma4:e2b", provider: str = "ollama",
                 multi_pass: bool = True, max_retries: int = 2):
        """
        Args:
            model: Model name (Ollama model name or Claude model ID)
            provider: "ollama" for local models, "claude" for Claude API
            multi_pass: Run encoding twice and reconcile disagreements
            max_retries: Max retries if LLM produces unparseable output
        """
        self.model = model
        self.provider = provider
        self.multi_pass = multi_pass
        self.max_retries = max_retries

    def encode(self, content: str, metadata: Optional[Dict] = None) -> np.ndarray:
        """Encode document to 81-dimensional ternary vector using LLM comprehension.

        The LLM reads the full document and assigns each of 81 trits based
        on its understanding of the content against the axis definitions.
        """
        # Truncate very long documents (LLM context limit)
        doc_text = content[:8000]

        user_prompt = ENCODING_USER_PROMPT.format(
            document=doc_text,
            axis_block=AXIS_BLOCK,
        )

        # Pass 1
        trits_1 = self._single_pass(user_prompt)
        if trits_1 is None:
            raise RuntimeError(f"LLM encoder failed to produce valid trits after {self.max_retries} retries")

        if not self.multi_pass:
            return trits_1

        # Pass 2
        trits_2 = self._single_pass(user_prompt)
        if trits_2 is None:
            return trits_1  # fall back to pass 1

        # Reconcile disagreements
        if np.array_equal(trits_1, trits_2):
            return trits_1

        return self._reconcile(doc_text, trits_1, trits_2)

    def _single_pass(self, user_prompt: str) -> Optional[np.ndarray]:
        """Run one encoding pass."""
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "ollama":
                    response = _call_ollama(
                        self.model,
                        [
                            {"role": "system", "content": ENCODING_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=500,
                    )
                elif self.provider == "claude":
                    response = _call_claude(
                        user_prompt,
                        system=ENCODING_SYSTEM_PROMPT,
                        model=self.model,
                    )
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")

                trits = _parse_trits(response)
                if trits is not None:
                    return trits
            except Exception as e:
                if attempt == self.max_retries:
                    raise
        return None

    def _reconcile(self, doc_text: str, trits_1: np.ndarray, trits_2: np.ndarray) -> np.ndarray:
        """Reconcile disagreements between two encoding passes.

        For axes where both passes agree → keep that value.
        For axes where they disagree → run a focused third pass on just those axes.
        """
        result = trits_1.copy()
        disagreements = []

        for i in range(N_DIMS):
            if trits_1[i] != trits_2[i]:
                ax = ALL_AXES[i]
                disagreements.append(
                    f"Axis {ax.id} ({ax.name}): Pass 1 said {trits_1[i]}, Pass 2 said {trits_2[i]}. "
                    f"Options: -1={ax.negative}, 0={ax.neutral}, +1={ax.positive}"
                )

        if not disagreements:
            return result

        # Third pass: focused on disagreements only
        reconcile_prompt = (
            f"Two encoding passes disagreed on {len(disagreements)} axes for this document.\n\n"
            f"Document:\n\"\"\"\n{doc_text[:4000]}\n\"\"\"\n\n"
            f"Disagreements:\n" + "\n".join(disagreements) + "\n\n"
            f"For each axis listed, give the correct value (-1, 0, or +1). "
            f"Output ONLY a JSON object mapping axis numbers to values, e.g. {{\"3\": 1, \"7\": -1}}"
        )

        try:
            if self.provider == "ollama":
                response = _call_ollama(
                    self.model,
                    [{"role": "user", "content": reconcile_prompt}],
                    temperature=0.1,
                    max_tokens=300,
                )
            else:
                response = _call_claude(reconcile_prompt, model=self.model)

            # Parse reconciliation
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                fixes = json.loads(text[start:end + 1])
                for axis_id_str, value in fixes.items():
                    idx = int(axis_id_str) - 1
                    if 0 <= idx < N_DIMS:
                        result[idx] = max(-1, min(1, int(value)))
        except Exception:
            # If reconciliation fails, use majority: where they agree keep it,
            # where they disagree default to 0 (conservative)
            for i in range(N_DIMS):
                if trits_1[i] != trits_2[i]:
                    result[i] = 0

        return result

    def infer(self, trits: np.ndarray) -> str:
        """Reconstruct document meaning from trit vector using LLM comprehension.

        The LLM reads the semantic barcode and describes what the document was about.
        """
        active_dims = []
        for i, t in enumerate(trits):
            if t != 0 and i < len(ALL_AXES):
                ax = ALL_AXES[i]
                pole = ax.positive if t == 1 else ax.negative
                active_dims.append(f"  {ax.name} = {pole} ({t:+d})")

        if not active_dims:
            return "Empty encoding — no semantic dimensions active."

        prompt = INFERENCE_PROMPT.format(active_dims="\n".join(active_dims))

        if self.provider == "ollama":
            return _call_ollama(
                self.model,
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
        else:
            return _call_claude(prompt, model=self.model)

    def model_info(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "multi_pass": self.multi_pass,
            "encoding_method": "llm_comprehension",
            "n_dims": N_DIMS,
        }

    @classmethod
    def from_config(cls, config: dict) -> "LLMEncoder":
        return cls(
            model=config.get("model_name", "gemma4:e2b"),
            provider=config.get("provider", "ollama"),
            multi_pass=config.get("multi_pass", True),
        )
