from __future__ import annotations
"""
backend_selector.py – Simplified backend decision tree
======================================================
This rewrite focuses on **deterministic, format‑first** routing.  
Given the myriad community checkpoints (GGUF, GPTQ, AWQ, ONNX, etc.) the
most reliable signal for the correct runtime is the *weight format* itself.  
We therefore drop heuristic rules that depended on model size or id and keep a
small, explicit map from *format* → *backend*.

If no rule matches we fall back to vanilla 🤗 *transformers* which is the
widest‑supported interface.
"""
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# Tasks handled by 🧨 diffusers
DIFFUSERS_TASKS = {
    "text-to-image", "image-to-image", "image-to-text",
    "text-to-video", "inpainting",
}

@dataclass
class Rule:
    """Predicate‑based rule used by the selector."""

    name: str
    pred: Callable[[Dict[str, Any]], bool]
    backend: str


class BackendSelector:
    """Very small rule engine → returns the first matching backend."""

    def __init__(self, info: Dict[str, Any]):
        self.info = info
        self._rules: List[Rule] = self._build_rules()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def recommend(self) -> str:
        """Return the first backend whose predicate matches *info*."""
        for rule in self._rules:
            if rule.pred(self.info):
                return rule.backend
        # Fallback – vanilla 🤗 transformers works for most un‑quantised fp16/fp32 checkpoints
        return "transformers"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_rules(self) -> List[Rule]:
        """Hard‑coded, format‑centric rules sorted by priority."""

        fmt = lambda *xs: (lambda i: i.get("weight_format", "") in xs)
        contains = lambda s: (lambda i: s in i.get("weight_format", ""))

        return [
            # 1️⃣   GGUF / GGML → llama‑cpp, the defacto runtime for these formats
            Rule("gguf_or_ggml", fmt("gguf", "ggml"), "llama-cpp-python"),

            # 2️⃣   GPTQ safetensors or .gptq → auto‑gptq
            Rule("gptq", contains("gptq"), "auto-gptq"),

            # 3️⃣   AWQ
            Rule("awq", fmt("awq"), "autoawq"),

            # 4️⃣   ExLlama 4‑bit safetensors
            Rule("exllama", contains("exllama"), "exllama"),

            # 5️⃣   Marlin 4‑bit
            Rule("marlin", fmt("marlin"), "marlin"),

            # 6️⃣   ONNX → onnxruntime / Optimum
            Rule("onnx", fmt("onnx"), "onnxruntime"),

            # 7️⃣   Diffusion pipelines (weight format is often irrelevant)
            Rule(
                "diffusers",
                lambda i: i.get("library") == "diffusers" or i.get("task") in DIFFUSERS_TASKS,
                "diffusers",
            ),
        ]
