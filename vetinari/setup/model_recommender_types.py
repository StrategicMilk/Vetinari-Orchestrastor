"""Shared setup model recommendation data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SetupModelRecommendation:
    """A recommended model for a given hardware profile.

    Attributes:
        name: Human-readable model name (e.g. "Llama 3.1 8B Q6_K").
        repo_id: HuggingFace repository ID for download.
        filename: Exact GGUF filename within the repository.
        size_gb: Approximate file size in gigabytes.
        quantization: Quantization level (e.g. "Q4_K_M", "Q6_K", "AWQ").
        parameter_count: Model parameter count (e.g. "7B", "14B").
        reason: Why this model is recommended for the detected hardware.
        is_primary: Whether this is the top recommendation for the tier.
        model_format: Serialization format — "gguf", "safetensors", "awq", or "gptq". Empty
            string means GGUF (the default llama-cpp format).
        backend: Inference backend — "nim", "vllm", or "llama_cpp". Empty string means
            llama_cpp (the default local backend).
        gpu_only: True when the model cannot run on CPU (AWQ/GPTQ via vLLM).
        best_for: Task categories this model excels at (e.g. "coding", "reasoning").
            Empty tuple means general-purpose with no specific strength.
    """

    name: str
    repo_id: str
    filename: str
    size_gb: float
    quantization: str
    parameter_count: str
    reason: str
    is_primary: bool = False
    model_format: str = "gguf"  # gguf, safetensors, awq, gptq
    backend: str = "llama_cpp"  # llama_cpp, vllm, nim
    gpu_only: bool = False  # True = must fit entirely in VRAM (no CPU offload)
    best_for: tuple[str, ...] = ()  # Task types this model excels at (coding, reasoning, etc.)

    def __repr__(self) -> str:
        return "ModelRecommendation(...)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dictionary.

        Returns:
            Dictionary representation of the recommendation.
        """
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "size_gb": self.size_gb,
            "quantization": self.quantization,
            "parameter_count": self.parameter_count,
            "reason": self.reason,
            "is_primary": self.is_primary,
            "model_format": self.model_format,
            "backend": self.backend,
            "gpu_only": self.gpu_only,
            "best_for": list(self.best_for),
        }


# ── VRAM-to-Model Matrix ─────────────────────────────────────────────────────
# Each tier defines models suitable for a VRAM range.  The first model in each
# tier is the primary recommendation.
