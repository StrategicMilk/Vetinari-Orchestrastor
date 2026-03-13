"""Speculative Decoding Support (C16).

====================================
Draft-verify pattern: small model drafts, large model verifies.

Uses a fast SLM to generate draft tokens, then a larger model to verify
and accept/reject them. Falls back to sequential decoding when
speculative decoding is not beneficial.

Requires LM Studio with vLLM backend or compatible API.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enabled: bool = False
    draft_model: str = ""  # e.g. "qwen2.5-coder-7b"
    verify_model: str = ""  # e.g. "qwen2.5-coder-32b"
    draft_tokens: int = 5  # tokens to draft before verification
    acceptance_threshold: float = 0.8  # min acceptance rate to keep using speculative
    fallback_after_failures: int = 3  # switch to sequential after N failures
    max_draft_attempts: int = 10  # max draft rounds per request


@dataclass
class SpeculativeResult:
    """Result of a speculative decoding run."""

    output: str
    total_tokens: int = 0
    draft_tokens_generated: int = 0
    draft_tokens_accepted: int = 0
    acceptance_rate: float = 0.0
    speedup_factor: float = 1.0
    used_speculative: bool = False
    duration_ms: float = 0.0


class SpeculativeDecoder:
    """Draft-verify speculative decoding engine.

    When enabled, uses a small model to draft tokens and a large model
    to verify them, achieving faster overall generation.
    """

    def __init__(self, config: SpeculativeConfig | None = None):
        self._config = config or SpeculativeConfig()
        self._consecutive_failures = 0
        self._total_requests = 0
        self._speculative_requests = 0
        self._total_speedup = 0.0

    @property
    def enabled(self) -> bool:
        return (
            self._config.enabled
            and bool(self._config.draft_model)
            and bool(self._config.verify_model)
            and self._consecutive_failures < self._config.fallback_after_failures
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> SpeculativeResult:
        """Generate text using speculative decoding if enabled.

        Falls back to standard generation if speculative decoding is
        disabled or unavailable.
        """
        self._total_requests += 1
        start = time.monotonic()

        if not self.enabled:
            # Standard sequential generation
            output = self._sequential_generate(
                self._config.verify_model or self._config.draft_model,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
            )
            duration = (time.monotonic() - start) * 1000
            return SpeculativeResult(
                output=output,
                total_tokens=len(output.split()),
                used_speculative=False,
                duration_ms=duration,
            )

        # Speculative: draft then verify
        self._speculative_requests += 1
        try:
            result = self._speculative_generate(
                prompt,
                system_prompt,
                max_tokens,
                temperature,
            )
            result.duration_ms = (time.monotonic() - start) * 1000
            if result.acceptance_rate < self._config.acceptance_threshold:
                self._consecutive_failures += 1
                logger.info(
                    "Speculative acceptance rate %.1f%% below threshold — failures=%d/%d",
                    result.acceptance_rate * 100,
                    self._consecutive_failures,
                    self._config.fallback_after_failures,
                )
            else:
                self._consecutive_failures = 0
            return result
        except Exception as e:
            logger.warning("Speculative decoding failed: %s — falling back", e)
            self._consecutive_failures += 1
            output = self._sequential_generate(
                self._config.verify_model,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
            )
            duration = (time.monotonic() - start) * 1000
            return SpeculativeResult(
                output=output,
                total_tokens=len(output.split()),
                used_speculative=False,
                duration_ms=duration,
            )

    def _speculative_generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> SpeculativeResult:
        """Draft-verify loop."""
        host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
        import requests

        generated = ""
        draft_total = 0
        accepted_total = 0
        rounds = 0

        while len(generated.split()) < max_tokens and rounds < self._config.max_draft_attempts:
            rounds += 1
            current_prompt = prompt + generated

            # Draft phase: get N tokens from small model
            draft_resp = requests.post(
                f"{host}/v1/completions",
                json={
                    "model": self._config.draft_model,
                    "prompt": current_prompt,
                    "max_tokens": self._config.draft_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            draft_resp.raise_for_status()
            draft_text = draft_resp.json()["choices"][0]["text"]
            draft_tokens = len(draft_text.split())
            draft_total += draft_tokens

            # Verify phase: check with large model
            verify_resp = requests.post(
                f"{host}/v1/completions",
                json={
                    "model": self._config.verify_model,
                    "prompt": current_prompt,
                    "max_tokens": draft_tokens,
                    "temperature": 0.0,  # greedy for verification
                },
                timeout=30,
            )
            verify_resp.raise_for_status()
            verify_text = verify_resp.json()["choices"][0]["text"]

            # Accept matching prefix
            draft_words = draft_text.split()
            verify_words = verify_text.split()
            accepted = 0
            for dw, vw in zip(draft_words, verify_words):
                if dw == vw:
                    accepted += 1
                else:
                    break
            accepted_total += accepted
            generated += " ".join(draft_words[: max(accepted, 1)]) + " "

            # Early stop if nothing accepted
            if accepted == 0:
                break

        acceptance_rate = accepted_total / max(draft_total, 1)
        return SpeculativeResult(
            output=generated.strip(),
            total_tokens=len(generated.split()),
            draft_tokens_generated=draft_total,
            draft_tokens_accepted=accepted_total,
            acceptance_rate=acceptance_rate,
            used_speculative=True,
        )

    def _sequential_generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Standard sequential generation via LM Studio."""
        try:
            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
            import requests

            resp = requests.post(
                f"{host}/v1/chat/completions",
                json={
                    "model": model_id or "default",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Sequential generation failed: %s", e)
            return ""

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_requests": self._total_requests,
            "speculative_requests": self._speculative_requests,
            "consecutive_failures": self._consecutive_failures,
            "config": {
                "draft_model": self._config.draft_model,
                "verify_model": self._config.verify_model,
                "draft_tokens": self._config.draft_tokens,
            },
        }


# ── Singleton ─────────────────────────────────────────────────────────

_decoder: SpeculativeDecoder | None = None


def get_speculative_decoder(config: SpeculativeConfig | None = None) -> SpeculativeDecoder:
    global _decoder
    if _decoder is None:
        _decoder = SpeculativeDecoder(config)
    return _decoder
