"""LLM Guard integration for ML-based input/output scanning.

Wraps the ``llm-guard`` library to provide ML-model-based safety scanning
as a complement to the regex-based checks in :mod:`vetinari.safety.guardrails`.
Gracefully degrades when ``llm-guard`` is not installed.

Configuration is loaded from ``config/llm_guard.yaml``.  Scanner instances
are constructed from a fixed allowlist — no dynamic module loading.

Usage::

    from vetinari.safety.llm_guard_scanner import get_llm_guard_scanner

    scanner = get_llm_guard_scanner()
    if scanner.available:
        result = scanner.scan_input("user prompt here")
        if not result.is_safe:
            print(result.findings)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_LLM_GUARD_AVAILABLE = False
try:
    import llm_guard  # noqa: F401

    _LLM_GUARD_AVAILABLE = True
    logger.debug("llm-guard available — ML-based scanners enabled")
except ImportError:
    logger.debug("llm-guard not installed — ML-based scanning disabled")

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "llm_guard.yaml"


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class ScanFinding:
    """A single finding from an LLM Guard scanner.

    Attributes:
        scanner_name: Name of the scanner that produced this finding.
        is_safe: Whether the scanned text passed this scanner.
        score: Confidence score (0.0-1.0).
    """

    scanner_name: str
    is_safe: bool
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {"scanner_name": self.scanner_name, "is_safe": self.is_safe, "score": round(self.score, 4)}


@dataclass
class ScanResult:
    """Aggregate result from multiple LLM Guard scanners.

    Attributes:
        is_safe: True when every scanner check passed.
        sanitized_text: Text after all sanitization passes.
        findings: Per-scanner findings.
        latency_ms: Total scan wall-clock time in milliseconds.
    """

    is_safe: bool
    sanitized_text: str
    findings: list[ScanFinding] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "is_safe": self.is_safe,
            "findings_count": len(self.findings),
            "unsafe_findings": [f.to_dict() for f in self.findings if not f.is_safe],
            "latency_ms": round(self.latency_ms, 2),
        }


# ── Scanner construction (fixed allowlist, no dynamic imports) ─────────


def _build_scanner(direction: str, name: str, settings: dict[str, Any]) -> Any | None:
    """Construct one scanner from the fixed allowlist.

    Uses direct imports from ``llm_guard.input_scanners`` or
    ``llm_guard.output_scanners`` — no ``importlib`` / dynamic loading.

    Args:
        direction: ``"input"`` or ``"output"``.
        name: Scanner identifier from config (e.g. ``"prompt_injection"``).
        settings: Scanner-specific YAML settings.

    Returns:
        Initialized scanner instance, or ``None`` if *name* is unrecognized.
    """
    threshold = settings.get("threshold", 0.5)
    use_onnx = settings.get("use_onnx", True)

    if direction == "input":
        return _build_input(name, settings, threshold, use_onnx)
    return _build_output(name, settings, threshold, use_onnx)


def _build_input(name: str, settings: dict[str, Any], threshold: float, use_onnx: bool) -> Any | None:
    """Construct an input scanner by name.

    Args:
        name: Scanner name from config.
        settings: Full scanner settings dict.
        threshold: Detection threshold.
        use_onnx: Whether to use ONNX runtime.

    Returns:
        Scanner instance or ``None``.
    """
    from llm_guard.input_scanners import (  # type: ignore[import-untyped]
        BanTopics,
        InvisibleText,
        PromptInjection,
        Secrets,
        Toxicity,
    )

    if name == "prompt_injection":
        return PromptInjection(threshold=threshold, use_onnx=use_onnx)
    if name == "ban_topics":
        return BanTopics(topics=settings.get("topics", []), threshold=threshold)
    if name == "toxicity":
        return Toxicity(threshold=threshold, use_onnx=use_onnx)
    if name == "secrets_detection":
        return Secrets(redact_mode="replace" if settings.get("redact", True) else "none")
    if name == "invisible_text":
        return InvisibleText()
    logger.warning("Unknown input scanner: %s", name)
    return None


def _build_output(name: str, settings: dict[str, Any], threshold: float, use_onnx: bool) -> Any | None:
    """Construct an output scanner by name.

    Args:
        name: Scanner name from config.
        settings: Full scanner settings dict.
        threshold: Detection threshold.
        use_onnx: Whether to use ONNX runtime.

    Returns:
        Scanner instance or ``None``.
    """
    from llm_guard.output_scanners import BanTopics, Bias, Sensitive, Toxicity  # type: ignore[import-untyped]

    if name == "ban_topics":
        return BanTopics(topics=settings.get("topics", []), threshold=threshold)
    if name == "sensitive":
        return Sensitive(entity_types=settings.get("entity_types", []), redact=settings.get("redact", True))
    if name == "toxicity":
        return Toxicity(threshold=threshold, use_onnx=use_onnx)
    if name == "bias":
        return Bias(threshold=threshold)
    logger.warning("Unknown output scanner: %s", name)
    return None


# ── LLMGuardScanner ───────────────────────────────────────────────────


class LLMGuardScanner:
    """ML-based safety scanner wrapping the llm-guard library.

    Loads scanner configuration from ``config/llm_guard.yaml`` and
    initializes the requested scanners.  Falls back to a no-op mode
    if the library is not installed.

    Use :func:`get_llm_guard_scanner` for the singleton instance.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._scanners: dict[str, list[Any]] = {"input": [], "output": []}
        self._available = _LLM_GUARD_AVAILABLE
        self._load_config()
        if self._available:
            self._init_scanners()

    @property
    def available(self) -> bool:
        """Whether llm-guard is installed and scanners are initialized."""
        return self._available

    def _load_config(self) -> None:
        """Load scanner configuration from the YAML config file."""
        if not self._config_path.exists():
            logger.info("LLM Guard config not found at %s", self._config_path)
            return
        try:
            with open(self._config_path, encoding="utf-8") as fh:
                self._config = yaml.safe_load(fh) or {}
        except Exception:
            logger.exception("Failed to load LLM Guard config")

    def _init_scanners(self) -> None:
        """Initialize scanners from config for both directions."""
        for direction in ("input", "output"):
            section = self._config.get(f"{direction}_scanners", {})
            for name, settings in section.items():
                if not settings.get("enabled", False):
                    continue
                try:
                    scanner = _build_scanner(direction, name, settings)
                    if scanner is not None:
                        self._scanners[direction].append(scanner)
                except Exception:
                    logger.exception("Failed to init %s scanner '%s'", direction, name)
        logger.info(
            "LLM Guard ready: %d input, %d output scanners",
            len(self._scanners["input"]),
            len(self._scanners["output"]),
        )

    # ── Scanning ───────────────────────────────────────────────────────

    def _run_scanners(
        self,
        direction: str,
        text: str,
        context: str,
        prompt: str = "",
    ) -> ScanResult:
        """Run scanners for one direction and return aggregated results.

        Args:
            direction: ``"input"`` or ``"output"``.
            text: Text to scan.
            context: Trust context for filtering.
            prompt: Original user prompt (output scanners only).

        Returns:
            ScanResult with verdict and per-scanner findings.
        """
        scanners = self._scanners[direction]
        if not self._available or not scanners:
            return ScanResult(is_safe=True, sanitized_text=text)

        allowed = self._context_filter(context, direction)
        max_len = self._config.get("performance", {}).get("max_text_length", 4096)
        sanitized = text[:max_len]
        start = time.monotonic()
        findings: list[ScanFinding] = []
        is_safe = True

        for scanner in scanners:
            cls_name = type(scanner).__name__
            if allowed and cls_name.lower() not in allowed:
                continue
            try:
                args = (prompt, sanitized) if direction == "output" else (sanitized,)
                sanitized, valid, score = scanner.scan(*args)
                findings.append(ScanFinding(scanner_name=cls_name, is_safe=valid, score=score))
                if not valid:
                    is_safe = False
                    logger.warning("LLM Guard %s '%s' flagged (score=%.3f)", direction, cls_name, score)
            except Exception:
                logger.exception("%s scanner '%s' error", direction.capitalize(), cls_name)

        elapsed = (time.monotonic() - start) * 1000
        return ScanResult(is_safe=is_safe, sanitized_text=sanitized, findings=findings, latency_ms=elapsed)

    def scan_input(self, text: str, context: str = "user_facing") -> ScanResult:
        """Scan user input through configured input scanners.

        Args:
            text: User input text.
            context: One of ``user_facing``, ``code_execution``, ``internal_agent``.

        Returns:
            ScanResult with safety verdict and findings.
        """
        return self._run_scanners("input", text, context)

    def scan_output(self, prompt: str, output: str, context: str = "user_facing") -> ScanResult:
        """Scan LLM output through configured output scanners.

        Args:
            prompt: Original input prompt (some scanners need it).
            output: LLM output text to scan.
            context: Trust context determining which scanners apply.

        Returns:
            ScanResult with safety verdict and findings.
        """
        return self._run_scanners("output", output, context, prompt=prompt)

    def _context_filter(self, context: str, direction: str) -> set[str]:
        """Return scanner names enabled for a context, or empty for all."""
        names = self._config.get("contexts", {}).get(context, {}).get(f"{direction}_scanners", [])
        return {n.lower() for n in names} if names else set()

    def get_stats(self) -> dict[str, Any]:
        """Return scanner statistics for dashboard display."""
        return {
            "available": self._available,
            "input_scanners": len(self._scanners["input"]),
            "output_scanners": len(self._scanners["output"]),
            "config_path": str(self._config_path),
        }


# ── Singleton ──────────────────────────────────────────────────────────

_scanner_instance: LLMGuardScanner | None = None
_scanner_lock = threading.Lock()


def get_llm_guard_scanner() -> LLMGuardScanner:
    """Return the process-global LLMGuardScanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        with _scanner_lock:
            if _scanner_instance is None:
                _scanner_instance = LLMGuardScanner()
    return _scanner_instance


def reset_llm_guard_scanner() -> None:
    """Reset the singleton (intended for testing)."""
    global _scanner_instance
    with _scanner_lock:
        _scanner_instance = None
