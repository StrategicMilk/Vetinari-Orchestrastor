"""NeMo Guardrails Integration — vetinari.safety.guardrails.

Provides safety rails at user-facing trust boundaries.
Gracefully degrades if NeMo Guardrails is not installed.

Usage
-----
    from vetinari.safety.guardrails import get_guardrails

    gr = get_guardrails()
    result = gr.check_input("user prompt here")
    if not result.allowed:
        logger.debug("Blocked:", result.violations)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single guardrail violation."""

    rail: str  # e.g. "jailbreak", "toxic", "sensitive_data"
    severity: str  # "high", "medium", "low"
    description: str
    matched_pattern: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rail": self.rail,
            "severity": self.severity,
            "description": self.description,
            "matched_pattern": self.matched_pattern,
        }


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool
    content: str  # original or modified content
    violations: list[Violation] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "content": self.content,
            "violations": [v.to_dict() for v in self.violations],
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Built-in pattern checks (no NeMo dependency required)
# ---------------------------------------------------------------------------

# Patterns that indicate prompt injection / jailbreak attempts
_JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
    re.compile(r"(system|admin)\s*prompt\s*[:=]", re.I),
    re.compile(r"disregard\s+(your|all|the)\s+(rules|instructions|guidelines)", re.I),
    re.compile(r"bypass\s+(safety|security|content)\s+(filter|check|restriction)", re.I),
]

# Patterns that indicate sensitive data in output
_SENSITIVE_DATA_PATTERNS = [
    re.compile(r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+", re.I),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style key
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub token
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),  # Private key
    # P6.2: Additional PII patterns
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email address
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),  # Credit card number
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone number
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"DefaultEndpointsProtocol=https;AccountName=\w+"),  # Azure connection string
    re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),  # JWT token
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}\b"),  # IP address with port
]


def _check_jailbreak(text: str) -> list[Violation]:
    violations = []
    for pattern in _JAILBREAK_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                Violation(
                    rail="jailbreak",
                    severity="high",
                    description="Potential jailbreak/prompt injection detected",
                    matched_pattern=match.group(0),
                )
            )
    return violations


def _check_sensitive_data(text: str) -> list[Violation]:
    violations = []
    for pattern in _SENSITIVE_DATA_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                Violation(
                    rail="sensitive_data",
                    severity="high",
                    description="Potential sensitive data detected in output",
                    matched_pattern=match.group(0)[:20] + "...",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# P6.1: Toxic content patterns for input scanning
# ---------------------------------------------------------------------------

# Each entry is (compiled_pattern, severity, description)
_TOXIC_PATTERNS: list[tuple] = [
    # Hate speech markers
    (
        re.compile(
            r"\b(kill|murder|exterminate|genocide|slaughter)\s+(all\s+)?(the\s+)?"
            r"(jews?|muslims?|christians?|blacks?|whites?|latinos?|asians?|gays?|"
            r"lesbians?|trans\w*|immigrants?|refugees?)\b",
            re.I,
        ),
        "high",
        "Hate speech — targeted violence against a group",
    ),
    (
        re.compile(
            r"\b(all|those|these)\s+(jews?|muslims?|christians?|blacks?|whites?|"
            r"latinos?|asians?|gays?|lesbians?|trans\w*|immigrants?|refugees?)\s+"
            r"(should|must|deserve\s+to)\s+(die|be\s+killed|be\s+eliminated|be\s+removed)\b",
            re.I,
        ),
        "high",
        "Hate speech — incitement to eliminate a group",
    ),
    (
        re.compile(
            r"\b(ni+g+[e]?r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg?[o0]t|tr[a4]nn[y]?)\b",
            re.I,
        ),
        "medium",
        "Hate speech — racial or homophobic slur",
    ),
    # Violence incitement markers
    (
        re.compile(
            r"\b(bomb|shoot|stab|attack|blow\s+up)\s+(the\s+)?(school|hospital|"
            r"church|mosque|synagogue|government|police|congress|parliament|crowd|crowd)\b",
            re.I,
        ),
        "high",
        "Violence incitement — attack on a specific target",
    ),
    (
        re.compile(
            r"\bhow\s+to\s+(make|build|construct|create)\s+a\s+(bomb|explosive|ied|"
            r"bioweapon|chemical\s+weapon|dirty\s+bomb)\b",
            re.I,
        ),
        "high",
        "Violence incitement — weapons manufacturing instructions",
    ),
    (
        re.compile(
            r"\b(mass\s+(shooting|stabbing|killing)|domestic\s+terrorism|lone\s+wolf\s+attack)\b",
            re.I,
        ),
        "high",
        "Violence incitement — mass violence planning language",
    ),
    # Self-harm markers
    (
        re.compile(
            r"\b(how\s+to|best\s+way\s+to|methods?\s+(for|of))\s+"
            r"(commit\s+suicide|kill\s+(myself|yourself)|end\s+my\s+life|self[\s-]?harm)\b",
            re.I,
        ),
        "high",
        "Self-harm — instructions for self-injury or suicide",
    ),
    (
        re.compile(
            r"\b(suicide|self[\s-]?harm|cut\s+(myself|yourself)|overdose)\s+"
            r"(methods?|instructions?|guide|tips?|ways?)\b",
            re.I,
        ),
        "high",
        "Self-harm — self-injury method request",
    ),
    (
        re.compile(
            r"\bi\s+(want\s+to|am\s+going\s+to|will)\s+(kill|hurt|harm)\s+(myself|me)\b",
            re.I,
        ),
        "medium",
        "Self-harm — first-person expression of self-harm intent",
    ),
]


def _check_toxic(text: str) -> list[Violation]:
    """Check text for toxic content patterns.

    Args:
        text: Input text to scan.

    Returns:
        List of Violation instances, one per matched pattern.
    """
    violations = []
    for pattern, severity, description in _TOXIC_PATTERNS:
        match = pattern.search(text)
        if match:
            violations.append(
                Violation(
                    rail="toxic",
                    severity=severity,
                    description=description,
                    matched_pattern=match.group(0)[:40],
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Context types for selective rail application
# ---------------------------------------------------------------------------


class RailContext:
    """Determines which rails to apply based on context."""

    USER_FACING = "user_facing"  # Full input + output rails
    INTERNAL_AGENT = "internal_agent"  # No rails (performance)
    CODE_EXECUTION = "code_execution"  # Input rails only


# ---------------------------------------------------------------------------
# GuardrailsManager
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "guardrails",
)


class GuardrailsManager:
    """Manages NeMo Guardrails with graceful degradation.

    If NeMo is not installed, falls back to built-in regex pattern checks.
    Singleton — use ``get_guardrails()`` to get the shared instance.
    """

    _instance: GuardrailsManager | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> GuardrailsManager:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._nemo_rails = None
        self._nemo_available = False
        self._config_dir = _DEFAULT_CONFIG_DIR
        self._init_nemo()

    def _init_nemo(self) -> None:
        """Try to initialize NeMo Guardrails. Fail silently if not available."""
        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore

            config = RailsConfig.from_path(self._config_dir)
            self._nemo_rails = LLMRails(config)
            self._nemo_available = True
            logger.info("NeMo Guardrails initialized from %s", self._config_dir)
        except ImportError:
            logger.info("NeMo Guardrails not installed — using built-in pattern checks")
            self._nemo_available = False
        except Exception as e:
            logger.warning("NeMo Guardrails init failed: %s — using built-in checks", e)
            self._nemo_available = False

    @property
    def is_nemo_available(self) -> bool:
        return self._nemo_available

    # ------------------------------------------------------------------
    # Input checking
    # ------------------------------------------------------------------

    def check_input(self, text: str, context: str = RailContext.USER_FACING) -> GuardrailResult:
        """Check user input against safety rails.

        Args:
            text: The user's input text.
            context: Rail context — determines which rails apply.

        Returns:
            GuardrailResult with allowed flag and any violations.
        """
        if context == RailContext.INTERNAL_AGENT:
            return GuardrailResult(allowed=True, content=text)

        start = time.monotonic()

        # Built-in pattern checks (always run)
        violations = _check_jailbreak(text)
        violations += _check_toxic(text)

        latency = (time.monotonic() - start) * 1000

        if violations:
            logger.warning("Input blocked: %d violation(s) detected", len(violations))
            return GuardrailResult(
                allowed=False,
                content=text,
                violations=violations,
                latency_ms=latency,
            )

        return GuardrailResult(allowed=True, content=text, latency_ms=latency)

    # ------------------------------------------------------------------
    # Output checking
    # ------------------------------------------------------------------

    def check_output(self, text: str, context: str = RailContext.USER_FACING) -> GuardrailResult:
        """Check bot output against safety rails.

        Args:
            text: The bot's output text.
            context: Rail context — determines which rails apply.

        Returns:
            GuardrailResult with allowed flag and any violations.
        """
        if context in (RailContext.INTERNAL_AGENT, RailContext.CODE_EXECUTION):
            return GuardrailResult(allowed=True, content=text)

        start = time.monotonic()

        violations = _check_sensitive_data(text)

        latency = (time.monotonic() - start) * 1000

        if violations:
            logger.warning("Output flagged: %d violation(s) — sensitive data detected", len(violations))
            return GuardrailResult(
                allowed=False,
                content="[Content filtered for safety]",
                violations=violations,
                latency_ms=latency,
            )

        return GuardrailResult(allowed=True, content=text, latency_ms=latency)

    # ------------------------------------------------------------------
    # PII redaction
    # ------------------------------------------------------------------

    def redact_pii(self, text: str) -> str:
        """Replace detected PII with ``[REDACTED]`` instead of blocking.

        Scans text against all ``_SENSITIVE_DATA_PATTERNS`` and substitutes
        each match with ``[REDACTED]``.  Useful when content should be passed
        through but sanitised rather than hard-blocked.

        Args:
            text: The text to redact.

        Returns:
            The text with any matched PII replaced by ``[REDACTED]``.
        """
        with self._lock:
            result = text
            for pattern in _SENSITIVE_DATA_PATTERNS:
                result = pattern.sub("[REDACTED]", result)
            return result

    # ------------------------------------------------------------------
    # Convenience: check input and output in one call
    # ------------------------------------------------------------------

    def check_both(
        self,
        input_text: str,
        output_text: str,
        context: str = RailContext.USER_FACING,
    ) -> tuple:
        """Run input and output checks and return both results.

        Args:
            input_text: The user's input text.
            output_text: The bot's output text.
            context: Rail context applied to both checks.

        Returns:
            Tuple of (input_result, output_result) as ``GuardrailResult`` instances.
        """
        input_result = self.check_input(input_text, context=context)
        output_result = self.check_output(output_text, context=context)
        return input_result, output_result

    # ------------------------------------------------------------------
    # Context-based rail selection
    # ------------------------------------------------------------------

    def get_rails_for_context(self, context_type: str) -> list[str]:
        """Return which rail categories apply for a given context type."""
        if context_type == RailContext.USER_FACING:
            return ["jailbreak", "toxic", "prompt_injection", "sensitive_data"]
        elif context_type == RailContext.CODE_EXECUTION:
            return ["jailbreak", "prompt_injection"]
        else:
            return []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "nemo_available": self._nemo_available,
            "config_dir": self._config_dir,
            "builtin_jailbreak_patterns": len(_JAILBREAK_PATTERNS),
            "builtin_sensitive_patterns": len(_SENSITIVE_DATA_PATTERNS),
            "builtin_toxic_patterns": len(_TOXIC_PATTERNS),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def get_guardrails() -> GuardrailsManager:
    return GuardrailsManager()


def reset_guardrails() -> None:
    with GuardrailsManager._class_lock:
        GuardrailsManager._instance = None
