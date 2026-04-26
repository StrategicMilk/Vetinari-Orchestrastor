"""Guardrails — vetinari.safety.guardrails.

Provides safety rails at user-facing trust boundaries using built-in
regex pattern checks.  Optionally augmented by LLM Guard when the
library is installed.

Usage
-----
    from vetinari.safety.guardrails import get_guardrails

    gr = get_guardrails()
    result = gr.check_input("user prompt here")
    if not result.allowed:
        logger.debug("Blocked: %s", result.violations)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from vetinari.utils.serialization import dataclass_to_dict

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

    def __repr__(self) -> str:
        return f"Violation(rail={self.rail!r}, severity={self.severity!r})"

    def to_dict(self) -> dict[str, Any]:
        """Converts violation fields to a JSON-serializable dict."""
        return dataclass_to_dict(self)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    allowed: bool
    content: str  # original or modified content
    violations: list[Violation] = field(default_factory=list)
    latency_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"GuardrailResult(allowed={self.allowed!r}, violations={len(self.violations)}, "
            f"latency_ms={self.latency_ms!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts guardrail result fields to a JSON-serializable dict, with nested violations expanded."""
        return dataclass_to_dict(self)


# ---------------------------------------------------------------------------
# Built-in pattern checks (no NeMo dependency required)
# ---------------------------------------------------------------------------

# Patterns that indicate prompt injection / jailbreak attempts
_JAILBREAK_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"(system|admin)\s*prompt\s*[:=]", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|the)\s+(rules|instructions|guidelines)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|security|content)\s+(filter|check|restriction)", re.IGNORECASE),
]

# Patterns that indicate sensitive data in output
_SENSITIVE_DATA_PATTERNS = [
    re.compile(r"(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style key
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub token
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
    re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----"),  # Private key
    # P6.2: Additional PII patterns
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # Email address
    re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),  # Credit card number
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone number
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"DefaultEndpointsProtocol=https;AccountName=\w+"),  # Azure connection string
    re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),  # JWT token
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}\b"),  # IP address with port
]


def _check_jailbreak(text: str) -> list[Violation]:
    """Scan text for prompt injection and jailbreak patterns.

    Args:
        text: Input text to check against jailbreak regex patterns.

    Returns:
        List of Violation instances for each matched jailbreak pattern.
    """
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
                ),
            )
    return violations


def _check_sensitive_data(text: str) -> list[Violation]:
    """Scan text for sensitive data patterns such as API keys, passwords, and PII.

    Args:
        text: Text to check against sensitive data regex patterns.

    Returns:
        List of Violation instances for each matched sensitive data pattern.
    """
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
                ),
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
            re.IGNORECASE,
        ),
        "high",
        "Hate speech — targeted violence against a group",
    ),
    (
        re.compile(
            r"\b(all|those|these)\s+(jews?|muslims?|christians?|blacks?|whites?|"
            r"latinos?|asians?|gays?|lesbians?|trans\w*|immigrants?|refugees?)\s+"
            r"(should|must|deserve\s+to)\s+(die|be\s+killed|be\s+eliminated|be\s+removed)\b",
            re.IGNORECASE,
        ),
        "high",
        "Hate speech — incitement to eliminate a group",
    ),
    (
        re.compile(
            r"\b(ni+g+[e]?r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg?[o0]t|tr[a4]nn[y]?)\b",
            re.IGNORECASE,
        ),
        "medium",
        "Hate speech — racial or homophobic slur",
    ),
    # Violence incitement markers
    (
        re.compile(
            r"\b(bomb|shoot|stab|attack|blow\s+up)\s+(the\s+)?(school|hospital|"
            r"church|mosque|synagogue|government|police|congress|parliament|crowd|crowd)\b",
            re.IGNORECASE,
        ),
        "high",
        "Violence incitement — attack on a specific target",
    ),
    (
        re.compile(
            r"\bhow\s+to\s+(make|build|construct|create)\s+a\s+(bomb|explosive|ied|"
            r"bioweapon|chemical\s+weapon|dirty\s+bomb)\b",
            re.IGNORECASE,
        ),
        "high",
        "Violence incitement — weapons manufacturing instructions",
    ),
    (
        re.compile(
            r"\b(mass\s+(shooting|stabbing|killing)|domestic\s+terrorism|lone\s+wolf\s+attack)\b",
            re.IGNORECASE,
        ),
        "high",
        "Violence incitement — mass violence planning language",
    ),
    # Self-harm markers
    (
        re.compile(
            r"\b(how\s+to|best\s+way\s+to|methods?\s+(for|of))\s+"
            r"(commit\s+suicide|kill\s+(myself|yourself)|end\s+my\s+life|self[\s-]?harm)\b",
            re.IGNORECASE,
        ),
        "high",
        "Self-harm — instructions for self-injury or suicide",
    ),
    (
        re.compile(
            r"\b(suicide|self[\s-]?harm|cut\s+(myself|yourself)|overdose)\s+"
            r"(methods?|instructions?|guide|tips?|ways?)\b",
            re.IGNORECASE,
        ),
        "high",
        "Self-harm — self-injury method request",
    ),
    (
        re.compile(
            r"\bi\s+(want\s+to|am\s+going\s+to|will)\s+(kill|hurt|harm)\s+(myself|me)\b",
            re.IGNORECASE,
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
                ),
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


class GuardrailsManager:
    """Built-in regex-based guardrails with optional LLM Guard augmentation.

    Enforces safety rules at user-facing trust boundaries using compiled
    regex pattern checks.  Optionally augmented by LLM Guard when the
    library is installed.  Singleton — use ``get_guardrails()`` to get
    the shared instance.
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

    # ------------------------------------------------------------------
    # Input checking
    # ------------------------------------------------------------------

    def check_input(self, text: str, context: str = RailContext.USER_FACING) -> GuardrailResult:
        """Check user input against safety rails.

        Applies jailbreak, toxic-content, and optional ML-based checks for all
        contexts.  The former INTERNAL_AGENT bypass has been removed so that
        inter-agent messages are subject to the same checks as user-facing input.

        Args:
            text: The user's input text.
            context: Rail context — determines which rails apply.  All contexts
                are now checked; the parameter is reserved for future
                context-specific tuning.

        Returns:
            GuardrailResult with allowed flag and any violations.
        """
        start = time.monotonic()

        # Built-in pattern checks (always run for every context)
        violations = _check_jailbreak(text)
        violations += _check_toxic(text)

        latency = (time.monotonic() - start) * 1000

        if violations:
            logger.warning("Input blocked: %d violation(s) detected", len(violations))
            try:
                from vetinari.audit import get_audit_logger

                get_audit_logger().log_guardrail_check(
                    check_type="input",
                    outcome="denied",
                    violations=[v.description for v in violations],
                )
            except Exception:
                logger.warning("Failed to log input-denied guardrail check")
            return GuardrailResult(
                allowed=False,
                content=text,
                violations=violations,
                latency_ms=latency,
            )

        # NeMo Guardrails (Colang) tier — runs after regex, before LLM Guard.
        # get_nemo_provider() returns None when nemoguardrails is not installed
        # OR when it was installed but failed to initialise.  The two cases are
        # distinguished by is_nemo_init_failed(): a broken install marks the
        # result as degraded; a clean missing-dep silently skips the tier.
        try:
            from vetinari.safety.nemo_provider import get_nemo_provider, is_nemo_init_failed

            nemo = get_nemo_provider()
            if nemo is None and is_nemo_init_failed():
                # NeMo was installed but the provider crashed on init — fail closed
                # rather than allowing the request through with only a degraded marker.
                logger.warning(
                    "NeMo Guardrails tier is unavailable — provider failed to initialise; "
                    "input blocked by fail-closed policy"
                )
                return GuardrailResult(
                    allowed=False,
                    content=text,
                    violations=[
                        Violation(
                            rail="nemo_colang",
                            severity="high",
                            description="NeMo Guardrails provider init failed — input blocked by fail-closed policy",
                            matched_pattern="",
                        )
                    ],
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            elif nemo is not None:
                nemo_result = nemo.check_input(text)
                if not nemo_result.allowed:
                    logger.warning(
                        "Input blocked by NeMo Guardrails: %d violation(s)",
                        len(nemo_result.violations),
                    )
                    try:
                        from vetinari.audit import get_audit_logger

                        get_audit_logger().log_guardrail_check(
                            check_type="input",
                            outcome="denied",
                            violations=[v.description for v in nemo_result.violations],
                        )
                    except Exception:
                        logger.warning("Failed to log NeMo input-denied guardrail check")
                    return GuardrailResult(
                        allowed=False,
                        content=text,
                        violations=nemo_result.violations,
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
        except ImportError:
            logger.debug("nemoguardrails not installed — NeMo tier skipped for input")
        except Exception:
            # Fail closed — unexpected errors in NeMo must block the input.
            logger.warning(
                "NeMo Guardrails check_input raised an unexpected error — input blocked as a precaution",
                exc_info=True,
            )
            return GuardrailResult(
                allowed=False,
                content=text,
                violations=[],
                latency_ms=(time.monotonic() - start) * 1000,
            )

        try:
            from vetinari.audit import get_audit_logger

            get_audit_logger().log_guardrail_check(
                check_type="input",
                outcome="allowed",
            )
        except Exception:
            logger.warning("Failed to log input-allowed guardrail check")
        return GuardrailResult(allowed=True, content=text, violations=violations, latency_ms=latency)

    # ------------------------------------------------------------------
    # Output checking
    # ------------------------------------------------------------------

    def check_output(self, text: str, context: str = RailContext.USER_FACING) -> GuardrailResult:
        """Check bot output against safety rails.

        CODE_EXECUTION output is exempt from toxic/content filtering because code
        output legitimately contains tokens that look like secrets (test keys, examples),
        but is NOT exempt from secrets scanning — actual credentials must be redacted even
        in code execution output.

        INTERNAL_AGENT output is subject to sensitive-data checks so that
        agent-to-agent messages cannot leak credentials between pipeline stages.

        Args:
            text: The bot's output text.
            context: Rail context — CODE_EXECUTION skips toxic/content checks but still
                runs secrets scanning; all other contexts run the full check suite.

        Returns:
            GuardrailResult with allowed flag and any violations.
        """
        # Code execution output: skip toxic/content filtering but still scan for secrets.
        # Real credentials must never leak through code output regardless of context.
        if context == RailContext.CODE_EXECUTION:
            from vetinari.security import get_secret_scanner

            scanner = get_secret_scanner()
            redacted = scanner.redact(text)
            if redacted != text:
                violations = [
                    Violation(
                        rail="sensitive_data",
                        severity="high",
                        description="Secrets detected in code execution output — redacted",
                        matched_pattern="",
                    )
                ]
                logger.warning("Secrets detected in CODE_EXECUTION output — redacted before returning")
                return GuardrailResult(allowed=True, content=redacted, violations=violations)
            return GuardrailResult(allowed=True, content=text)

        start = time.monotonic()

        violations = _check_sensitive_data(text)
        violations += _check_toxic(text)

        latency = (time.monotonic() - start) * 1000

        if violations:
            logger.warning("Output flagged: %d violation(s) — sensitive data detected", len(violations))
            try:
                from vetinari.audit import get_audit_logger

                get_audit_logger().log_guardrail_check(
                    check_type="output",
                    outcome="denied",
                    violations=[v.description for v in violations],
                )
            except Exception:
                logger.warning("Failed to log output-denied guardrail check")
            return GuardrailResult(
                allowed=False,
                content="[Content filtered for safety]",
                violations=violations,
                latency_ms=latency,
            )

        # NeMo Guardrails (Colang) tier — runs after regex, before LLM Guard.
        # When the provider failed to init, mark the result as degraded rather
        # than silently skipping the tier.
        try:
            from vetinari.safety.nemo_provider import get_nemo_provider, is_nemo_init_failed

            nemo = get_nemo_provider()
            if nemo is None and is_nemo_init_failed():
                # NeMo was installed but the provider crashed on init — fail closed
                # rather than allowing the output through with only a degraded marker.
                logger.warning(
                    "NeMo Guardrails tier is unavailable — provider failed to initialise; "
                    "output blocked by fail-closed policy"
                )
                return GuardrailResult(
                    allowed=False,
                    content="[Content blocked by NeMo Guardrails fail-closed policy]",
                    violations=[
                        Violation(
                            rail="nemo_colang",
                            severity="high",
                            description="NeMo Guardrails provider init failed — output blocked by fail-closed policy",
                            matched_pattern="",
                        )
                    ],
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            elif nemo is not None:
                nemo_result = nemo.check_output(text)
                if not nemo_result.allowed:
                    logger.warning(
                        "Output blocked by NeMo Guardrails: %d violation(s)",
                        len(nemo_result.violations),
                    )
                    try:
                        from vetinari.audit import get_audit_logger

                        get_audit_logger().log_guardrail_check(
                            check_type="output",
                            outcome="denied",
                            violations=[v.description for v in nemo_result.violations],
                        )
                    except Exception:
                        logger.warning("Failed to log NeMo output-denied guardrail check")
                    return GuardrailResult(
                        allowed=False,
                        content="[Content filtered by NeMo Guardrails]",
                        violations=nemo_result.violations,
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
        except ImportError:
            logger.debug("nemoguardrails not installed — NeMo tier skipped for output")
        except Exception:
            # Fail closed — unexpected errors in NeMo must block the output.
            logger.warning(
                "NeMo Guardrails check_output raised an unexpected error — output blocked as a precaution",
                exc_info=True,
            )
            return GuardrailResult(
                allowed=False,
                content="[Content filtering error — output blocked for safety]",
                violations=[],
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # Augment with ML-based LLM Guard scanner when available.
        try:
            from vetinari.safety.llm_guard_scanner import get_llm_guard_scanner

            scanner = get_llm_guard_scanner()
            if scanner.available:
                scan_result = scanner.scan_output(prompt="", output=text, context=context)
                if not scan_result.is_safe:
                    logger.warning(
                        "LLM Guard scanner flagged output (%d finding(s)) — output blocked",
                        len(scan_result.findings),
                    )
                    return GuardrailResult(
                        allowed=False,
                        content="[Content filtered by ML scanner]",
                        violations=[],
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
        except ImportError:
            logger.debug("llm-guard not installed — ML content scan skipped")
        except Exception:
            # Security checks fail closed — unexpected scanner errors block output
            logger.warning(
                "LLM Guard scanner raised an unexpected error — output blocked as a precaution",
                exc_info=True,
            )
            return GuardrailResult(
                allowed=False,
                content="[Content filtering error — output blocked for safety]",
                violations=[],
                latency_ms=(time.monotonic() - start) * 1000,
            )

        try:
            from vetinari.audit import get_audit_logger

            get_audit_logger().log_guardrail_check(
                check_type="output",
                outcome="allowed",
            )
        except Exception:
            logger.warning("Failed to log output-allowed guardrail check")
        return GuardrailResult(allowed=True, content=text, violations=violations, latency_ms=latency)

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
        """Return which rail categories apply for a given context type.

        Args:
            context_type: One of ``RailContext.USER_FACING``, ``INTERNAL_AGENT``,
                or ``CODE_EXECUTION``.

        Returns:
            List of rail category names applicable to the given context.
        """
        if context_type == RailContext.USER_FACING:
            return ["jailbreak", "toxic", "prompt_injection", "sensitive_data"]
        if context_type == RailContext.INTERNAL_AGENT:
            # Phase 6.42: INTERNAL_AGENT bypass removed — same rails as USER_FACING
            return ["jailbreak", "toxic", "prompt_injection", "sensitive_data"]
        if context_type == RailContext.CODE_EXECUTION:
            return ["jailbreak", "prompt_injection"]
        return []

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return introspection statistics about the guardrails configuration.

        Returns:
            Dictionary with pattern counts for each built-in rail category.
        """
        return {
            "builtin_jailbreak_patterns": len(_JAILBREAK_PATTERNS),
            "builtin_sensitive_patterns": len(_SENSITIVE_DATA_PATTERNS),
            "builtin_toxic_patterns": len(_TOXIC_PATTERNS),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def get_guardrails() -> GuardrailsManager:
    """Return the singleton GuardrailsManager instance.

    Returns:
        The shared GuardrailsManager, creating it on first call.
    """
    return GuardrailsManager()


def reset_guardrails() -> None:
    """Destroy the singleton GuardrailsManager so the next call recreates it."""
    with GuardrailsManager._class_lock:
        GuardrailsManager._instance = None


def guard_inference_input(prompt: str) -> str:
    """Check a prompt against guardrails before sending to inference.

    Calls ``GuardrailsManager.check_input()`` on the prompt. On policy
    violation, raises ``SecurityError``. On internal exception, fails
    CLOSED by raising ``SecurityError`` (never fails open).

    Args:
        prompt: The raw prompt text to validate.

    Returns:
        The prompt unchanged if it passes all checks.

    Raises:
        SecurityError: If the prompt violates a guardrail policy or if
            an internal error occurs (fail-closed semantics).
    """
    from vetinari.exceptions import SecurityError

    try:
        gm = get_guardrails()
        result = gm.check_input(prompt)
        if not result.allowed:
            reasons = ", ".join(v.description for v in result.violations) if result.violations else "policy violation"
            raise SecurityError(f"Input blocked by guardrails: {reasons}")
    except SecurityError:
        raise
    except Exception as exc:
        logger.warning("Guardrails check_input failed — fail-closed: %s", exc)
        raise SecurityError("Input blocked by guardrails (fail-closed on internal error)") from exc
    return prompt


def guard_inference_output(response: str) -> str:
    """Check an inference response against guardrails before returning it.

    Calls ``GuardrailsManager.check_output()`` on the response. On policy
    violation, raises ``SecurityError``. On internal exception, fails
    CLOSED by raising ``SecurityError`` (never fails open).

    Args:
        response: The raw inference response text to validate.

    Returns:
        The response unchanged if it passes all checks.

    Raises:
        SecurityError: If the response violates a guardrail policy or if
            an internal error occurs (fail-closed semantics).
    """
    from vetinari.exceptions import SecurityError

    try:
        gm = get_guardrails()
        result = gm.check_output(response)
        if not result.allowed:
            reasons = ", ".join(v.description for v in result.violations) if result.violations else "policy violation"
            raise SecurityError(f"Output blocked by guardrails: {reasons}")
    except SecurityError:
        raise
    except Exception as exc:
        logger.warning("Guardrails check_output failed — fail-closed: %s", exc)
        raise SecurityError("Output blocked by guardrails (fail-closed on internal error)") from exc
    return response
