"""Secrets detection filter — prevents sensitive data from entering training records.

Scans text for API keys, tokens, PEM blocks, and high-entropy strings before
the training data collector persists them.  This is the last line of defense
against accidentally fine-tuning a model on credentials.

This is step 0 of the training pipeline: **Filter** → Record → Export → Train.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, human-readable label)
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    (re.compile(r"(?:aws_secret_access_key|secret_key)\s*[:=]\s*[A-Za-z0-9/+=]{40}"), "aws_secret_key"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "github_pat_classic"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{82}"), "github_pat_fine_grained"),
    (re.compile(r"gho_[A-Za-z0-9]{36}"), "github_oauth_token"),
    (re.compile(r"ghs_[A-Za-z0-9]{36}"), "github_server_token"),
    (re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"), "pem_private_key"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai_api_key"),
    (re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"), "anthropic_api_key"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "google_api_key"),
    (re.compile(r"xox[bpoa]-[0-9A-Za-z\-]{10,}"), "slack_token"),
    (
        re.compile(
            r"(?:password|passwd|pwd|secret|token|api_key|apikey)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?", re.IGNORECASE
        ),
        "generic_secret_assignment",
    ),
    (re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"), "jwt_token"),
    (re.compile(r"DefaultEndpointsProtocol=https;AccountName=\w+"), "azure_connection_string"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
]

_BLOCKED_FILE_EXTENSIONS: frozenset[str] = frozenset({
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".htpasswd",
    ".netrc",
    ".pgpass",
    ".credentials",
    ".keystore",
})

_BLOCKED_FILENAMES: frozenset[str] = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "service-account.json",
    "secrets.yaml",
    "secrets.yml",
    "docker-compose.yml",
    "docker-compose.yaml",
})

_BLOCKED_NAME_SUBSTRINGS: tuple[str, ...] = (
    "credentials",
    "secret",
    "private_key",
    "api_key",
)

_HIGH_ENTROPY_THRESHOLD = 4.5
_MIN_TOKEN_LENGTH_FOR_ENTROPY = 20


@dataclass(frozen=True, slots=True)
class SecretDetection:
    """A single detected secret in training data."""

    pattern_label: str
    matched_text: str
    position: int

    def __repr__(self) -> str:
        return f"SecretDetection(label={self.pattern_label!r}, pos={self.position})"


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy in bits per character."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _find_high_entropy_tokens(text: str) -> list[SecretDetection]:
    """Find tokens with suspiciously high Shannon entropy."""
    detections: list[SecretDetection] = []
    for match in re.finditer(r"\S+", text):
        token = match.group()
        if len(token) >= _MIN_TOKEN_LENGTH_FOR_ENTROPY:
            entropy = _shannon_entropy(token)
            if entropy >= _HIGH_ENTROPY_THRESHOLD:
                detections.append(
                    SecretDetection(
                        pattern_label="high_entropy",
                        matched_text=token[:20] + "...",
                        position=match.start(),
                    )
                )
    return detections


def scan_text(text: str) -> list[SecretDetection]:
    """Scan text for secrets, credentials, and high-entropy strings.

    Returns:
        All SecretDetection findings in the text; empty list when nothing is found.
    """
    if not text:
        return []
    detections: list[SecretDetection] = []
    detections.extend(
        SecretDetection(
            pattern_label=label,
            matched_text=match.group(0)[:20] + "...",
            position=match.start(),
        )
        for pattern, label in _SECRET_PATTERNS
        for match in pattern.finditer(text)
    )
    detections.extend(_find_high_entropy_tokens(text))
    return detections


def is_blocked_file(filepath: str | Path) -> bool:
    """Check if a file path refers to a file that should never enter training data.

    Returns:
        True when the file has a blocked extension, name, or name substring.
    """
    p = Path(filepath)
    name_lower = p.name.lower()
    if p.suffix.lower() in _BLOCKED_FILE_EXTENSIONS:
        return True
    if name_lower in _BLOCKED_FILENAMES:
        return True
    return any(sub in name_lower for sub in _BLOCKED_NAME_SUBSTRINGS)


def redact_secrets(text: str) -> str:
    """Replace detected secrets with redaction placeholders.

    Applies two passes: first replaces all known credential patterns with
    ``[REDACTED]``, then locates any remaining high-entropy tokens (≥20 chars
    with Shannon entropy ≥4.5 bits/char) and replaces them with
    ``[HIGH_ENTROPY_REDACTED]``.  High-entropy detection supplements pattern
    matching — it will not catch patterns already replaced in the first pass,
    which is intentional to avoid double-substitution artefacts.

    Returns:
        Copy of the input text with all detected secrets substituted.
    """
    result = text
    for pattern, _label in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)

    # Second pass: replace any high-entropy tokens the pattern list missed.
    # Work right-to-left so that positional offsets stay valid as we substitute.
    detections = _find_high_entropy_tokens(result)
    if detections:
        # Sort descending by position so substitutions don't shift later offsets.
        for det in sorted(detections, key=lambda d: d.position, reverse=True):
            # Re-locate the full token at the recorded position rather than
            # relying on the truncated matched_text stored in SecretDetection.
            match = re.match(r"\S+", result[det.position :])
            if match:
                token = match.group()
                result = result[: det.position] + "[HIGH_ENTROPY_REDACTED]" + result[det.position + len(token) :]
    return result


def filter_training_record(
    prompt: str,
    response: str,
) -> tuple[bool, list[SecretDetection]]:
    """Check whether a training record is safe to persist.

    Args:
        prompt: The prompt text submitted for training.
        response: The model response text to validate.

    Returns:
        ``(True, [])`` when no secrets are found; ``(False, detections)`` listing each finding
        when the record must be blocked.
    """
    detections = scan_text(prompt) + scan_text(response)
    if detections:
        logger.warning(
            "[SecretsFilter] Blocked training record: %d secret(s) detected — %s",
            len(detections),
            ", ".join(d.pattern_label for d in detections[:5]),
        )
        return False, detections
    return True, []
