"""Tests hardening the secrets filter against high-entropy tokens and tuple traversal.

Covers two previously unguarded paths:
1. ``redact_secrets()`` in ``vetinari.learning.secrets_filter`` now runs a
   high-entropy pass after pattern substitution.
2. ``SecretScanner.sanitize_dict()`` in ``vetinari.security`` now recurses into
   tuple values instead of returning them verbatim.
"""

from __future__ import annotations

import pytest

from vetinari.learning.secrets_filter import redact_secrets
from vetinari.security import SecretScanner, sanitize_dict_for_memory

# A deterministic token with guaranteed Shannon entropy > 4.5 bits/char.
# Using 32 unique mixed-case alphanumeric characters each appearing exactly once
# gives entropy = log2(32) = 5.0 bits/char — well above the 4.5 threshold.
# This avoids test flakiness from randomly-generated tokens that may fall below
# the threshold by chance.
_HIGH_ENTROPY_TOKEN = "xK9mZ2nQ8pL4rV7wA1sT6yH3cF0uE5jB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scanner() -> SecretScanner:
    """Return a fresh SecretScanner for each test to avoid singleton state."""
    return SecretScanner()


# ---------------------------------------------------------------------------
# 1. redact_secrets() redacts high-entropy tokens
# ---------------------------------------------------------------------------


class TestRedactSecretsHighEntropy:
    """redact_secrets() must redact high-entropy tokens, not just pattern matches."""

    def test_redacts_high_entropy_token(self) -> None:
        """A 32-char token with all unique characters has entropy=5.0 bits/char,
        well above the 4.5 threshold — it must be redacted by the entropy pass."""
        token = _HIGH_ENTROPY_TOKEN
        text = f"Authorization header value: {token}"
        result = redact_secrets(text)

        assert token not in result, f"High-entropy token was NOT redacted.\nInput:  {text!r}\nOutput: {result!r}"
        assert "[HIGH_ENTROPY_REDACTED]" in result, (
            f"Expected [HIGH_ENTROPY_REDACTED] marker in output, got: {result!r}"
        )

    def test_does_not_redact_short_tokens(self) -> None:
        """Tokens shorter than 20 chars must never be flagged as high-entropy."""
        text = "Hello world from test"
        result = redact_secrets(text)
        assert result == text, f"Short plain text was mutated: {result!r}"

    def test_does_not_redact_low_entropy_long_token(self) -> None:
        """A long token made of repeated characters has near-zero entropy."""
        text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 42 chars, 1 unique symbol
        result = redact_secrets(text)
        # Low entropy — should NOT be marked as high-entropy
        assert "[HIGH_ENTROPY_REDACTED]" not in result, f"Low-entropy token was incorrectly redacted: {result!r}"

    def test_pattern_match_still_works_alongside_entropy(self) -> None:
        """Pattern-matched secrets must still be caught after high-entropy pass added."""
        aws_key = "AKIAIOSFODNN7EXAMPLE"  # canonical AWS access key test vector
        text = f"key={aws_key}"
        result = redact_secrets(text)
        assert aws_key not in result, "AWS access key was not redacted by pattern match"
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# 2. SecretScanner.redact() catches high-entropy tokens
# ---------------------------------------------------------------------------


class TestSecretScannerRedactHighEntropy:
    """SecretScanner.redact() delegates to redact_secrets() — must inherit entropy fix."""

    def test_redact_catches_high_entropy_token(self) -> None:
        """SecretScanner.redact() must redact a deterministic high-entropy token
        (entropy=5.0 bits/char, 32 unique characters)."""
        scanner = _make_scanner()
        token = _HIGH_ENTROPY_TOKEN
        text = f"Bearer {token}"
        result = scanner.redact(text)

        assert token not in result, (
            f"High-entropy token survived SecretScanner.redact().\nInput:  {text!r}\nOutput: {result!r}"
        )

    def test_redact_known_patterns_unchanged(self) -> None:
        """Pattern-based detection in SecretScanner.redact() must still work."""
        scanner = _make_scanner()
        github_token = "ghp_" + "A" * 36
        result = scanner.redact(f"token={github_token}")
        assert github_token not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# 3. sanitize_dict() redacts secrets inside tuples
# ---------------------------------------------------------------------------


class TestSanitizeDictTuples:
    """sanitize_dict() must traverse tuple values, not return them verbatim."""

    def test_redacts_string_inside_tuple(self) -> None:
        """A string secret nested inside a tuple value must be redacted."""
        scanner = _make_scanner()
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        data = {"metadata": (f"key={aws_key}", "safe value")}
        result = scanner.sanitize_dict(data)

        assert isinstance(result["metadata"], tuple), "tuple type must be preserved"
        assert aws_key not in result["metadata"][0], f"AWS key survived tuple traversal: {result['metadata']!r}"
        assert "[REDACTED]" in result["metadata"][0]
        assert result["metadata"][1] == "safe value"

    def test_redacts_dict_inside_tuple(self) -> None:
        """A dict nested inside a tuple must be recursively sanitized.

        The sensitive inner dict has its secret redacted; the non-sensitive
        inner dict is returned with its original value intact.
        """
        scanner = _make_scanner()
        data = {
            "items": (
                {"password": "super_secret_password"},
                {"description": "harmless text"},
            )
        }
        result = scanner.sanitize_dict(data)

        assert isinstance(result["items"], tuple)
        assert result["items"][0]["password"] == "[REDACTED]"
        assert result["items"][1]["description"] == "harmless text"

    def test_preserves_non_string_tuple_elements(self) -> None:
        """Non-string, non-dict tuple elements must be passed through unchanged."""
        scanner = _make_scanner()
        data = {"counts": (1, 2, 3)}
        result = scanner.sanitize_dict(data)
        assert result["counts"] == (1, 2, 3)

    def test_sensitive_key_takes_priority_over_tuple(self) -> None:
        """If the key itself is sensitive, the entire value is [REDACTED] regardless."""
        scanner = _make_scanner()
        data = {"secret": ("value_a", "value_b")}
        result = scanner.sanitize_dict(data)
        assert result["secret"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# 4. sanitize_dict_for_memory() redacts tuple-nested secrets
# ---------------------------------------------------------------------------


class TestSanitizeDictForMemoryTuples:
    """sanitize_dict_for_memory() must propagate tuple traversal fix."""

    def test_sanitize_dict_for_memory_tuple_nested_secret(self) -> None:
        """Public API sanitize_dict_for_memory() must redact tuple-nested secrets."""
        aws_key = "AKIAIOSFODNN7EXAMPLE"
        data = {"headers": (f"Authorization: {aws_key}", "Content-Type: application/json")}
        result = sanitize_dict_for_memory(data)

        assert isinstance(result["headers"], tuple)
        assert aws_key not in result["headers"][0], (
            f"AWS key survived sanitize_dict_for_memory() tuple traversal: {result['headers']!r}"
        )


# ---------------------------------------------------------------------------
# 5. Known patterns still redacted (regression guard)
# ---------------------------------------------------------------------------


class TestKnownPatternRedaction:
    """Ensure no known-pattern regression was introduced by the entropy changes."""

    @pytest.mark.parametrize(
        "label,text",
        [
            ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
            ("github_pat_classic", "ghp_" + "A" * 36),
            ("openai_api_key", "sk-" + "x" * 30),
            (
                "jwt_token",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            ),
        ],
    )
    def test_pattern_redacted_by_redact_secrets(self, label: str, text: str) -> None:
        """``redact_secrets()`` must redact the given known-credential pattern."""
        result = redact_secrets(text)
        assert text not in result, f"{label} was NOT redacted.\nInput:  {text!r}\nOutput: {result!r}"

    @pytest.mark.parametrize(
        "label,text",
        [
            ("aws_access_key", "AKIAIOSFODNN7EXAMPLE"),
            ("github_pat_classic", "ghp_" + "A" * 36),
        ],
    )
    def test_pattern_redacted_by_scanner_redact(self, label: str, text: str) -> None:
        """``SecretScanner.redact()`` must redact the given known-credential pattern."""
        scanner = _make_scanner()
        result = scanner.redact(text)
        assert text not in result, (
            f"{label} was NOT redacted by SecretScanner.redact().\nInput:  {text!r}\nOutput: {result!r}"
        )
