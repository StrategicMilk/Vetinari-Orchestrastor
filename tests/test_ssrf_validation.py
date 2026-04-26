"""Tests for SSRF validation in vetinari.security.

Covers standard private/loopback addresses, alternative IP encodings that
Python's ipaddress module does not natively parse (decimal integer, hex,
octal-octet), and the explicit hostname blocklist.  Both the legacy
``vetinari.security`` (flat module) and the package-form
``vetinari.security`` (via ``__init__``) expose the same public API so
tests import from the package path only.
"""

from __future__ import annotations

import pytest

from vetinari.exceptions import SecurityError
from vetinari.security import validate_url_no_ssrf

# ---------------------------------------------------------------------------
# Blocked URLs — must raise SecurityError
# ---------------------------------------------------------------------------


class TestBlockedUrls:
    """validate_url_no_ssrf must block all known SSRF vectors."""

    @pytest.mark.parametrize(
        "url",
        [
            # Explicit blocklist entries
            "http://localhost/",
            "http://localhost:8080/api",
            "http://host.docker.internal/metadata",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.internal/",
            # Standard loopback
            "http://127.0.0.1/",
            "http://127.0.0.1:9200/",
            # Private ranges (RFC 1918)
            "http://10.0.0.1/",
            "http://10.255.255.255/",
            "http://172.16.0.1/",
            "http://172.31.255.255/",
            "http://192.168.1.1/",
            "http://192.168.255.254/",
            # Link-local
            "http://169.254.0.1/",
            # IPv6 loopback
            "http://[::1]/",
            # Decimal integer encoding of 127.0.0.1 == 2130706433
            "http://2130706433/",
            # Hex encoding of 127.0.0.1 == 0x7f000001
            "http://0x7f000001/",
            # Octal-octet notation of 127.0.0.1
            "http://0177.0.0.1/",
            # Decimal integer encoding of a private address 10.0.0.1 == 167772161
            "http://167772161/",
        ],
    )
    def test_blocked_url_raises(self, url: str) -> None:
        """Every SSRF vector must raise SecurityError, not silently pass."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf(url)


# ---------------------------------------------------------------------------
# Allowed URLs — must pass without raising
# ---------------------------------------------------------------------------


class TestAllowedUrls:
    """validate_url_no_ssrf must allow legitimate public URLs."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/",
            "https://example.com/path?q=1",
            "http://8.8.8.8/",
            "http://1.1.1.1/",
            "https://api.openai.com/v1/chat",
            "https://huggingface.co/models",
        ],
    )
    def test_allowed_url_returns_unchanged(self, url: str) -> None:
        """Public URLs must be returned unchanged (no SecurityError)."""
        result = validate_url_no_ssrf(url)
        assert result == url


# ---------------------------------------------------------------------------
# Scheme enforcement
# ---------------------------------------------------------------------------


class TestSchemeEnforcement:
    """Only http and https schemes are permitted."""

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/file",
            "file:///etc/passwd",
            "gopher://example.com/",
            "dict://example.com/",
        ],
    )
    def test_disallowed_scheme_raises(self, url: str) -> None:
        """Non-http/https schemes must raise SecurityError."""
        with pytest.raises(SecurityError, match="not allowed"):
            validate_url_no_ssrf(url)


# ---------------------------------------------------------------------------
# Edge-case inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that must not silently pass."""

    def test_no_hostname_raises(self) -> None:
        """A URL with no hostname must raise SecurityError."""
        with pytest.raises(SecurityError, match="no hostname"):
            validate_url_no_ssrf("http:///path")

    def test_decimal_loopback_exact_value(self) -> None:
        """2130706433 is 127.0.0.1 in decimal — must be blocked."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf("http://2130706433/")

    def test_hex_loopback_lowercase(self) -> None:
        """0x7f000001 is 127.0.0.1 in hex — must be blocked."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf("http://0x7f000001/")

    def test_hex_loopback_uppercase(self) -> None:
        """0X7F000001 (uppercase) is also a hex IP — must be blocked."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf("http://0X7F000001/")

    def test_octal_loopback(self) -> None:
        """0177.0.0.1 is 127.0.0.1 in octal-octet — must be blocked."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf("http://0177.0.0.1/")

    def test_decimal_private_10_block(self) -> None:
        """167772161 is 10.0.0.1 in decimal — private, must be blocked."""
        with pytest.raises(SecurityError):
            validate_url_no_ssrf("http://167772161/")
