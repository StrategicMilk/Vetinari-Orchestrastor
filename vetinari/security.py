"""Security Module for Vetinari.

Provides secret detection and filtering to prevent sensitive credentials
to prevent sensitive credentials persisting in memory backends.

Features:
- Regex-based detection of common secret patterns
- Keyword-based detection (api_key, password, token, etc.)
- Integration with CredentialManager for known credentials
- Safe sanitization of sensitive content

Usage:
    from vetinari.security import SecretScanner

    scanner = SecretScanner()
    sanitized = scanner.sanitize_content("My API key is sk-abc123xyz")
    # Result: "My API key is [REDACTED]"
"""

from __future__ import annotations

import ipaddress
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from vetinari.exceptions import SecurityError

logger = logging.getLogger(__name__)


@dataclass
class SecretPattern:
    """Represents a regex pattern for detecting secrets."""

    name: str
    pattern: str
    description: str


class SecretScanner:
    """Scanner for detecting and sanitizing sensitive information.

    Uses multiple detection strategies:
    1. Regex patterns (API keys, tokens, etc.)
    2. Keyword-based detection (fields containing 'password', 'token', etc.)
    3. Known credentials from CredentialManager
    """

    # Common secret patterns
    PATTERNS = [
        # OpenAI API keys (sk-* and sk-proj-*)
        # Minimum total length ~20 chars; after "sk-" we allow 8+ chars
        SecretPattern(
            name="openai_api_key",
            pattern=r"sk-(?:proj-)?[a-zA-Z0-9\-_]{8,}",
            description="OpenAI API key pattern",
        ),
        # GitHub tokens (ghp_*, ghu_*, ghs_*, gho_*, ghr_*)
        SecretPattern(
            name="github_token",
            pattern=r"gh[pousr]_[a-zA-Z0-9]{16,255}",
            description="GitHub personal access token",
        ),
        # AWS credentials (AKIA*)
        SecretPattern(name="aws_access_key", pattern=r"AKIA[0-9A-Z]{16}", description="AWS access key ID"),
        # Google API keys
        SecretPattern(name="google_api_key", pattern=r"AIza[0-9A-Za-z\-_]{35}", description="Google API key"),
        # Anthropic API keys (claude-* or sk-ant-*)
        SecretPattern(
            name="anthropic_api_key",
            pattern=r"(sk-ant-[a-zA-Z0-9_-]{20,}|claude-key-[a-zA-Z0-9]+)",
            description="Anthropic API key",
        ),
        # Cohere API keys
        SecretPattern(name="cohere_api_key", pattern=r"(co_[a-zA-Z0-9]{32})", description="Cohere API key"),
        # Gemini API keys
        SecretPattern(name="gemini_api_key", pattern=r"AIza[0-9A-Za-z\-_]{35}", description="Gemini API key"),
        # Generic bearer tokens
        SecretPattern(name="bearer_token", pattern=r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", description="HTTP Bearer token"),
        # Database connection strings (basic)
        SecretPattern(
            name="db_connection_string",
            pattern=r"(mongodb|postgresql|mysql|sqlite)://[^\s]+@[^\s]+",
            description="Database connection string",
        ),
        # JWT tokens (basic pattern)
        SecretPattern(
            name="jwt_token",
            pattern=r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
            description="JWT token",
        ),
        # Private SSH keys (basic)
        SecretPattern(
            name="ssh_private_key",
            pattern=r"-----BEGIN[^-]+-PRIVATE KEY-----",
            description="SSH private key",
        ),
        # Hex-encoded secrets (32+ chars of hex)
        SecretPattern(name="hex_secret", pattern=r"[a-f0-9]{32,}", description="Long hex string (possible secret)"),
        # Path traversal attempts (../ or ..\)
        SecretPattern(
            name="path_traversal",
            pattern=r"(?:\.\./|\.\.\\)+",
            description="Path traversal attempt",
        ),
    ]

    # Keywords that indicate sensitive fields
    SENSITIVE_KEYWORDS = [
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "api-key",
        "access_key",
        "private_key",
        "private-key",
        "privatekeyid",
        "auth",
        "credential",
        "credentials",
        "bearer",
        "refresh_token",
        "refreshtoken",
        "refresh-token",
        "oauth_token",
        "oauthtoken",
        "oauth-token",
        "encryption_key",
        "encryptionkey",
        "encryption-key",
        "signing_key",
        "signingkey",
        "signing-key",
        "hmac",
        "api_secret",
        "apisecret",
        "api-secret",
        "client_secret",
        "clientsecret",
        "client-secret",
        "webhook_secret",
        "webhooksecret",
        "webhook-secret",
        "stripe_key",
        "stripekey",
        "stripe-key",
        "aws_secret",
        "awssecret",
        "aws-secret",
        "connection_string",
        "connectionstring",
        "connection-string",
    ]

    def __init__(self, additional_patterns: list[SecretPattern] | None = None):
        """Build the scanner, compiling all regex patterns for efficient repeated scanning.

        Args:
            additional_patterns: Extra ``SecretPattern`` entries to append to the built-in list.
        """
        self.patterns = list(self.PATTERNS)
        if additional_patterns:
            self.patterns.extend(additional_patterns)

        # Compile regex patterns for performance
        self._compiled_patterns = {p.name: re.compile(p.pattern, re.IGNORECASE) for p in self.patterns}

        self._sensitive_keywords_lower = [kw.lower() for kw in self.SENSITIVE_KEYWORDS]

        logger.info("SecretScanner initialized with %s patterns", len(self.patterns))

    def scan(self, content: str, max_matches: int = 100) -> dict[str, list[str]]:
        """Scan content for secrets and return detected patterns.

        Args:
            content: Text to scan
            max_matches: Maximum number of matches to return per pattern

        Returns:
            Dictionary mapping pattern names to detected secrets
        """
        if not isinstance(content, str):
            return {}

        detected = {}

        for pattern_name, compiled_pattern in self._compiled_patterns.items():
            matches = compiled_pattern.findall(content)
            if matches:
                # Handle both groups and full matches
                if isinstance(matches[0], tuple):
                    matches = [m for group in matches for m in group if group]
                detected[pattern_name] = list(set(matches[:max_matches]))

        return detected

    def scan_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Scan a dictionary for secrets in both keys and values.

        Args:
            data: Dictionary to scan

        Returns:
            Dictionary with detected secrets (pattern name -> list of values)
        """
        detected = {}

        for key, value in data.items():
            # Check if the key itself is sensitive
            if self._is_sensitive_key(key):
                detected[f"key_{key}"] = [str(value)]

            # Scan the value
            if isinstance(value, str):
                value_scan = self.scan(value)
                detected.update(value_scan)
            elif isinstance(value, dict):
                nested_scan = self.scan_dict(value)
                detected.update(nested_scan)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        item_scan = self.scan(item)
                        detected.update(item_scan)
                    elif isinstance(item, dict):
                        nested_scan = self.scan_dict(item)
                        detected.update(nested_scan)

        return detected

    def sanitize(self, content: str, placeholder: str = "[REDACTED]") -> str:
        """Sanitize content by replacing detected secrets.

        Args:
            content: Text to sanitize
            placeholder: Replacement string for secrets

        Returns:
            Sanitized content with secrets replaced

        Raises:
            ValueError: If *content* is not a string.
        """
        if not isinstance(content, str):
            raise SecurityError(f"sanitize() requires str input, got {type(content).__name__}")

        sanitized = content

        # Apply all patterns except hex_secret (too broad, causes false positives
        # on the placeholder string itself)
        for pattern_name, compiled_pattern in self._compiled_patterns.items():
            if pattern_name == "hex_secret":
                continue
            sanitized = compiled_pattern.sub(placeholder, sanitized)

        return sanitized

    def sanitize_dict(self, data: dict[str, Any], placeholder: str = "[REDACTED]") -> dict[str, Any]:
        """Sanitize a dictionary by replacing detected secrets.

        Args:
            data: Dictionary to sanitize
            placeholder: Replacement string for secrets

        Returns:
            New dictionary with secrets replaced
        """
        result = {}

        for key, value in data.items():
            # If key is sensitive and value is a plain string, redact the value
            if self._is_sensitive_key(key) and isinstance(value, str):
                result[key] = placeholder  # noqa: VET034 — 'placeholder' is a parameter name, not a placeholder string
            elif isinstance(value, str):
                result[key] = self.sanitize(value, placeholder)
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value, placeholder)
            elif isinstance(value, (list, tuple)):
                sanitized_items = []
                for item in value:
                    if isinstance(item, str):
                        sanitized_item = self.sanitize(item, placeholder)
                        sanitized_items.append(sanitized_item)
                    elif isinstance(item, dict):
                        sanitized_items.append(self.sanitize_dict(item, placeholder))
                    else:
                        sanitized_items.append(item)
                result[key] = sanitized_items
            else:
                result[key] = value

        return result

    def sanitize_object(self, obj: Any, placeholder: str = "[REDACTED]") -> Any:
        """Sanitize an arbitrary object (string, dict, list, or dataclass).

        Args:
            obj: Object to sanitize
            placeholder: Replacement string for secrets

        Returns:
            Sanitized object
        """
        if isinstance(obj, str):
            return self.sanitize(obj, placeholder)
        if isinstance(obj, dict):
            return self.sanitize_dict(obj, placeholder)
        if isinstance(obj, (list, tuple)):
            return [self.sanitize_object(item, placeholder) for item in obj]
        if hasattr(obj, "__dict__"):
            # Handle dataclass or object with __dict__
            try:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(obj):
                    sanitized_dict = self.sanitize_dict(asdict(obj), placeholder)
                    return type(obj)(**sanitized_dict)
            except Exception as e:
                logger.warning("Could not sanitize dataclass: %s", e)

        return obj

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key indicates a sensitive field."""
        key_lower = key.lower().replace("-", "_")
        return any(kw in key_lower for kw in self._sensitive_keywords_lower)

    def add_pattern(self, pattern: SecretPattern) -> None:
        """Add a custom secret pattern."""
        self.patterns.append(pattern)
        self._compiled_patterns[pattern.name] = re.compile(pattern.pattern, re.IGNORECASE)
        logger.info("Added custom pattern: %s", pattern.name)


# Global singleton instance
_scanner: SecretScanner | None = None
_scanner_lock = threading.Lock()

# Hostnames that are known cloud metadata endpoints, container-host aliases, or
# loopback aliases that Python's ipaddress module does not parse as IPs —
# and therefore must be explicitly named here rather than caught by the IP range check.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({
    "localhost",  # Loopback alias — not parsed by ipaddress.ip_address()
    "metadata.google.internal",
    "169.254.169.254",  # AWS/GCP/Azure IMDS endpoint (also caught by link-local check)
    "metadata.internal",
    "host.docker.internal",  # Docker bridge alias — resolves to host on most platforms
})


def get_secret_scanner() -> SecretScanner:
    """Get or create the global secret scanner instance (thread-safe singleton).

    Returns:
        The shared SecretScanner instance.
    """
    global _scanner
    if _scanner is None:
        with _scanner_lock:
            if _scanner is None:
                _scanner = SecretScanner()
    return _scanner


def sanitize_for_memory(content: str) -> str:
    """Convenience function to sanitize content before storing in memory.

    Args:
        content: Content to sanitize

    Returns:
        Sanitized content
    """
    scanner = get_secret_scanner()
    return scanner.sanitize(content)


def sanitize_dict_for_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to sanitize a dictionary before storing in memory.

    Args:
        data: Dictionary to sanitize

    Returns:
        Sanitized dictionary
    """
    scanner = get_secret_scanner()
    return scanner.sanitize_dict(data)


def _normalize_hostname_to_ip(hostname: str) -> str:
    """Convert alternative IP representations to dotted-decimal for safe parsing.

    Python's ``ipaddress.ip_address()`` only accepts standard dotted-decimal (IPv4)
    or colon-hex (IPv6) notation.  Attackers can bypass naive checkers by encoding
    the loopback address as a decimal integer (2130706433), hex (0x7f000001), or
    octal-octet notation (0177.0.0.1) — all of which ``ipaddress`` would reject with
    ``ValueError``, causing the caller to treat them as DNS names and allow them.

    This helper detects each encoding and converts it to canonical form so the
    ``ipaddress`` module can apply its normal private/loopback/link-local checks.

    Args:
        hostname: The raw hostname extracted from the URL (lowercase, no port).

    Returns:
        A canonicalised hostname string ready for ``ipaddress.ip_address()``.
        Returns the original hostname unchanged if no alternative encoding is detected.

    Raises:
        SecurityError: If the hostname matches an alternative-IP encoding pattern but
            cannot be parsed (e.g. numeric value out of IPv4 range).
    """
    # Decimal integer IP: e.g. 2130706433 == 127.0.0.1
    # A pure decimal hostname with no dots is unambiguously an encoded IPv4 address.
    if hostname.isdigit():
        int_val = int(hostname)
        if 0 <= int_val <= 0xFFFFFFFF:
            return str(ipaddress.ip_address(int_val))
        raise SecurityError(f"Numeric hostname {hostname!r} is out of the valid IPv4 range")

    # Hex-encoded IPv4: e.g. 0x7f000001 == 127.0.0.1
    if hostname.lower().startswith("0x"):
        try:
            int_val = int(hostname, 16)
            return str(ipaddress.ip_address(int_val))
        except (ValueError, OverflowError) as exc:
            raise SecurityError(f"Hex hostname {hostname!r} could not be parsed as an IP address") from exc

    # Octal-octet notation: e.g. 0177.0.0.1 — each octet begins with a leading zero
    # and contains only digits (so it is ambiguously decimal or octal in some parsers).
    parts = hostname.split(".")
    if len(parts) == 4 and any(p.startswith("0") and len(p) > 1 and p.isdigit() for p in parts):
        try:
            decimal_parts = [str(int(p, 8)) for p in parts]
            return ".".join(decimal_parts)
        except (ValueError, OverflowError) as exc:
            raise SecurityError(f"Octal-notation hostname {hostname!r} could not be parsed as an IP address") from exc

    return hostname


def validate_url_no_ssrf(url: str) -> str:
    """Validate a URL is not targeting internal or private networks.

    Blocks private IPs, link-local addresses, loopback, and cloud metadata
    endpoints to prevent SSRF attacks.  Handles alternative IP encodings
    (decimal integer, hex, octal octet) that Python's ``ipaddress`` module
    does not natively parse, preventing bypass via encoding tricks.

    Only ``http`` and ``https`` schemes are permitted.

    Args:
        url: The URL to validate.

    Returns:
        The validated URL string (unchanged) when it passes all checks.

    Raises:
        SecurityError: If the URL targets a blocked address space, uses a
            disallowed scheme, or contains an unrecognised IP encoding.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SecurityError(f"URL scheme '{parsed.scheme}' not allowed — only http/https")

    hostname = parsed.hostname
    if not hostname:
        raise SecurityError("URL has no hostname")

    # Explicit hostname blocklist — catches aliases that ipaddress cannot parse.
    if hostname in _BLOCKED_HOSTNAMES:
        raise SecurityError(f"Hostname '{hostname}' is blocked")

    # Normalise alternative IP encodings before passing to ipaddress so that
    # decimal, hex, and octal representations are checked against private ranges.
    hostname_for_check = _normalize_hostname_to_ip(hostname)

    try:
        addr = ipaddress.ip_address(hostname_for_check)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise SecurityError(f"URL targets non-routable address: {hostname}")
    except ValueError:
        # ipaddress.ip_address() raised — hostname is a DNS name, not a raw IP.
        # This is the normal case for most public URLs; allow it.
        logger.debug("Hostname %r is a DNS name, not a raw IP — allowed", hostname)

    return url
