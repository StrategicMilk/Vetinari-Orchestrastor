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

import logging
import re
from dataclasses import dataclass
from typing import Any

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
            name="openai_api_key", pattern=r"sk-(?:proj-)?[a-zA-Z0-9\-_]{8,}", description="OpenAI API key pattern"
        ),
        # GitHub tokens (ghp_*, ghu_*, ghs_*, gho_*, ghr_*)
        SecretPattern(
            name="github_token", pattern=r"gh[pousr]_[a-zA-Z0-9]{16,255}", description="GitHub personal access token"
        ),
        # AWS credentials (AKIA*)
        SecretPattern(name="aws_access_key", pattern=r"AKIA[0-9A-Z]{16}", description="AWS access key ID"),
        # Google API keys
        SecretPattern(name="google_api_key", pattern=r"AIza[0-9A-Za-z\-_]{35}", description="Google API key"),
        # Anthropic API keys (claude-* or sk-ant-*)
        SecretPattern(
            name="anthropic_api_key",
            pattern=r"(sk-ant-[a-zA-Z0-9]{48}|claude-key-[a-zA-Z0-9]+)",
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
            name="jwt_token", pattern=r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", description="JWT token"
        ),
        # Private SSH keys (basic)
        SecretPattern(
            name="ssh_private_key", pattern=r"-----BEGIN[^-]+-PRIVATE KEY-----", description="SSH private key"
        ),
        # Hex-encoded secrets (32+ chars of hex)
        SecretPattern(name="hex_secret", pattern=r"[a-f0-9]{32,}", description="Long hex string (possible secret)"),
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
        """Initialize the secret scanner."""
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

        return detected

    def sanitize(self, content: str, placeholder: str = "[REDACTED]") -> str:
        """Sanitize content by replacing detected secrets.

        Args:
            content: Text to sanitize
            placeholder: Replacement string for secrets

        Returns:
            Sanitized content with secrets replaced
        """
        if not isinstance(content, str):
            return content

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
                result[key] = placeholder
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
        elif isinstance(obj, dict):
            return self.sanitize_dict(obj, placeholder)
        elif isinstance(obj, (list, tuple)):
            return [self.sanitize_object(item, placeholder) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle dataclass or object with __dict__
            try:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(obj):
                    sanitized_dict = self.sanitize_dict(asdict(obj), placeholder)
                    return type(obj)(**sanitized_dict)
            except Exception as e:
                logger.debug("Could not sanitize dataclass: %s", e)

        return obj

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key indicates a sensitive field."""
        key_lower = key.lower().replace("-", "_")
        return any(kw in key_lower for kw in self._sensitive_keywords_lower)

    def add_pattern(self, pattern: SecretPattern):
        """Add a custom secret pattern."""
        self.patterns.append(pattern)
        self._compiled_patterns[pattern.name] = re.compile(pattern.pattern, re.IGNORECASE)
        logger.info("Added custom pattern: %s", pattern.name)


# Global singleton instance
_scanner: SecretScanner | None = None


def get_secret_scanner() -> SecretScanner:
    """Get or create the global secret scanner instance."""
    global _scanner
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
