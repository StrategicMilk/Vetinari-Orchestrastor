"""Security pattern definitions for the quality skill."""

from __future__ import annotations

OWASP_TOP_10 = [
    "A01:2021 - Broken Access Control",
    "A02:2021 - Cryptographic Failures",
    "A03:2021 - Injection",
    "A04:2021 - Insecure Design",
    "A05:2021 - Security Misconfiguration",
    "A06:2021 - Vulnerable and Outdated Components",
    "A07:2021 - Identification and Authentication Failures",
    "A08:2021 - Software and Data Integrity Failures",
    "A09:2021 - Security Logging and Monitoring Failures",
    "A10:2021 - Server-Side Request Forgery (SSRF)",
]


def build_security_patterns() -> dict[str, dict[str, str]]:
    """Build vulnerability signatures searched for in reviewed code.

    Returns:
        Mapping from suspicious code fragment to severity and taxonomy metadata.
    """
    return {
        "yaml.load(": {
            "severity": "high",
            "cwe": "CWE-502",
            "owasp": "A02:2021",
            "desc": "Unsafe YAML load - use yaml.safe_load()",
        },
        "md5(": {
            "severity": "medium",
            "cwe": "CWE-328",
            "owasp": "A02:2021",
            "desc": "MD5 is cryptographically broken",
        },
        "debug=True": {
            "severity": "high",
            "cwe": "CWE-489",
            "owasp": "A05:2021",
            "desc": "Debug mode enabled in configuration",
        },
        "password =": {
            "severity": "critical",
            "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded password assignment",
        },
        "api_key =": {
            "severity": "critical",
            "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded API key assignment",
        },
        "secret =": {
            "severity": "critical",
            "cwe": "CWE-798",
            "owasp": "A07:2021",
            "desc": "Possible hardcoded secret assignment",
        },
    }
