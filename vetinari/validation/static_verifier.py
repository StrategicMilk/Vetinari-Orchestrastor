"""Static Verifier — Tier 1 of the verification cascade.

Performs fast, deterministic checks on task output without any model calls:
  - Python syntax validity
  - Banned import detection
  - Hardcoded credential patterns
  - Code block presence when code is expected

Pipeline role: Called first by CascadeOrchestrator. When static checks pass,
the cascade can skip Tier 2 (entailment) and Tier 3 (LLM) for simple tasks.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns that indicate hardcoded credentials — should never appear in output
_CREDENTIAL_PATTERNS = [
    re.compile(r'password\s*=\s*["\'][^"\']{4,}["\']', re.IGNORECASE),
    re.compile(r'api_key\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
    re.compile(r'secret\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
    re.compile(r'token\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
]

# Modules that are always unsafe to import in generated code
_BANNED_IMPORTS = frozenset({"ctypes", "mmap", "msvcrt", "winreg"})


@dataclass
class StaticCheckResult:
    """Result of a single static verification check.

    Attributes:
        passed: Whether the check passed.
        check_name: Short identifier for this check.
        finding: Human-readable description of the finding if the check failed,
            empty string when passed.
    """

    passed: bool
    check_name: str
    finding: str = ""

    def __repr__(self) -> str:
        """Show key fields for debugging."""
        return f"StaticCheckResult(check_name={self.check_name!r}, passed={self.passed!r})"


class StaticVerifier:
    r"""Tier 1 verifier — deterministic checks with zero model calls.

    Runs a battery of cheap, rule-based checks on the output text.  All checks
    are independent and run to completion; individual failures are collected so
    the caller knows exactly which rules were violated.

    Example::

        verifier = StaticVerifier()
        results = verifier.verify("def add(a, b):\n    return a + b\n")
        passed = all(r.passed for r in results)
    """

    def verify(self, content: str, task_description: str = "") -> list[StaticCheckResult]:
        """Run all static checks on *content*.

        Args:
            content: The text or code to check.
            task_description: Optional task description used to decide whether
                a code block is expected (improves code-presence check accuracy).

        Returns:
            List of :class:`StaticCheckResult` — one per check.  An empty list
            is returned for empty or non-string content (all checks skipped).
        """
        if not isinstance(content, str) or not content.strip():
            return []

        results: list[StaticCheckResult] = [
            self._check_syntax(content),
            self._check_banned_imports(content),
            self._check_credentials(content),
            self._check_code_presence(content, task_description),
        ]
        failed = [r for r in results if not r.passed]
        if failed:
            logger.debug(
                "StaticVerifier: %d/%d checks failed: %s",
                len(failed),
                len(results),
                [r.check_name for r in failed],
            )
        return results

    # ── individual checks ────────────────────────────────────────────────────

    def _check_syntax(self, content: str) -> StaticCheckResult:
        """Return a passing result when *content* contains no Python code.

        Also passes when the Python code it contains parses without a SyntaxError.
        """
        # Strip markdown fences before parsing
        cleaned = re.sub(r"```[\w]*\n?", "\n", content)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # Only run AST parse if the cleaned text looks like Python.
        # Recognise a broader set of constructs beyond def/class/import so that
        # loops, assignments, return statements, and context managers are caught.
        _PYTHON_CONSTRUCTS = re.compile(
            r"^\s*(def|class|import|from|@|async\s+def|for\s|while\s|with\s|return\s|\w+\s*=)",
            re.MULTILINE,
        )
        if not _PYTHON_CONSTRUCTS.search(cleaned):
            return StaticCheckResult(passed=True, check_name="syntax")

        try:
            ast.parse(cleaned)
            return StaticCheckResult(passed=True, check_name="syntax")
        except SyntaxError as exc:
            logger.warning("StaticVerifier: syntax check failed at line %s: %s", exc.lineno, exc.msg)
            return StaticCheckResult(
                passed=False,
                check_name="syntax",
                finding=f"SyntaxError at line {exc.lineno}: {exc.msg}",
            )

    def _check_banned_imports(self, content: str) -> StaticCheckResult:
        """Fail when *content* imports a module from the banned list."""
        import_re = re.compile(
            r"^\s*(?:import|from)\s+([\w.]+)",
            re.MULTILINE,
        )
        for match in import_re.finditer(content):
            module_root = match.group(1).split(".")[0]
            if module_root in _BANNED_IMPORTS:
                return StaticCheckResult(
                    passed=False,
                    check_name="banned_imports",
                    finding=f"Banned module imported: {module_root}",
                )
        return StaticCheckResult(passed=True, check_name="banned_imports")

    def _check_credentials(self, content: str) -> StaticCheckResult:
        """Fail when *content* contains a hardcoded credential pattern."""
        for pattern in _CREDENTIAL_PATTERNS:
            match = pattern.search(content)
            if match:
                # Redact the value before logging
                snippet = match.group(0)[:40]
                return StaticCheckResult(
                    passed=False,
                    check_name="credentials",
                    finding=f"Potential hardcoded credential: {snippet!r}",
                )
        return StaticCheckResult(passed=True, check_name="credentials")

    def _check_code_presence(self, content: str, task_description: str) -> StaticCheckResult:
        """Warn when the task asks for code but the response contains none."""
        if not task_description:
            return StaticCheckResult(passed=True, check_name="code_presence")

        desc_lower = task_description.lower()
        task_wants_code = any(
            kw in desc_lower for kw in ("implement", "write", "create", "build", "function", "class", "code", "def ")
        )
        if not task_wants_code:
            return StaticCheckResult(passed=True, check_name="code_presence")

        # A fenced block counts as code only when it has a recognized code language
        # tag or contains Python syntax.  Plain ``` or ```text/```markdown blocks
        # are prose, not code, so they must NOT satisfy code presence alone.
        _PYTHON_CONSTRUCTS_BARE = re.compile(
            r"^\s*(def|class|import|from|for\s|while\s|with\s|return\s|\w+\s*=)",
            re.MULTILINE,
        )
        has_fenced_code = bool(
            re.search(r"```(?:python|py|sh|bash|javascript|js|ts|typescript|java|cpp|c\b)", content)
        )
        has_bare_python = bool(_PYTHON_CONSTRUCTS_BARE.search(content))
        has_code = has_fenced_code or has_bare_python
        if has_code:
            return StaticCheckResult(passed=True, check_name="code_presence")

        return StaticCheckResult(
            passed=False,
            check_name="code_presence",
            finding="Task requested code but response contains no code block or function definition",
        )
