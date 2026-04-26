"""Tests for vetinari.validation.static_verifier.StaticVerifier (Tier 1 cascade)."""

from __future__ import annotations

import pytest

from vetinari.validation.static_verifier import StaticCheckResult, StaticVerifier


class TestSyntaxCheck:
    """_check_syntax identifies Python code with syntax errors."""

    def test_valid_python_passes(self) -> None:
        """Well-formed Python code passes the syntax check."""
        verifier = StaticVerifier()
        results = verifier.verify("def add(a, b):\n    return a + b\n")
        syntax = next(r for r in results if r.check_name == "syntax")
        assert syntax.passed is True
        assert syntax.finding == ""

    def test_syntax_error_fails(self) -> None:
        """Code with a SyntaxError fails the syntax check with a finding."""
        verifier = StaticVerifier()
        results = verifier.verify("def broken(\n    return None\n", task_description="implement something")
        syntax = next(r for r in results if r.check_name == "syntax")
        assert syntax.passed is False
        assert "SyntaxError" in syntax.finding or "syntax" in syntax.finding.lower()

    def test_prose_without_code_passes_syntax(self) -> None:
        """Plain prose text with no Python constructs passes the syntax check."""
        verifier = StaticVerifier()
        results = verifier.verify("Here is a description of what I did.")
        syntax = next(r for r in results if r.check_name == "syntax")
        assert syntax.passed is True

    def test_markdown_fenced_code_parsed(self) -> None:
        """Python code wrapped in markdown fences is parsed for syntax."""
        verifier = StaticVerifier()
        valid_fenced = "```python\ndef greet(name):\n    return f'Hello {name}'\n```"
        results = verifier.verify(valid_fenced)
        syntax = next(r for r in results if r.check_name == "syntax")
        assert syntax.passed is True


class TestBannedImportsCheck:
    """_check_banned_imports blocks known dangerous modules."""

    def test_ctypes_import_fails(self) -> None:
        """Importing ctypes triggers banned_imports failure."""
        verifier = StaticVerifier()
        results = verifier.verify("import ctypes\ndef hello(): pass\n")
        banned = next(r for r in results if r.check_name == "banned_imports")
        assert banned.passed is False
        assert "ctypes" in banned.finding

    def test_mmap_import_fails(self) -> None:
        """Importing mmap triggers banned_imports failure."""
        verifier = StaticVerifier()
        results = verifier.verify("import mmap\nx = 1\n")
        banned = next(r for r in results if r.check_name == "banned_imports")
        assert banned.passed is False

    def test_safe_imports_pass(self) -> None:
        """Standard library imports like os and re pass the check."""
        verifier = StaticVerifier()
        results = verifier.verify("import os\nimport re\nfrom pathlib import Path\n")
        banned = next(r for r in results if r.check_name == "banned_imports")
        assert banned.passed is True

    def test_from_import_also_checked(self) -> None:
        """'from ctypes import Structure' also triggers the check."""
        verifier = StaticVerifier()
        results = verifier.verify("from ctypes import Structure\n")
        banned = next(r for r in results if r.check_name == "banned_imports")
        assert banned.passed is False


class TestCredentialsCheck:
    """_check_credentials detects hardcoded secrets."""

    def test_hardcoded_password_fails(self) -> None:
        """A hardcoded password assignment triggers the check."""
        verifier = StaticVerifier()
        results = verifier.verify('password = "supersecret123"\ndef connect(): pass\n')
        creds = next(r for r in results if r.check_name == "credentials")
        assert creds.passed is False

    def test_hardcoded_api_key_fails(self) -> None:
        """A hardcoded api_key assignment triggers the check."""
        verifier = StaticVerifier()
        results = verifier.verify('api_key = "sk-1234567890abcdef"\n')
        creds = next(r for r in results if r.check_name == "credentials")
        assert creds.passed is False

    def test_no_credentials_passes(self) -> None:
        """Code with no credential patterns passes the check."""
        verifier = StaticVerifier()
        results = verifier.verify("def compute(x, y):\n    return x + y\n")
        creds = next(r for r in results if r.check_name == "credentials")
        assert creds.passed is True

    def test_env_var_password_not_flagged(self) -> None:
        """Reading a password from an env variable is not flagged."""
        verifier = StaticVerifier()
        results = verifier.verify("import os\npassword = os.environ.get('DB_PASSWORD')\n")
        creds = next(r for r in results if r.check_name == "credentials")
        assert creds.passed is True


class TestCodePresenceCheck:
    """_check_code_presence warns when code was expected but not provided."""

    def test_no_task_description_always_passes(self) -> None:
        """Without a task_description the code_presence check always passes."""
        verifier = StaticVerifier()
        results = verifier.verify("some prose response")
        presence = next(r for r in results if r.check_name == "code_presence")
        assert presence.passed is True

    def test_code_task_with_code_passes(self) -> None:
        """Task asking for code + response with a function passes."""
        verifier = StaticVerifier()
        results = verifier.verify(
            "def add(a, b):\n    return a + b\n",
            task_description="implement an add function",
        )
        presence = next(r for r in results if r.check_name == "code_presence")
        assert presence.passed is True

    def test_code_task_without_code_fails(self) -> None:
        """Task asking for code + prose-only response fails the check."""
        verifier = StaticVerifier()
        results = verifier.verify(
            "Here is my answer to your question about adding numbers.",
            task_description="implement an add function",
        )
        presence = next(r for r in results if r.check_name == "code_presence")
        assert presence.passed is False

    def test_non_code_task_not_checked(self) -> None:
        """A task that does not mention code does not trigger the check."""
        verifier = StaticVerifier()
        results = verifier.verify(
            "The answer is 42.",
            task_description="what is the answer",
        )
        presence = next(r for r in results if r.check_name == "code_presence")
        assert presence.passed is True


class TestVerifyReturnsAllChecks:
    """verify() always returns exactly four StaticCheckResult objects."""

    def test_returns_four_results(self) -> None:
        """verify() returns four check results for any non-empty input."""
        verifier = StaticVerifier()
        results = verifier.verify("x = 1\n")
        assert len(results) == 4

    def test_empty_content_returns_empty_list(self) -> None:
        """verify() returns an empty list for empty/whitespace-only content."""
        verifier = StaticVerifier()
        assert verifier.verify("") == []
        assert verifier.verify("   ") == []

    def test_all_results_are_static_check_result(self) -> None:
        """All returned objects are StaticCheckResult instances."""
        verifier = StaticVerifier()
        results = verifier.verify("def f(): pass\n")
        for r in results:
            assert isinstance(r, StaticCheckResult)

    def test_check_names_are_unique(self) -> None:
        """Each check has a distinct name."""
        verifier = StaticVerifier()
        results = verifier.verify("x = 1\n")
        names = [r.check_name for r in results]
        assert len(names) == len(set(names))
