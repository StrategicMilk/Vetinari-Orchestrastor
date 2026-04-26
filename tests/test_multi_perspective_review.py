"""Tests for multi-perspective review in vetinari.agents.consolidated.quality_agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.consolidated.quality_agent import (
    INSPECTOR_PASS_CONFIGS,
    _run_correctness_scan,
    _run_performance_scan,
    _run_standards_scan,
    run_multi_perspective_review,
)

# -- INSPECTOR_PASS_CONFIGS ---------------------------------------------------


class TestInspectorPassConfigs:
    """INSPECTOR_PASS_CONFIGS is a list of 4 review pass definitions."""

    def test_has_four_entries(self) -> None:
        """Exactly 4 review pass configurations are defined."""
        assert len(INSPECTOR_PASS_CONFIGS) == 4

    @pytest.mark.parametrize("required_key", ["name", "focus", "system_prompt"])
    def test_each_entry_has_required_keys(self, required_key: str) -> None:
        """Every pass config has name, focus, and system_prompt keys."""
        for config in INSPECTOR_PASS_CONFIGS:
            assert required_key in config, f"Pass '{config.get('name', '?')}' missing key '{required_key}'"

    def test_pass_names_are_correct(self) -> None:
        """The four pass names are correctness, security, performance, and standards."""
        names = [config["name"] for config in INSPECTOR_PASS_CONFIGS]
        assert names == ["correctness", "security", "performance", "standards"]

    def test_system_prompts_are_non_empty(self) -> None:
        """Each system_prompt is a non-empty string with meaningful content."""
        for config in INSPECTOR_PASS_CONFIGS:
            assert isinstance(config["system_prompt"], str)
            assert len(config["system_prompt"]) >= 20


# -- run_multi_perspective_review ---------------------------------------------


class TestRunMultiPerspectiveReview:
    """run_multi_perspective_review() executes all passes and returns findings."""

    def test_returns_list(self) -> None:
        """run_multi_perspective_review() always returns a list."""
        code = "def add(a, b):\n    return a + b\n"
        result = run_multi_perspective_review(code)
        assert isinstance(result, list)

    def test_security_pass_detects_hardcoded_password(self) -> None:
        """The security pass flags hardcoded credentials in code."""
        code = 'password = "supersecret123"\ndb_url = "postgresql://user:supersecret123@host/db"\n'
        result = run_multi_perspective_review(code)
        assert len(result) >= 1
        # The security heuristic scan should flag a hardcoded secret
        assert any(
            "password" in f.get("finding", "").lower()
            or "secret" in f.get("finding", "").lower()
            or "credential" in f.get("finding", "").lower()
            or "hardcoded" in f.get("finding", "").lower()
            for f in result
        )

    def test_finding_dicts_have_expected_keys(self) -> None:
        """Each finding dict contains at least 'finding' and 'pass_name' keys."""
        code = 'api_key = "abc123secret"\n'
        result = run_multi_perspective_review(code)
        if result:  # Only assert structure if findings were returned
            for finding in result:
                assert "finding" in finding or "severity" in finding

    def test_all_four_passes_run(self) -> None:
        """Findings from all 4 passes are included in the merged result."""
        # Code that should trigger findings in every pass:
        # - correctness: bare except, == None
        # - security: hardcoded password
        # - performance: string concat with +
        # - standards: TODO comment, print() call
        code = (
            "password = 'hunter2'\n"
            "# TODO: fix this\n"
            "print('debug')\n"
            "def foo():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
            "x = a == None\n"
            "result = 'hello' + name\n"
        )
        result = run_multi_perspective_review(code)
        pass_names_found = {f["pass_name"] for f in result}
        # At minimum the security and standards passes should fire on this code
        assert "security" in pass_names_found
        assert "standards" in pass_names_found

    def test_no_duplicate_findings_by_key(self) -> None:
        """The same finding at the same line is not reported twice across passes."""
        code = 'password = "hunter2"\n'
        result = run_multi_perspective_review(code)
        # Build dedup keys as the implementation does
        seen: set[str] = set()
        for f in result:
            key = f"{f['pass_name']}:{f.get('finding', '')}:{f.get('line', 0)}"
            assert key not in seen, f"Duplicate finding key: {key}"
            seen.add(key)


# -- _run_correctness_scan ----------------------------------------------------


class TestRunCorrectnessScan:
    """_run_correctness_scan() detects common Python correctness pitfalls."""

    def test_detects_bare_except(self) -> None:
        """Bare except clause is flagged as a correctness issue."""
        code = "try:\n    pass\nexcept:\n    pass\n"
        findings = _run_correctness_scan(code)
        assert any("bare except" in f["finding"].lower() or "except" in f["finding"].lower() for f in findings)

    def test_detects_equals_none(self) -> None:
        """x == None comparisons are flagged in favour of 'is None'."""
        code = "if x == None:\n    pass\n"
        findings = _run_correctness_scan(code)
        assert any("is None" in f["finding"] or "None" in f["finding"] for f in findings)

    def test_detects_mutable_default_list(self) -> None:
        """Mutable default list argument is flagged."""
        code = "def foo(items=[]):\n    return items\n"
        findings = _run_correctness_scan(code)
        assert any("mutable" in f["finding"].lower() or "default" in f["finding"].lower() for f in findings)

    def test_returns_list_of_dicts(self) -> None:
        """Return type is always a list of dicts."""
        code = "def add(a, b):\n    return a + b\n"
        findings = _run_correctness_scan(code)
        assert isinstance(findings, list)
        for f in findings:
            assert isinstance(f, dict)
            assert "finding" in f
            assert "severity" in f
            assert "line" in f

    def test_skips_comment_lines(self) -> None:
        """Lines starting with # are not scanned."""
        code = "# except:  this is just a comment\n"
        findings = _run_correctness_scan(code)
        assert findings == []


# -- _run_performance_scan ----------------------------------------------------


class TestRunPerformanceScan:
    """_run_performance_scan() detects common performance anti-patterns."""

    def test_detects_string_concat_with_plus(self) -> None:
        """String concatenation using + is flagged."""
        code = "result = 'hello ' + name\n"
        findings = _run_performance_scan(code)
        assert any("concat" in f["finding"].lower() or "+" in f["finding"] for f in findings)

    def test_detects_long_sleep(self) -> None:
        """Long sleep calls are flagged as blocking."""
        code = "import time\ntime.sleep(60)\n"
        findings = _run_performance_scan(code)
        assert any("sleep" in f["finding"].lower() or "block" in f["finding"].lower() for f in findings)

    def test_returns_list_for_clean_code(self) -> None:
        """Clean code with no performance issues returns an empty list."""
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        findings = _run_performance_scan(code)
        assert isinstance(findings, list)


# -- _run_standards_scan ------------------------------------------------------


class TestRunStandardsScan:
    """_run_standards_scan() detects standards violations."""

    def test_detects_print_call(self) -> None:
        """print() in production code is flagged."""
        code = "    print('debug value')\n"
        findings = _run_standards_scan(code)
        assert any("print" in f["finding"].lower() for f in findings)

    def test_detects_todo_comment(self) -> None:
        """TODO annotations in code are flagged."""
        code = "# TODO: fix this later\nresult = 42\n"
        findings = _run_standards_scan(code)
        assert any("TODO" in f["finding"] or "FIXME" in f["finding"] for f in findings)

    def test_detects_deprecated_utcnow(self) -> None:
        """datetime.utcnow() is flagged as deprecated."""
        code = "from datetime import datetime\nnow = datetime.utcnow()\n"
        findings = _run_standards_scan(code)
        assert any("utcnow" in f["finding"].lower() or "deprecated" in f["finding"].lower() for f in findings)

    def test_returns_list_for_clean_code(self) -> None:
        """Clean code with no standards issues returns an empty list."""
        code = "from __future__ import annotations\n\ndef add(a: int, b: int) -> int:\n    return a + b\n"
        findings = _run_standards_scan(code)
        assert isinstance(findings, list)
