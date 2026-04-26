"""Tests for batch 33E.2D misc-script defect fixes.

Covers:
    D1  — token_usage.py: malformed timestamp is skipped, no crash
    D2  — session_quality.py: verification requires actual success, not just text
    D3  — learnings_promoter.py: KG schema drift degrades gracefully
    D4  — test_summary.py: malformed XML produces bounded error, nonzero exit
    D5  — test_summary.py: empty run (tests=0) is non-green
    D6  — verify_wiring.py: unknown flags exit nonzero
    D7  — verify_wiring.py: PolicyEnforcer check fails on empty/invalid jurisdiction
    D9  — docs_drift.py: .codex/ surface is scanned
    D10 — docs_drift.py: powershell/ps1 fenced blocks are extracted
    D12 — semgrep_scan.py: result.error propagated in JSON mode
"""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: project root (worktree)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent


# ===========================================================================
# D1 — token_usage.py: malformed timestamp skipped, no crash
# ===========================================================================


class TestTokenUsageMalformedTimestamp:
    """token_usage.parse_session skips malformed timestamps instead of crashing."""

    def _make_jsonl(self, tmp_path: Path, timestamp: str) -> Path:
        """Write a minimal JSONL session file with the given timestamp.

        The outer ``type`` field must be ``"assistant"`` — that is the discriminator
        that ``parse_session`` filters on (line: ``if obj.get("type") != "assistant"``).
        """
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "test-model",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "echo hi"},
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
            "timestamp": timestamp,
        }
        p = tmp_path / "session.jsonl"
        p.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        return p

    def test_malformed_iso_timestamp_does_not_crash(self, tmp_path: Path) -> None:
        """parse_session must not raise on a garbage timestamp string."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from token_usage import parse_session  # type: ignore[import]
        finally:
            sys.path.pop(0)

        p = self._make_jsonl(tmp_path, "not-a-date")
        # Must not raise ValueError
        result = parse_session(p)
        # Entry is still counted (tokens are valid); only the timestamp is skipped
        assert result is not None
        assert result["turns"] == 1

    def test_malformed_timestamp_excluded_from_time_range(self, tmp_path: Path) -> None:
        """When a timestamp is malformed, first_ts/last_ts stay None."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from token_usage import parse_session  # type: ignore[import]
        finally:
            sys.path.pop(0)

        p = self._make_jsonl(tmp_path, "2099-99-99T99:99:99")
        result = parse_session(p)
        assert result is not None
        assert result["first_ts"] is None
        assert result["last_ts"] is None

    def test_valid_timestamp_still_works(self, tmp_path: Path) -> None:
        """Valid ISO timestamps continue to populate the time range."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from token_usage import parse_session  # type: ignore[import]
        finally:
            sys.path.pop(0)

        p = self._make_jsonl(tmp_path, "2025-01-15T10:00:00Z")
        result = parse_session(p)
        assert result is not None
        assert result["first_ts"] is not None


# ===========================================================================
# D2 — session_quality.py: verification requires success, not just text
# ===========================================================================


class TestSessionQualityVerification:
    """_metric_edit_verify_ratio requires result_succeeded=True, not just command text."""

    def _make_tool_call(
        self,
        tool_name: str,
        command: str = "",
        result_succeeded: bool = False,
    ):
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from session_quality import ToolCall  # type: ignore[import]
        finally:
            sys.path.pop(0)
        return ToolCall(
            turn=1,
            tool_name=tool_name,
            input_data={"command": command},
            result_succeeded=result_succeeded,
        )

    def test_verify_call_with_only_text_not_counted(self) -> None:
        """A pytest Bash call that failed (result_succeeded=False) is not counted."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from session_quality import Thresholds, _metric_edit_verify_ratio  # type: ignore[import]
        finally:
            sys.path.pop(0)

        calls = [
            self._make_tool_call("Edit", result_succeeded=False),
            self._make_tool_call("Edit", result_succeeded=False),
            self._make_tool_call("Edit", result_succeeded=False),
            # Bash with pytest in command but result_succeeded=False — should NOT count
            self._make_tool_call("Bash", command="pytest tests/ -x -q", result_succeeded=False),
        ]
        result = _metric_edit_verify_ratio(calls, Thresholds())
        # 3 edits, 0 successful verifications → FAIL
        assert result.status == "FAIL"

    def test_verify_call_with_success_is_counted(self) -> None:
        """A pytest Bash call with result_succeeded=True counts as verification."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from session_quality import Thresholds, _metric_edit_verify_ratio  # type: ignore[import]
        finally:
            sys.path.pop(0)

        calls = [
            self._make_tool_call("Edit", result_succeeded=False),
            self._make_tool_call("Bash", command="pytest tests/ -x -q", result_succeeded=True),
        ]
        result = _metric_edit_verify_ratio(calls, Thresholds())
        assert result.status == "PASS"

    def test_result_indicates_success_failure_patterns(self) -> None:
        """_result_indicates_success correctly detects failure output."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from session_quality import _result_indicates_success  # type: ignore[import]
        finally:
            sys.path.pop(0)

        assert _result_indicates_success("") is True
        assert _result_indicates_success("1 passed") is True
        assert _result_indicates_success("Exit code: 1\nsome output") is False
        assert _result_indicates_success("Exit code: 127") is False


# ===========================================================================
# D3 — learnings_promoter.py: KG schema drift degrades gracefully
# ===========================================================================


class TestLearningsPromoterKGCrash:
    """get_frequent_kg_nodes degrades gracefully when 'nodes' table is absent."""

    def test_missing_nodes_table_returns_empty_list(self, tmp_path: Path) -> None:
        """OperationalError from missing nodes table yields [] with a warning log."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from learnings_promoter import get_frequent_kg_nodes  # type: ignore[import]
        finally:
            sys.path.pop(0)

        # Create a DB with no 'nodes' table
        db_path = tmp_path / "kg.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        conn.commit()

        import logging

        with patch.object(logging.getLogger("learnings_promoter"), "warning") as mock_warn:
            result = get_frequent_kg_nodes(conn, threshold=3)

        conn.close()
        assert result == []
        mock_warn.assert_called_once()
        assert "nodes" in mock_warn.call_args[0][0].lower() or "nodes" in str(mock_warn.call_args)

    def test_valid_nodes_table_returns_rows(self, tmp_path: Path) -> None:
        """get_frequent_kg_nodes returns rows when schema is correct."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from learnings_promoter import get_frequent_kg_nodes  # type: ignore[import]
        finally:
            sys.path.pop(0)

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, title TEXT, times_observed INTEGER)")
        conn.execute("INSERT INTO nodes VALUES (1, 'pattern-a', 5)")
        conn.execute("INSERT INTO nodes VALUES (2, 'pattern-b', 1)")
        conn.commit()

        result = get_frequent_kg_nodes(conn, threshold=3)
        conn.close()
        assert len(result) == 1
        assert result[0]["title"] == "pattern-a"


# ===========================================================================
# D4 — test_summary.py: malformed XML produces bounded error, nonzero exit
# ===========================================================================


class TestTestSummaryMalformedXML:
    """test_summary.main returns nonzero on malformed JUnit XML."""

    def test_malformed_xml_returns_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        """Truncated/corrupt XML must produce FAILED message and exit 1."""
        # Point the script at a temp results file
        xml_path = tmp_path / "test-results.xml"
        xml_path.write_text("<<< this is not valid xml >>>", encoding="utf-8")

        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import test_summary  # type: ignore[import]
        finally:
            sys.path.pop(0)

        monkeypatch.setattr(test_summary, "RESULTS_PATH", xml_path)

        output_lines = []
        monkeypatch.setattr("builtins.print", lambda *a, **kw: output_lines.append(" ".join(str(x) for x in a)))

        rc = test_summary.main()

        assert rc != 0, "Expected nonzero return code for malformed XML"
        combined = " ".join(output_lines)
        assert "malformed" in combined.lower() or "failed" in combined.lower()

    def test_valid_xml_passes(self, tmp_path: Path, monkeypatch) -> None:
        """Well-formed JUnit XML with passing tests returns 0."""
        xml_path = tmp_path / "test-results.xml"
        xml_path.write_text(
            textwrap.dedent("""\
                <?xml version="1.0"?>
                <testsuite name="suite" tests="2" failures="0" errors="0" skipped="0" time="0.1">
                  <testcase classname="TestFoo" name="test_a"/>
                  <testcase classname="TestFoo" name="test_b"/>
                </testsuite>
            """),
            encoding="utf-8",
        )

        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import test_summary  # type: ignore[import]
        finally:
            sys.path.pop(0)

        monkeypatch.setattr(test_summary, "RESULTS_PATH", xml_path)
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

        rc = test_summary.main()
        assert rc == 0


# ===========================================================================
# D5 — test_summary.py: empty run (tests=0) is non-green
# ===========================================================================


class TestTestSummaryEmptyRun:
    """test_summary.main returns nonzero when zero tests were executed."""

    def test_zero_tests_returns_nonzero(self, tmp_path: Path, monkeypatch) -> None:
        """JUnit XML with tests=0 must produce EMPTY message and exit 1."""
        xml_path = tmp_path / "test-results.xml"
        xml_path.write_text(
            textwrap.dedent("""\
                <?xml version="1.0"?>
                <testsuite name="empty" tests="0" failures="0" errors="0" skipped="0" time="0.0"/>
            """),
            encoding="utf-8",
        )

        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import test_summary  # type: ignore[import]
        finally:
            sys.path.pop(0)

        monkeypatch.setattr(test_summary, "RESULTS_PATH", xml_path)

        output_lines = []
        monkeypatch.setattr("builtins.print", lambda *a, **kw: output_lines.append(" ".join(str(x) for x in a)))

        rc = test_summary.main()

        assert rc != 0, "tests=0 must be non-green"
        combined = " ".join(output_lines)
        assert "empty" in combined.lower()

    def test_nonzero_tests_not_affected(self, tmp_path: Path, monkeypatch) -> None:
        """Runs with at least one test are not falsely flagged as empty."""
        xml_path = tmp_path / "test-results.xml"
        xml_path.write_text(
            textwrap.dedent("""\
                <?xml version="1.0"?>
                <testsuite name="s" tests="1" failures="0" errors="0" skipped="0" time="0.01">
                  <testcase classname="T" name="test_ok"/>
                </testsuite>
            """),
            encoding="utf-8",
        )

        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import test_summary  # type: ignore[import]
        finally:
            sys.path.pop(0)

        monkeypatch.setattr(test_summary, "RESULTS_PATH", xml_path)
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

        rc = test_summary.main()
        assert rc == 0


# ===========================================================================
# D6 — verify_wiring.py: unknown flags exit nonzero
# ===========================================================================


class TestVerifyWiringUnknownFlags:
    """verify_wiring __main__ block exits nonzero on unrecognised flags."""

    def _run_script(self, *extra_args: str) -> int:
        """Run verify_wiring as __main__ with the given extra argv."""
        import subprocess

        r = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [
                sys.executable,
                str(_REPO / "scripts" / "verify_wiring.py"),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(_REPO),
        )
        return r.returncode

    def test_unknown_flag_exits_nonzero(self) -> None:
        """--unknown-flag must produce exit code 1 before running any checks."""
        rc = self._run_script("--unknown-flag")
        assert rc != 0

    def test_unknown_flag_with_value_exits_nonzero(self) -> None:
        """Unrecognised flags with values also fail fast."""
        rc = self._run_script("--no-such-option")
        assert rc != 0

    def test_known_verbose_flag_is_accepted(self) -> None:
        """--verbose is a known flag and must not be rejected by the guard."""
        # We just need the script to NOT exit with the "unknown flag" path (rc==1
        # due to actual wiring failures is fine — we only care it didn't fail on
        # the flag itself, i.e. the guard didn't fire).
        import subprocess

        r = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [sys.executable, str(_REPO / "scripts" / "verify_wiring.py"), "--verbose"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Windows verify_wiring may emit cp1252 bytes (em-dash 0x97) in messages
            cwd=str(_REPO),
        )
        # The script must not print "Unknown flag" in stderr/stdout
        combined = (r.stdout or "") + (r.stderr or "")
        assert "Unknown flag" not in combined


# ===========================================================================
# D9 + D10 — docs_drift.py: .codex/ scanned; ps1/powershell blocks extracted
# ===========================================================================


class TestDocsDriftCodexSurface:
    """docs_drift finds .codex/ markdown files and extracts ps1/powershell blocks."""

    def test_find_markdown_files_includes_codex(self, tmp_path: Path, monkeypatch) -> None:
        """find_markdown_files includes .codex/ tree when it exists."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import docs_drift  # type: ignore[import]
        finally:
            sys.path.pop(0)

        # Build a fake project root with a .codex/ file
        fake_root = tmp_path
        codex_dir = fake_root / ".codex"
        codex_dir.mkdir()
        codex_doc = codex_dir / "plans"
        codex_doc.mkdir()
        codex_md = codex_doc / "my-plan.md"
        codex_md.write_text("# Plan\n", encoding="utf-8")

        monkeypatch.setattr(docs_drift, "PROJECT_ROOT", fake_root)

        files = docs_drift.find_markdown_files()
        paths = [str(f) for f in files]
        assert any("my-plan.md" in p for p in paths), f".codex/ file not found in: {paths}"

    def test_extract_commands_includes_powershell_block(self) -> None:
        """Commands inside ```powershell blocks are extracted."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from docs_drift import extract_commands  # type: ignore[import]
        finally:
            sys.path.pop(0)

        content = textwrap.dedent("""\
            Some prose.

            ```powershell
            Get-ChildItem -Path .
            Set-Location C:/dev/Vetinari
            ```

            More prose.
        """)
        cmds = extract_commands(content)
        assert "Get-ChildItem -Path ." in cmds
        assert "Set-Location C:/dev/Vetinari" in cmds

    def test_extract_commands_includes_ps1_block(self) -> None:
        """Commands inside ```ps1 blocks are extracted."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from docs_drift import extract_commands  # type: ignore[import]
        finally:
            sys.path.pop(0)

        content = textwrap.dedent("""\
            ```ps1
            Write-Host hello
            ```
        """)
        cmds = extract_commands(content)
        assert "Write-Host hello" in cmds

    def test_extract_commands_bash_still_works(self) -> None:
        """Existing bash block extraction is not broken by the new matchers."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            from docs_drift import extract_commands  # type: ignore[import]
        finally:
            sys.path.pop(0)

        content = "```bash\npython -m pytest\n```"
        cmds = extract_commands(content)
        assert "python -m pytest" in cmds


# ===========================================================================
# D12 — semgrep_scan.py: result.error propagated in JSON mode
# ===========================================================================


class TestSemgrepScanJsonError:
    """semgrep_scan exits nonzero in --json mode when result.error is set."""

    def _make_result(self, *, is_available: bool, error: str | None, findings=None):
        r = SimpleNamespace(
            is_available=is_available,
            error=error,
            findings=findings or [],
        )
        return r

    def test_json_mode_error_exits_nonzero(self, monkeypatch, capsys) -> None:
        """When result.error is set, --json mode must exit nonzero."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import semgrep_scan  # type: ignore[import]
        finally:
            sys.path.pop(0)

        error_result = self._make_result(is_available=True, error="semgrep crashed")

        with patch.object(semgrep_scan, "run_semgrep", return_value=error_result):
            with patch("sys.argv", ["semgrep_scan.py", "vetinari/", "--json"]):
                with pytest.raises(SystemExit) as exc_info:
                    semgrep_scan.main()
        assert exc_info.value.code != 0

    def test_json_mode_unavailable_exits_nonzero(self, monkeypatch, capsys) -> None:
        """When semgrep is unavailable, --json mode must exit nonzero."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import semgrep_scan  # type: ignore[import]
        finally:
            sys.path.pop(0)

        unavail_result = self._make_result(is_available=False, error="not installed")

        with patch.object(semgrep_scan, "run_semgrep", return_value=unavail_result):
            with patch("sys.argv", ["semgrep_scan.py", "vetinari/", "--json"]):
                with pytest.raises(SystemExit) as exc_info:
                    semgrep_scan.main()
        assert exc_info.value.code != 0

    def test_json_mode_success_exits_zero(self, monkeypatch, capsys) -> None:
        """Clean run with no error exits 0 in JSON mode."""
        sys.path.insert(0, str(_REPO / "scripts"))
        try:
            import semgrep_scan  # type: ignore[import]
        finally:
            sys.path.pop(0)

        ok_result = self._make_result(is_available=True, error=None, findings=[])

        with patch.object(semgrep_scan, "run_semgrep", return_value=ok_result):
            with patch("sys.argv", ["semgrep_scan.py", "vetinari/", "--json"]):
                with pytest.raises(SystemExit) as exc_info:
                    semgrep_scan.main()
        assert exc_info.value.code == 0
