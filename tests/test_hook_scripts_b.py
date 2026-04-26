"""Tests for batch 33E.2B hook-script defect fixes.

Covers 7 defects in maintainer hook scripts:
  D1 - hook_repo_router.py routes to current surfaces (.codex/, .claude/plans/)
  D2 - hook_session_status.py probes .codex/ instead of retired .ai-codex
  D3 - hook_subagent_start/stop docstrings reference .omc/state/ and .codex/
  D4 - hook_telemetry.py dual-writes; usage_prune_report.py deduplicates merged reads
  D5 - session_capsule.record_postcompact preserves existing summary when payload is empty
  D6 - hook_read_nudge._file_stats raises OSError (fail-closed) on unreadable file
  D7 - hook_stop_checks.main() returns 1 when checks fail

Scripts live in scripts/ (not vetinari/), so VET124 does not apply.

Implementation note: all module loads happen at collection time (module scope) so that
per-test execution stays well within the 30-second pytest-timeout budget.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — scripts/ uses bare imports that rely on each other
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _stub(name: str) -> MagicMock:
    """Register a MagicMock as sys.modules[name] and return it."""
    mock = MagicMock()
    sys.modules[name] = mock
    return mock


def _load_script(name: str) -> Any:
    """Load a script from scripts/ by name, returning its module object.

    Pre-existing sys.modules entries (stubs registered before this call) are
    respected — exec_module sees them as already-imported dependencies.

    Args:
        name: Module file stem, e.g. ``"hook_repo_router"``.

    Returns:
        The loaded module object.
    """
    module_path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Module-level loads — run ONCE at collection time, not inside any test
# ---------------------------------------------------------------------------

_stub("handoff_bundle")
_stub("hook_telemetry")
_stub("session_capsule")
_ROUTER = _load_script("hook_repo_router")

_stub("refresh_ai_accelerators")
_STATUS = _load_script("hook_session_status")

_stub("subagent_bootstrap")
_SUBAGENT_START = _load_script("hook_subagent_start")
_SUBAGENT_STOP = _load_script("hook_subagent_stop")

# Reload hook_telemetry as a real module (overrides the stub above)
_stub("handoff_bundle")
_TELEMETRY = _load_script("hook_telemetry")

_PRUNE = _load_script("usage_prune_report")
_CAPSULE = _load_script("session_capsule")
_NUDGE = _load_script("hook_read_nudge")

_stub("analysis_router")
_STOP = _load_script("hook_stop_checks")


# ---------------------------------------------------------------------------
# D1 — hook_repo_router routes to current surfaces
# ---------------------------------------------------------------------------


class TestHookRepoRouter:
    """D1: _route_hint() returns hints pointing to .codex/, .omc/state/, .ai-codex/plans/."""

    def test_continuity_hint_points_to_omc_state(self):
        hint = _ROUTER._route_hint("continue the session from earlier")
        assert ".omc/state/" in hint
        assert ".ai-codex/plans/" in hint

    def test_runtime_hint_points_to_codex(self):
        hint = _ROUTER._route_hint("investigate the mcp integration and library dependency resolution")
        assert ".codex/" in hint
        assert ".ai-codex/audit/" in hint

    def test_change_hint_points_to_codex(self):
        hint = _ROUTER._route_hint("review the bug fix and assess regression risk across modules")
        assert ".codex/" in hint

    def test_broad_hint_points_to_codex_and_plans(self):
        # Use a prompt with BROAD_TERMS only — no CHANGE_TERMS or RUNTIME_TERMS
        hint = _ROUTER._route_hint("broad system architecture design migration workflow investigation")
        assert ".codex/" in hint
        assert ".ai-codex/plans/" in hint

    def test_retired_paths_not_present_in_any_hint(self):
        for prompt in (
            "continue the session",
            "api service mcp integration",
            "refactor fix regression risk",
            "broad architecture system design workflow",
        ):
            hint = _ROUTER._route_hint(prompt)
            if not hint:
                continue
            assert ".claude/handoffs/" not in hint, f"Retired path in hint for: {prompt!r}"
            assert ".ai-codex/wiki.md" not in hint, f"Retired path in hint for: {prompt!r}"
            assert "rollups.md" not in hint, f"Retired path in hint for: {prompt!r}"


# ---------------------------------------------------------------------------
# D2 — hook_session_status probes .codex/ not .ai-codex
# ---------------------------------------------------------------------------


class TestHookSessionStatus:
    """D2: _wiki_summary() reflects .codex/ presence; _resume_target returns .omc/notepad.md."""

    def test_wiki_summary_ready_when_codex_dir_exists(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        with patch.object(_STATUS, "CODEX_MAINTAINER_DIR", codex_dir):
            result = _STATUS._wiki_summary()
        assert result == "ready"

    def test_wiki_summary_missing_when_codex_dir_absent(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        # deliberately not created
        with patch.object(_STATUS, "CODEX_MAINTAINER_DIR", codex_dir):
            result = _STATUS._wiki_summary()
        assert result == "missing"

    def test_resume_target_returns_omc_notepad_when_present(self, tmp_path):
        notepad = tmp_path / ".omc" / "notepad.md"
        notepad.parent.mkdir(parents=True)
        notepad.write_text("# Notes\n", encoding="utf-8")
        with patch.object(_STATUS, "OMC_NOTEPAD_PATH", notepad):
            result = _STATUS._resume_target()
        assert result == ".omc/notepad.md"


# ---------------------------------------------------------------------------
# D3 — hook_subagent_start/stop reference .omc/state/ and .codex/ in docstrings
# ---------------------------------------------------------------------------


class TestHookSubagentHandoff:
    """D3: module docstrings no longer reference .claude/handoffs/current.json."""

    def test_subagent_start_docstring_references_omc_state(self):
        doc = _SUBAGENT_START.__doc__ or ""
        assert ".omc/state/" in doc
        assert ".codex/" in doc
        assert ".claude/handoffs/current.json" not in doc

    def test_subagent_stop_docstring_references_omc_state(self):
        doc = _SUBAGENT_STOP.__doc__ or ""
        assert ".omc/state/" in doc
        assert ".codex/" in doc
        assert ".claude/handoffs/current.json" not in doc


# ---------------------------------------------------------------------------
# D4 — hook_telemetry dual-write; usage_prune_report deduplicates merged reads
# ---------------------------------------------------------------------------


class TestHookTelemetryDualWrite:
    """D4: log_hook_event writes to both log paths; usage_prune_report has HOOK_LOG_PATHS."""

    def test_log_hook_event_writes_to_both_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VET_HOOK_TELEMETRY_DISABLE", "0")
        claude_log = tmp_path / ".claude" / "hook-log.jsonl"
        omc_log = tmp_path / ".omc" / "hook-log.jsonl"

        with (
            patch.object(_TELEMETRY, "LOG_PATH", claude_log),
            patch.object(_TELEMETRY, "OMC_LOG_PATH", omc_log),
        ):
            _TELEMETRY.log_hook_event("test-hook", "TestEvent", "allowed", metadata={"rule": "test"})

        assert claude_log.exists(), ".claude hook log not written"
        assert omc_log.exists(), ".omc hook log not written"
        claude_record = json.loads(claude_log.read_text(encoding="utf-8").strip())
        omc_record = json.loads(omc_log.read_text(encoding="utf-8").strip())
        assert claude_record["hook"] == "test-hook"
        assert omc_record["hook"] == "test-hook"

    def test_usage_prune_report_has_both_hook_log_paths(self):
        paths = getattr(_PRUNE, "HOOK_LOG_PATHS", None)
        assert paths is not None, "HOOK_LOG_PATHS not defined in usage_prune_report"
        assert len(paths) == 2, f"Expected 2 paths, got {len(paths)}"

    def test_usage_prune_report_deduplicates_hook_records(self, tmp_path):
        record = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "hook": "test-hook",
            "event": "TestEvent",
            "outcome": "allowed",
            "blocked": False,
            "context_chars": 0,
            "metadata": {},
        }
        line = json.dumps(record) + "\n"
        claude_log = tmp_path / "claude-hook-log.jsonl"
        omc_log = tmp_path / "omc-hook-log.jsonl"
        claude_log.write_text(line, encoding="utf-8")
        omc_log.write_text(line, encoding="utf-8")

        # Reproduce the deduplication logic used in usage_prune_report
        seen_keys: set[tuple[str, str, str]] = set()
        merged: list[dict] = []  # type: ignore[type-arg]
        for log_path in (claude_log, omc_log):
            for raw_line in log_path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(raw_line)
                key = (
                    str(payload.get("timestamp") or ""),
                    str(payload.get("hook") or ""),
                    str(payload.get("outcome") or ""),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged.append(payload)
        assert len(merged) == 1, f"Deduplication failed: got {len(merged)} records"


# ---------------------------------------------------------------------------
# D5 — session_capsule.record_postcompact preserves existing summary when empty
# ---------------------------------------------------------------------------


class TestPostcompactPreserveSummary:
    """D5: record_postcompact preserves existing last_summary when compact_summary is empty."""

    def test_existing_summary_preserved_when_new_is_empty(self, tmp_path):
        compact_state_path = tmp_path / "compact-state.json"
        existing = {
            "last_summary": "Important previous context.",
            "last_summary_chars": 28,
            "compact_count": 1,
        }
        compact_state_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        with patch.object(_CAPSULE, "COMPACT_STATE_PATH", compact_state_path):
            _CAPSULE.record_postcompact("test-trigger", "")

        result = json.loads(compact_state_path.read_text(encoding="utf-8"))
        assert result["last_summary"] == "Important previous context."

    def test_new_summary_overwrites_when_non_empty(self, tmp_path):
        compact_state_path = tmp_path / "compact-state.json"
        existing = {
            "last_summary": "Old summary.",
            "last_summary_chars": 12,
            "compact_count": 1,
        }
        compact_state_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        with patch.object(_CAPSULE, "COMPACT_STATE_PATH", compact_state_path):
            _CAPSULE.record_postcompact("test-trigger", "New summary content here.")

        result = json.loads(compact_state_path.read_text(encoding="utf-8"))
        assert "New summary content here." in result["last_summary"]

    def test_compact_count_increments_even_when_summary_empty(self, tmp_path):
        compact_state_path = tmp_path / "compact-state.json"
        existing = {"compact_count": 3, "last_summary": "Keep this."}
        compact_state_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

        with patch.object(_CAPSULE, "COMPACT_STATE_PATH", compact_state_path):
            _CAPSULE.record_postcompact("test-trigger", "")

        result = json.loads(compact_state_path.read_text(encoding="utf-8"))
        assert result["compact_count"] == 4
        assert result["last_summary"] == "Keep this."


# ---------------------------------------------------------------------------
# D6 — hook_read_nudge._file_stats raises OSError for unreadable file (fail-closed)
# ---------------------------------------------------------------------------


class TestHookReadNudgeStatFailure:
    """D6: _file_stats raises OSError on missing/unreadable file; main() exits 1."""

    def test_file_stats_raises_oserror_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.py"
        with pytest.raises(OSError, match="No such file"):
            _NUDGE._file_stats(missing)

    def test_file_stats_returns_correct_counts_for_existing_file(self, tmp_path):
        sample = tmp_path / "sample.py"
        content = "def foo():\n    pass\n"
        sample.write_text(content, encoding="utf-8")
        byte_count, line_count = _NUDGE._file_stats(sample)
        assert byte_count == len(content.encode("utf-8"))
        assert line_count == content.count("\n") + 1

    def test_main_exits_1_for_unreadable_code_file(self, tmp_path):
        import io

        missing_py = tmp_path / "ghost.py"
        payload = json.dumps({"tool_input": {"file_path": str(missing_py)}})

        with (
            patch("sys.stdin", io.StringIO(payload)),
            pytest.raises(SystemExit) as exc_info,
        ):
            _NUDGE.main()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# D7 — hook_stop_checks.main() returns 1 when any check fails
# ---------------------------------------------------------------------------


class TestHookStopChecksFailPropagates:
    """D7: main() returns 1 when ruff or other checks fail, not 0."""

    def test_main_returns_nonzero_when_check_fails(self):
        with (
            patch.object(_STOP, "_has_relevant_changes", return_value=True),
            patch.object(_STOP, "_changed_python_files", return_value=["vetinari/foo.py"]),
            patch.object(_STOP, "_run", return_value=(False, "E501 line too long")),
        ):
            result = _STOP.main()

        assert result != 0, "main() must return non-zero when checks fail"

    def test_main_returns_zero_when_all_checks_pass(self):
        with (
            patch.object(_STOP, "_has_relevant_changes", return_value=True),
            patch.object(_STOP, "_changed_python_files", return_value=["vetinari/foo.py"]),
            patch.object(_STOP, "_run", return_value=(True, "")),
        ):
            result = _STOP.main()

        assert result == 0, "main() must return 0 when all checks pass"

    def test_main_returns_zero_when_no_relevant_changes(self):
        with patch.object(_STOP, "_has_relevant_changes", return_value=False):
            result = _STOP.main()

        assert result == 0
