"""Regression tests for SESSION-33E.2 batch A hook script defects.

Covers 9 specific defects across 5 hook scripts:
  - hook_delegation_guard.py: defects 1 (case-insensitive anchor), 2 (fail closed)
  - hook_bash_guard.py:       defects 3 (rm whitelist bypass), 4 (fail closed)
  - hook_agent_gate.py:       defects 5 (oh-my-Codex families), 6 (fail closed)
  - hook_post_edit_format.py: defect  7 (formatter failure exits non-zero)
  - hook_lsp_nudge.py:        defect  8 (builder subagent exclusion)
  - hook_pre_commit_gate.py:  defect  9 (no sys.executable fallback)
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to path so we can import hook modules directly.
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helper: run a hook's main() with fabricated stdin and capture exit code
# ---------------------------------------------------------------------------


def _run_main(module_name: str, stdin_data: str | dict) -> int:
    """Import hook module, inject stdin, call main(), return sys.exit code.

    Args:
        module_name: Bare module name inside scripts/ (no .py extension).
        stdin_data: Dict (JSON-encoded for stdin) or raw string for stdin.

    Returns:
        The integer exit code from sys.exit(), or 0 if main() returned normally.
    """
    if isinstance(stdin_data, dict):
        raw = json.dumps(stdin_data)
    else:
        raw = stdin_data

    mod = importlib.import_module(module_name)
    importlib.reload(mod)  # reset module state between tests

    import io

    with patch("sys.stdin", io.StringIO(raw)):
        try:
            mod.main()
            return 0
        except SystemExit as exc:
            return int(exc.code) if exc.code is not None else 0


# ===========================================================================
# Defect 1 — hook_delegation_guard: case-insensitive repo-root anchor
# ===========================================================================


class TestDelegationGuardCaseInsensitiveAnchor:
    """Defect 1: path-casing variants must not bypass the delegation guard."""

    def test_upper_case_path_is_blocked(self) -> None:
        """C:/DEV/VETINARI/... must still be treated as an implementation file."""
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "C:/DEV/VETINARI/scripts/hook_bash_guard.py"},
        }
        # No agent_id → main agent → should be blocked
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 2, f"Expected exit 2 (blocked), got {rc}"

    def test_mixed_case_path_is_blocked(self) -> None:
        """Mixed casing like C:/dev/Vetinari (capital V) must also block."""
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "C:/dev/Vetinari/vetinari/security/test_guard.py"},
        }
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 2, f"Expected exit 2 (blocked), got {rc}"

    def test_lowercase_path_is_blocked(self) -> None:
        """All-lowercase c:/dev/vetinari/... must also block (not bypass)."""
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "c:/dev/vetinari/vetinari/routing/core.py"},
        }
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 2, f"Expected exit 2 (blocked), got {rc}"


# ===========================================================================
# Defect 2 — hook_delegation_guard: fail closed on malformed stdin / missing file_path
# ===========================================================================


class TestDelegationGuardFailClosed:
    """Defect 2: malformed JSON and missing file_path must fail closed (exit 2)."""

    def test_malformed_json_exits_nonzero(self) -> None:
        """Garbage stdin must not be silently allowed."""
        rc = _run_main("hook_delegation_guard", "NOT VALID JSON }{")
        assert rc != 0, "Expected non-zero exit for malformed JSON"

    def test_missing_file_path_in_edit_exits_nonzero(self) -> None:
        """An Edit payload with no file_path must fail closed."""
        payload = {"tool_name": "Edit", "tool_input": {}}
        rc = _run_main("hook_delegation_guard", payload)
        assert rc != 0, "Expected non-zero exit for Edit with missing file_path"


# ===========================================================================
# Defect 3 — hook_bash_guard: rm must NOT bypass blocked-pattern checks
# ===========================================================================


class TestBashGuardRmWhitelist:
    """Defect 3: 'rm' is no longer on the first-word whitelist."""

    def test_rm_rf_does_not_early_exit(self) -> None:
        """rm -rf ... must reach blocked-pattern checks, not exit 0 immediately."""
        # We test that rm is not silently allowed.  The exact exit code depends
        # on whether rm itself matches a BLOCKED pattern — what matters is that
        # it doesn't short-circuit to 0 via the whitelist.
        import hook_bash_guard

        importlib.reload(hook_bash_guard)
        assert "rm" not in hook_bash_guard.WHITELIST_FIRST_WORD, (
            "'rm' must be removed from WHITELIST_FIRST_WORD"
        )

    def test_plain_rm_command_reaches_guard(self) -> None:
        """A bare 'rm somefile' payload should not immediately exit 0."""
        import hook_bash_guard

        importlib.reload(hook_bash_guard)
        # rm is not in BLOCKED patterns, so it will exit 0 after full check —
        # the critical point is it does NOT short-circuit before block checks.
        # We verify by checking the whitelist directly (structural test).
        assert "rm" not in hook_bash_guard.WHITELIST_FIRST_WORD


# ===========================================================================
# Defect 4 — hook_bash_guard: fail closed on malformed stdin / empty command
# ===========================================================================


class TestBashGuardFailClosed:
    """Defect 4: malformed stdin and missing command must fail closed."""

    def test_malformed_json_exits_nonzero(self) -> None:
        """Garbage stdin must not be treated as an ordinary allow."""
        rc = _run_main("hook_bash_guard", "{{invalid json")
        assert rc != 0, "Expected non-zero exit for malformed JSON"

    def test_missing_command_exits_nonzero(self) -> None:
        """A Bash hook payload with no command must fail closed."""
        payload = {"tool_name": "Bash", "tool_input": {}}
        rc = _run_main("hook_bash_guard", payload)
        assert rc != 0, "Expected non-zero exit for Bash hook with missing command"


# ===========================================================================
# Defect 5 — hook_agent_gate: oh-my-Codex:* families hit enforcement
# ===========================================================================


class TestAgentGateCodexFamily:
    """Defect 5: oh-my-Codex:* agents must trigger delegation-quality enforcement."""

    def test_omc_codex_executor_too_short_is_blocked(self) -> None:
        """oh-my-Codex:executor with a tiny prompt must be blocked like oh-my-claudecode."""
        payload = {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "oh-my-Codex:executor",
                "prompt": "fix it",  # way below IMPL_MIN_CHARS
            },
        }
        rc = _run_main("hook_agent_gate", payload)
        assert rc == 2, f"Expected exit 2 (blocked), got {rc}"

    def test_omc_codex_code_reviewer_short_is_blocked(self) -> None:
        """oh-my-Codex:code-reviewer with short prompt must be blocked."""
        payload = {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "oh-my-Codex:code-reviewer",
                "prompt": "review",
            },
        }
        rc = _run_main("hook_agent_gate", payload)
        assert rc == 2, f"Expected exit 2 (blocked), got {rc}"

    def test_omc_codex_executor_adequate_prompt_is_allowed(self) -> None:
        """oh-my-Codex:executor with a sufficiently detailed prompt must pass."""
        long_prompt = (
            "Mode: build. "
            "Modify vetinari/foo.py to add a new function called compute_score "
            "that takes a list of float values and returns their weighted average. "
            "Add corresponding tests in tests/test_foo.py. "
            "Ensure ruff and pytest pass after the change. "
            "This is needed because the scoring subsystem requires weighted output."
        )
        payload = {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "oh-my-Codex:executor",
                "prompt": long_prompt,
            },
        }
        rc = _run_main("hook_agent_gate", payload)
        assert rc == 0, f"Expected exit 0 (allowed), got {rc}"


# ===========================================================================
# Defect 6 — hook_agent_gate: fail closed on malformed stdin
# ===========================================================================


class TestAgentGateFailClosed:
    """Defect 6: malformed stdin must fail closed in hook_agent_gate."""

    def test_malformed_json_exits_nonzero(self) -> None:
        """Garbage stdin must not silently pass the agent gate."""
        rc = _run_main("hook_agent_gate", "[broken json")
        assert rc != 0, "Expected non-zero exit for malformed JSON"


# ===========================================================================
# Defect 7 — hook_post_edit_format: formatter failure exits non-zero
# ===========================================================================


class TestPostEditFormatFailurePropagates:
    """Defect 7: a formatter failure must cause non-zero exit from _handle_path."""

    def test_handle_path_returns_nonzero_on_formatter_failure(self) -> None:
        """When _format_python_file returns (False, [...]), _handle_path must return != 0."""
        import hook_post_edit_format

        importlib.reload(hook_post_edit_format)

        with patch.object(
            hook_post_edit_format,
            "_format_python_file",
            return_value=(False, ["ruff-check-fix: failed"]),
        ):
            result = hook_post_edit_format._handle_path("vetinari/some_file.py")
        assert result != 0, f"Expected non-zero return from _handle_path on failure, got {result}"

    def test_handle_path_returns_zero_on_success(self) -> None:
        """When _format_python_file succeeds, _handle_path must return 0."""
        import hook_post_edit_format

        importlib.reload(hook_post_edit_format)

        with patch.object(
            hook_post_edit_format,
            "_format_python_file",
            return_value=(True, []),
        ):
            result = hook_post_edit_format._handle_path("vetinari/some_file.py")
        assert result == 0, f"Expected 0 return from _handle_path on success, got {result}"


# ===========================================================================
# Defect 8 — hook_lsp_nudge: builder subagent payloads are not nudged
# ===========================================================================


class TestLspNudgeBuilderExclusion:
    """Defect 8: payloads with a builder agent_id must be skipped (no nudge)."""

    def test_builder_agent_id_skips_nudge(self) -> None:
        """When agent_id contains 'builder', main() must return without nudging."""
        import hook_lsp_nudge

        importlib.reload(hook_lsp_nudge)

        # Patch _read_payload to return a builder-tagged payload with a concept-search pattern
        builder_payload = {
            "agent_id": "builder-abc123",
            "tool_name": "Grep",
            "tool_input": {
                "pattern": "how does the plan execution pipeline work",  # concept search
                "type": "py",
            },
        }
        with patch.object(hook_lsp_nudge, "_read_payload", return_value=builder_payload):
            # Should return without calling sys.exit or printing to stderr
            try:
                hook_lsp_nudge.main()
                exited = False
            except SystemExit:
                exited = True

        assert not exited, "Builder subagent payload must not trigger sys.exit(2)"

    def test_non_builder_agent_id_is_not_skipped(self) -> None:
        """When agent_id does not contain 'builder', normal nudge logic applies."""
        import hook_lsp_nudge

        importlib.reload(hook_lsp_nudge)

        # A non-builder agent with a hard-block pattern should still be blocked
        non_builder_payload = {
            "agent_id": "architect-xyz",
            "tool_name": "Grep",
            "tool_input": {
                # Hard-block pattern: searching for a def by exact name triggers block
                "pattern": r"def my_function",
                "type": "py",
            },
        }
        with patch.object(hook_lsp_nudge, "_read_payload", return_value=non_builder_payload):
            try:
                hook_lsp_nudge.main()
                exited_2 = False
            except SystemExit as exc:
                exited_2 = exc.code == 2

        # We don't assert exited_2 strictly because pattern may not match BLOCK_PATTERNS —
        # the key property is that the builder exclusion path was NOT taken.
        # We verify this via the builder exclusion unit test above.
        # This test confirms the non-builder path reaches the guard logic at all.
        assert exited_2 is True


# ===========================================================================
# Defect 9 — hook_pre_commit_gate: no sys.executable fallback
# ===========================================================================


class TestPreCommitGateNoSysExecutableFallback:
    """Defect 9: when venv is not found, must fail closed rather than use sys.executable."""

    def test_get_venv_python_returns_none_when_not_found(self) -> None:
        """_get_venv_python() must return None when .venv312 is absent."""
        import hook_pre_commit_gate

        importlib.reload(hook_pre_commit_gate)

        with (
            patch.dict("os.environ", {"CLAUDE_PROJECT_DIR": "/nonexistent/path"}, clear=False),
            patch("os.getcwd", return_value="/also/nonexistent"),
            patch("os.path.isfile", return_value=False),
        ):
            result = hook_pre_commit_gate._get_venv_python()
        assert result is None, f"Expected None when venv not found, got {result!r}"

    def test_main_fails_closed_when_venv_missing(self) -> None:
        """main() must exit non-zero (not exit 0) when venv Python cannot be located."""
        import hook_pre_commit_gate

        importlib.reload(hook_pre_commit_gate)

        git_commit_payload = {"tool_input": {"command": "git commit -m 'test'"}}
        import io

        with (
            patch("sys.stdin", io.StringIO(json.dumps(git_commit_payload))),
            patch.object(hook_pre_commit_gate, "_get_venv_python", return_value=None),
        ):
            try:
                hook_pre_commit_gate.main()
                rc = 0
            except SystemExit as exc:
                rc = int(exc.code) if exc.code is not None else 0

        assert rc != 0, f"Expected non-zero exit when venv missing, got {rc}"


class TestDelegationGuardOrdinaryImplementationAllowed:
    """Ordinary implementation files should no longer force delegation."""

    def test_regular_vetinari_module_is_allowed(self) -> None:
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "C:/dev/Vetinari/vetinari/core.py"},
        }
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 0, f"Expected exit 0 (allowed), got {rc}"

    def test_tests_file_is_allowed(self) -> None:
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "C:/dev/Vetinari/tests/test_core.py"},
        }
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 0, f"Expected exit 0 (allowed), got {rc}"

    def test_non_hook_script_is_allowed(self) -> None:
        payload = {
            "tool_name": "Edit",
            "tool_input": {"file_path": "C:/dev/Vetinari/scripts/fix_vet123.py"},
        }
        rc = _run_main("hook_delegation_guard", payload)
        assert rc == 0, f"Expected exit 0 (allowed), got {rc}"
