"""Focused tests for hook_session_observability emission behavior."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _reload_module():
    import hook_session_observability

    return importlib.reload(hook_session_observability)


def test_user_visible_signals_filters_low_priority_noise() -> None:
    mod = _reload_module()
    signals = [
        mod.Signal("unverified-edits", "too many edits without verification"),
        mod.Signal("high-cost-session", "estimated cost is high"),
        mod.Signal("implementation-stall", "implementation appears stalled"),
        mod.Signal("active-agent-dirty-worktree", "dirty files overlap with active agents"),
    ]

    visible = mod._user_visible_signals(signals)

    assert [signal.key for signal in visible] == [
        "implementation-stall",
        "active-agent-dirty-worktree",
    ]


def test_emit_only_prints_user_visible_signals(capsys) -> None:
    mod = _reload_module()
    signals = [
        mod.Signal("unverified-edits", "too many edits without verification"),
        mod.Signal("implementation-stall", "implementation appears stalled"),
    ]

    mod._emit(signals)
    captured = capsys.readouterr()

    assert "[session_observability] Check session health:" in captured.out
    assert "implementation appears stalled" in captured.out
    assert "too many edits without verification" not in captured.out
