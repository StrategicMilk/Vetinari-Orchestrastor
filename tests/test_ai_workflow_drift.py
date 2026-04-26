"""Regression tests for AI workflow drift policy checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "check_ai_workflow_drift.py"
    spec = importlib.util.spec_from_file_location("check_ai_workflow_drift", path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_worktree_policy_accepts_shared_repo_root_for_filesystem_mcp() -> None:
    """A worktree may intentionally expose the shared repo root without failing drift."""
    mod = _load_script()
    worktree_root = Path("C:/dev/Vetinari/.claude/worktrees/release-signoff")
    mod.APPROVED_FS_ROOTS = mod.compute_approved_fs_roots(worktree_root)

    findings = mod.check_mcp_servers(
        Path("C:/Users/darst/.codex/config.toml"),
        {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem@2026.1.14",
                    "C:/dev/Vetinari",
                ],
            }
        },
    )

    assert findings == []


def test_settings_json_provides_auto_memory_policy_when_local_file_is_absent(tmp_path, monkeypatch) -> None:
    """Missing settings.local.json must not erase an explicit repo policy in settings.json."""
    mod = _load_script()
    base = tmp_path / "settings.json"
    local = tmp_path / "settings.local.json"
    base.write_text(json.dumps({"autoMemoryEnabled": True, "permissions": {"allow": []}}), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_SETTINGS_BASE", base)
    monkeypatch.setattr(mod, "PROJECT_SETTINGS_LOCAL", local)

    assert mod.check_claude_settings() == []
