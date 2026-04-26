"""Publication boundary checks for GitHub-ready repository exports."""

from __future__ import annotations

from scripts.release.check_publication_boundary import (
    BlobRecord,
    inspect_blob_records,
    inspect_file_contents,
    inspect_tracked_paths,
)


def test_tracked_boundary_rejects_internal_ai_workbench_paths() -> None:
    findings = inspect_tracked_paths(
        [
            "vetinari/__init__.py",
            ".codex/config.toml",
            ".claire/worktrees/cleanup-A-redo/file.py",
            ".mcp.json",
            ".pytest-probe3/tmp/x.txt",
            ".ai-codex/audit/full-spectrum/SUMMARY.md",
            ".claude/settings.json",
            ".agents/skills/workflow-router/SKILL.md",
            "docs/internal/ai-workflows/agent-routing.yaml",
            "docs/audit/MASTER-INDEX.md",
            "docs/benchmarks/session-16-llm-reduction.md",
            "docs/archive/claude-commands/qa-session-01.md",
            "docs/plans/MASTER-PLAN.md",
            "docs/planning/vetinari-factory-plan.md",
            "docs/research/2026-04-06-action-plan.md",
            "docs/STATE-OF-VETINARI.md",
            "adr/ADR-0103.json",
            "prompts/t1_run.txt",
            "AGENTS.md",
            "vetinari/web/CLAUDE.md",
            ".github/workflows/vetinari-ci.yml",
        ]
    )

    paths = {finding.path for finding in findings}
    assert "vetinari/__init__.py" not in paths
    assert {
        ".codex/config.toml",
        ".claire/worktrees/cleanup-A-redo/file.py",
        ".mcp.json",
        ".pytest-probe3/tmp/x.txt",
        ".ai-codex/audit/full-spectrum/SUMMARY.md",
        ".claude/settings.json",
        ".agents/skills/workflow-router/SKILL.md",
        "docs/internal/ai-workflows/agent-routing.yaml",
        "docs/audit/MASTER-INDEX.md",
        "docs/benchmarks/session-16-llm-reduction.md",
        "docs/archive/claude-commands/qa-session-01.md",
        "docs/plans/MASTER-PLAN.md",
        "docs/planning/vetinari-factory-plan.md",
        "docs/research/2026-04-06-action-plan.md",
        "docs/STATE-OF-VETINARI.md",
        "adr/ADR-0103.json",
        "prompts/t1_run.txt",
        "AGENTS.md",
        "vetinari/web/CLAUDE.md",
        ".github/workflows/vetinari-ci.yml",
    }.issubset(paths)


def test_tracked_boundary_rejects_model_blobs_and_root_probe_artifacts() -> None:
    findings = inspect_tracked_paths(
        [
            "README.md",
            "ui/svelte/models/model-00001-of-00016.safetensors",
            "tools/rg.exe",
            "dpo_general_20260419_1254.jsonl",
            "fuzz_out.txt",
            "dict[str",
            "{})",
            "vetinari.egg-info/SOURCES.txt",
            "scripts/hook_pre_commit_gate.py",
            "scripts/claude_md_focus_updater.py",
            "scripts/refresh_ai_accelerators.py",
            "scripts/memory/cli.py",
        ]
    )

    paths = {finding.path for finding in findings}
    assert "README.md" not in paths
    assert {
        "ui/svelte/models/model-00001-of-00016.safetensors",
        "tools/rg.exe",
        "dpo_general_20260419_1254.jsonl",
        "fuzz_out.txt",
        "dict[str",
        "{})",
        "vetinari.egg-info/SOURCES.txt",
        "scripts/hook_pre_commit_gate.py",
        "scripts/claude_md_focus_updater.py",
        "scripts/refresh_ai_accelerators.py",
        "scripts/memory/cli.py",
    }.issubset(paths)


def test_content_boundary_rejects_internal_only_markers(tmp_path) -> None:
    internal_doc = tmp_path / "docs" / "STATE-OF-VETINARI.md"
    assistant_tool = tmp_path / "scripts" / "analysis_router.py"
    public_doc = tmp_path / "README.md"
    internal_doc.parent.mkdir(parents=True)
    assistant_tool.parent.mkdir(parents=True)
    internal_doc.write_text("> INTERNAL-ONLY: maintainer note\n", encoding="utf-8")
    assistant_tool.write_text("This is a tool FOR the AI assistant (Claude), not runtime code.\n", encoding="utf-8")
    public_doc.write_text("# Public docs\n", encoding="utf-8")

    findings = inspect_file_contents(
        tmp_path,
        [
            "docs/STATE-OF-VETINARI.md",
            "scripts/analysis_router.py",
            "README.md",
        ],
    )

    assert [finding.path for finding in findings] == [
        "docs/STATE-OF-VETINARI.md",
        "scripts/analysis_router.py",
    ]


def test_history_boundary_rejects_blobs_over_publication_limit() -> None:
    findings = inspect_blob_records(
        [
            BlobRecord(size_bytes=99, object_id="small", path="README.md"),
            BlobRecord(size_bytes=101, object_id="large", path="models/model.safetensors"),
        ],
        max_blob_bytes=100,
    )

    assert [finding.path for finding in findings] == ["models/model.safetensors"]
    assert "large" in findings[0].message
