#!/usr/bin/env python3
"""Fail closed when the Git publication boundary contains private or oversized content."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_BLOB_BYTES = 100 * 1024 * 1024
GIT = shutil.which("git") or "git"

FORBIDDEN_TRACKED_PREFIXES = (
    ".claire/",
    ".agents/",
    ".ai-codex/",
    ".claude/",
    ".codex/",
    ".omc/",
    ".pytest-probe",
    "adr/",
    "docs/archive/",
    "docs/audit/",
    "docs/benchmarks/",
    "docs/internal/",
    "docs/planning/",
    "docs/plans/",
    "docs/research/",
    "prompts/",
    "simpo_output/",
    "ui/svelte/models/",
    "vetinari.egg-info/",
)

FORBIDDEN_TRACKED_EXACT = {
    ".claudeignore",
    ".ignore",
    ".lumenignore",
    ".mcp.json",
    ".github/workflows/audit-prevention.yml",
    ".github/workflows/vetinari-ci.yml",
    "AGENTS.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    "CLEANUP_PLAN.md",
    "docs/MIGRATION_INDEX.md",
    "docs/STATE-OF-VETINARI.md",
    "docs/security/security-audit-report.md",
    "docs/security/windows-report.md",
    "audit_output.json",
    "coverage.json",
    "progress.txt",
    "rules_out.txt",
    "skills-lock.json",
    "training_data.jsonl",
    "verify_bypass_detector_logic.py",
}

FORBIDDEN_TRACKED_SUFFIXES = (
    ".ckpt",
    ".db",
    ".db-shm",
    ".db-wal",
    ".dll",
    ".dylib",
    ".egg-info",
    ".exe",
    ".gguf",
    ".node",
    ".onnx",
    ".pt",
    ".pth",
    ".pyd",
    ".safetensors",
    ".so",
)

FORBIDDEN_ROOT_PREFIXES = (
    "dpo_general_",
    "fuzz_",
    "pytest_",
    "test_",
)

FORBIDDEN_SCRIPT_NAMES = {
    "ai_accelerators.py",
    "ai_wiki.py",
    "agent_corrections_updater.py",
    "agent_scorecard.py",
    "agent_token_cost_audit.py",
    "analysis_router.py",
    "check_agent_routing.py",
    "check_ai_workflow_drift.py",
    "check_handoff_bundle.py",
    "claude_md_focus_updater.py",
    "codex_code_intel_mcp.py",
    "create_adrs.py",
    "create_dept9_adrs.py",
    "generate_ai_codex_indexes.py",
    "generate_session_evidence_matrix.py",
    "handoff_bundle.py",
    "memory_cli.py",
    "memory_lint.py",
    "session_capsule.py",
    "session_continuity_audit.py",
    "session_quality.py",
    "session_trends.py",
    "reapply_omc_mods.py",
    "refresh_ai_accelerators.py",
    "skill_command_budget_doctor.py",
    "skill_description_tuner.py",
    "subagent_bootstrap.py",
    "subagent_prompt_audit.py",
    "token_usage.py",
    "tool_discovery_audit.py",
    "tool_pair_audit.py",
    "tool_value_audit.py",
    "usage_prune_report.py",
    "verification_compliance_audit.py",
}

FORBIDDEN_SCRIPT_PREFIXES = (
    "hook_",
    "memory/",
)

FORBIDDEN_CONTENT_MARKERS = (
    "INTERNAL-ONLY",
    "This is a tool FOR the AI assistant",
)

CONTENT_MARKER_ALLOWLIST = {
    "scripts/release/check_publication_boundary.py",
    "tests/test_publication_boundary.py",
}


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    message: str

    def format(self) -> str:
        return f"{self.check}: {self.path}: {self.message}"


@dataclass(frozen=True)
class BlobRecord:
    size_bytes: int
    object_id: str
    path: str


def _normalise_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def inspect_tracked_paths(paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for raw_path in paths:
        path = _normalise_path(raw_path)
        if not path:
            continue
        lower = path.lower()
        root_name = lower.split("/", 1)[0]
        basename = lower.rsplit("/", 1)[-1]

        if path in FORBIDDEN_TRACKED_EXACT or basename in {"agents.md", "claude.md", "claude.local.md"}:
            findings.append(Finding("tracked-public-boundary", path, "internal or generated root file is tracked"))
            continue
        if any(lower.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            findings.append(Finding("tracked-public-boundary", path, "internal/private tree is tracked"))
            continue
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_TRACKED_SUFFIXES):
            findings.append(Finding("tracked-public-boundary", path, "binary/generated artifact is tracked"))
            continue
        if lower.startswith("scripts/"):
            script_rel = lower.split("/", 1)[1]
            script_basename = script_rel.rsplit("/", 1)[-1]
            if script_basename in FORBIDDEN_SCRIPT_NAMES or any(
                script_rel.startswith(prefix) or script_basename.startswith(prefix)
                for prefix in FORBIDDEN_SCRIPT_PREFIXES
            ):
                findings.append(Finding("tracked-public-boundary", path, "internal AI-workflow script is tracked"))
                continue
        if "/" not in path and any(root_name.startswith(prefix) for prefix in FORBIDDEN_ROOT_PREFIXES):
            findings.append(Finding("tracked-public-boundary", path, "root-level probe or test-output artifact is tracked"))
            continue
        if "/" not in path and any(char in path for char in "[]{}()"):
            findings.append(Finding("tracked-public-boundary", path, "malformed root artifact is tracked"))
    return findings


def inspect_file_contents(root: Path, paths: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for raw_path in paths:
        path = _normalise_path(raw_path)
        if not path:
            continue
        if path in CONTENT_MARKER_ALLOWLIST:
            continue
        full_path = root / path
        if not full_path.is_file():
            continue
        try:
            text = full_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for marker in FORBIDDEN_CONTENT_MARKERS:
            if marker in text:
                findings.append(
                    Finding("content-public-boundary", path, f"file content contains private marker {marker!r}")
                )
                break
    return findings


def inspect_blob_records(records: list[BlobRecord], *, max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES) -> list[Finding]:
    findings: list[Finding] = []
    for record in records:
        if record.size_bytes <= max_blob_bytes:
            continue
        size_mib = record.size_bytes / (1024 * 1024)
        limit_mib = max_blob_bytes / (1024 * 1024)
        findings.append(
            Finding(
                "git-history-size",
                record.path,
                f"{size_mib:.2f} MiB blob {record.object_id} exceeds {limit_mib:.2f} MiB publication limit",
            )
        )
    return findings


def _run_git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - argv is assembled from fixed git subcommands in this module.
        [GIT, *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _tracked_paths(root: Path) -> list[str]:
    return [line for line in _run_git(root, ["ls-files"]).splitlines() if line.strip()]


def _blob_records(root: Path) -> list[BlobRecord]:
    object_rows = _run_git(root, ["rev-list", "--objects", "--all"])
    if not object_rows.strip():
        return []

    completed = subprocess.run(  # noqa: S603 - fixed git subcommand with stdin from rev-list output.
        [GIT, "cat-file", "--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)"],
        cwd=root,
        input=object_rows,
        check=True,
        text=True,
        capture_output=True,
    )
    records: list[BlobRecord] = []
    for line in completed.stdout.splitlines():
        parts = line.split(" ", 3)
        if len(parts) != 4 or parts[0] != "blob":
            continue
        records.append(BlobRecord(size_bytes=int(parts[2]), object_id=parts[1], path=parts[3]))
    return records


def run_checks(root: Path, *, max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES) -> list[Finding]:
    paths = _tracked_paths(root)
    findings = inspect_tracked_paths(paths)
    findings.extend(inspect_file_contents(root, paths))
    findings.extend(inspect_blob_records(_blob_records(root), max_blob_bytes=max_blob_bytes))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Git publication boundary before pushing/exporting.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to inspect.")
    parser.add_argument("--max-blob-bytes", type=int, default=DEFAULT_MAX_BLOB_BYTES)
    args = parser.parse_args(argv)

    try:
        findings = run_checks(args.root.resolve(), max_blob_bytes=args.max_blob_bytes)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"Publication boundary check could not inspect git state: {stderr}", file=sys.stderr)
        return 2

    if findings:
        print(f"Publication boundary check failed with {len(findings)} finding(s):", file=sys.stderr)
        for finding in findings:
            print(f"- {finding.format()}", file=sys.stderr)
        return 1

    print("Publication boundary check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
