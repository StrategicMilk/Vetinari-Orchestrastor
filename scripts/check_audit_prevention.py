#!/usr/bin/env python3
"""Fail closed on repeatable full-spectrum audit regression patterns.

This checker covers findings that are cheap to prove locally and broad enough
to deserve a hard gate: corrupt resource baselines, stale active runbooks,
stale ASGI entry points, and privileged workflow actions that are not pinned.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in CI/dev.
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]

RESOURCE_BASELINES = (
    Path("config/resource_baselines.json"),
    Path("vetinari/config/runtime/resource_baselines.json"),
)
REQUIRED_BASELINE_WORKLOADS = {
    "small_task_end_to_end",
    "medium_task_end_to_end",
    "complex_task_5_subtasks",
    "long_running_project",
    "startup_cost",
    "idle_cost",
}

ACTIVE_DOC_FILES = (Path("README.md"),)
ACTIVE_DOC_DIRS = (
    Path("docs/runbooks"),
    Path("docs/reference"),
    Path("docs/getting-started"),
    Path("docs/security"),
)
PLAN_ADMIN_TOKEN_RE = re.compile(r"\bPLAN_ADMIN_TOKEN\b")
PLAN_ADMIN_TOKEN_ALLOWED_RE = re.compile(
    r"\b(legacy|deprecated|historical|not accepted|not current|not a current|status[- ]only)\b",
    re.IGNORECASE,
)

UVICORN_TARGET_RE = re.compile(r"\buvicorn\s+([A-Za-z_][\w.]*:[A-Za-z_]\w*)[^\n`]*")
FACTORY_TARGET_NAMES = {"create_app", "get_app"}

WRITE_PERMISSION_SCOPES = {"contents", "pull-requests"}
OFFICIAL_ACTION_OWNERS = {"actions"}
ACTION_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

CI_WORKFLOW = Path(".github/workflows/vetinari-ci.yml")
AUDIT_PREVENTION_WORKFLOW = Path(".github/workflows/audit-prevention.yml")


@dataclass(frozen=True)
class Finding:
    """A blocking audit-prevention finding."""

    path: Path
    message: str
    line: int | None = None

    def format(self) -> str:
        if self.line is None:
            return f"{self.path}: {self.message}"
        return f"{self.path}:{self.line}: {self.message}"


def _display_path(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _line_number(text: str, needle: str) -> int | None:
    index = text.find(needle)
    if index < 0:
        return None
    return text.count("\n", 0, index) + 1


def _iter_active_docs(root: Path) -> list[Path]:
    docs: list[Path] = []
    for rel_path in ACTIVE_DOC_FILES:
        path = root / rel_path
        if path.exists():
            docs.append(path)
    for rel_dir in ACTIVE_DOC_DIRS:
        directory = root / rel_dir
        if directory.exists():
            docs.extend(sorted(directory.rglob("*.md")))
    return sorted(set(docs))


def check_resource_baselines(root: Path) -> list[Finding]:
    """Validate resource baseline files as JSON with the expected workload skeleton."""

    findings: list[Finding] = []
    for rel_path in RESOURCE_BASELINES:
        path = root / rel_path
        display_path = _display_path(root, path)
        if not path.exists():
            findings.append(Finding(display_path, "resource baseline file is missing"))
            continue

        raw = path.read_bytes()
        if not raw.strip():
            findings.append(Finding(display_path, "resource baseline file is empty"))
            continue
        if b"\x00" in raw:
            findings.append(Finding(display_path, "resource baseline contains NUL bytes; regenerate valid JSON"))
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            findings.append(Finding(display_path, f"resource baseline is not UTF-8 JSON: {exc}"))
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            findings.append(Finding(display_path, f"resource baseline is invalid JSON: {exc.msg}"))
            continue

        if not isinstance(data, dict):
            findings.append(Finding(display_path, "resource baseline root must be a JSON object"))
            continue

        baseline_version = data.get("baseline_version")
        if not isinstance(baseline_version, int) or baseline_version < 0:
            findings.append(Finding(display_path, "baseline_version must be a non-negative integer"))

        workloads = data.get("workloads")
        if not isinstance(workloads, dict):
            findings.append(Finding(display_path, "workloads must be a JSON object"))
            continue

        missing = sorted(REQUIRED_BASELINE_WORKLOADS - set(workloads))
        if missing:
            findings.append(Finding(display_path, f"resource baseline omits workloads: {', '.join(missing)}"))

        for workload_name in sorted(REQUIRED_BASELINE_WORKLOADS & set(workloads)):
            if not isinstance(workloads[workload_name], dict):
                findings.append(Finding(display_path, f"workload {workload_name!r} must be a JSON object"))

        if baseline_version and (not data.get("recorded_at") or not data.get("machine_profile")):
            findings.append(
                Finding(
                    display_path,
                    "measured baselines require recorded_at and machine_profile; use version 0 for placeholders",
                )
            )

    return findings


def check_admin_token_docs(root: Path) -> list[Finding]:
    """Reject active docs that still tell operators to use PLAN_ADMIN_TOKEN."""

    findings: list[Finding] = []
    for path in _iter_active_docs(root):
        display_path = _display_path(root, path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not PLAN_ADMIN_TOKEN_RE.search(line):
                continue
            if PLAN_ADMIN_TOKEN_ALLOWED_RE.search(line):
                continue
            findings.append(
                Finding(
                    display_path,
                    "active operator docs mention PLAN_ADMIN_TOKEN as current auth; use VETINARI_ADMIN_TOKEN",
                    line_number,
                )
            )
    return findings


def _resolve_uvicorn_target(root: Path, target: str) -> str | None:
    module_name, attr_name = target.split(":", 1)
    root_text = str(root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised through message path.
        return f"documented uvicorn target {target!r} is not importable: {type(exc).__name__}: {exc}"
    finally:
        if inserted:
            with suppress(ValueError):
                sys.path.remove(root_text)

    if not hasattr(module, attr_name):
        return f"documented uvicorn target {target!r} does not exist"
    return None


def check_uvicorn_docs(root: Path) -> list[Finding]:
    """Validate ASGI entry points documented in active operator docs."""

    findings: list[Finding] = []
    seen_targets: dict[str, str | None] = {}
    for path in _iter_active_docs(root):
        display_path = _display_path(root, path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in UVICORN_TARGET_RE.finditer(line):
                command = match.group(0)
                target = match.group(1)
                _, attr_name = target.split(":", 1)
                if attr_name in FACTORY_TARGET_NAMES and "--factory" not in command:
                    findings.append(
                        Finding(display_path, f"uvicorn factory target {target!r} must use --factory", line_number)
                    )
                    continue
                if target not in seen_targets:
                    seen_targets[target] = _resolve_uvicorn_target(root, target)
                error = seen_targets[target]
                if error:
                    findings.append(Finding(display_path, error, line_number))
    return findings


def _workflow_paths(root: Path) -> list[Path]:
    workflows = root / ".github" / "workflows"
    if not workflows.exists():
        return []
    return sorted([*workflows.glob("*.yml"), *workflows.glob("*.yaml")])


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None:
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _has_write_permissions(permissions: Any) -> bool:
    if isinstance(permissions, str):
        return "write" in permissions.lower()
    if not isinstance(permissions, dict):
        return False
    for scope, value in permissions.items():
        if str(value).lower() != "write":
            continue
        if str(scope) in WRITE_PERMISSION_SCOPES:
            return True
    return False


def _is_third_party_action(uses_value: str) -> bool:
    if uses_value.startswith("./") or uses_value.startswith("../"):
        return False
    action_name, _, _ref = uses_value.partition("@")
    owner = action_name.split("/", 1)[0].lower()
    return owner not in OFFICIAL_ACTION_OWNERS


def _is_sha_pinned(uses_value: str) -> bool:
    _action_name, separator, ref = uses_value.partition("@")
    return bool(separator and ACTION_SHA_RE.fullmatch(ref))


def check_privileged_workflow_actions(root: Path) -> list[Finding]:
    """Require SHA pins for third-party actions in jobs with write-token scopes."""

    findings: list[Finding] = []
    for path in _workflow_paths(root):
        display_path = _display_path(root, path)
        workflow = _load_yaml(path)
        if workflow is None:
            findings.append(Finding(display_path, "PyYAML is required to inspect GitHub workflow actions"))
            continue

        workflow_permissions = workflow.get("permissions")
        jobs = workflow.get("jobs", {})
        if not isinstance(jobs, dict):
            continue
        text = path.read_text(encoding="utf-8")

        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            permissions = job.get("permissions", workflow_permissions)
            if not _has_write_permissions(permissions):
                continue
            steps = job.get("steps", [])
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                uses_value = step.get("uses")
                if not isinstance(uses_value, str) or not _is_third_party_action(uses_value):
                    continue
                if _is_sha_pinned(uses_value):
                    continue
                name = step.get("name", uses_value)
                findings.append(
                    Finding(
                        display_path,
                        f"{job_id}/{name} uses third-party action {uses_value!r} with write permissions; pin a 40-char SHA",
                        _line_number(text, uses_value),
                    )
                )
    return findings


def check_audit_prevention_workflow_wiring(root: Path) -> list[Finding]:
    """Ensure the promotion workflow runs this guard before opening a PR."""

    path = root / AUDIT_PREVENTION_WORKFLOW
    display_path = _display_path(root, path)
    if not path.exists():
        return [Finding(display_path, "audit-prevention workflow is missing")]
    text = path.read_text(encoding="utf-8")
    if "scripts/check_audit_prevention.py" not in text:
        return [Finding(display_path, "audit-prevention workflow must run scripts/check_audit_prevention.py")]
    return []


def run_checks(root: Path) -> list[Finding]:
    checks = [
        check_resource_baselines,
        check_admin_token_docs,
        check_uvicorn_docs,
        check_privileged_workflow_actions,
        check_audit_prevention_workflow_wiring,
    ]
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate full-spectrum audit prevention invariants.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to check.")
    args = parser.parse_args(argv)

    findings = run_checks(args.root.resolve())
    if findings:
        print(f"Audit prevention check failed with {len(findings)} finding(s):")
        for finding in findings:
            print(f"- {finding.format()}")
        return 1

    print("Audit prevention check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
