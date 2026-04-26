#!/usr/bin/env python3
"""Fail closed on CI workflow proof gaps.

This checker is intentionally static. It proves that release-proof lanes are
present and blocking before those lanes are allowed to count as release
evidence.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = Path(".github/workflows/vetinari-ci.yml")
AUDIT_WORKFLOW = Path(".github/workflows/audit-prevention.yml")
PYPROJECT = Path("pyproject.toml")

NODE24_ENV = "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24"
UNSECURE_NODE_ENV = "ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION"

REQUIRED_RELEASE_PROOF_FRAGMENTS = {
    "ruff lint": "ruff check vetinari tests scripts",
    "test-quality gate": "scripts/check_test_quality.py",
    "VET promoted-rule gate": "scripts/check_vetinari_rules.py --yaml config/vet_rules.yaml",
    "audit prevention regression guard": "scripts/check_audit_prevention.py",
    "coverage collection": "--cov=vetinari",
    "coverage JSON": "--cov-report=json",
    "coverage gate": "scripts/check_coverage_gate.py --report coverage.json",
    "route-auth and degraded-health proof": "tests/test_litestar_dashboard_governance.py",
    "admin auth proof": "tests/test_admin_auth_hardening.py",
    "runtime config web proof": "tests/runtime/test_config_matrix_web.py",
    "package build": "python -m build --sdist --wheel",
    "artifact hygiene gate": "scripts/check_release_artifacts.py --dist-dir dist",
    "installed wrapper smoke": "vetinari --help",
    "registry validator": "validate_registry",
    "workflow proof self-check": "scripts/check_ci_release_proof.py",
    "workflow drift checker": "scripts/check_ai_workflow_drift.py",
}

REQUIRED_CHILD_SHARD_FRAGMENTS = {
    "frontend API parity child shard": "docs/plans/sessions/SESSION-34C.md",
    "Markdown/XSS child shard": "docs/plans/sessions/SESSION-34F1.md",
    "config/schema child shard": "docs/plans/sessions/SESSION-34G1.md",
    "model-download child shard": "docs/plans/sessions/SESSION-34I3.md",
}

FORBIDDEN_PARENT_PLAN_PATTERNS = {
    "34D coordination parent": r"docs/plans/sessions/SESSION-34D\.md",
    "34F coordination parent": r"docs/plans/sessions/SESSION-34F\.md",
    "34G coordination parent": r"docs/plans/sessions/SESSION-34G\.md",
    "34I coordination parent": r"docs/plans/sessions/SESSION-34I\.md",
}


@dataclass(frozen=True)
class Finding:
    path: Path
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _workflow_paths(root: Path) -> list[Path]:
    return [root / CI_WORKFLOW, root / AUDIT_WORKFLOW]


def _normalise_command(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _workflow_text(root: Path, rel_path: Path) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def _job_steps(workflow: dict[str, Any], job_id: str) -> list[dict[str, Any]]:
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return []
    job = jobs.get(job_id, {})
    if not isinstance(job, dict):
        return []
    steps = job.get("steps", [])
    return steps if isinstance(steps, list) else []


def _job_run_text(workflow: dict[str, Any], job_id: str) -> str:
    runs = []
    for step in _job_steps(workflow, job_id):
        run = step.get("run")
        if isinstance(run, str):
            runs.append(run)
    return _normalise_command("\n".join(runs))


def _all_run_steps(workflow: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return []
    results: list[tuple[str, str, dict[str, Any]]] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str):
                results.append((str(job_id), run, step))
    return results


def _all_steps(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    jobs = workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return []
    results: list[tuple[str, dict[str, Any]]] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps", [])
        if isinstance(steps, list):
            results.extend((str(job_id), step) for step in steps if isinstance(step, dict))
    return results


def _declared_python_versions(root: Path) -> set[str]:
    pyproject = tomllib.loads((root / PYPROJECT).read_text(encoding="utf-8"))
    classifiers = pyproject.get("project", {}).get("classifiers", [])
    versions = set()
    for classifier in classifiers:
        match = re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", str(classifier))
        if match:
            versions.add(match.group(1))
    return versions


def _ci_matrix_versions(workflow: dict[str, Any]) -> set[str]:
    unit_job = workflow.get("jobs", {}).get("unit-tests", {})
    matrix = unit_job.get("strategy", {}).get("matrix", {}) if isinstance(unit_job, dict) else {}
    versions = matrix.get("python-version", []) if isinstance(matrix, dict) else []
    return {str(version) for version in versions}


def check_required_release_proof(root: Path) -> list[Finding]:
    path = root / CI_WORKFLOW
    workflow = _load_yaml(path)
    run_text = _job_run_text(workflow, "release-proof")
    findings: list[Finding] = []
    if not run_text:
        return [Finding(path, "missing blocking release-proof job")]

    for label, fragment in REQUIRED_RELEASE_PROOF_FRAGMENTS.items():
        if fragment not in run_text:
            findings.append(Finding(path, f"release-proof job is missing {label}: {fragment!r}"))
    for label, fragment in REQUIRED_CHILD_SHARD_FRAGMENTS.items():
        if fragment not in run_text:
            findings.append(Finding(path, f"release-proof job is missing {label}: {fragment!r}"))
    return findings


def check_python_matrix(root: Path) -> list[Finding]:
    path = root / CI_WORKFLOW
    workflow = _load_yaml(path)
    declared = _declared_python_versions(root)
    covered = _ci_matrix_versions(workflow)
    missing = sorted(declared - covered)
    if missing:
        return [Finding(path, f"unit-test matrix omits declared Python minors: {', '.join(missing)}")]
    return []


def check_continue_on_error(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in _workflow_paths(root):
        workflow = _load_yaml(path)
        for job_id, step in _all_steps(workflow):
            if step.get("continue-on-error") is not True:
                continue
            name = str(step.get("name", "unnamed step"))
            proof_context = "proof" in job_id.lower() or "proof" in name.lower() or "release" in name.lower()
            advisory_label = "advisory" in name.lower() or "telemetry" in name.lower()
            if proof_context or not advisory_label:
                findings.append(Finding(path, f"{job_id}/{name} uses continue-on-error without advisory-only labeling"))
    return findings


def check_vet_errors_only(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in _workflow_paths(root):
        workflow = _load_yaml(path)
        for job_id, run, step in _all_run_steps(workflow):
            if "scripts/check_vetinari_rules.py" in run and "--errors-only" in run:
                name = str(step.get("name", "unnamed step"))
                findings.append(Finding(path, f"{job_id}/{name} hides warning-level VET findings with --errors-only"))
    return findings


def check_workflow_dispatch_inputs(root: Path) -> list[Finding]:
    path = root / AUDIT_WORKFLOW
    workflow = _load_yaml(path)
    findings: list[Finding] = []
    for job_id, run, step in _all_run_steps(workflow):
        if re.search(r"\$\{\{\s*(github\.event\.inputs|inputs)\.", run):
            name = str(step.get("name", "unnamed step"))
            findings.append(Finding(path, f"{job_id}/{name} interpolates workflow_dispatch input directly in shell"))
    return findings


def check_node_runtime_horizon(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    node20 = re.compile(r"\bnode20\b|node-version:\s*['\"]?20['\"]?", re.IGNORECASE)
    for rel_path in (CI_WORKFLOW, AUDIT_WORKFLOW):
        path = root / rel_path
        text = _workflow_text(root, rel_path)
        if UNSECURE_NODE_ENV in text:
            findings.append(Finding(path, f"{UNSECURE_NODE_ENV} would keep expired Node20 actions alive"))
        if node20.search(text):
            findings.append(Finding(path, "workflow pins or references Node20 after the 2026-04-30 EOL horizon"))
        if NODE24_ENV not in text or not re.search(rf"{NODE24_ENV}:\s*['\"]?true['\"]?", text):
            findings.append(
                Finding(path, f"workflow does not force JavaScript actions through Node24 with {NODE24_ENV}")
            )
    return findings


def check_parent_coordination_plans(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel_path in (CI_WORKFLOW, AUDIT_WORKFLOW):
        path = root / rel_path
        text = _workflow_text(root, rel_path)
        for label, pattern in FORBIDDEN_PARENT_PLAN_PATTERNS.items():
            if re.search(pattern, text):
                findings.append(Finding(path, f"workflow schedules coordination-only parent shard: {label}"))
    return findings


def check_observability_extra(root: Path) -> list[Finding]:
    path = root / CI_WORKFLOW
    workflow = _load_yaml(path)
    run_text = _job_run_text(workflow, "telemetry-tracing")
    findings: list[Finding] = []
    if ".[observability]" not in run_text and ".[dev,observability]" not in run_text:
        findings.append(Finding(path, "telemetry-tracing job does not install the observability extra"))
    for test_path in ("tests/test_otel_backend.py", "tests/test_otel_genai_integration.py"):
        if test_path not in run_text:
            findings.append(Finding(path, f"telemetry-tracing job does not exercise shipped OTEL helper: {test_path}"))
    return findings


def check_drift_coverage_gate(root: Path) -> list[Finding]:
    path = root / CI_WORKFLOW
    workflow = _load_yaml(path)
    run_text = _job_run_text(workflow, "drift-control")
    required = "scripts/check_coverage_gate.py --report coverage.json"
    if required not in run_text:
        return [Finding(path, f"drift-control job generates coverage without blocking gate: {required!r}")]
    return []


def run_checks(root: Path) -> list[Finding]:
    checks = [
        check_required_release_proof,
        check_python_matrix,
        check_continue_on_error,
        check_vet_errors_only,
        check_workflow_dispatch_inputs,
        check_node_runtime_horizon,
        check_parent_coordination_plans,
        check_observability_extra,
        check_drift_coverage_gate,
    ]
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate release-proof CI workflow invariants.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root to check.")
    args = parser.parse_args(argv)

    findings = run_checks(args.root.resolve())
    if findings:
        print(f"CI release proof check failed with {len(findings)} finding(s):")
        for item in findings:
            print(f"- {item.format()}")
        return 1
    print("CI release proof check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
