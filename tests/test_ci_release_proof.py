"""Static proof tests for release-bearing CI workflow gates."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "check_ci_release_proof.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_ci_release_proof", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_ci_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    shutil.copy2(ROOT / ".github" / "workflows" / "vetinari-ci.yml", workflows / "vetinari-ci.yml")
    shutil.copy2(ROOT / ".github" / "workflows" / "audit-prevention.yml", workflows / "audit-prevention.yml")
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    return root


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_current_ci_workflows_satisfy_release_proof_contract() -> None:
    checker = _load_checker()

    findings = checker.run_checks(ROOT)

    assert findings == []


def test_direct_workflow_dispatch_shell_interpolation_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "audit-prevention.yml"
    _replace(workflow, '--deployment-path "$DEPLOYMENT_PATH"', '--deployment-path "${{ inputs.deployment_path }}"')

    findings = checker.check_workflow_dispatch_inputs(root)

    assert findings
    assert "interpolates workflow_dispatch input directly in shell" in findings[0].message


def test_unlabelled_continue_on_error_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "vetinari-ci.yml"
    _replace(workflow, "continue-on-error: false", "continue-on-error: true")

    findings = checker.check_continue_on_error(root)

    assert findings
    assert "continue-on-error" in findings[0].message


def test_unsecure_node20_action_runtime_escape_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "vetinari-ci.yml"
    _replace(
        workflow,
        'FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"',
        'FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"\n  ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION: "true"',
    )

    findings = checker.check_node_runtime_horizon(root)

    assert findings
    assert "expired Node20" in findings[0].message


def test_missing_drift_coverage_gate_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "vetinari-ci.yml"
    _replace(workflow, "        python scripts/check_coverage_gate.py --report coverage.json\n", "")

    findings = checker.check_drift_coverage_gate(root)

    assert findings
    assert "coverage without blocking gate" in findings[0].message


def test_missing_audit_prevention_guard_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "vetinari-ci.yml"
    _replace(workflow, "    - name: Run audit prevention regression guard\n      run: python scripts/check_audit_prevention.py\n\n", "")
    text = workflow.read_text(encoding="utf-8")
    workflow.write_text(text.replace("        python scripts/check_audit_prevention.py\n", ""), encoding="utf-8")

    findings = checker.check_required_release_proof(root)

    assert findings
    assert "audit prevention regression guard" in findings[0].message


def test_missing_artifact_hygiene_gate_is_rejected(tmp_path: Path) -> None:
    checker = _load_checker()
    root = _copy_ci_inputs(tmp_path)
    workflow = root / ".github" / "workflows" / "vetinari-ci.yml"
    text = workflow.read_text(encoding="utf-8")
    updated = text.replace("        python scripts/check_release_artifacts.py --dist-dir dist\n", "", 1)
    assert updated != text
    workflow.write_text(updated, encoding="utf-8")

    findings = checker.check_required_release_proof(root)

    assert findings
    assert "artifact hygiene gate" in findings[0].message
