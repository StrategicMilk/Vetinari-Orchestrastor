"""Regression tests for full-spectrum audit prevention gates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "quality" / "check_audit_prevention.py"
VALID_ACTION_SHA = "c5a7806660adbe173f04e3e038b0ccdcd758773c"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_audit_prevention", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_valid_baseline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """{
  "baseline_version": 0,
  "recorded_at": null,
  "machine_profile": null,
  "workloads": {
    "small_task_end_to_end": {},
    "medium_task_end_to_end": {},
    "complex_task_5_subtasks": {},
    "long_running_project": {},
    "startup_cost": {},
    "idle_cost": {}
  }
}
""",
        encoding="utf-8",
    )


def _write_all_valid_baselines(root: Path) -> None:
    _write_valid_baseline(root / "config" / "resource_baselines.json")
    _write_valid_baseline(root / "vetinari" / "config" / "runtime" / "resource_baselines.json")


def _write_workflow(root: Path, text: str) -> Path:
    path = root / ".github" / "workflows" / "audit-prevention.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_resource_baseline_rejects_null_bytes(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_all_valid_baselines(tmp_path)
    (tmp_path / "config" / "resource_baselines.json").write_bytes(b"\x00\x00\x00")

    findings = checker.check_resource_baselines(tmp_path)

    assert findings
    assert "NUL bytes" in findings[0].message


def test_resource_baseline_accepts_placeholder_schema(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_all_valid_baselines(tmp_path)

    findings = checker.check_resource_baselines(tmp_path)

    assert findings == []


def test_active_docs_reject_current_plan_admin_token_usage(tmp_path: Path) -> None:
    checker = _load_checker()
    doc = tmp_path / "docs" / "runbooks" / "operator.md"
    doc.parent.mkdir(parents=True)
    doc.write_text('-H "Authorization: Bearer $PLAN_ADMIN_TOKEN"\n', encoding="utf-8")

    findings = checker.check_admin_token_docs(tmp_path)

    assert findings
    assert "VETINARI_ADMIN_TOKEN" in findings[0].message


def test_legacy_plan_admin_token_reference_is_allowed(tmp_path: Path) -> None:
    checker = _load_checker()
    doc = tmp_path / "docs" / "reference" / "config.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(
        "| PLAN_ADMIN_TOKEN | (none) | Legacy plan-memory/status flag; not accepted by Litestar admin_guard |\n",
        encoding="utf-8",
    )

    findings = checker.check_admin_token_docs(tmp_path)

    assert findings == []


def test_docs_reject_stale_litestar_app_target(tmp_path: Path) -> None:
    checker = _load_checker()
    doc = tmp_path / "docs" / "runbooks" / "server.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("uvicorn vetinari.web.litestar_app:app --port 5000\n", encoding="utf-8")

    findings = checker.check_uvicorn_docs(tmp_path)

    assert findings
    assert "does not exist" in findings[0].message


def test_docs_require_factory_flag_for_app_factories(tmp_path: Path) -> None:
    checker = _load_checker()
    doc = tmp_path / "docs" / "runbooks" / "server.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("uvicorn vetinari.web.litestar_app:create_app --port 5000\n", encoding="utf-8")

    findings = checker.check_uvicorn_docs(tmp_path)

    assert findings
    assert "--factory" in findings[0].message


def test_docs_accept_current_uvicorn_factory_target(tmp_path: Path) -> None:
    checker = _load_checker()
    doc = tmp_path / "docs" / "runbooks" / "server.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("uvicorn vetinari.web.litestar_app:get_app --factory --port 5000\n", encoding="utf-8")

    findings = checker.check_uvicorn_docs(tmp_path)

    assert findings == []


def test_write_token_workflow_requires_sha_for_third_party_action(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_workflow(
        tmp_path,
        """
name: Audit Prevention
on: workflow_dispatch
jobs:
  promote:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: peter-evans/create-pull-request@v6
""",
    )

    findings = checker.check_privileged_workflow_actions(tmp_path)

    assert findings
    assert "pin a 40-char SHA" in findings[0].message


def test_write_token_workflow_accepts_sha_pinned_third_party_action(tmp_path: Path) -> None:
    checker = _load_checker()
    _write_workflow(
        tmp_path,
        f"""
name: Audit Prevention
on: workflow_dispatch
jobs:
  promote:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: peter-evans/create-pull-request@{VALID_ACTION_SHA}
""",
    )

    findings = checker.check_privileged_workflow_actions(tmp_path)

    assert findings == []
