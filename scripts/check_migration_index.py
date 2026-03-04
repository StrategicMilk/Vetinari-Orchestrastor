#!/usr/bin/env python3
"""
Migration Index Status Checker — Phase 7 Drift Control

Verifies that MIGRATION_INDEX.md reflects actual codebase state across
all phases (0-7).

Usage:
    python scripts/check_migration_index.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _check_file_exists(rel_path: str) -> bool:
    return (project_root / rel_path).exists()


# ─── Per-phase artifact checks ────────────────────────────────────────────────

def check_phase0_artifacts() -> bool:
    _section("PHASE 0 — Foundations")
    artifacts = {
        "vetinari/agents/contracts.py":            "Data contracts & registry",
        "vetinari/agents/base_agent.py":           "Base agent class",
        "vetinari/agents/planner_agent.py":        "Planner agent",
        "vetinari/orchestration/agent_graph.py":   "AgentGraph orchestration",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase2_artifacts() -> bool:
    _section("PHASE 2 — Tool Interface Migration")
    artifacts = {
        "vetinari/tool_interface.py":    "Tool/ToolResult protocol",
        "vetinari/execution_context.py": "ExecutionContext",
        "tests/test_tool_interface.py":  "Tool interface tests",
        "tests/test_execution_context.py": "Context tests",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase3_artifacts() -> bool:
    _section("PHASE 3 — Observability & Security")
    artifacts = {
        "vetinari/telemetry.py":          "Telemetry collector",
        "vetinari/security.py":           "Secret scanner",
        "vetinari/structured_logging.py": "Structured logging + tracing",
        "tests/test_telemetry.py":        "Telemetry tests",
        "tests/test_security.py":         "Security tests",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase4_artifacts() -> bool:
    _section("PHASE 4 — Dashboard & Monitoring")
    artifacts = {
        "vetinari/dashboard/api.py":              "Dashboard API",
        "vetinari/dashboard/rest_api.py":         "Flask REST API",
        "vetinari/dashboard/alerts.py":           "Alert engine",
        "vetinari/dashboard/log_aggregator.py":   "Log aggregator",
        "ui/templates/dashboard.html":            "Dashboard UI",
        "tests/test_dashboard_api.py":            "Dashboard API tests",
        "tests/test_dashboard_alerts.py":         "Alert tests",
        "tests/test_dashboard_log_aggregator.py": "Log aggregator tests",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase5_artifacts() -> bool:
    _section("PHASE 5 — Advanced Analytics")
    artifacts = {
        "vetinari/analytics/anomaly.py":              "Anomaly detector",
        "vetinari/analytics/cost.py":                 "Cost tracker",
        "vetinari/analytics/sla.py":                  "SLA tracker",
        "vetinari/analytics/forecasting.py":          "Forecaster",
        "tests/test_analytics_anomaly.py":            "Anomaly tests",
        "tests/test_analytics_cost.py":               "Cost tests",
        "tests/test_analytics_sla.py":                "SLA tests",
        "tests/test_analytics_forecasting.py":        "Forecasting tests",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase6_artifacts() -> bool:
    _section("PHASE 6 — Production Readiness")
    artifacts = {
        ".github/workflows/vetinari-ci.yml":         "CI/CD pipeline",
        "tests/regression/test_regression_phase4.py":"Phase 4 regression",
        "tests/regression/test_regression_phase5.py":"Phase 5 regression",
        "templates/migrations/new_skill_template.py":"Migration template",
        "docs/onboarding/ONBOARDING.md":             "Onboarding guide",
        "docs/PRODUCTION.md":                        "Production guide",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_phase7_artifacts() -> bool:
    _section("PHASE 7 — Drift Control")
    artifacts = {
        "vetinari/drift/__init__.py":               "Drift package",
        "vetinari/drift/contract_registry.py":      "Contract registry",
        "vetinari/drift/capability_auditor.py":     "Capability auditor",
        "vetinari/drift/schema_validator.py":       "Schema validator",
        "vetinari/drift/monitor.py":                "Drift monitor",
        "scripts/check_doc_contract_alignment.py":  "Alignment checker script",
        "scripts/check_migration_index.py":         "Migration index checker",
        "scripts/check_agent_capabilities.py":      "Capability checker script",
        "scripts/check_coverage_gate.py":           "Coverage gate script",
        "tests/test_drift_control.py":              "Drift control tests",
    }
    ok = True
    for path, desc in artifacts.items():
        exists = _check_file_exists(path)
        print(f"  {'OK' if exists else 'MISS'}  {path}  ({desc})")
        if not exists:
            ok = False
    return ok


def check_agent_implementations() -> bool:
    _section("ALL 15 AGENT IMPLEMENTATIONS")
    agent_files = [
        "vetinari/agents/planner_agent.py",
        "vetinari/agents/explorer_agent.py",
        "vetinari/agents/librarian_agent.py",
        "vetinari/agents/oracle_agent.py",
        "vetinari/agents/researcher_agent.py",
        "vetinari/agents/evaluator_agent.py",
        "vetinari/agents/synthesizer_agent.py",
        "vetinari/agents/builder_agent.py",
        "vetinari/agents/ui_planner_agent.py",
        "vetinari/agents/security_auditor_agent.py",
        "vetinari/agents/data_engineer_agent.py",
        "vetinari/agents/documentation_agent.py",
        "vetinari/agents/cost_planner_agent.py",
        "vetinari/agents/test_automation_agent.py",
        "vetinari/agents/experimentation_manager_agent.py",
    ]
    found = sum(1 for p in agent_files if _check_file_exists(p))
    for p in agent_files:
        print(f"  {'OK' if _check_file_exists(p) else 'MISS'}  {p}")
    print(f"\n  {found}/{len(agent_files)} agents implemented")
    return found == len(agent_files)


def check_migration_index_content() -> bool:
    _section("MIGRATION_INDEX.md CONTENT")
    path = project_root / "docs" / "MIGRATION_INDEX.md"
    if not path.exists():
        print("  FAIL: MIGRATION_INDEX.md not found")
        return False

    content = path.read_text()
    required_markers = [
        ("Phase 2 Complete",  "Tool Interface Migration | **Complete**"),
        ("Phase 3 Complete",  "Observability & Security | **Complete**"),
        ("Phase 4 Complete",  "Dashboard & Monitoring | **Complete**"),
        ("Phase 5 Complete",  "Advanced Analytics | **Complete**"),
        ("Phase 6 Complete",  "Production Readiness | **Complete**"),
    ]
    ok = True
    for label, marker in required_markers:
        present = marker in content
        print(f"  {'OK' if present else 'MISS'}  {label}")
        if not present:
            ok = False
    return ok


def check_test_suite_health() -> bool:
    _section("TEST SUITE HEALTH")
    critical_test_files = [
        "tests/test_agent_contracts.py",
        "tests/test_base_agent.py",
        "tests/test_agent_graph.py",
        "tests/test_tool_interface.py",
        "tests/test_telemetry.py",
        "tests/test_security.py",
        "tests/test_dashboard_api.py",
        "tests/test_analytics_anomaly.py",
        "tests/regression/test_regression_phase4.py",
        "tests/regression/test_regression_phase5.py",
        "tests/test_drift_control.py",
    ]
    missing = [p for p in critical_test_files if not _check_file_exists(p)]
    for p in critical_test_files:
        print(f"  {'OK' if _check_file_exists(p) else 'MISS'}  {p}")
    if missing:
        print(f"\n  FAIL: {len(missing)} critical test file(s) missing.")
        return False
    print(f"\n  PASS: All {len(critical_test_files)} critical test files present.")
    return True


def main() -> int:
    print("\n" + "#" * 70)
    print("# Migration Index Status Checker — All Phases (0-7)")
    print("#" * 70)

    checks = [
        ("Phase 0 Artifacts",         check_phase0_artifacts),
        ("Phase 2 Artifacts",         check_phase2_artifacts),
        ("Phase 3 Artifacts",         check_phase3_artifacts),
        ("Phase 4 Artifacts",         check_phase4_artifacts),
        ("Phase 5 Artifacts",         check_phase5_artifacts),
        ("Phase 6 Artifacts",         check_phase6_artifacts),
        ("Phase 7 Artifacts",         check_phase7_artifacts),
        ("All 15 Agents",             check_agent_implementations),
        ("Migration Index Content",   check_migration_index_content),
        ("Test Suite Health",         check_test_suite_health),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  ERROR in {name}: {exc}")
            results[name] = False

    _section("FINAL SUMMARY")
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{total} checks passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
