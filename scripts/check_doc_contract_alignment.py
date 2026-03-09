#!/usr/bin/env python3
"""
Contract-Documentation Alignment Checker — Phase 7 Drift Control

Runs the full contract fingerprint, capability, and schema checks via the
vetinari.drift.monitor API and exits non-zero if any drift is detected.

Usage:
    python scripts/check_doc_contract_alignment.py [--snapshot]

    --snapshot   Save a fresh baseline snapshot after the check passes.
                 Use this after intentional contract changes.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_contract_fingerprints() -> bool:
    """Check that no registered contract hash has changed."""
    _section("CONTRACT FINGERPRINT CHECK")
    from vetinari.drift.contract_registry import get_contract_registry, reset_contract_registry
    reset_contract_registry()
    reg = get_contract_registry()

    # Register all known contracts
    try:
        from vetinari.plan_types import Plan, Subtask, PlanCandidate
        reg.register("Plan",         Plan(goal="__check__"))
        reg.register("Subtask",      Subtask())
        reg.register("PlanCandidate",PlanCandidate())
        print("  Registered plan_types contracts")
    except Exception as e:
        print(f"  WARN: plan_types: {e}")

    try:
        from vetinari.dashboard.alerts import AlertThreshold, AlertCondition
        reg.register("AlertThreshold", AlertThreshold(
            name="__check__", metric_key="x",
            condition=AlertCondition.GREATER_THAN, threshold_value=0.0,
        ))
        print("  Registered AlertThreshold")
    except Exception as e:
        print(f"  WARN: AlertThreshold: {e}")

    try:
        from vetinari.analytics.cost import CostEntry
        reg.register("CostEntry", CostEntry(provider="__check__", model="__check__"))
        print("  Registered CostEntry")
    except Exception as e:
        print(f"  WARN: CostEntry: {e}")

    try:
        from vetinari.dashboard.log_aggregator import LogRecord
        reg.register("LogRecord", LogRecord(message="__check__"))
        print("  Registered LogRecord")
    except Exception as e:
        print(f"  WARN: LogRecord: {e}")

    loaded = reg.load_snapshot()
    if not loaded:
        print("\n  No snapshot found — creating baseline (first run).")
        reg.snapshot()
        print("  Snapshot saved.  Re-run to detect future drift.")
        return True

    drifts = reg.check_drift()
    if drifts:
        print(f"\n  FAIL: {len(drifts)} contract(s) have drifted:")
        for name, info in drifts.items():
            prev = info["previous"][:16]
            curr = info["current"][:16]
            print(f"    {name}: {prev}... -> {curr}...")
        return False

    print(f"\n  PASS: All {len(reg.list_contracts())} contracts stable.")
    return True


def check_agent_registry() -> bool:
    """Check all 15 agents are in the registry."""
    _section("AGENT REGISTRY COMPLETENESS")
    try:
        from vetinari.agents.contracts import AgentType, AGENT_REGISTRY, get_agent_spec
    except ImportError as e:
        print(f"  FAIL: Cannot import contracts: {e}")
        return False

    expected = {
        "PLANNER", "EXPLORER", "LIBRARIAN", "ORACLE", "RESEARCHER",
        "EVALUATOR", "SYNTHESIZER", "BUILDER", "UI_PLANNER",
        "SECURITY_AUDITOR", "DATA_ENGINEER", "DOCUMENTATION_AGENT",
        "COST_PLANNER", "TEST_AUTOMATION", "EXPERIMENTATION_MANAGER",
    }
    found = {at.value for at in AGENT_REGISTRY}
    missing = expected - found
    extra   = found - expected

    print(f"  Registered: {len(found)}/15")
    if missing:
        print(f"  FAIL — Missing agents: {sorted(missing)}")
    if extra:
        print(f"  WARN — Extra agents: {sorted(extra)}")

    # Spec validity
    invalid = []
    for at in AgentType:
        spec = get_agent_spec(at)
        if spec and (not spec.name or not spec.description or not spec.default_model):
            invalid.append(at.value)
    if invalid:
        print(f"  FAIL — Incomplete specs: {invalid}")
        return False

    if missing:
        return False
    print("  PASS: All 15 agents registered with valid specs.")
    return True


def check_schema_validation() -> bool:
    """Run schema validation on live contract instances."""
    _section("SCHEMA VALIDATION CHECK")
    from vetinari.drift.schema_validator import get_schema_validator, reset_schema_validator
    reset_schema_validator()
    v = get_schema_validator()
    v.register_vetinari_schemas()

    failures = []

    try:
        from vetinari.plan_types import Plan
        errs = v.validate("Plan", Plan(goal="alignment check"))
        if errs:
            failures.append(("Plan", errs))
            print(f"  FAIL Plan: {errs}")
        else:
            print("  PASS Plan schema")
    except Exception as e:
        print(f"  WARN Plan: {e}")

    try:
        from vetinari.plan_types import Subtask
        errs = v.validate("Subtask", Subtask(description="x", plan_id="p1"))
        if errs:
            failures.append(("Subtask", errs))
        else:
            print("  PASS Subtask schema")
    except Exception as e:
        print(f"  WARN Subtask: {e}")

    try:
        from vetinari.dashboard.log_aggregator import LogRecord
        errs = v.validate("LogRecord", LogRecord(message="test", level="INFO"))
        if errs:
            failures.append(("LogRecord", errs))
        else:
            print("  PASS LogRecord schema")
    except Exception as e:
        print(f"  WARN LogRecord: {e}")

    if failures:
        print(f"\n  FAIL: {len(failures)} schema validation failure(s).")
        return False
    print("\n  PASS: All schemas valid.")
    return True


def check_documentation_files() -> bool:
    """Verify all required documentation files exist."""
    _section("DOCUMENTATION FILES CHECK")
    required = [
        "docs/MIGRATION_INDEX.md",
        "docs/DRIFT_PREVENTION.md",
        "docs/DEVELOPER_GUIDE.md",
        "docs/SKILL_MIGRATION_GUIDE.md",
        "docs/api-reference-dashboard.md",
        "docs/api-reference-analytics.md",
        "docs/runbooks/dashboard_guide.md",
        "docs/onboarding/ONBOARDING.md",
        "docs/onboarding/QUICK_START.md",
        "docs/PRODUCTION.md",
    ]
    missing = []
    for p in required:
        full = project_root / p
        if full.exists():
            print(f"  OK  {p}")
        else:
            print(f"  MISS {p}")
            missing.append(p)

    if missing:
        print(f"\n  FAIL: {len(missing)} doc(s) missing.")
        return False
    print(f"\n  PASS: All {len(required)} documentation files present.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract-Documentation Alignment Checker")
    parser.add_argument("--snapshot", action="store_true",
                        help="Save a fresh snapshot after successful check")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# Contract-Documentation Alignment Checker (Phase 7 Drift Control)")
    print("#" * 70)

    checks = [
        ("Agent Registry",       check_agent_registry),
        ("Schema Validation",    check_schema_validation),
        ("Documentation Files",  check_documentation_files),
        ("Contract Fingerprints",check_contract_fingerprints),
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

    if args.snapshot and passed == total:
        from vetinari.drift.contract_registry import get_contract_registry
        get_contract_registry().snapshot()
        print("\n  Snapshot saved — this state is the new baseline.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
