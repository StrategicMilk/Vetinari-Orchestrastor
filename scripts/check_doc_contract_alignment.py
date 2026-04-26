#!/usr/bin/env python3
"""Contract-Documentation Alignment Checker — bounded governance script.

Verifies four bounded conditions and exits non-zero on any failure:
  1. Agent registry completeness (3 factory-pipeline agents present)
  2. Schema validation on live contract instances
  3. Documentation file presence
  4. Contract fingerprint stability (requires an existing snapshot)

Usage:
    python scripts/check_doc_contract_alignment.py [--snapshot]

    --snapshot   Save a fresh snapshot after the check passes.
                 Use this after intentional contract changes.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

GOVERNED_CONTRACT_BASELINE: str | Path = project_root / "config" / "drift_baselines" / "contracts.json"


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def governed_contract_baseline_path() -> Path:
    """Return the repo-owned governed baseline path for contract fingerprints."""
    return Path(GOVERNED_CONTRACT_BASELINE)


def register_governed_contracts(reg) -> None:
    """Register the governed contract inventory using deterministic fixture values."""
    # Register all known contracts
    try:
        from vetinari.planning.plan_types import Plan, PlanCandidate, Subtask

        plan_candidate = PlanCandidate(
            plan_id="plan_candidate_check",
            created_at="2026-01-01T00:00:00+00:00",
        )
        subtask = Subtask(
            subtask_id="subtask_check",
            plan_id="plan_check",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        plan = Plan(
            plan_id="plan_check",
            goal="__check__",
            plan_candidates=[plan_candidate],
            subtasks=[subtask],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        reg.register("Plan", plan)
        reg.register("Subtask", subtask)
        reg.register("PlanCandidate", plan_candidate)
        print("  Registered plan_types contracts")
    except Exception as e:
        print(f"  WARN: plan_types: {e}")

    try:
        from vetinari.dashboard.alerts import AlertCondition, AlertThreshold

        reg.register(
            "AlertThreshold",
            AlertThreshold(
                name="__check__",
                metric_key="x",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=0.0,
            ),
        )
        print("  Registered AlertThreshold")
    except Exception as e:
        print(f"  WARN: AlertThreshold: {e}")

    try:
        from vetinari.analytics.cost import CostEntry

        reg.register("CostEntry", CostEntry(provider="__check__", model="__check__", timestamp=0.0))
        print("  Registered CostEntry")
    except Exception as e:
        print(f"  WARN: CostEntry: {e}")

    try:
        from vetinari.dashboard.log_aggregator import LogRecord

        reg.register("LogRecord", LogRecord(message="__check__", timestamp=0.0))
        print("  Registered LogRecord")
    except Exception as e:
        print(f"  WARN: LogRecord: {e}")


def check_contract_fingerprints() -> bool:
    """Check that no registered contract hash has changed."""
    _section("CONTRACT FINGERPRINT CHECK")
    from vetinari.drift.contract_registry import get_contract_registry, reset_contract_registry

    reset_contract_registry()
    reg = get_contract_registry()
    register_governed_contracts(reg)

    baseline_path = governed_contract_baseline_path()
    if not baseline_path.exists():
        print("\n  FAIL: No governed contract baseline found.")
        print(f"  Expected: {baseline_path}")
        print("  Run with --snapshot after intentional contract review to create the explicit baseline.")
        return False

    loaded = reg.load_snapshot(str(baseline_path))
    if not loaded:
        print("\n  FAIL: Could not load the governed contract baseline.")
        print(f"  Path: {baseline_path}")
        return False

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


def save_contract_snapshot() -> bool:
    """Write the explicit governed contract baseline for the current repo state."""
    from vetinari.drift.contract_registry import get_contract_registry

    reg = get_contract_registry()
    register_governed_contracts(reg)
    snapshot_path = governed_contract_baseline_path()
    payload = {
        "hashes": {name: reg.get_hash(name) for name in reg.list_contracts()},
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n  Snapshot saved to {snapshot_path} - this state is the governed baseline.")
    return True


def check_agent_registry() -> bool:
    """Check all factory-pipeline agents are in the registry with valid specs."""
    _section("AGENT REGISTRY COMPLETENESS")
    try:
        from vetinari.agents.contracts import AGENT_REGISTRY, get_agent_spec
        from vetinari.types import AgentType
    except ImportError as e:
        print(f"  FAIL: Cannot import contracts: {e}")
        return False

    # v0.5.0: 3 factory-pipeline agents
    expected = {
        "FOREMAN",
        "WORKER",
        "INSPECTOR",
    }
    found = {at.value for at in AGENT_REGISTRY}
    missing = expected - found
    extra = found - expected

    print(f"  Active agents: {len(found)}/3")
    if missing:
        print(f"  FAIL — Missing agents: {sorted(missing)}")
    if extra:
        print(f"  INFO — Additional registry entries: {len(extra)}")

    # Spec validity for active agents only
    invalid = []
    for at in AgentType:
        if at.value in expected:
            spec = get_agent_spec(at)
            if spec and (not spec.name or not spec.description or not spec.default_model):
                invalid.append(at.value)
    if invalid:
        print(f"  FAIL — Incomplete specs: {invalid}")
        return False

    if missing:
        return False
    print(f"  PASS: All {len(expected)} factory-pipeline agents registered with valid specs.")
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
        from vetinari.planning.plan_types import Plan

        errs = v.validate("Plan", Plan(goal="alignment check"))
        if errs:
            failures.append(("Plan", errs))
            print(f"  FAIL Plan: {errs}")
        else:
            print("  PASS Plan schema")
    except Exception as e:
        print(f"  WARN Plan: {e}")

    try:
        from vetinari.planning.plan_types import Subtask

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
        "docs/planning/drift-prevention.md",
        "docs/archive/skill-migration-guide.md",
        "docs/api/dashboard.md",
        "docs/api/analytics.md",
        "docs/runbooks/dashboard-guide.md",
        "docs/getting-started/onboarding.md",
        "docs/getting-started/quick-start.md",
        "docs/reference/production.md",
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
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save the current governed contract baseline after the non-contract checks pass",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# Contract-Documentation Alignment Checker")
    print("#" * 70)

    checks = [
        ("Agent Registry", check_agent_registry),
        ("Schema Validation", check_schema_validation),
        ("Documentation Files", check_documentation_files),
        ("Contract Fingerprints", check_contract_fingerprints),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  ERROR in {name}: {exc}")
            results[name] = False

    if args.snapshot and all(ok for name, ok in results.items() if name != "Contract Fingerprints") and save_contract_snapshot():
        results["Contract Fingerprints"] = check_contract_fingerprints()

    _section("FINAL SUMMARY")
    passed = sum(v for v in results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{total} checks passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
