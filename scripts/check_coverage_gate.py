#!/usr/bin/env python3
"""
Coverage Gate Script — Phase 7 Drift Control

Parses a pytest coverage JSON report (coverage.json) and fails CI if
any module falls below its configured minimum coverage threshold.

Usage:
    # Run pytest with coverage first:
    python -m pytest tests/ --cov=vetinari --cov-report=json -q

    # Then run this gate:
    python scripts/check_coverage_gate.py [--report coverage.json] [--fail-under 70]

Exit codes:
    0   All modules meet their thresholds.
    1   One or more modules are below threshold.
    2   The coverage report was not found.
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent

# ── Per-module minimum coverage thresholds ───────────────────────────────────
# Modules not listed here fall back to the global minimum.
MODULE_THRESHOLDS = {
    # Phase 3
    "vetinari/telemetry.py": 80,
    "vetinari/security.py": 60,
    "vetinari/structured_logging.py": 75,
    # Phase 4
    "vetinari/dashboard/api.py": 85,
    "vetinari/dashboard/alerts.py": 85,
    "vetinari/dashboard/log_aggregator.py": 85,
    # Phase 5
    "vetinari/analytics/anomaly.py": 85,
    "vetinari/analytics/cost.py": 85,
    "vetinari/analytics/sla.py": 85,
    "vetinari/analytics/forecasting.py": 85,
    # Phase 7
    "vetinari/drift/contract_registry.py": 85,
    "vetinari/drift/capability_auditor.py": 80,
    "vetinari/drift/schema_validator.py": 85,
    "vetinari/drift/monitor.py": 80,
    # Adapters
    "vetinari/adapters/base.py": 80,
    "vetinari/adapters/registry.py": 80,
    # Contracts
    "vetinari/agents/contracts.py": 80,
    "vetinari/plan_types.py": 80,
    "vetinari/memory/interfaces.py": 80,
}

GLOBAL_MIN = 50  # unlisted modules must meet this baseline; use MODULE_THRESHOLDS for per-module overrides


def _normalise(path: str) -> str:
    """Convert Windows backslashes and leading separators to forward slashes."""
    return path.replace("\\", "/").lstrip("/")


def load_report(report_path: Path) -> dict:
    if not report_path.exists():
        print(f"ERROR: Coverage report not found at {report_path}")
        print("Run:  python -m pytest tests/ --cov=vetinari --cov-report=json -q")
        sys.exit(2)
    report = json.loads(report_path.read_text())
    files = report.get("files")
    if not isinstance(files, dict) or not files:
        print(f"ERROR: Coverage report at {report_path} has no file data")
        sys.exit(2)
    return report


def check_coverage(report: dict, global_min: int) -> list:
    """Return list of (module_path, actual_pct, required_pct) for failures."""
    failures = []
    for raw_path, data in report.get("files", {}).items():
        normalised = _normalise(raw_path)
        # Strip leading project root
        for prefix in ("vetinari/", "C:/", "/"):
            if normalised.startswith(prefix):
                break

        # Find the shortest matching key in MODULE_THRESHOLDS
        required = global_min
        for key in MODULE_THRESHOLDS:
            if normalised.endswith(key) or key in normalised:
                required = MODULE_THRESHOLDS[key]
                break

        actual = data["summary"]["percent_covered"]
        if actual < required:
            failures.append((normalised, actual, required))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Coverage Gate")
    parser.add_argument("--report", default="coverage.json", help="Path to pytest coverage JSON report")
    parser.add_argument(
        "--fail-under", type=float, default=GLOBAL_MIN, dest="fail_under", help="Global minimum coverage percent"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    report_path = project_root / args.report
    report = load_report(report_path)
    overall_pct = report.get("totals", {}).get("percent_covered", 0)

    print("\n" + "#" * 70)
    print("# Coverage Gate — Phase 7 Drift Control")
    print("#" * 70)
    print(f"\n  Report:   {report_path}")
    print(f"  Overall:  {overall_pct:.1f}%  (global min: {args.fail_under}%)")
    print(f"  Files:    {len(report.get('files', {}))}")

    failures = check_coverage(report, args.fail_under)

    # Classify modules as governed (in MODULE_THRESHOLDS) vs ungoverned (using global min)
    # so the output honestly discloses which modules are under per-module oversight
    # vs just the baseline floor.
    governed = []
    ungoverned = []
    for raw_path in report.get("files", {}):
        normalised = _normalise(raw_path)
        is_governed = any(
            normalised.endswith(key) or key in normalised
            for key in MODULE_THRESHOLDS
        )
        if is_governed:
            governed.append(normalised)
        else:
            ungoverned.append(normalised)

    print(f"\n  Governed modules (per-module threshold): {len(governed)}")
    print(f"  Ungoverned modules (global min {args.fail_under}%): {len(ungoverned)}")
    if ungoverned and args.verbose:
        for p in sorted(ungoverned)[:10]:
            print(f"    {p}")
        if len(ungoverned) > 10:
            print(f"    ... and {len(ungoverned) - 10} more")

    if args.verbose:
        print("\n  Per-module results:")
        for raw_path, data in sorted(report.get("files", {}).items()):
            pct = data["summary"]["percent_covered"]
            normalised = _normalise(raw_path)
            required = args.fail_under
            for key in MODULE_THRESHOLDS:
                if key in normalised:
                    required = MODULE_THRESHOLDS[key]
                    break
            flag = "FAIL" if pct < required else "ok  "
            print(f"    {flag}  {pct:5.1f}% / {required}%  {normalised}")

    if failures:
        print(f"\n  FAIL: {len(failures)} module(s) below threshold:")
        for path, actual, required in sorted(failures, key=lambda x: x[1]):
            print(f"    {actual:5.1f}% / {required}%  {path}")
        return 1

    governed_count = len(governed)
    ungoverned_count = len(ungoverned)
    if ungoverned_count > 0:
        print(
            f"\n  PASS: {governed_count} governed module(s) meet per-module thresholds; "
            f"{ungoverned_count} ungoverned module(s) meet global min ({args.fail_under}%)."
        )
    else:
        print(f"\n  PASS: All {governed_count} modules meet coverage thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
