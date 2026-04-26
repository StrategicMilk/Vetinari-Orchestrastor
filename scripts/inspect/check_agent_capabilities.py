#!/usr/bin/env python3
"""
Agent Capability Audit Script — Phase 7 Drift Control

Verifies that every agent's ``get_capabilities()`` output is:
  1. Non-empty (each agent must declare at least one capability)
  2. Stable across two runs (no random ordering)
  3. Consistent with the AgentType enum (agent exists and loads)

Exits 0 when all agents pass; non-zero on any failure.

Usage:
    python scripts/inspect/check_agent_capabilities.py [--verbose]
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


AGENT_MAP = {
    "FOREMAN": "vetinari.agents.planner_agent.ForemanAgent",
    "WORKER": "vetinari.agents.consolidated.worker_agent.WorkerAgent",
    "INSPECTOR": "vetinari.agents.consolidated.quality_agent.InspectorAgent",
}


def _load(dotpath: str) -> object:
    import importlib

    mod_path, cls_name = dotpath.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    return cls()


def audit_agent(name: str, dotpath: str, verbose: bool) -> dict:
    result = {"name": name, "ok": False, "caps": [], "error": None}
    try:
        agent = _load(dotpath)
        caps = agent.get_capabilities()
        result["caps"] = caps

        errors = []
        if not caps:
            errors.append("get_capabilities() returned empty list")
        if not isinstance(caps, list):
            errors.append(f"get_capabilities() returned {type(caps).__name__}, expected list")
        # Stability: call twice and compare the exact emitted sequence. Consumers
        # depend on stable capability ordering unless the API explicitly says it
        # is order-insensitive.
        caps2 = agent.get_capabilities()
        if caps != caps2:
            errors.append("get_capabilities() emitted a non-deterministic order")
        if not errors:
            result["ok"] = True
            if verbose:
                print(f"  PASS {name:30} {caps}")
        else:
            result["error"] = "; ".join(errors)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent Capability Audit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        dest="save_baseline",
        help="Save current capabilities as governed baseline",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# Agent Capability Audit — Phase 7 Drift Control")
    print("#" * 70)

    results = []
    for name, dotpath in sorted(AGENT_MAP.items()):
        r = audit_agent(name, dotpath, args.verbose)
        results.append(r)
        if not r["ok"]:
            print(f"  FAIL {name:30} — {r['error']}")
        elif not args.verbose:
            print(f"  PASS {name}")

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    failed = [r for r in results if not r["ok"]]

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed}/{total} agents passed")

    if failed:
        print("\n  Failed agents:")
        for r in failed:
            print(f"    {r['name']}: {r['error']}")
        return 1

    print("  All agent capabilities are defined and stable.")

    if args.save_baseline:
        from vetinari.drift.capability_auditor import get_capability_auditor, reset_capability_auditor

        reset_capability_auditor()
        auditor = get_capability_auditor()
        auditor.save_baseline()
        print("  Capability baseline saved.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
