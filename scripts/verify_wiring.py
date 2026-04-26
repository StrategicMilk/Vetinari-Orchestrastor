#!/usr/bin/env python3
"""Cross-stream wiring verification — checks that safety subsystems are connected.

Usage:
    python scripts/verify_wiring.py
    python scripts/verify_wiring.py --verbose
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable


def _check(label: str, check_fn: Callable[[], bool], verbose: bool = False) -> bool:
    """Run a single wiring check."""
    try:
        ok = check_fn()
        if ok:
            if verbose:
                print(f"  PASS: {label}")
            return True
        print(f"  FAIL: {label}")
        return False
    except Exception as exc:
        print(f"  FAIL: {label} — {exc}")
        return False


def verify_all(verbose: bool = False) -> int:
    """Run all wiring verification checks. Returns number of failures."""
    print("Vetinari Wiring Verification")
    print("=" * 40)

    checks = [
        (
            "GuardrailsManager singleton importable",
            lambda: importlib.import_module("vetinari.safety.guardrails").get_guardrails() is not None,
        ),
        (
            "GuardrailsManager.check_input callable",
            lambda: callable(
                getattr(importlib.import_module("vetinari.safety.guardrails").get_guardrails(), "check_input", None)
            ),
        ),
        (
            "PolicyEnforcer singleton importable",
            lambda: importlib.import_module("vetinari.safety.policy_enforcer").get_policy_enforcer() is not None,
        ),
        (
            "PolicyEnforcer.enforce_all callable",
            lambda: callable(
                getattr(
                    importlib.import_module("vetinari.safety.policy_enforcer").get_policy_enforcer(),
                    "enforce_all",
                    None,
                )
            ),
        ),
        (
            "PolicyEnforcer uses AgentType enums",
            lambda: (
                # D7: verify the jurisdiction map is non-empty AND is keyed by
                # lowercased AgentType values (not bare strings like "builder").
                # An empty map would pass the old "builder not in map" check trivially.
                len(importlib.import_module("vetinari.safety.policy_enforcer")._JURISDICTION) >= 3
                and all(
                    key in {
                        at.value.lower()
                        for at in importlib.import_module("vetinari.types").AgentType
                    }
                    for key in importlib.import_module(
                        "vetinari.safety.policy_enforcer"
                    )._JURISDICTION
                )
            ),
        ),
        (
            "AgentMonitor singleton importable",
            lambda: importlib.import_module("vetinari.safety.agent_monitor").get_agent_monitor() is not None,
        ),
        (
            "SecretsFilter importable",
            lambda: importlib.import_module("vetinari.learning.secrets_filter").scan_text is not None,
        ),
        (
            "SecretsFilter detects AWS keys",
            lambda: (
                len(importlib.import_module("vetinari.learning.secrets_filter").scan_text("AKIAIOSFODNN7EXAMPLE")) > 0
            ),
        ),
        (
            "ResourceMonitor importable",
            lambda: importlib.import_module("vetinari.system.resource_monitor").get_resource_monitor() is not None,
        ),
        (
            "ChatTemplates importable",
            lambda: importlib.import_module("vetinari.models.chat_templates").validate_template is not None,
        ),
        (
            "Unified permission arbitration importable",
            lambda: importlib.import_module("vetinari.execution_context").check_permission_unified is not None,
        ),
        (
            "safety exports GuardrailsManager",
            lambda: hasattr(importlib.import_module("vetinari.safety"), "GuardrailsManager"),
        ),
        (
            "safety exports PolicyEnforcer",
            lambda: hasattr(importlib.import_module("vetinari.safety"), "PolicyEnforcer"),
        ),
        ("safety exports AgentMonitor", lambda: hasattr(importlib.import_module("vetinari.safety"), "AgentMonitor")),
    ]

    failed = 0
    for label, fn in checks:
        if not _check(label, fn, verbose=verbose):
            failed += 1

    print()
    if failed:
        print(f"FAILED: {failed}/{len(checks)} checks")
    else:
        print(f"ALL PASSED: {len(checks)} checks")
    return failed


if __name__ == "__main__":
    # D6: Fail fast on unknown flags so callers don't silently get wrong results.
    _known_flags = {"--verbose", "-v"}
    _unknown = [a for a in sys.argv[1:] if a.startswith("-") and a not in _known_flags]
    if _unknown:
        print("Usage: python scripts/verify_wiring.py [--verbose|-v]")
        print(f"Unknown flag(s): {', '.join(_unknown)}")
        sys.exit(1)
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    failures = verify_all(verbose=verbose)
    sys.exit(1 if failures else 0)
