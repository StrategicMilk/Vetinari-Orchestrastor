#!/usr/bin/env python3
"""Thin CLI wrapper around Vetinari's Semgrep helper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vetinari.tools.semgrep_tool import run_semgrep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Semgrep as an on-demand security accelerator")
    parser.add_argument("target", help="File or directory to scan")
    parser.add_argument("--config", default="auto", help="Semgrep config selector (default: auto)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_semgrep(Path(args.target), config=args.config)

    if args.json:
        payload = {
            "available": result.is_available,
            "error": result.error,
            "findings": [
                {
                    "rule_id": finding.rule_id,
                    "file": finding.file,
                    "line": finding.line,
                    "message": finding.message,
                    "severity": finding.severity,
                }
                for finding in result.findings
            ],
        }
        print(json.dumps(payload, indent=2))
        # D12: propagate execution errors even in JSON mode — a Semgrep run that
        # returned an error must not exit 0, which would falsely signal success.
        if not result.is_available:
            sys.exit(2)
        if result.error:
            sys.exit(1)
        sys.exit(0)

    if not result.is_available:
        print(f"Semgrep unavailable: {result.error}")
        sys.exit(2)
    if result.error:
        print(f"Semgrep error: {result.error}")
        sys.exit(1)
    if not result.findings:
        print("Semgrep: no findings")
        return

    print(f"Semgrep: {len(result.findings)} finding(s)")
    for finding in result.findings[:50]:
        print(
            f"- {finding.severity} {finding.rule_id} "
            f"{finding.file}:{finding.line} {finding.message}"
        )


if __name__ == "__main__":
    main()
