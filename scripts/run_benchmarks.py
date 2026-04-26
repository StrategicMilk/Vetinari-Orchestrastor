#!/usr/bin/env python3
"""
CI Benchmark Runner — Vetinari
================================

CLI entry point for the lightweight CI benchmark suite.  Runs all three
fast probes (plan latency, decomposition quality, token optimisation) and
reports results as JSON.

Usage::

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --output results.json

Exit codes:
  0 — all benchmarks passed
  1 — one or more benchmarks failed
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Vetinari CI benchmark suite and print results as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON results to FILE in addition to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    """Run CI benchmarks, print results, and return an exit code.

    Returns:
        0 if all benchmarks passed, 1 otherwise.
    """
    args = _parse_args()

    from vetinari.benchmarks import run_ci_benchmarks

    results = run_ci_benchmarks()
    output_json = json.dumps(results, indent=2)

    print(output_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Results written to {out_path}", file=sys.stderr)

    return 0 if results["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
