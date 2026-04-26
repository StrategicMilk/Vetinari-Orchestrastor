"""Read JUnit XML test results and print a clean one-line summary.

Usage::

    python.cmd scripts/run_tests.py      # writes .vetinari/test-results.xml
    python.cmd scripts/test_summary.py   # reads it and prints summary

This exists because pytest's stdout summary gets corrupted by logging
errors during atexit on Windows (colorama + closed stream handles),
and the Bash tool truncates large outputs. A structured XML file
sidesteps both problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

from defusedxml import ElementTree as ET

RESULTS_PATH = Path(".vetinari/test-results.xml")
RAW_OUTPUT_PATH = Path(".vetinari/pytest-last-output.txt")


def _print_raw_output_tail(max_lines: int = 60) -> None:
    """Print captured pytest output when the structured XML is unavailable."""
    if not RAW_OUTPUT_PATH.exists():
        return
    try:
        lines = RAW_OUTPUT_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if not lines:
        return
    print(f"Raw pytest output tail from {RAW_OUTPUT_PATH}:")
    for line in lines[-max_lines:]:
        print(f"  {line}")


def main() -> int:
    """Parse JUnit XML and print a concise summary."""
    if not RESULTS_PATH.exists():
        print(f"No test results found at {RESULTS_PATH}")
        print("Run: python.cmd scripts/run_tests.py")
        _print_raw_output_tail()
        return 1

    try:
        tree = ET.parse(RESULTS_PATH)
    except ET.ParseError as exc:
        print(f"FAILED (malformed XML): could not parse {RESULTS_PATH} — {exc}")
        _print_raw_output_tail()
        return 1
    root = tree.getroot()

    # JUnit XML: <testsuites> or <testsuite> at root
    if root.tag == "testsuites":
        suites = list(root)
    else:
        suites = [root]

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0
    failed_names: list[str] = []
    error_names: list[str] = []

    for suite in suites:
        total_tests += int(suite.get("tests", 0))
        total_failures += int(suite.get("failures", 0))
        total_errors += int(suite.get("errors", 0))
        total_skipped += int(suite.get("skipped", 0))
        total_time += float(suite.get("time", 0))

        for tc in suite.iter("testcase"):
            name = f"{tc.get('classname', '')}.{tc.get('name', '')}"
            if tc.find("failure") is not None:
                failed_names.append(name)
            if tc.find("error") is not None:
                error_names.append(name)

    # D5: An empty run (zero tests collected) is not green — it signals a
    # broken collection phase or no tests found at all.
    if total_tests == 0:
        print(f"EMPTY (no tests executed) in {total_time:.1f}s")
        return 1

    passed = total_tests - total_failures - total_errors - total_skipped

    # One-line summary matching pytest format
    parts = [f"{passed} passed"]
    if total_failures:
        parts.append(f"{total_failures} failed")
    if total_errors:
        parts.append(f"{total_errors} errors")
    if total_skipped:
        parts.append(f"{total_skipped} skipped")

    status = "PASSED" if (total_failures + total_errors) == 0 else "FAILED"
    print(f"{status}: {', '.join(parts)} in {total_time:.1f}s")

    # Show failed test names (compact)
    if failed_names:
        print(f"\nFailed ({len(failed_names)}):")
        for name in failed_names[:30]:
            print(f"  {name}")
        if len(failed_names) > 30:
            print(f"  ... and {len(failed_names) - 30} more")

    if error_names:
        print(f"\nErrors ({len(error_names)}):")
        for name in error_names[:15]:
            print(f"  {name}")
        if len(error_names) > 15:
            print(f"  ... and {len(error_names) - 15} more")

    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
