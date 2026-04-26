#!/usr/bin/env python3
"""Run the Vetinari test suite through the canonical project interpreter.

The wrapper clears stale JUnit XML before launch, captures raw pytest output to
``.vetinari/pytest-last-output.txt``, then prints the structured summary from
``scripts/test_summary.py``. This keeps agent output readable even when pytest
produces more text than the UI can safely display.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / ".vetinari" / "test-results.xml"
RAW_OUTPUT_PATH = PROJECT_ROOT / ".vetinari" / "pytest-last-output.txt"
RUN_META_PATH = PROJECT_ROOT / ".vetinari" / "pytest-last-run.json"
DEFAULT_ARGS = ["tests/", "-x", "-q", "--tb=line"]


def _candidate_project_pythons() -> list[Path]:
    env_python = os.environ.get("VETINARI_PYTHON")
    candidates = []
    if env_python:
        candidates.append(Path(env_python))
    candidates.extend(
        [
            PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe",
            PROJECT_ROOT / ".venv312" / "bin" / "python",
        ]
    )
    return candidates


def _project_python() -> str:
    for candidate in _candidate_project_pythons():
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _tail(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _pytest_mode_does_not_execute_tests(args: list[str]) -> bool:
    return any(arg in {"--collect-only", "--co", "--fixtures", "--version", "-h", "--help"} for arg in args)


def _write_run_metadata(started_at: float, finished_at: float, argv: list[str], returncode: int) -> None:
    payload = {
        "started_at_epoch": started_at,
        "finished_at_epoch": finished_at,
        "argv": argv,
        "returncode": returncode,
        "results_path": str(RESULTS_PATH),
        "results_exists": RESULTS_PATH.exists(),
        "raw_output_path": str(RAW_OUTPUT_PATH),
    }
    if RESULTS_PATH.exists():
        payload["results_mtime_epoch"] = RESULTS_PATH.stat().st_mtime
    RUN_META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run pytest and print a bounded structured summary."""
    args = argv if argv is not None else sys.argv[1:]
    pytest_args = args or DEFAULT_ARGS
    PROJECT_ROOT.joinpath(".vetinari").mkdir(exist_ok=True)

    with suppress(OSError):
        RESULTS_PATH.unlink()

    python = _project_python()
    command = [python, "-m", "pytest", *pytest_args]
    started_at = time.time()
    result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    finished_at = time.time()

    combined = (result.stdout or "") + (result.stderr or "")
    RAW_OUTPUT_PATH.write_text(combined, encoding="utf-8")
    _write_run_metadata(started_at, finished_at, command, result.returncode)

    if _pytest_mode_does_not_execute_tests(pytest_args):
        tail = _tail(combined)
        if tail:
            print(tail)
        return result.returncode

    summary = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
        [python, str(PROJECT_ROOT / "scripts" / "test_summary.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if summary.stdout:
        print(summary.stdout.rstrip())
    if summary.stderr:
        print(summary.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0 and not RESULTS_PATH.exists():
        print(f"\npytest did not produce {RESULTS_PATH.relative_to(PROJECT_ROOT)}.")
        print(f"Raw output saved to {RAW_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
        tail = _tail(combined)
        if tail:
            print("\nLast pytest output:")
            print(tail)

    return result.returncode if result.returncode != 0 else summary.returncode


if __name__ == "__main__":
    sys.exit(main())
