"""Extract fuzz test failures for a given keyword filter."""

from __future__ import annotations

import pathlib
import subprocess
import sys

# Repository root derived from this file's location (scripts/ -> repo root).
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(keyword: str) -> None:
    import os
    import tempfile

    out_file = os.path.join(tempfile.gettempdir(), "_fuzz_out.txt")
    with pathlib.Path(out_file).open("w", encoding="utf-8") as output:
        subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/runtime/test_mutating_route_fuzz.py::TestTypeMutationFuzz",
                "-k",
                keyword,
                "--tb=line",
                "--no-header",
                "-q",
                f"--junit-xml={out_file}.xml",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            cwd=str(_REPO_ROOT),
        )
    combined = pathlib.Path(out_file).read_text(encoding="utf-8")

    failures = []
    for line in combined.splitlines():
        if "AssertionError:" in line and "returned" in line:
            failures.append(line.strip())
    if failures:
        print(f"\n{len(failures)} failures for keyword '{keyword}':")
        for f in sorted(set(failures)):
            print(" ", f)
    else:
        # Print last 3000 chars for context
        print(combined[-3000:])


if __name__ == "__main__":
    kw = sys.argv[1] if len(sys.argv) > 1 else "empty_object"
    main(kw)
