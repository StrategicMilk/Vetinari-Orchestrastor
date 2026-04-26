"""VET141 BAD fixture — imports sandbox but writes without a guard.

This module imports vetinari.security.sandbox (so the rule is in-scope) and
calls Path.write_text without a preceding enforce_blocked_paths call in the
same function body. VET141 must flag the write line.
"""

from __future__ import annotations

from pathlib import Path

from vetinari.security.sandbox import enforce_blocked_paths


def save_thing(target: Path, content: str) -> None:
    """Write a string to *target* without guarding with the sandbox first."""
    target.write_text(content, encoding="utf-8")  # VET141 should flag this line
