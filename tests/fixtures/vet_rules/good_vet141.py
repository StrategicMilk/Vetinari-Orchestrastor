"""VET141 GOOD fixture — every write is preceded by enforce_blocked_paths.

This module imports vetinari.security.sandbox and calls Path.write_text, but
every write is preceded by enforce_blocked_paths(target) inside the same
function body. VET141 must NOT flag this module.
"""

from __future__ import annotations

from pathlib import Path

from vetinari.security.sandbox import enforce_blocked_paths


def save_thing(target: Path, content: str) -> None:
    """Guard the target path before writing."""
    enforce_blocked_paths(target)
    target.write_text(content, encoding="utf-8")


def save_binary(target: Path, payload: bytes) -> None:
    """Guard the target path before a binary write."""
    enforce_blocked_paths(target)
    target.write_bytes(payload)


def save_via_open(target: Path, content: str) -> None:
    """Guard the target path before using open() in write mode."""
    enforce_blocked_paths(target)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(content)
