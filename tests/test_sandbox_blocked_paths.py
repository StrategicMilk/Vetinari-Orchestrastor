"""Tests for vetinari.security.sandbox.enforce_blocked_paths.

Exercises the Rule 2 closure from SESSION-03 SHARD-02: blocked_paths declared
in sandbox_policy.yaml are consulted at the write site, and any failure in
the check path (missing policy, malformed YAML, unresolvable target) fails
closed rather than defaulting to allow.

Uses a temporary policy yaml + tmp_path fixtures; never touches the real
policy file. Calls the REAL helper (no monkeypatching the function itself —
that would be the Harness-wide monkeypatches anti-pattern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.exceptions import SandboxPolicyViolation
from vetinari.security.sandbox import (
    enforce_blocked_paths,
    reset_blocked_paths_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Reset the blocked-paths module cache before and after each test."""
    reset_blocked_paths_cache()
    yield
    reset_blocked_paths_cache()


def _write_policy(tmp_path: Path, blocked: list[str]) -> Path:
    """Write a minimal sandbox_policy.yaml with the given blocked paths.

    Uses YAML single-quoted strings so Windows backslashes are NOT interpreted
    as YAML unicode escapes. Single-quoted YAML only needs ``''`` escaping of
    inner single quotes, which is fine for filesystem paths.

    Args:
        tmp_path: Pytest tmp_path fixture root.
        blocked: List of blocked path strings to embed.

    Returns:
        Path to the generated policy file.
    """
    policy_path = tmp_path / "sandbox_policy.yaml"
    body = "rules:\n  blocked_paths:\n"
    for entry in blocked:
        escaped = entry.replace("'", "''")
        body += f"    - '{escaped}'\n"
    policy_path.write_text(body, encoding="utf-8")
    return policy_path


class TestBlockedPathsEnforcement:
    """enforce_blocked_paths blocks writes to denylisted paths."""

    def test_exact_blocked_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "secret.key"
        target.write_text("mock key material", encoding="utf-8")
        policy = _write_policy(tmp_path, [str(target)])

        with pytest.raises(SandboxPolicyViolation) as excinfo:
            enforce_blocked_paths(target, policy_path=policy)

        message = str(excinfo.value)
        assert str(target) in message, f"Exception message must name the attempted path. Got: {message}"

    def test_file_inside_blocked_directory_raises(self, tmp_path: Path) -> None:
        blocked_dir = tmp_path / "protected"
        blocked_dir.mkdir()
        target = blocked_dir / "nested.txt"
        policy = _write_policy(tmp_path, [str(blocked_dir)])

        with pytest.raises(SandboxPolicyViolation):
            enforce_blocked_paths(target, policy_path=policy)

    def test_allowed_sibling_path_passes(self, tmp_path: Path) -> None:
        blocked_dir = tmp_path / "protected"
        blocked_dir.mkdir()
        allowed = tmp_path / "ok.txt"
        policy = _write_policy(tmp_path, [str(blocked_dir)])

        # No exception expected.
        enforce_blocked_paths(allowed, policy_path=policy)

    def test_symlink_into_blocked_directory_raises(self, tmp_path: Path) -> None:
        blocked_dir = tmp_path / "protected"
        blocked_dir.mkdir()
        real_target = blocked_dir / "sensitive.txt"
        policy = _write_policy(tmp_path, [str(blocked_dir)])

        symlink = tmp_path / "looks_innocent"
        try:
            symlink.symlink_to(real_target)
        except OSError:
            pytest.skip("Symlinks not supported on this filesystem / permissions set")

        with pytest.raises(SandboxPolicyViolation):
            enforce_blocked_paths(symlink, policy_path=policy)

    def test_glob_pattern_pem_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "private_key.pem"
        policy = _write_policy(tmp_path, ["*.pem"])

        with pytest.raises(SandboxPolicyViolation) as excinfo:
            enforce_blocked_paths(target, policy_path=policy)

        assert "*.pem" in str(excinfo.value) or "pem" in str(excinfo.value).lower()


class TestFailClosedSemantics:
    """Any error in the check path MUST fail closed, never default to allow."""

    def test_missing_policy_file_raises(self, tmp_path: Path) -> None:
        missing_policy = tmp_path / "does_not_exist.yaml"
        target = tmp_path / "write.txt"

        with pytest.raises(SandboxPolicyViolation) as excinfo:
            enforce_blocked_paths(target, policy_path=missing_policy)

        assert "policy" in str(excinfo.value).lower()

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        policy = tmp_path / "bad.yaml"
        policy.write_text("rules: [not: closed", encoding="utf-8")  # malformed
        target = tmp_path / "write.txt"

        with pytest.raises(SandboxPolicyViolation) as excinfo:
            enforce_blocked_paths(target, policy_path=policy)

        assert "parse" in str(excinfo.value).lower() or "evaluated" in str(excinfo.value).lower()

    def test_blocked_paths_not_a_list_raises(self, tmp_path: Path) -> None:
        policy = tmp_path / "bad_shape.yaml"
        policy.write_text("rules:\n  blocked_paths: 'not-a-list'\n", encoding="utf-8")
        target = tmp_path / "write.txt"

        with pytest.raises(SandboxPolicyViolation):
            enforce_blocked_paths(target, policy_path=policy)
