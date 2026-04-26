"""Tests for vetinari.paths — path resolution helpers and traversal guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.paths import (
    resolve_project_path,
    resolve_state_path,
    resolve_user_path,
)


class TestResolveStatePath:
    """Tests for resolve_state_path() traversal guard."""

    def test_no_segments_returns_state_root(self) -> None:
        """Called with no arguments, returns the state root directory."""
        from vetinari.constants import VETINARI_STATE_DIR

        result = resolve_state_path()
        assert result == VETINARI_STATE_DIR

    def test_valid_segments_appended(self) -> None:
        """Normal path segments are appended under the state root."""
        from vetinari.constants import VETINARI_STATE_DIR

        result = resolve_state_path("checkpoints", "run-001")
        assert result.is_relative_to(VETINARI_STATE_DIR.resolve())

    def test_dotdot_segment_raises(self) -> None:
        """'..' segments that escape the state root are rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            resolve_state_path("..", "etc", "passwd")

    def test_absolute_like_escape_raises(self) -> None:
        """Segments that resolve outside the state root are rejected."""
        # Build a traversal that escapes by stepping up from state root
        from vetinari.constants import VETINARI_STATE_DIR

        depth = len(VETINARI_STATE_DIR.parts)
        traversal = ("..",) * depth + ("etc",)
        with pytest.raises(ValueError, match="path traversal"):
            resolve_state_path(*traversal)


class TestResolveUserPath:
    """Tests for resolve_user_path() traversal guard."""

    def test_no_segments_returns_user_root(self, tmp_path: Path, monkeypatch) -> None:
        """Called with no arguments, returns the user dir from env."""
        monkeypatch.setenv("VETINARI_USER_DIR", str(tmp_path / "user"))
        result = resolve_user_path()
        assert result == tmp_path / "user"

    def test_valid_segments_appended(self, tmp_path: Path, monkeypatch) -> None:
        """Normal path segments are appended under the user root."""
        user_root = tmp_path / "user"
        monkeypatch.setenv("VETINARI_USER_DIR", str(user_root))
        result = resolve_user_path("models", "llama")
        assert result.is_relative_to(user_root)

    def test_dotdot_segment_raises(self, tmp_path: Path, monkeypatch) -> None:
        """'..' segments that escape the user root are rejected."""
        monkeypatch.setenv("VETINARI_USER_DIR", str(tmp_path / "user"))
        with pytest.raises(ValueError, match="path traversal"):
            resolve_user_path("..", "etc", "shadow")

    def test_env_override_respected(self, tmp_path: Path, monkeypatch) -> None:
        """resolve_user_path() re-reads VETINARI_USER_DIR on every call."""
        first = tmp_path / "first"
        second = tmp_path / "second"

        monkeypatch.setenv("VETINARI_USER_DIR", str(first))
        assert resolve_user_path() == first

        monkeypatch.setenv("VETINARI_USER_DIR", str(second))
        assert resolve_user_path() == second


class TestResolveProjectPath:
    """Tests for resolve_project_path() traversal guard."""

    def test_valid_project_returns_path(self) -> None:
        """A normal project_id builds a path under PROJECTS_DIR."""
        from vetinari.constants import PROJECTS_DIR

        result = resolve_project_path("my-project")
        assert result.is_relative_to(PROJECTS_DIR)

    def test_empty_project_id_raises(self) -> None:
        """An empty project_id is rejected immediately."""
        with pytest.raises(ValueError, match="project_id must not be empty"):
            resolve_project_path("")

    def test_dotdot_project_id_raises(self) -> None:
        """A project_id with '..' is rejected to prevent traversal."""
        with pytest.raises(ValueError, match="path traversal"):
            resolve_project_path("../../etc/passwd")

    def test_segments_appended(self) -> None:
        """Extra segments are appended after the project directory."""
        from vetinari.constants import PROJECTS_DIR

        result = resolve_project_path("proj-abc", "results", "output.json")
        assert result.is_relative_to(PROJECTS_DIR)
        assert result.name == "output.json"
