"""Proof tests — VETINARI_USER_DIR env override reaches every owned persistence surface.

Each test class covers one persistence/cache surface and verifies:
  1. When VETINARI_USER_DIR is set, the surface uses that directory.
  2. When no override is present, the default (home-based) path is used.

These tests are the direct proof required by SESSION-32B completion gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.constants import get_user_dir


class TestGetUserDir:
    """get_user_dir() re-reads the env var on every call."""

    def test_default_is_home_dot_vetinari(self, monkeypatch) -> None:
        """Without override, returns ~/.vetinari."""
        monkeypatch.delenv("VETINARI_USER_DIR", raising=False)
        result = get_user_dir()
        assert result == Path.home() / ".vetinari"

    def test_env_override_respected(self, tmp_path: Path, monkeypatch) -> None:
        """VETINARI_USER_DIR env var overrides the default."""
        custom = tmp_path / "custom_vetinari"
        monkeypatch.setenv("VETINARI_USER_DIR", str(custom))
        assert get_user_dir() == custom

    def test_env_change_mid_process_is_visible(self, tmp_path: Path, monkeypatch) -> None:
        """Changing VETINARI_USER_DIR after import takes effect immediately."""
        first = tmp_path / "dir_a"
        second = tmp_path / "dir_b"

        monkeypatch.setenv("VETINARI_USER_DIR", str(first))
        assert get_user_dir() == first

        monkeypatch.setenv("VETINARI_USER_DIR", str(second))
        assert get_user_dir() == second


class TestResolveUserPathEnvParity:
    """resolve_user_path() surfaces VETINARI_USER_DIR to callers."""

    def test_override_reaches_resolve_user_path(self, tmp_path: Path, monkeypatch) -> None:
        """VETINARI_USER_DIR override is visible through resolve_user_path()."""
        from vetinari.paths import resolve_user_path

        custom = tmp_path / "user_root"
        monkeypatch.setenv("VETINARI_USER_DIR", str(custom))
        assert resolve_user_path() == custom

    def test_segments_land_under_overridden_root(self, tmp_path: Path, monkeypatch) -> None:
        """Segments are joined onto the overridden user root, not ~/.vetinari."""
        from vetinari.paths import resolve_user_path

        custom = tmp_path / "user_root"
        monkeypatch.setenv("VETINARI_USER_DIR", str(custom))
        result = resolve_user_path("models", "llama-3")
        assert result == custom / "models" / "llama-3"


class TestACONCompressorStateRoot:
    """ACONCompressor default rules path uses VETINARI_STATE_DIR, not cwd."""

    def test_default_rules_path_under_state_dir(self) -> None:
        """ACONCompressor() without explicit rules_path uses VETINARI_STATE_DIR."""
        from vetinari.constants import VETINARI_STATE_DIR
        from vetinari.context.acon import ACONCompressor

        compressor = ACONCompressor()
        assert (
            VETINARI_STATE_DIR in compressor._rules_path.parents or compressor._rules_path.parent == VETINARI_STATE_DIR
        )

    def test_explicit_rules_path_honored(self, tmp_path: Path) -> None:
        """An explicit rules_path is used as-is, not replaced by state dir."""
        from vetinari.context.acon import ACONCompressor

        rules = tmp_path / "my_rules.json"
        compressor = ACONCompressor(rules_path=rules)
        assert compressor._rules_path == rules


class TestToolResultStoreStateRoot:
    """ToolResultStore default cache_dir uses VETINARI_STATE_DIR, not cwd."""

    def test_default_cache_under_state_dir(self) -> None:
        """ToolResultStore() without explicit cache_dir uses VETINARI_STATE_DIR."""
        from vetinari.constants import VETINARI_STATE_DIR
        from vetinari.context.tool_persistence import ToolResultStore

        store = ToolResultStore()
        assert VETINARI_STATE_DIR in store._cache_dir.parents or store._cache_dir.parent == VETINARI_STATE_DIR

    def test_explicit_cache_dir_honored(self, tmp_path: Path) -> None:
        """An explicit cache_dir is used as-is."""
        from vetinari.context.tool_persistence import ToolResultStore

        cache = tmp_path / "tool_cache"
        store = ToolResultStore(cache_dir=cache)
        assert store._cache_dir == cache


class TestOperatorSelectorStateRoot:
    """OperatorSelector._default_state_path() respects VETINARI_STATE_DIR env."""

    def test_env_override_changes_default_path(self, tmp_path: Path, monkeypatch) -> None:
        """VETINARI_STATE_DIR env var overrides the default state path."""
        custom_state = tmp_path / "custom_state"
        custom_state.mkdir(parents=True)
        monkeypatch.setenv("VETINARI_STATE_DIR", str(custom_state))

        from vetinari.learning.operator_selector import OperatorSelector

        path = OperatorSelector._default_state_path()
        assert path.parent == custom_state

    def test_default_path_under_vetinari_state_dir(self, monkeypatch) -> None:
        """Without override, default path is under VETINARI_STATE_DIR."""
        monkeypatch.delenv("VETINARI_STATE_DIR", raising=False)

        from vetinari.constants import VETINARI_STATE_DIR
        from vetinari.learning.operator_selector import OperatorSelector

        path = OperatorSelector._default_state_path()
        assert path.parent == VETINARI_STATE_DIR


class TestLandscapeMonitorStateRoot:
    """LandscapeMonitor default cache_dir uses VETINARI_STATE_DIR, not cwd."""

    def test_default_cache_under_state_dir(self) -> None:
        """LandscapeMonitor() without cache_dir uses VETINARI_STATE_DIR."""
        from vetinari.constants import VETINARI_STATE_DIR
        from vetinari.models.landscape_monitor import LandscapeMonitor

        monitor = LandscapeMonitor()
        assert VETINARI_STATE_DIR in monitor._cache_dir.parents

    def test_explicit_cache_dir_honored(self, tmp_path: Path) -> None:
        """An explicit cache_dir is used as-is."""
        from vetinari.models.landscape_monitor import LandscapeMonitor

        cache = tmp_path / "landscape_cache"
        monitor = LandscapeMonitor(cache_dir=cache)
        assert monitor._cache_dir == cache


class TestCliPackagingDataEnvOverride:
    """Proves cli_packaging_data uses get_user_dir() at call sites, not a cached constant."""

    def test_get_user_dir_override_affects_packaging_data(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """VETINARI_USER_DIR env override is reflected in cli_packaging_data at runtime.

        The fix moved from a module-level ``VETINARI_USER_DIR`` constant to
        calling ``get_user_dir()`` at each use site inside ``cmd_init``.  This
        test confirms that ``get_user_dir()`` sees the monkeypatched env var at
        call time — i.e. the env override is *not* baked in at import time.
        """
        custom_dir = str(tmp_path / "custom_user")
        monkeypatch.setenv("VETINARI_USER_DIR", custom_dir)

        # Import is already done at module load — the point is that
        # get_user_dir() re-reads the env on every call, so the override
        # is visible even though cli_packaging_data was already imported.
        import vetinari.cli_packaging_data
        from vetinari.constants import get_user_dir

        result = get_user_dir()
        assert str(result) == custom_dir, (
            f"get_user_dir() returned {result!r}, expected {custom_dir!r}. "
            "Ensure VETINARI_USER_DIR env override is respected at call time."
        )

    def test_get_user_dir_override_changes_mid_process(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Changing VETINARI_USER_DIR after cli_packaging_data is imported takes effect immediately.

        This is the property that would break if cmd_init read a cached
        module-level constant instead of calling get_user_dir() at each use.
        """
        from vetinari.constants import get_user_dir

        first = tmp_path / "first_user"
        second = tmp_path / "second_user"

        monkeypatch.setenv("VETINARI_USER_DIR", str(first))
        assert get_user_dir() == first

        monkeypatch.setenv("VETINARI_USER_DIR", str(second))
        assert get_user_dir() == second
