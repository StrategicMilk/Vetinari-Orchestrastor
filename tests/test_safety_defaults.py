"""Tests for vetinari.safety.safety_defaults — YAML config loader for lifecycle stores.

Covers:
- load_safety_defaults() returns correct values from the canonical YAML
- Cache hit: second call returns the same object (lru_cache)
- Missing YAML raises ConfigurationError with a clear message
- Malformed YAML raises ConfigurationError with a clear message
- RecycleStore() with no args picks up YAML grace_hours and recycle_root
- ArchiveStore() with no args picks up YAML recent_days / cooling_days / archive_root
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vetinari.exceptions import ConfigurationError
from vetinari.lifecycle.archive import ArchiveStore
from vetinari.safety import safety_defaults as sd_module
from vetinari.safety.recycle import RecycleStore
from vetinari.safety.safety_defaults import SafetyDefaults, load_safety_defaults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_YAML = """
recycle_policy:
  grace_hours: 72
  recycle_root: outputs/recycle
archive_policy:
  recent_days: 7
  cooling_days: 30
  archive_root: outputs/archive
"""


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the lru_cache before and after each test for isolation."""
    load_safety_defaults.cache_clear()
    yield
    load_safety_defaults.cache_clear()


@pytest.fixture
def canonical_yaml_path(tmp_path: Path) -> Path:
    """Write the canonical safety_defaults.yaml to tmp_path and redirect the module."""
    p = tmp_path / "config" / "safety_defaults.yaml"
    _write_yaml(p, _VALID_YAML)
    return p


# ---------------------------------------------------------------------------
# Tests: load_safety_defaults()
# ---------------------------------------------------------------------------


class TestLoadSafetyDefaults:
    def test_returns_correct_values_from_canonical_yaml(
        self, canonical_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parsed values match the canonical YAML content."""
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", canonical_yaml_path)

        result = load_safety_defaults()

        assert isinstance(result, SafetyDefaults)
        assert result.grace_hours == 72
        assert result.recent_days == 7
        assert result.cooling_days == 30
        assert result.recycle_root.parts[-2:] == ("outputs", "recycle")
        assert result.archive_root.parts[-2:] == ("outputs", "archive")

    def test_cache_hit_returns_same_instance(self, canonical_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Second call returns the identical object from the cache."""
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", canonical_yaml_path)

        first = load_safety_defaults()
        second = load_safety_defaults()

        assert first is second

    def test_missing_yaml_raises_configuration_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing YAML file raises ConfigurationError with a descriptive message."""
        missing = tmp_path / "config" / "no_such_file.yaml"
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", missing)

        with pytest.raises(ConfigurationError, match="Safety defaults config not found"):
            load_safety_defaults()

    def test_malformed_yaml_raises_configuration_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A YAML file with invalid syntax raises ConfigurationError, not a bare YAMLError."""
        bad_path = tmp_path / "config" / "safety_defaults.yaml"
        _write_yaml(bad_path, "key: [unclosed bracket")
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", bad_path)

        with pytest.raises(ConfigurationError, match="malformed"):
            load_safety_defaults()

    def test_missing_required_field_raises_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """YAML missing a required key raises ConfigurationError naming the missing fields."""
        incomplete = tmp_path / "config" / "safety_defaults.yaml"
        _write_yaml(
            incomplete,
            "recycle_policy:\n  grace_hours: 48\n  recycle_root: outputs/recycle\n",
        )
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", incomplete)

        with pytest.raises(ConfigurationError, match="missing required fields"):
            load_safety_defaults()


# ---------------------------------------------------------------------------
# Tests: RecycleStore() no-args picks up YAML values
# ---------------------------------------------------------------------------


class TestRecycleStoreYamlDefaults:
    def test_no_args_uses_yaml_grace_hours(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RecycleStore() with no args picks up grace_hours from the YAML."""
        yaml_path = tmp_path / "config" / "safety_defaults.yaml"
        custom_yaml = yaml.dump({
            "recycle_policy": {
                "grace_hours": 48,
                "recycle_root": str(tmp_path / "recycle"),
            },
            "archive_policy": {
                "recent_days": 7,
                "cooling_days": 30,
                "archive_root": str(tmp_path / "archive"),
            },
        })
        _write_yaml(yaml_path, custom_yaml)
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", yaml_path)

        store = RecycleStore()

        assert store._policy.grace_hours == 48

    def test_no_args_uses_yaml_recycle_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """RecycleStore() with no args uses recycle_root from the YAML."""
        recycle_root = tmp_path / "my_recycle"
        yaml_path = tmp_path / "config" / "safety_defaults.yaml"
        custom_yaml = yaml.dump({
            "recycle_policy": {
                "grace_hours": 72,
                "recycle_root": str(recycle_root),
            },
            "archive_policy": {
                "recent_days": 7,
                "cooling_days": 30,
                "archive_root": str(tmp_path / "archive"),
            },
        })
        _write_yaml(yaml_path, custom_yaml)
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", yaml_path)

        store = RecycleStore()

        assert store._store._root == recycle_root

    def test_explicit_args_override_yaml(self, canonical_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit constructor args take precedence over YAML defaults."""
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", canonical_yaml_path)
        custom_root = Path("/tmp/custom_recycle")

        store = RecycleStore(root=custom_root, grace_hours=24)

        assert store._policy.grace_hours == 24
        assert store._store._root == custom_root


# ---------------------------------------------------------------------------
# Tests: ArchiveStore() no-args picks up YAML values
# ---------------------------------------------------------------------------


class TestArchiveStoreYamlDefaults:
    def test_no_args_uses_yaml_recent_and_cooling_days(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ArchiveStore() with no args picks up recent_days and cooling_days from the YAML."""
        yaml_path = tmp_path / "config" / "safety_defaults.yaml"
        custom_yaml = yaml.dump({
            "recycle_policy": {
                "grace_hours": 72,
                "recycle_root": str(tmp_path / "recycle"),
            },
            "archive_policy": {
                "recent_days": 14,
                "cooling_days": 60,
                "archive_root": str(tmp_path / "archive"),
            },
        })
        _write_yaml(yaml_path, custom_yaml)
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", yaml_path)

        store = ArchiveStore()

        assert store._policy.recent_days == 14
        assert store._policy.cooling_days == 60

    def test_no_args_uses_yaml_archive_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ArchiveStore() with no args uses archive_root from the YAML."""
        archive_root = tmp_path / "my_archive"
        yaml_path = tmp_path / "config" / "safety_defaults.yaml"
        custom_yaml = yaml.dump({
            "recycle_policy": {
                "grace_hours": 72,
                "recycle_root": str(tmp_path / "recycle"),
            },
            "archive_policy": {
                "recent_days": 7,
                "cooling_days": 30,
                "archive_root": str(archive_root),
            },
        })
        _write_yaml(yaml_path, custom_yaml)
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", yaml_path)

        store = ArchiveStore()

        assert store._store._root == archive_root

    def test_explicit_args_override_yaml(self, canonical_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit constructor args take precedence over YAML defaults."""
        monkeypatch.setattr(sd_module, "_SAFETY_DEFAULTS_PATH", canonical_yaml_path)
        custom_root = Path("/tmp/custom_archive")

        store = ArchiveStore(root=custom_root, recent_days=3, cooling_days=10)

        assert store._policy.recent_days == 3
        assert store._policy.cooling_days == 10
        assert store._store._root == custom_root
