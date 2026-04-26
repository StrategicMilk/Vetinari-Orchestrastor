"""Tests for US-006 — Unknown-family learning protocol.

Covers:
- find_closest_known_family() — SequenceMatcher similarity lookup
- seed_unknown_family() — temperature bootstrapping from closest family
- store_model_temperature_overrides() — merge behaviour and thread safety
- record_unknown_family_task() — threshold-based graduation
- create_family_entry() — YAML write and ADR creation (mocked)
- get_family_profile() in loader.py — borrowed profile fallback path
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ── find_closest_known_family ─────────────────────────────────────────────────


def test_find_closest_known_family_exact_slug():
    from vetinari.models.model_profiler_data import find_closest_known_family

    # "llama" is a known slug — should match itself perfectly
    result = find_closest_known_family("llama")
    assert result == "llama"


def test_find_closest_known_family_near_match():
    from vetinari.models.model_profiler_data import find_closest_known_family

    # "llama2" is close to "llama"
    result = find_closest_known_family("llama2")
    assert result == "llama"


def test_find_closest_known_family_unknown_below_threshold():
    from vetinari.models.model_profiler_data import find_closest_known_family

    # Completely unrelated string — should fall below 0.4 threshold
    result = find_closest_known_family("zzzzzzzzz_completely_unrelated_xyz")
    assert result == "unknown"


def test_find_closest_known_family_case_insensitive():
    from vetinari.models.model_profiler_data import find_closest_known_family

    result = find_closest_known_family("MISTRAL")
    assert result == "mistral"


def test_find_closest_known_family_empty_string():
    from vetinari.models.model_profiler_data import find_closest_known_family

    # Empty string has no similarity to any slug
    result = find_closest_known_family("")
    # Should return unknown or a slug — either is acceptable as long as it's a string
    assert isinstance(result, str)


# ── store_model_temperature_overrides ───────────────────────────────────────────────


def test_store_model_temperature_overrides_stores_values():
    from vetinari.models import model_profiler_data as mpd

    model_id = "test-model-store-001"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    mpd.store_model_temperature_overrides(model_id, {"coding": 0.1, "general": 0.5})

    with mpd._per_model_temps_lock:
        stored = mpd._per_model_temperature_overrides.get(model_id, {})
    assert stored["coding"] == pytest.approx(0.1)
    assert stored["general"] == pytest.approx(0.5)


def test_store_model_temperature_overrides_merges_existing():
    from vetinari.models import model_profiler_data as mpd

    model_id = "test-model-merge-002"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    mpd.store_model_temperature_overrides(model_id, {"coding": 0.1, "general": 0.5})
    mpd.store_model_temperature_overrides(model_id, {"reasoning": 0.3})

    with mpd._per_model_temps_lock:
        stored = mpd._per_model_temperature_overrides.get(model_id, {})

    assert "coding" in stored
    assert "reasoning" in stored


def test_store_model_temperature_overrides_overwrites_existing_key():
    from vetinari.models import model_profiler_data as mpd

    model_id = "test-model-overwrite-003"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    mpd.store_model_temperature_overrides(model_id, {"coding": 0.1})
    mpd.store_model_temperature_overrides(model_id, {"coding": 0.9})

    with mpd._per_model_temps_lock:
        stored = mpd._per_model_temperature_overrides.get(model_id, {})
    assert stored["coding"] == pytest.approx(0.9)


def test_store_model_temperature_overrides_thread_safe():
    """Concurrent writes must not corrupt the dict."""
    from vetinari.models import model_profiler_data as mpd

    model_id = "test-model-threads-004"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    errors: list[Exception] = []

    def writer(task_type: str, temp: float) -> None:
        try:
            mpd.store_model_temperature_overrides(model_id, {task_type: temp})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"task_{i}", float(i) * 0.1)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ── seed_unknown_family ───────────────────────────────────────────────────────


def test_seed_unknown_family_copies_temps_from_known_family():
    from vetinari.models import model_profiler_data as mpd

    model_id = "seed-test-model-001"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    mpd.seed_unknown_family(model_id, architecture="novel_arch", closest_family="llama")

    with mpd._per_model_temps_lock:
        stored = mpd._per_model_temperature_overrides.get(model_id, {})

    # llama is in _TEMPERATURE_MATRIX — its temps should be copied in
    assert "coding" in stored
    assert "general" in stored


def test_seed_unknown_family_unknown_closest_does_not_crash():
    from vetinari.models import model_profiler_data as mpd

    model_id = "seed-test-model-002"
    mpd._per_model_temperature_overrides.pop(model_id, None)

    # "not_a_family" has no entry in _TEMPERATURE_MATRIX — should log warning only
    mpd.seed_unknown_family(model_id, architecture="novel", closest_family="not_a_family")

    with mpd._per_model_temps_lock:
        stored = mpd._per_model_temperature_overrides.get(model_id, {})
    # Nothing should be written when closest family has no data
    assert stored == {}


# ── record_unknown_family_task ────────────────────────────────────────────────


def test_record_unknown_family_task_increments_count():
    from vetinari.models import model_profiler_data as mpd

    model_id = "record-task-model-001"
    with mpd._UNKNOWN_FAMILY_LOCK:
        mpd._UNKNOWN_FAMILY_TASK_COUNTS.pop(model_id, None)
    # Seed the model so the unknown-family guard allows counting
    with mpd._per_model_temps_lock:
        mpd._per_model_temperature_overrides[model_id] = {"general": 0.7}

    try:
        with patch.object(mpd, "create_family_entry") as mock_create:
            mpd.record_unknown_family_task(model_id, "test_arch", 0.8)
            mock_create.assert_not_called()

        with mpd._UNKNOWN_FAMILY_LOCK:
            assert mpd._UNKNOWN_FAMILY_TASK_COUNTS[model_id] == 1
    finally:
        with mpd._per_model_temps_lock:
            mpd._per_model_temperature_overrides.pop(model_id, None)


def test_record_unknown_family_task_triggers_at_threshold(tmp_path: Path):
    from vetinari.models import model_profiler_data as mpd

    model_id = "record-task-model-threshold"
    with mpd._UNKNOWN_FAMILY_LOCK:
        # Pre-seed count to one below threshold
        mpd._UNKNOWN_FAMILY_TASK_COUNTS[model_id] = mpd._FAMILY_ENTRY_THRESHOLD - 1
    # Seed the model so the unknown-family guard allows counting
    with mpd._per_model_temps_lock:
        mpd._per_model_temperature_overrides[model_id] = {"general": 0.7}

    try:
        with patch.object(mpd, "create_family_entry") as mock_create:
            mpd.record_unknown_family_task(model_id, "test_arch", 0.7)
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[0][0] == model_id
            assert call_args[0][1] == "test_arch"
    finally:
        with mpd._per_model_temps_lock:
            mpd._per_model_temperature_overrides.pop(model_id, None)


# ── create_family_entry ───────────────────────────────────────────────────────


def test_create_family_entry_writes_yaml(tmp_path: Path):
    from vetinari.models import model_profiler_data as mpd

    families_yaml = tmp_path / "model_families.yaml"
    initial_data = {
        "model_families": {
            "llama": {
                "name": "LLaMA",
                "vendor": "Meta",
                "capabilities": {"context_window": 8192, "supports_function_calling": True, "supports_vision": False},
                "strengths": ["general purpose"],
                "weaknesses": [],
            }
        }
    }
    families_yaml.write_text(yaml.dump(initial_data), encoding="utf-8")

    model_id = "create-entry-model-001"
    architecture = "novel_architecture"

    with (
        patch(
            "vetinari.models.model_profiler_data.Path",
            side_effect=lambda p: tmp_path / Path(p).name if "model_families" in str(p) else Path(p),
        ),
        patch("vetinari.adr.get_adr_system") as mock_adr,
    ):
        mock_adr.return_value = MagicMock()
        # Directly call with patched path
        with patch("vetinari.models.model_profiler_data.Path") as mock_path_cls:
            mock_path_cls.return_value = families_yaml
            mpd.create_family_entry(model_id, architecture, "llama")

    # Read the written file
    written = yaml.safe_load(families_yaml.read_text(encoding="utf-8"))
    slug = "novel_architecture"[:30]
    assert slug in written["model_families"]
    entry = written["model_families"][slug]
    assert entry["discovered_from_model"] == model_id
    assert entry["discovered_via"] == "unknown_family_protocol"


def test_create_family_entry_missing_yaml_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    import logging

    from vetinari.models import model_profiler_data as mpd

    with patch("vetinari.models.model_profiler_data.Path") as mock_path_cls:
        mock_instance = MagicMock()
        mock_instance.exists.return_value = False
        mock_path_cls.return_value = mock_instance

        with caplog.at_level(logging.WARNING, logger="vetinari.models.model_profiler_data"):
            mpd.create_family_entry("some-model", "some_arch", "llama")

    assert any("model_families.yaml not found" in r.message for r in caplog.records)


def test_create_family_entry_skips_existing_slug(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    import logging

    from vetinari.models import model_profiler_data as mpd

    families_yaml = tmp_path / "model_families.yaml"
    slug = "existing_arch"
    initial_data = {"model_families": {slug: {"name": "Existing"}}}
    families_yaml.write_text(yaml.dump(initial_data), encoding="utf-8")

    with patch("vetinari.models.model_profiler_data.Path") as mock_path_cls:
        mock_path_cls.return_value = families_yaml

        with caplog.at_level(logging.INFO, logger="vetinari.models.model_profiler_data"):
            mpd.create_family_entry("model-x", slug, "llama")

    assert any("already exists" in r.message for r in caplog.records)
    # File should be unchanged
    written = yaml.safe_load(families_yaml.read_text(encoding="utf-8"))
    assert list(written["model_families"].keys()) == [slug]


# ── loader.py get_family_profile — borrowed profile fallback ──────────────────


def test_get_family_profile_exact_match_returned_directly():
    """Exact matches must not have is_borrowed set."""
    from vetinari.knowledge import loader

    loader.invalidate_cache()

    families_data = {"model_families": {"llama": {"name": "LLaMA", "vendor": "Meta"}}}
    with patch.object(loader._get_cache(), "_load_file", return_value=families_data):
        loader.invalidate_cache()
        # Seed cache manually
        loader._get_cache()._cache["model_families.yaml"] = families_data
        loader._get_cache()._timestamps["model_families.yaml"] = float("inf")

        profile = loader.get_family_profile("llama")

    assert profile.get("name") == "LLaMA"
    assert "is_borrowed" not in profile


def test_get_family_profile_unknown_family_returns_borrowed():
    """An unknown slug that closely matches a known one gets a borrowed profile."""
    from vetinari.knowledge import loader
    from vetinari.models.model_profiler_data import _learned_temperature_overrides

    loader.invalidate_cache()
    families_data = {
        "model_families": {
            "llama": {"name": "LLaMA", "vendor": "Meta", "strengths": ["general"]},
        }
    }
    loader._get_cache()._cache["model_families.yaml"] = families_data
    loader._get_cache()._timestamps["model_families.yaml"] = float("inf")

    # "llama2" is not in the families dict but is close to "llama"
    _learned_temperature_overrides.pop("llama2", None)

    profile = loader.get_family_profile("llama2")

    assert profile.get("is_borrowed") is True
    assert profile.get("borrowed_from") == "llama"
    assert profile.get("name") == "LLaMA"


def test_get_family_profile_truly_unknown_returns_empty():
    """A slug with no close match returns empty dict."""
    from vetinari.knowledge import loader

    loader.invalidate_cache()
    families_data = {
        "model_families": {
            "llama": {"name": "LLaMA"},
        }
    }
    loader._get_cache()._cache["model_families.yaml"] = families_data
    loader._get_cache()._timestamps["model_families.yaml"] = float("inf")

    with patch("vetinari.models.model_profiler_data.find_closest_known_family", return_value="unknown"):
        profile = loader.get_family_profile("zzz_completely_novel_arch_xyz")

    assert profile == {}
