"""Tests for model landscape monitor — cache, HTTP fallback, stale detection."""

from __future__ import annotations

import json
import time
import unittest.mock as mock
import urllib.error
from pathlib import Path

import pytest
import yaml

from vetinari.models.landscape_monitor import LandscapeMonitor


def _make_monitor(tmp_path: Path, ttl: int = 7 * 86400) -> LandscapeMonitor:
    """Build a LandscapeMonitor that writes cache under tmp_path."""
    return LandscapeMonitor(cache_dir=tmp_path / "cache", cache_ttl_seconds=ttl)


# -- test_cache_hit -----------------------------------------------------------


def test_cache_hit(tmp_path: Path) -> None:
    """_load_cache() returns data when the cache file exists and is valid JSON."""
    monitor = _make_monitor(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    data = [{"modelId": "meta-llama/Llama-2-7b", "sha": "abc123"}]
    (cache_dir / "huggingface.json").write_text(json.dumps(data), encoding="utf-8")

    result = monitor._load_cache("huggingface")

    assert result == data


# -- test_cache_miss_returns_none ---------------------------------------------


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    """_load_cache() returns None when no cache file exists."""
    monitor = _make_monitor(tmp_path)

    result = monitor._load_cache("huggingface")

    assert result is None


# -- test_cache_write_and_read ------------------------------------------------


def test_cache_write_and_read(tmp_path: Path) -> None:
    """_save_cache() writes valid JSON that _load_cache() can read back."""
    monitor = _make_monitor(tmp_path)
    payload = [{"tag_name": "b3000", "name": "b3000", "published_at": "2026-01-01"}]

    monitor._save_cache("llama_cpp", payload)
    result = monitor._load_cache("llama_cpp")

    assert result == payload


# -- test_offline_fallback ----------------------------------------------------


def test_offline_fallback(tmp_path: Path) -> None:
    """When urlopen raises URLError, cached data is returned instead of failing."""
    # Pre-seed a stale cache (TTL=1 second so age check skips the fast path)
    monitor = _make_monitor(tmp_path, ttl=1)
    cached_data = [{"tag_name": "v0.1", "name": "v0.1", "published_at": "2025-01-01"}]
    monitor._save_cache("llama_cpp", cached_data)

    # Expire the cache so the monitor tries the network
    cache_file = tmp_path / "cache" / "llama_cpp.json"
    # Back-date mtime by 10 seconds so age > ttl
    old_time = time.time() - 10
    import os

    os.utime(cache_file, (old_time, old_time))

    with mock.patch(
        "vetinari.models.landscape_monitor.urllib.request.urlopen",
        side_effect=urllib.error.URLError("network down"),
    ):
        releases = monitor.check_llama_cpp()

    # Should fall back to cached data and return parsed releases
    assert len(releases) == 1
    assert releases[0].name == "v0.1"
    assert releases[0].source == "llama_cpp"


# -- test_flag_stale_knowledge ------------------------------------------------


def test_flag_stale_knowledge(tmp_path: Path) -> None:
    """flag_stale_knowledge() flags families absent from cached HF data."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    # Two families: one present in HF cache, one missing
    families_yaml = {
        "model_families": {
            "llama": {"description": "Meta Llama family"},
            "completely_unknown_xyz": {"description": "Unknown model"},
        }
    }
    (knowledge_dir / "model_families.yaml").write_text(yaml.dump(families_yaml), encoding="utf-8")

    # Seed HF cache with a llama model (so "llama" is found)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hf_data = [{"modelId": "meta-llama/Llama-2-7b", "sha": "abc"}]
    (cache_dir / "huggingface.json").write_text(json.dumps(hf_data), encoding="utf-8")

    monitor = LandscapeMonitor(cache_dir=cache_dir, knowledge_dir=knowledge_dir)
    stale = monitor.flag_stale_knowledge()

    stale_keys = {s.entry_key for s in stale}
    assert "completely_unknown_xyz" in stale_keys
    assert "llama" not in stale_keys


# -- test_compare_rankings_empty ----------------------------------------------


def test_compare_rankings_empty(tmp_path: Path) -> None:
    """compare_rankings() returns a LandscapeReport with empty lists when all fetches fail."""
    monitor = _make_monitor(tmp_path)

    with mock.patch(
        "vetinari.models.landscape_monitor.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        report = monitor.compare_rankings()

    # No cache, no network → releases_found is empty but report is returned
    assert report.releases_found == []
    # sources_checked may be empty (errors) or may list attempted sources
    assert isinstance(report.sources_checked, list)
    assert isinstance(report.errors, list)


# -- test_cache_ttl_expired ---------------------------------------------------


def test_cache_ttl_expired(tmp_path: Path) -> None:
    """When cache exists but is older than TTL, a network fetch is attempted."""
    # TTL = 1 second so any existing file is immediately expired
    monitor = _make_monitor(tmp_path, ttl=1)
    old_data = [{"tag_name": "old", "name": "old", "published_at": "2020-01-01"}]
    monitor._save_cache("vllm", old_data)

    # Back-date the file so it is definitely expired
    cache_file = tmp_path / "cache" / "vllm.json"
    old_time = time.time() - 10
    import os

    os.utime(cache_file, (old_time, old_time))

    fresh_data = [{"tag_name": "new", "name": "new", "published_at": "2026-01-01"}]
    fresh_response = json.dumps(fresh_data).encode("utf-8")

    mock_resp = mock.MagicMock()
    mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
    mock_resp.__exit__ = mock.Mock(return_value=False)
    mock_resp.read.return_value = fresh_response

    with mock.patch(
        "vetinari.models.landscape_monitor.urllib.request.urlopen",
        return_value=mock_resp,
    ):
        releases = monitor.check_vllm()

    # Should have fetched fresh data
    assert len(releases) == 1
    assert releases[0].name == "new"


# -- test_from_cache_stale_field -----------------------------------------------


def test_from_cache_stale_field_defaults_false_when_fresh(tmp_path: Path) -> None:
    """LandscapeReport.from_cache is False when all sources returned live data."""
    monitor = _make_monitor(tmp_path)

    fresh_data = [{"tag_name": "v1.0", "name": "v1.0", "published_at": "2026-01-01"}]
    fresh_response = json.dumps(fresh_data).encode("utf-8")

    mock_resp = mock.MagicMock()
    mock_resp.__enter__ = mock.Mock(return_value=mock_resp)
    mock_resp.__exit__ = mock.Mock(return_value=False)
    mock_resp.read.return_value = fresh_response

    with mock.patch(
        "vetinari.models.landscape_monitor.urllib.request.urlopen",
        return_value=mock_resp,
    ):
        report = monitor.compare_rankings()

    assert report.from_cache is False


def test_from_cache_stale_field_true_when_old(tmp_path: Path) -> None:
    """LandscapeReport.from_cache is True when stale cache satisfied a source."""
    import os

    # Seed expired caches for all three sources
    monitor = _make_monitor(tmp_path, ttl=1)
    stale = [{"tag_name": "old", "name": "old", "published_at": "2020-01-01"}]
    for source in ("llama_cpp", "vllm"):
        monitor._save_cache(source, stale)
        cache_file = tmp_path / "cache" / f"{source}.json"
        old_time = time.time() - 10
        os.utime(cache_file, (old_time, old_time))

    # Seed HF cache as expired too
    hf_stale = [{"modelId": "meta-llama/old", "sha": "aaa"}]
    monitor._save_cache("huggingface", hf_stale)
    hf_file = tmp_path / "cache" / "huggingface.json"
    old_time = time.time() - 10
    os.utime(hf_file, (old_time, old_time))

    with mock.patch(
        "vetinari.models.landscape_monitor.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        report = monitor.compare_rankings()

    # At least one source fell back to stale cache → from_cache must be True
    assert report.from_cache is True
