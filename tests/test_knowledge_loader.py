"""Tests for vetinari.knowledge.loader — cached YAML knowledge access."""

from __future__ import annotations

import logging
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import vetinari.knowledge.loader as _loader_mod
from vetinari.knowledge.loader import (
    _get_cache,
    _index_benchmark_list,
    _KnowledgeCache,
    apply_self_corrections,
    get_architecture_info,
    get_benchmark_info,
    get_family_profile,
    get_parameter_guide,
    get_quant_recommendation,
    invalidate_cache,
    record_knowledge_outcome,
)

# -- Helpers --

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "config" / "knowledge"


def _files_exist(*names: str) -> bool:
    """Return True only if every named YAML file exists in config/knowledge/."""
    return all((_KNOWLEDGE_DIR / name).exists() for name in names)


def _call_with_dir(fn: Any, tmp_path: Path, *args: Any, **kwargs: Any) -> Any:
    """Call a loader public function with _KNOWLEDGE_DIR and singleton redirected."""
    fresh_cache = _KnowledgeCache(ttl_seconds=3600)
    with (
        patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path),
        patch("vetinari.knowledge.loader._instance", fresh_cache),
    ):
        return fn(*args, **kwargs)


# -- Tests: unknown keys return empty dict --


def test_get_benchmark_info_returns_empty_for_unknown() -> None:
    """Unknown benchmark IDs must return an empty dict, never raise."""
    result = get_benchmark_info("nonexistent_benchmark_xyz")
    assert result == {}


def test_get_family_profile_returns_empty_for_unknown() -> None:
    """Unknown family slugs must return an empty dict, never raise."""
    result = get_family_profile("nonexistent_family_xyz")
    assert result == {}


def test_get_parameter_guide_returns_dict_for_unknown() -> None:
    """Unrecognised task types must return a dict (possibly empty), never raise."""
    result = get_parameter_guide("nonexistent_task_xyz")
    assert isinstance(result, dict)


def test_get_architecture_info_returns_empty_for_unknown() -> None:
    """Unknown architecture type must return an empty dict, never raise."""
    result = get_architecture_info("nonexistent_arch_xyz")
    assert result == {}


def test_get_quant_recommendation_returns_dict_for_unknown() -> None:
    """Unknown task type must return a dict (may be empty), never raise."""
    result = get_quant_recommendation("nonexistent_task_xyz")
    assert isinstance(result, dict)


# -- Tests: benchmark list indexing --


def test_index_benchmark_list_slug_lookup() -> None:
    """Benchmark list is indexed by lowercased slug for fast lookup."""
    benchmarks = [
        {"name": "HumanEval", "measures": "Python code"},
        {"name": "MMLU", "measures": "multi-domain knowledge"},
    ]
    index = _index_benchmark_list(benchmarks)
    assert "humaneval" in index
    assert "mmlu" in index
    assert index["humaneval"]["measures"] == "Python code"


def test_index_benchmark_list_exact_name_lookup() -> None:
    """Benchmark list is also indexed by original display name."""
    benchmarks = [{"name": "HumanEval", "measures": "code"}]
    index = _index_benchmark_list(benchmarks)
    assert "HumanEval" in index


def test_index_benchmark_list_skips_non_dicts() -> None:
    """Non-dict entries in the benchmarks list are ignored gracefully."""
    benchmarks: list[Any] = [{"name": "HumanEval"}, "bad_entry", None, 42]
    index = _index_benchmark_list(benchmarks)
    assert list(index.keys()) == ["humaneval", "HumanEval"]


def test_index_benchmark_list_skips_entries_without_name() -> None:
    """Entries missing the 'name' field are skipped silently."""
    benchmarks = [{"measures": "something"}, {"name": "MMLU"}]
    index = _index_benchmark_list(benchmarks)
    assert "mmlu" in index
    assert len(index) == 2  # slug + original name for MMLU only


# -- Tests: singleton behaviour --


def test_cache_singleton_is_same_instance() -> None:
    """_get_cache() must return the identical object on every call."""
    a = _get_cache()
    b = _get_cache()
    assert a is b


def test_cache_singleton_thread_safe() -> None:
    """Concurrent first-calls must still yield a single shared instance."""
    results: list[_KnowledgeCache] = []
    barrier = threading.Barrier(8)

    def _grab() -> None:
        barrier.wait()
        results.append(_get_cache())

    threads = [threading.Thread(target=_grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is results[0] for r in results)


# -- Tests: invalidate cache --


def test_invalidate_cache_clears_specific_file() -> None:
    """Invalidating a specific file removes it from cache so next get reloads."""
    cache = _KnowledgeCache(ttl_seconds=3600)
    cache._cache["benchmarks.yaml"] = {"benchmarks": []}
    cache._timestamps["benchmarks.yaml"] = 1.0

    cache.invalidate("benchmarks.yaml")

    assert "benchmarks.yaml" not in cache._cache
    assert "benchmarks.yaml" not in cache._timestamps


def test_invalidate_cache_clears_all() -> None:
    """Invalidating with None clears every cached file."""
    cache = _KnowledgeCache(ttl_seconds=3600)
    cache._cache["a.yaml"] = {}
    cache._cache["b.yaml"] = {}
    cache._timestamps["a.yaml"] = 1.0
    cache._timestamps["b.yaml"] = 2.0

    cache.invalidate(None)

    assert cache._cache == {}
    assert cache._timestamps == {}


def test_invalidate_cache_public_api() -> None:
    """invalidate_cache() public function must not raise."""
    cache = _get_cache()
    cache._cache["benchmarks.yaml"] = {"benchmarks": []}
    cache._timestamps["benchmarks.yaml"] = 1.0

    invalidate_cache("benchmarks.yaml")
    assert "benchmarks.yaml" not in cache._cache
    assert "benchmarks.yaml" not in cache._timestamps

    cache._cache["benchmarks.yaml"] = {"benchmarks": []}
    cache._timestamps["benchmarks.yaml"] = 1.0
    invalidate_cache(None)
    assert cache._cache == {}
    assert cache._timestamps == {}

    invalidate_cache()
    assert cache._cache == {}
    assert cache._timestamps == {}


# -- Tests: TTL expiry --


def test_cache_ttl_expired_reloads(tmp_path: Path) -> None:
    """After TTL expires, the cache must reload from disk."""
    yaml_content = "benchmarks:\n  - name: HumanEval\n    measures: code\n"
    (tmp_path / "benchmarks.yaml").write_text(yaml_content, encoding="utf-8")

    cache = _KnowledgeCache(ttl_seconds=0.0)  # always expired

    with patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path):
        first = cache.get("benchmarks.yaml")
        second = cache.get("benchmarks.yaml")

    assert first == second
    assert "benchmarks" in first


# -- Tests: graceful fallback on missing/corrupt files --


def test_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    """A missing YAML file must return {} and not raise."""
    cache = _KnowledgeCache()
    with patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path):
        result = cache.get("benchmarks.yaml")
    assert result == {}


def test_corrupt_yaml_returns_empty_dict(tmp_path: Path) -> None:
    """A YAML parse error must return {} and not propagate."""
    (tmp_path / "benchmarks.yaml").write_text("key: [\nbad yaml {{{\n", encoding="utf-8")

    cache = _KnowledgeCache()
    with patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path):
        result = cache.get("benchmarks.yaml")
    assert result == {}


def test_non_dict_yaml_returns_empty_dict(tmp_path: Path) -> None:
    """A YAML file whose root is a list (not dict) must return {}."""
    (tmp_path / "benchmarks.yaml").write_text("- item1\n- item2\n", encoding="utf-8")

    cache = _KnowledgeCache()
    with patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path):
        result = cache.get("benchmarks.yaml")
    assert result == {}


def test_missing_top_level_key_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A YAML file missing its expected top-level key must log a warning."""
    (tmp_path / "benchmarks.yaml").write_text("other_key: value\n", encoding="utf-8")
    cache = _KnowledgeCache()
    with patch("vetinari.knowledge.loader._KNOWLEDGE_DIR", tmp_path):
        with caplog.at_level(logging.WARNING, logger="vetinari.knowledge.loader"):
            cache.get("benchmarks.yaml")

    assert any("missing expected top-level key" in r.message for r in caplog.records)


# -- Synthetic YAML fixture --


@pytest.fixture
def synthetic_knowledge_dir(tmp_path: Path) -> Path:
    """Populate tmp_path with minimal but valid YAML knowledge files.

    Schema mirrors the real config/knowledge/ files exactly:
    - benchmarks.yaml: benchmarks is a list of dicts
    - quantization.yaml: quantization_methods is a keyed dict + task_recommendations
    - model_families.yaml: model_families is a keyed dict
    - parameters.yaml: parameters is a keyed dict + parameter_presets
    - architecture.yaml: architecture_types, attention_mechanisms, optimization_features
    """
    benchmarks = textwrap.dedent("""\
        benchmarks:
          - name: HumanEval
            measures: Python code generation correctness
            predicts:
              - python_function_completion
            does_not_predict:
              - multi_file_coding
            score_interpretation:
              "0-50": "Poor"
              "50-100": "Good"
            saturation_threshold: 95
            weight_for_task:
              coding: 0.8
            contamination_risk: high
          - name: MMLU
            measures: multi-domain knowledge
            predicts:
              - general_knowledge
            does_not_predict:
              - creative_writing
            score_interpretation:
              "0-50": "Poor"
              "50-100": "Good"
            saturation_threshold: 90
            weight_for_task:
              reasoning: 0.7
            contamination_risk: medium
    """)

    quantization = textwrap.dedent("""\
        quantization_methods:
          q4_k_m:
            name: Q4_K_M
            bits_per_weight: 4.5
            quality_retention: 0.92
            speed_factor: 1.6
            vram_factor: 0.28
            recommended_for:
              - coding
              - general
            not_recommended_for: []
            temperature_offset: 0.02
          q8_0:
            name: Q8_0
            bits_per_weight: 8.0
            quality_retention: 0.99
            speed_factor: 1.1
            vram_factor: 0.50
            recommended_for:
              - general
              - reasoning
            not_recommended_for: []
            temperature_offset: 0.0
        task_recommendations:
          coding:
            preferred: q4_k_m
            notes: good balance of speed and quality
          general:
            preferred: q4_k_m
            notes: default choice
    """)

    model_families = textwrap.dedent("""\
        model_families:
          llama:
            name: Llama (3.x series)
            vendor: Meta
            architecture_type: dense
            available_sizes: [8, 70]
            native_capabilities:
              - function_calling
              - code_generation
            strengths:
              - Well-rounded generalist
            weaknesses:
              - High VRAM for large sizes
            recommended_roles: [worker, foreman]
          qwen2:
            name: Qwen2
            vendor: Alibaba
            architecture_type: dense
            available_sizes: [7, 72]
            native_capabilities:
              - multilingual
              - math
            strengths:
              - Strong math and multilingual
            weaknesses:
              - Creative writing
            recommended_roles: [worker]
    """)

    parameters = textwrap.dedent("""\
        parameters:
          temperature:
            name: Temperature
            what_it_controls: Controls randomness of output
            type: float
            range:
              min: 0.0
              max: 2.0
              default: 0.7
            task_recommendations:
              coding: "0.05-0.1"
              creative: "0.7-0.9"
          top_p:
            name: Top-P
            what_it_controls: Nucleus sampling cutoff
            type: float
            range:
              min: 0.0
              max: 1.0
              default: 0.9
            task_recommendations:
              coding: "0.85-0.95"
              creative: "0.9-0.95"
        parameter_presets:
          code:
            temperature: 0.1
            top_p: 0.95
            repeat_penalty: 1.1
            description: Low temperature for deterministic code output
          reasoning:
            temperature: 0.3
            top_p: 0.9
            description: Focused but slightly creative
          creative:
            temperature: 0.8
            top_p: 0.95
            description: High creativity for open-ended generation
          deterministic:
            temperature: 0.05
            top_p: 1.0
            description: Near-deterministic for classification
    """)

    architecture = textwrap.dedent("""\
        architecture_types:
          dense:
            description: All parameters activated for every token
            vram_usage: Proportional to total parameter count
            strengths:
              - Predictable VRAM requirements
            weaknesses:
              - Higher compute per token
          moe:
            description: Mixture of Experts — only a subset of parameters active per token
            vram_usage: Must load ALL experts into memory
            strengths:
              - Higher quality per compute FLOP
            weaknesses:
              - VRAM must hold ALL experts
        attention_mechanisms:
          flash_attention:
            description: Memory-efficient attention using tiling
            benefit: Reduced VRAM for long contexts
        optimization_features:
          sliding_window:
            description: Limits attention to a local context window
            benefit: O(n) memory instead of O(n^2)
    """)

    (tmp_path / "benchmarks.yaml").write_text(benchmarks, encoding="utf-8")
    (tmp_path / "quantization.yaml").write_text(quantization, encoding="utf-8")
    (tmp_path / "model_families.yaml").write_text(model_families, encoding="utf-8")
    (tmp_path / "parameters.yaml").write_text(parameters, encoding="utf-8")
    (tmp_path / "architecture.yaml").write_text(architecture, encoding="utf-8")

    return tmp_path


# -- Tests using synthetic fixtures --


def test_benchmark_info_loads_humaneval(synthetic_knowledge_dir: Path) -> None:
    """humaneval entry loaded from synthetic YAML must have 'measures' key."""
    result = _call_with_dir(get_benchmark_info, synthetic_knowledge_dir, "humaneval")
    assert "measures" in result


def test_benchmark_info_loads_by_display_name(synthetic_knowledge_dir: Path) -> None:
    """Benchmark lookup by original display name must also work."""
    result = _call_with_dir(get_benchmark_info, synthetic_knowledge_dir, "HumanEval")
    assert "measures" in result


def test_family_profile_loads_llama(synthetic_knowledge_dir: Path) -> None:
    """llama entry loaded from synthetic YAML must have 'strengths' key."""
    result = _call_with_dir(get_family_profile, synthetic_knowledge_dir, "llama")
    assert "strengths" in result


def test_parameter_guide_coding_returns_preset(
    synthetic_knowledge_dir: Path,
) -> None:
    """coding task must resolve to the 'code' preset with a temperature key."""
    result = _call_with_dir(get_parameter_guide, synthetic_knowledge_dir, "coding")
    assert result.get("preset") == "code"
    assert "temperature" in result


def test_parameter_guide_math_maps_to_reasoning_preset(
    synthetic_knowledge_dir: Path,
) -> None:
    """math task must resolve to the 'reasoning' preset."""
    result = _call_with_dir(get_parameter_guide, synthetic_knowledge_dir, "math")
    assert result.get("preset") == "reasoning"


def test_parameter_guide_unknown_falls_back_to_per_param(
    synthetic_knowledge_dir: Path,
) -> None:
    """An unrecognised task type should return a dict (may be empty)."""
    result = _call_with_dir(get_parameter_guide, synthetic_knowledge_dir, "custom_task_xyz")
    assert isinstance(result, dict)


def test_architecture_info_dense(synthetic_knowledge_dir: Path) -> None:
    """dense architecture info must contain a 'description' key."""
    result = _call_with_dir(get_architecture_info, synthetic_knowledge_dir, "dense")
    assert "description" in result


def test_architecture_info_attention_mechanism(
    synthetic_knowledge_dir: Path,
) -> None:
    """flash_attention falls under attention_mechanisms and must be found."""
    result = _call_with_dir(get_architecture_info, synthetic_knowledge_dir, "flash_attention")
    assert "description" in result


def test_architecture_info_optimization_feature(
    synthetic_knowledge_dir: Path,
) -> None:
    """sliding_window falls under optimization_features and must be found."""
    result = _call_with_dir(get_architecture_info, synthetic_knowledge_dir, "sliding_window")
    assert "description" in result


def test_quant_recommendation_coding(synthetic_knowledge_dir: Path) -> None:
    """coding task must yield a result with suitable_methods."""
    result = _call_with_dir(get_quant_recommendation, synthetic_knowledge_dir, "coding")
    assert isinstance(result, dict)
    assert "suitable_methods" in result
    assert len(result["suitable_methods"]) > 0


def test_quant_recommendation_includes_vram_when_provided(
    synthetic_knowledge_dir: Path,
) -> None:
    """available_vram_gb parameter must be echoed back in the result."""
    result = _call_with_dir(
        get_quant_recommendation,
        synthetic_knowledge_dir,
        "coding",
        available_vram_gb=8.0,
    )
    assert result.get("available_vram_gb") == 8.0


# -- Tests: real files (skipped when files absent) --


@pytest.mark.skipif(
    not _files_exist("benchmarks.yaml"),
    reason="config/knowledge/benchmarks.yaml not present",
)
def test_benchmark_info_humaneval_real() -> None:
    """If benchmarks.yaml is present, humaneval must have a 'measures' key."""
    result = get_benchmark_info("humaneval")
    assert "measures" in result


@pytest.mark.skipif(
    not _files_exist("benchmarks.yaml"),
    reason="config/knowledge/benchmarks.yaml not present",
)
def test_benchmark_info_humaneval_real_display_name() -> None:
    """If benchmarks.yaml is present, 'HumanEval' display name lookup works."""
    result = get_benchmark_info("HumanEval")
    assert "measures" in result


@pytest.mark.skipif(
    not _files_exist("model_families.yaml"),
    reason="config/knowledge/model_families.yaml not present",
)
def test_family_profile_llama_real() -> None:
    """If model_families.yaml is present, llama entry must have 'strengths'."""
    result = get_family_profile("llama")
    assert "strengths" in result


@pytest.mark.skipif(
    not _files_exist("parameters.yaml"),
    reason="config/knowledge/parameters.yaml not present",
)
def test_parameter_guide_coding_real() -> None:
    """If parameters.yaml is present, coding task must return a non-empty dict."""
    result = get_parameter_guide("coding")
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.skipif(
    not _files_exist("architecture.yaml"),
    reason="config/knowledge/architecture.yaml not present",
)
def test_architecture_info_dense_real() -> None:
    """If architecture.yaml is present, dense arch must have 'description'."""
    result = get_architecture_info("dense")
    assert "description" in result


# -- record_knowledge_outcome / apply_self_corrections --


@pytest.fixture
def clean_prediction_records():
    """Clear the global prediction buffer before and after each test."""
    with _loader_mod._tracker_lock:
        _loader_mod._prediction_records.clear()
    yield
    with _loader_mod._tracker_lock:
        _loader_mod._prediction_records.clear()


class TestRecordKnowledgeOutcome:
    def test_record_is_stored(self, clean_prediction_records) -> None:
        record_knowledge_outcome("benchmarks", "mmlu", 0.8, 0.85)
        with _loader_mod._tracker_lock:
            assert len(_loader_mod._prediction_records) == 1

    def test_record_content(self, clean_prediction_records) -> None:
        record_knowledge_outcome("benchmarks", "mmlu", 0.8, 0.85)
        with _loader_mod._tracker_lock:
            source, item_id, predicted, actual = _loader_mod._prediction_records[0]
        assert source == "benchmarks"
        assert item_id == "mmlu"
        assert predicted == 0.8
        assert actual == 0.85

    def test_multiple_records_accumulate(self, clean_prediction_records) -> None:
        for _ in range(5):
            record_knowledge_outcome("benchmarks", "mmlu", 0.8, 0.9)
        with _loader_mod._tracker_lock:
            assert len(_loader_mod._prediction_records) == 5

    def test_float_coercion(self, clean_prediction_records) -> None:
        record_knowledge_outcome("benchmarks", "mmlu", 1, 0)
        with _loader_mod._tracker_lock:
            _, _, predicted, actual = _loader_mod._prediction_records[0]
        assert isinstance(predicted, float)
        assert isinstance(actual, float)


class TestApplySelfCorrections:
    def test_returns_empty_when_too_few_records(self, clean_prediction_records) -> None:
        for _ in range(10):
            record_knowledge_outcome("benchmarks", "mmlu", 0.5, 0.9)
        result = apply_self_corrections()
        assert result == []

    def test_returns_empty_when_divergence_below_threshold(self, clean_prediction_records) -> None:
        # 50 records with tiny divergence (0.01) — below the 0.15 threshold
        for _ in range(50):
            record_knowledge_outcome("benchmarks", "mmlu", 0.80, 0.81)
        result = apply_self_corrections()
        assert result == []

    def test_patches_file_when_divergence_exceeds_threshold(
        self, clean_prediction_records, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(_loader_mod, "_KNOWLEDGE_DIR", tmp_path)
        (tmp_path / "benchmarks.yaml").write_text("mmlu:\n  description: test\n", encoding="utf-8")
        for _ in range(50):
            record_knowledge_outcome("benchmarks", "mmlu", 0.5, 0.9)  # 0.4 divergence
        result = apply_self_corrections()
        assert "benchmarks.yaml" in result

    def test_patched_yaml_contains_calibration_section(self, clean_prediction_records, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(_loader_mod, "_KNOWLEDGE_DIR", tmp_path)
        yaml_path = tmp_path / "benchmarks.yaml"
        yaml_path.write_text("mmlu:\n  description: test\n", encoding="utf-8")
        for _ in range(50):
            record_knowledge_outcome("benchmarks", "mmlu", 0.5, 0.9)
        apply_self_corrections()
        content = yaml_path.read_text(encoding="utf-8")
        assert "calibration" in content

    def test_records_cleared_after_patch(self, clean_prediction_records, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(_loader_mod, "_KNOWLEDGE_DIR", tmp_path)
        (tmp_path / "benchmarks.yaml").write_text("mmlu:\n  description: test\n", encoding="utf-8")
        for _ in range(50):
            record_knowledge_outcome("benchmarks", "mmlu", 0.5, 0.9)
        apply_self_corrections()
        with _loader_mod._tracker_lock:
            remaining = [r for r in _loader_mod._prediction_records if r[0] == "benchmarks"]
        assert remaining == []
