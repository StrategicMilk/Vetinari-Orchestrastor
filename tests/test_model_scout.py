"""Tests for vetinari.models.model_scout."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_model_candidate
from vetinari.models.model_scout import (
    ModelRecommendation,
    ModelScout,
    get_model_scout,
    reset_model_scout,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ModelScout singleton before and after each test."""
    reset_model_scout()
    yield
    reset_model_scout()


@pytest.fixture
def scout() -> ModelScout:
    """Return a fresh ModelScout instance."""
    return ModelScout()


# ---------------------------------------------------------------------------
# ModelRecommendation.to_dict
# ---------------------------------------------------------------------------


class TestModelRecommendationToDict:
    def test_serializes_all_fields(self):
        rec = ModelRecommendation(
            model_name="llama-3-8b",
            source="huggingface",
            task_type="coding",
            estimated_quality=0.823,
            vram_estimate_gb=8.0,
            reason="Strong benchmark score.",
        )
        result = rec.to_dict()
        assert result["model_name"] == "llama-3-8b"
        assert result["source"] == "huggingface"
        assert result["task_type"] == "coding"
        assert result["estimated_quality"] == 0.823
        assert result["vram_estimate_gb"] == 8.0
        assert result["reason"] == "Strong benchmark score."
        assert result["recommended_backend"] == "vllm"
        assert result["recommended_format"] == "safetensors"

    def test_rounds_estimated_quality_to_3_places(self):
        rec = ModelRecommendation(estimated_quality=0.123456789)
        result = rec.to_dict()
        assert result["estimated_quality"] == 0.123

    def test_rounds_vram_to_1_place(self):
        rec = ModelRecommendation(vram_estimate_gb=7.678)
        result = rec.to_dict()
        assert result["vram_estimate_gb"] == 7.7

    def test_default_values(self):
        rec = ModelRecommendation()
        result = rec.to_dict()
        assert result["model_name"] == ""
        assert result["estimated_quality"] == 0.0


# ---------------------------------------------------------------------------
# ModelScout.is_underperforming
# ---------------------------------------------------------------------------


class TestIsUnderperforming:
    def test_returns_false_when_no_thompson_data(self, scout):
        """Empty rankings mean no underperformance signal."""
        mock_selector = MagicMock()
        mock_selector.get_rankings.return_value = []

        # The method does `from vetinari.learning.model_selector import get_thompson_selector`
        # so we patch the function in that module.
        with patch("vetinari.learning.model_selector.get_thompson_selector", return_value=mock_selector):
            result = scout.is_underperforming("coding")

        assert result is False

    def test_returns_true_when_all_models_have_low_mean(self, scout):
        """All models below threshold triggers underperformance."""
        mock_selector = MagicMock()
        mock_selector.get_rankings.return_value = [
            ("model-a", 0.3),
            ("model-b", 0.4),
            ("model-c", 0.2),
        ]

        with patch("vetinari.learning.model_selector.get_thompson_selector", return_value=mock_selector):
            result = scout.is_underperforming("coding")

        assert result is True

    def test_returns_false_when_at_least_one_model_has_good_mean(self, scout):
        """One model above threshold means not all underperforming."""
        mock_selector = MagicMock()
        mock_selector.get_rankings.return_value = [
            ("model-a", 0.3),
            ("model-b", 0.8),  # above 0.5 threshold
        ]

        with patch("vetinari.learning.model_selector.get_thompson_selector", return_value=mock_selector):
            result = scout.is_underperforming("coding")

        assert result is False

    def test_returns_false_when_exception_raised(self, scout):
        """Exceptions during Thompson lookup return False safely."""
        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            side_effect=RuntimeError("unavailable"),
        ):
            result = scout.is_underperforming("coding")

        assert result is False


# ---------------------------------------------------------------------------
# ModelScout.scout_for_task / get_recommendations
# ---------------------------------------------------------------------------


_DISCOVERY_PATCH = "vetinari.model_discovery.ModelDiscovery"


class TestScoutForTask:
    def test_returns_recommendations_from_discovery(self, scout):
        """Results from ModelDiscovery are mapped to ModelRecommendation."""
        candidates = [
            make_model_candidate(name="LLaMA-3", source_type="huggingface", final_score=0.85, memory_gb=8),
            make_model_candidate(name="Mistral-7B", source_type="reddit", final_score=0.72, memory_gb=6),
        ]

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = candidates

            recs = scout.scout_for_task("coding")

        assert len(recs) == 2
        # After sort by quality desc, LLaMA-3 (0.85) is first
        assert recs[0].model_name == "LLaMA-3"
        assert recs[0].source == "huggingface"
        assert recs[0].task_type == "coding"
        assert recs[0].estimated_quality == 0.85
        assert recs[0].vram_estimate_gb == 8.0
        assert recs[0].recommended_backend == "vllm"
        assert recs[0].recommended_format == "safetensors"

    def test_infers_gguf_backend_for_llama_cpp_candidates(self, scout):
        """Scout recommendations keep GGUF candidates on llama.cpp."""
        candidates = [
            make_model_candidate(name="Qwen2.5-7B-Instruct-GGUF", source_type="huggingface", final_score=0.85),
        ]

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = candidates

            recs = scout.scout_for_task("coding")

        assert recs[0].recommended_backend == "llama_cpp"
        assert recs[0].recommended_format == "gguf"

    def test_results_sorted_by_quality_descending(self, scout):
        """Recommendations are returned highest quality first."""
        candidates = [
            make_model_candidate(name="Low", source_type="huggingface", final_score=0.3),
            make_model_candidate(name="High", source_type="reddit", final_score=0.9),
            make_model_candidate(name="Mid", source_type="github", final_score=0.6),
        ]

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = candidates

            recs = scout.scout_for_task("reasoning")

        assert recs[0].estimated_quality >= recs[1].estimated_quality >= recs[2].estimated_quality

    def test_handles_discovery_unavailable_gracefully(self, scout):
        """Raises no exception and returns empty list when ModelDiscovery import fails."""
        with patch(_DISCOVERY_PATCH, side_effect=ImportError("no module")):
            recs = scout.scout_for_task("coding")

        assert recs == []

    def test_handles_discovery_search_exception_gracefully(self, scout):
        """Returns empty list when discovery.search() raises."""
        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.side_effect = RuntimeError("network error")

            recs = scout.scout_for_task("coding")

        assert recs == []

    def test_respects_max_recommendations(self, scout):
        """At most MAX_RECOMMENDATIONS results are returned."""
        candidates = [
            make_model_candidate(name=f"Model-{i}", source_type="huggingface", final_score=0.5 + i * 0.01)
            for i in range(20)
        ]

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = candidates

            recs = scout.scout_for_task("general")

        assert len(recs) <= ModelScout.MAX_RECOMMENDATIONS

    def test_uses_mapped_query_for_known_task_types(self, scout):
        """Known task types use predefined search queries."""
        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = []

            scout.scout_for_task("coding")

            instance.search.assert_called_once_with(
                "best coding LLM vLLM NIM safetensors AWQ GPTQ local 2026"
            )

    def test_uses_fallback_query_for_unknown_task_types(self, scout):
        """Unknown task types fall back to generic query pattern."""
        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = []

            scout.scout_for_task("translation")

            instance.search.assert_called_once_with(
                "best translation LLM vLLM NIM safetensors AWQ GPTQ local"
            )


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    def test_get_recommendations_uses_cache_on_second_call(self, scout):
        """Second call to get_recommendations must not re-invoke ModelDiscovery."""
        from vetinari.model_discovery import ModelCandidate

        c = ModelCandidate(
            id="test",
            name="TestModel",
            source_type="huggingface",
            memory_gb=4,
            final_score=0.7,
            short_rationale="Test",
        )

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = [c]

            scout.get_recommendations("coding")
            scout.get_recommendations("coding")

            # search should only be called once despite two get_recommendations calls
            assert instance.search.call_count == 1

    def test_clear_cache_forces_fresh_search(self, scout):
        """After clear_cache, the next call must re-invoke ModelDiscovery."""
        from vetinari.model_discovery import ModelCandidate

        c = ModelCandidate(
            id="test",
            name="TestModel",
            source_type="huggingface",
            memory_gb=4,
            final_score=0.7,
            short_rationale="Test",
        )

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = [c]

            scout.get_recommendations("coding")
            scout.clear_cache()
            scout.get_recommendations("coding")

            assert instance.search.call_count == 2

    def test_clear_cache_removes_all_task_types(self, scout):
        """clear_cache removes entries for every cached task type."""
        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = []

            scout.get_recommendations("coding")
            scout.get_recommendations("reasoning")

        assert len(scout._cache) == 2
        scout.clear_cache()
        assert len(scout._cache) == 0


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_returns_ok_true(self, scout):
        status = scout.get_status()
        assert status["ok"] is True

    def test_reflects_cached_task_types(self, scout):
        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = []
            scout.get_recommendations("coding")
            scout.get_recommendations("reasoning")

        status = scout.get_status()
        assert "coding" in status["cached_task_types"]
        assert "reasoning" in status["cached_task_types"]

    def test_total_recommendations_counts_all_cached(self, scout):
        from vetinari.model_discovery import ModelCandidate

        def make_c(name):
            return ModelCandidate(
                id=name,
                name=name,
                source_type="hf",
                memory_gb=4,
                final_score=0.5,
                short_rationale="",
            )

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.side_effect = [
                [make_c("A"), make_c("B")],
                [make_c("C")],
            ]
            scout.get_recommendations("coding")
            scout.get_recommendations("reasoning")

        status = scout.get_status()
        assert status["total_recommendations"] == 3


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_model_scout_returns_same_instance(self):
        a = get_model_scout()
        b = get_model_scout()
        assert a is b

    def test_reset_model_scout_creates_fresh_instance(self):
        a = get_model_scout()
        reset_model_scout()
        b = get_model_scout()
        assert a is not b

    def test_get_model_scout_returns_model_scout_instance(self):
        instance = get_model_scout()
        assert isinstance(instance, ModelScout)


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestModelScoutGetRecommendationsStateIsolation:
    """Defect 4: get_recommendations must return defensive copies so callers cannot
    mutate the internal cache state."""

    def test_appending_to_returned_list_does_not_mutate_cache(self, scout):
        """Mutating the list returned by get_recommendations must not affect _cache.

        The defensive copy contract: each call returns a new list of new
        ModelRecommendation objects. Appending to the caller's list must leave
        the internal cache length unchanged.
        """
        from vetinari.model_discovery import ModelCandidate

        candidate = ModelCandidate(
            id="test-model",
            name="TestModel",
            source_type="huggingface",
            memory_gb=8,
            final_score=0.85,
            short_rationale="Strong benchmark score.",
        )

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = [candidate]

            first_result = scout.get_recommendations("coding")

        # Confirm the cache was populated with exactly 1 item.
        assert len(first_result) == 1
        cache_len_before = len(scout._cache["coding"])

        # Mutate the returned list — add a fake entry.
        first_result.append(
            ModelRecommendation(
                model_name="intruder",
                source="fake",
                task_type="coding",
                estimated_quality=0.1,
                vram_estimate_gb=1.0,
                reason="Should not appear in cache.",
            )
        )

        # The internal cache must still have the original count.
        assert len(scout._cache["coding"]) == cache_len_before, (
            f"Mutating the returned list changed the internal cache: "
            f"expected {cache_len_before} items, got {len(scout._cache['coding'])}"
        )

    def test_mutating_returned_recommendation_does_not_mutate_cache(self, scout):
        """Modifying a field on a returned ModelRecommendation must not alter the
        cached copy — each returned object is a separate dataclass instance."""
        from vetinari.model_discovery import ModelCandidate

        candidate = ModelCandidate(
            id="test-model",
            name="OriginalName",
            source_type="huggingface",
            memory_gb=4,
            final_score=0.7,
            short_rationale="Test.",
        )

        with patch(_DISCOVERY_PATCH) as MockDiscovery:
            instance = MockDiscovery.return_value
            instance.search.return_value = [candidate]

            first_result = scout.get_recommendations("coding")

        original_name = scout._cache["coding"][0].model_name

        # Forcibly replace the field on the returned copy.
        import dataclasses

        first_result[0] = dataclasses.replace(first_result[0], model_name="MutatedName")

        # The cache must still hold the original name.
        assert scout._cache["coding"][0].model_name == original_name, (
            "Replacing a returned recommendation mutated the cached entry"
        )


class TestImport:
    def test_direct_import(self):
        from vetinari.models.model_scout import ModelScout as MS

        assert MS is ModelScout

    def test_package_import(self):
        from vetinari.models import ModelRecommendation, ModelScout, get_model_scout

        assert callable(ModelScout)
        assert callable(ModelRecommendation)
        assert callable(get_model_scout)
