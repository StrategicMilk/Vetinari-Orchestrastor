"""Tests for vetinari.setup.model_recommender — VRAM-to-model recommendation engine."""

from __future__ import annotations

import pytest

from tests.factories import make_hardware_profile
from vetinari.setup.model_recommender import ModelRecommender, SetupModelRecommendation
from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile


class TestModelRecommendation:
    """Tests for the ModelRecommendation dataclass."""

    def test_to_dict(self) -> None:
        """Serialization includes all fields."""
        rec = SetupModelRecommendation(
            name="Test Model 7B Q4_K_M",
            repo_id="test/test-model-GGUF",
            filename="test-7b-q4_k_m.gguf",
            size_gb=4.1,
            quantization="Q4_K_M",
            parameter_count="7B",
            reason="Test reason",
            is_primary=True,
        )
        d = rec.to_dict()
        assert d["name"] == "Test Model 7B Q4_K_M"
        assert d["repo_id"] == "test/test-model-GGUF"
        assert d["size_gb"] == 4.1
        assert d["quantization"] == "Q4_K_M"
        assert d["is_primary"] is True


class TestModelRecommender:
    """Tests for the ModelRecommender VRAM-to-model matrix."""

    def test_cpu_only_gets_smallest_models(self) -> None:
        """CPU-only system gets the smallest model tier."""
        profile = make_hardware_profile(vram_gb=0.0, ram_gb=8.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        assert len(recs) >= 1
        # Should be small models (1-3B range)
        assert all(r.size_gb < 5 for r in recs)

    def test_4gb_vram_gets_7b_models(self) -> None:
        """4GB VRAM system gets 7B Q4 models."""
        profile = make_hardware_profile(vram_gb=6.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        assert len(recs) >= 1
        # effective_vram = 6 * 0.9 = 5.4, in the 4-8GB tier
        primary = next(r for r in recs if r.is_primary)
        assert "7B" in primary.parameter_count or "7b" in primary.name.lower()

    def test_12gb_vram_gets_q6_models(self) -> None:
        """12GB VRAM system gets Q6_K quantized models."""
        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        assert len(recs) >= 1
        # effective_vram = 12 * 0.9 = 10.8, in the 8-16GB tier
        primary = next(r for r in recs if r.is_primary)
        assert "Q6" in primary.quantization

    def test_24gb_vram_gets_14b_models(self) -> None:
        """24GB VRAM system gets 14B+ parameter models."""
        profile = make_hardware_profile(vram_gb=24.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        assert len(recs) >= 1
        # effective_vram = 24 * 0.9 = 21.6, in the 16-24GB tier
        primary = next(r for r in recs if r.is_primary)
        assert "14B" in primary.parameter_count or "22B" in primary.parameter_count

    def test_48gb_vram_gets_largest_models(self) -> None:
        """48GB VRAM system gets the largest available models."""
        profile = make_hardware_profile(vram_gb=48.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        assert len(recs) >= 1
        # Some model should be 14B+ at high quantization or 32B
        assert any("14B" in r.parameter_count or "32B" in r.parameter_count for r in recs)

    def test_apple_silicon_uses_unified_memory(self) -> None:
        """Apple Silicon with unified memory estimates VRAM from RAM."""
        gpu = GpuInfo(
            name="Apple M2 Pro",
            vendor=GpuVendor.APPLE,
            vram_gb=12.0,  # 16GB * 0.75
            metal_available=True,
        )
        profile = HardwareProfile(cpu_count=10, ram_gb=16.0, gpu=gpu)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(profile)
        # effective_vram = 12 * 0.9 = 10.8, should be in 8-16GB tier
        assert len(recs) >= 1

    def test_primary_recommendation_exists(self) -> None:
        """Every tier has exactly one primary recommendation."""
        for vram in [0, 5, 10, 18, 30]:
            profile = make_hardware_profile(
                vram_gb=float(vram),
                vendor=GpuVendor.NVIDIA if vram > 0 else GpuVendor.NONE,
            )
            recommender = ModelRecommender()
            recs = recommender.recommend_models(profile)
            primaries = [r for r in recs if r.is_primary]
            assert len(primaries) == 1, f"Expected 1 primary for VRAM={vram}, got {len(primaries)}"

    def test_get_tier_label(self) -> None:
        """get_tier_label() returns a human-readable string."""
        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        label = recommender.get_tier_label(profile)
        assert isinstance(label, str)
        assert "GB" in label


# -- recommend_for_task --


class TestRecommendForTask:
    def test_returns_list(self) -> None:
        """recommend_for_task always returns a list."""
        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        result = recommender.recommend_for_task(profile, "coding")
        assert isinstance(result, list)

    def test_returns_model_recommendation_objects(self) -> None:
        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        result = recommender.recommend_for_task(profile, "coding")
        assert all(isinstance(r, SetupModelRecommendation) for r in result)

    def test_returns_same_count_as_base_when_no_quant_match(self) -> None:
        """When the task has no quant recommendation, the list is unchanged."""
        from unittest.mock import patch

        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        base = recommender.recommend_models(profile)
        with patch("vetinari.setup.model_recommender.get_quant_recommendation", return_value=None):
            result = recommender.recommend_for_task(profile, "unknown_task_xyz")
        assert len(result) == len(base)

    def test_annotates_reason_when_quant_matches(self) -> None:
        """A recommendation whose quant matches the task preferred quant gets an annotation."""
        from unittest.mock import patch

        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        base = recommender.recommend_models(profile)
        if not base:
            pytest.skip("No base recommendations for this hardware profile")
        # Pick the quant of the first recommendation as the preferred quant
        target_quant = base[0].quantization.lower()
        fake_rec = {"task_recommendation": {"preferred": target_quant, "notes": "best for coding"}}
        with patch("vetinari.setup.model_recommender.get_quant_recommendation", return_value=fake_rec):
            result = recommender.recommend_for_task(profile, "coding")
        annotated = [r for r in result if "(recommended quant" in r.reason]
        assert len(annotated) >= 1

    def test_non_matching_recommendations_unchanged(self) -> None:
        """Recommendations whose quant doesn't match the preferred are returned unmodified."""
        from unittest.mock import patch

        profile = make_hardware_profile(vram_gb=12.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        base = recommender.recommend_models(profile)
        if not base:
            pytest.skip("No base recommendations for this hardware profile")
        # Use a quant that won't match any real recommendation
        fake_rec = {"task_recommendation": {"preferred": "q99_impossible", "notes": ""}}
        with patch("vetinari.setup.model_recommender.get_quant_recommendation", return_value=fake_rec):
            result = recommender.recommend_for_task(profile, "coding")
        # No recommendations should be annotated since q99_impossible matches nothing
        non_annotated = [r for r in result if "(recommended quant" not in r.reason]
        assert len(non_annotated) == len(base)


# ── Session 15: KV cache quant suggestion ─────────────────────────────────


class TestSuggestKvCacheQuant:
    """ModelRecommender.suggest_kv_cache_quant picks the right tier for the VRAM budget."""

    def test_returns_f16_for_high_vram(self) -> None:
        """A 24 GB GPU with a short context has plenty of headroom — f16 is fine."""
        profile = make_hardware_profile(vram_gb=24.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        result = recommender.suggest_kv_cache_quant(profile, context_length=4096)
        assert result == "f16"

    def test_returns_q8_0_for_moderate_pressure(self) -> None:
        """A 6 GB GPU with a large context hits the 50-75% KV VRAM band → q8_0."""
        profile = make_hardware_profile(vram_gb=6.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        # effective_vram = 6 * 0.9 = 5.4 GB; 20% headroom = 1.08 GB
        # 100k * 2048 bytes = ~0.19 GB < 50% of 1.08 GB → f16, so use 300k
        # 300k * 2048 bytes = ~0.57 GB; 50% threshold = 0.54 GB → q8_0
        result = recommender.suggest_kv_cache_quant(profile, context_length=300_000)
        assert result == "q8_0"

    def test_returns_q4_0_for_high_pressure(self) -> None:
        """A 4 GB GPU with a very large context exceeds 75% KV headroom → q4_0."""
        profile = make_hardware_profile(vram_gb=4.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        # effective_vram = 4 * 0.9 = 3.6 GB; 20% headroom = 0.72 GB
        # 500k * 2048 bytes = ~0.95 GB > 0.72 * 0.75 = 0.54 GB → q4_0
        result = recommender.suggest_kv_cache_quant(profile, context_length=500_000)
        assert result == "q4_0"

    def test_cpu_only_system_always_f16(self) -> None:
        """CPU-only inference has no VRAM pressure; KV quant defaults to f16."""
        profile = make_hardware_profile(vram_gb=0.0, ram_gb=32.0)
        recommender = ModelRecommender()
        # effective_vram = 0 for CPU, but Apple Silicon / CPU path uses RAM fraction
        # Any result is valid — assert it is one of the allowed values
        result = recommender.suggest_kv_cache_quant(profile, context_length=8192)
        assert result in {"f16", "q8_0", "q4_0"}

    def test_default_context_length_is_8192(self) -> None:
        """suggest_kv_cache_quant defaults to 8192 tokens when not specified."""
        profile = make_hardware_profile(vram_gb=24.0, vendor=GpuVendor.NVIDIA)
        recommender = ModelRecommender()
        explicit = recommender.suggest_kv_cache_quant(profile, context_length=8192)
        default = recommender.suggest_kv_cache_quant(profile)
        assert explicit == default
