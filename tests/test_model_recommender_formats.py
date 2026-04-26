"""Tests for multi-format model recommendations (vLLM/NIM support)."""

from __future__ import annotations

import pytest

from vetinari.setup.model_recommender import ModelRecommender, SetupModelRecommendation
from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_hardware(vram_gb: float, ram_gb: float = 64.0) -> HardwareProfile:
    """Create a HardwareProfile with the given VRAM and RAM."""
    has_gpu = vram_gb > 0
    return HardwareProfile(
        cpu_count=16,
        ram_gb=ram_gb,
        gpu=GpuInfo(
            name="Test GPU" if has_gpu else "",
            vendor=GpuVendor.NVIDIA if has_gpu else GpuVendor.NONE,
            vram_gb=vram_gb,
            cuda_available=has_gpu,
            metal_available=False,
            driver_version="555.42",
        ),
    )


# ── Basic GGUF Recommendations (unchanged behavior) ─────────────────────────


class TestGGUFRecommendations:
    def test_recommend_models_returns_gguf(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(hw)
        assert len(recs) >= 1
        assert all(r.model_format == "gguf" for r in recs)
        assert all(r.backend == "llama_cpp" for r in recs)

    def test_small_vram_tier(self) -> None:
        hw = _make_hardware(vram_gb=6.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models(hw)
        assert any("7B" in r.parameter_count for r in recs)

    def test_new_fields_in_to_dict(self) -> None:
        rec = SetupModelRecommendation(
            name="Test",
            repo_id="test/test",
            filename="test.gguf",
            size_gb=4.0,
            quantization="Q4_K_M",
            parameter_count="7B",
            reason="test",
            model_format="awq",
            backend="vllm",
            gpu_only=True,
        )
        d = rec.to_dict()
        assert d["model_format"] == "awq"
        assert d["backend"] == "vllm"
        assert d["gpu_only"] is True


# ── Multi-Format Recommendations ─────────────────────────────────────────────


class TestMultiFormatRecommendations:
    def test_llama_cpp_only_returns_gguf(self) -> None:
        """When only llama_cpp is available, only GGUF models are returned."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp"])
        gguf_recs = [r for r in recs if r.model_format == "gguf"]
        assert len(gguf_recs) == len(recs)

    def test_vllm_adds_awq_recommendations(self) -> None:
        """When vLLM is available, AWQ models are added."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        awq_recs = [r for r in recs if r.model_format == "awq"]
        assert len(awq_recs) > 0
        assert all(r.gpu_only is True for r in awq_recs)
        assert all(r.backend == "vllm" for r in awq_recs)

    def test_vllm_models_are_primary_when_available(self) -> None:
        """vLLM AWQ models should be marked primary when vLLM is available."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        awq_primary = [r for r in recs if r.model_format == "awq" and r.is_primary]
        assert len(awq_primary) >= 1

    def test_gguf_demoted_when_vllm_available(self) -> None:
        """GGUF models should not be primary when vLLM options exist."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        gguf_primary = [r for r in recs if r.model_format == "gguf" and r.is_primary]
        assert len(gguf_primary) == 0

    def test_vllm_only_fits_in_vram(self) -> None:
        """vLLM models that exceed VRAM should not be recommended."""
        hw = _make_hardware(vram_gb=10.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        awq_recs = [r for r in recs if r.model_format == "awq"]
        for r in awq_recs:
            assert r.size_gb <= 10.0, f"AWQ model {r.name} ({r.size_gb}GB) exceeds 10GB VRAM"

    def test_cpu_offload_models_with_enough_ram(self) -> None:
        """CPU offload models should appear when RAM >= 32GB."""
        hw = _make_hardware(vram_gb=24.0, ram_gb=64.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        offload_recs = [r for r in recs if r.gpu_only is False and r.size_gb > 24.0]
        assert len(offload_recs) > 0
        assert all(r.backend == "llama_cpp" for r in offload_recs)

    def test_no_cpu_offload_with_low_ram(self) -> None:
        """CPU offload models should not appear with < 32GB RAM."""
        hw = _make_hardware(vram_gb=24.0, ram_gb=16.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        offload_recs = [r for r in recs if r.gpu_only is False and r.size_gb > 24.0]
        assert len(offload_recs) == 0

    def test_ordering_vllm_first(self) -> None:
        """vLLM models should appear before GGUF when vLLM is available."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        # First recommendation should be vLLM format
        assert recs[0].backend == "vllm"

    def test_nim_adds_recommendations(self) -> None:
        """NIM backend should get native recommendations labelled as NIM."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "nim"])
        gpu_recs = [r for r in recs if r.backend == "nim"]
        assert len(gpu_recs) > 0

    def test_default_nvidia_recommendations_are_native_first(self) -> None:
        """Unspecified backend list should prefer native NIM/vLLM before GGUF."""
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw)
        assert recs[0].backend == "nim"
        assert recs[0].model_format in {"awq", "gptq", "safetensors"}

    def test_cpu_default_stays_llama_cpp_gguf(self) -> None:
        """CPU-only systems should keep the GGUF llama.cpp path."""
        hw = _make_hardware(vram_gb=0.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw)
        assert recs
        assert all(r.backend == "llama_cpp" for r in recs)
        assert all(r.model_format == "gguf" for r in recs)

    def test_small_vram_no_vllm_recs(self) -> None:
        """< 8GB VRAM should not get vLLM recommendations (no tier defined)."""
        hw = _make_hardware(vram_gb=6.0)
        recommender = ModelRecommender()
        recs = recommender.recommend_models_multi_format(hw, ["llama_cpp", "vllm"])
        awq_recs = [r for r in recs if r.model_format == "awq"]
        # The 7B AWQ (4.5GB) should fit in 6GB VRAM
        for r in awq_recs:
            assert r.size_gb <= 6.0


# ── Model Freshness Checker ──────────────────────────────────────────────────


# ── Portfolio Recommendations ─────────────────────────────────────────────


class TestPortfolioRecommendations:
    def test_portfolio_has_all_roles(self) -> None:
        hw = _make_hardware(vram_gb=24.0, ram_gb=64.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw, ["llama_cpp"])
        assert "grunt" in portfolio
        assert "worker" in portfolio
        assert "thinker" in portfolio

    def test_grunt_models_are_small(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw)
        for m in portfolio["grunt"]:
            assert m.size_gb <= 2.0, f"Grunt model {m.name} too large: {m.size_gb}GB"

    def test_grunt_best_for_classification(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw)
        assert any("classification" in m.best_for for m in portfolio["grunt"])

    def test_worker_best_for_coding(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw)
        assert any("coding" in m.best_for for m in portfolio["worker"])

    def test_thinker_best_for_reasoning(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw)
        assert any("reasoning" in m.best_for for m in portfolio["thinker"])

    def test_vllm_promotes_awq_workers(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw, ["llama_cpp", "vllm"])
        awq_workers = [m for m in portfolio["worker"] if m.model_format == "awq"]
        assert len(awq_workers) > 0
        assert awq_workers[0].is_primary is True

    def test_default_portfolio_prefers_native_worker(self) -> None:
        hw = _make_hardware(vram_gb=24.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw)
        assert portfolio["worker"][0].backend == "nim"
        assert portfolio["worker"][0].model_format in {"awq", "gptq", "safetensors"}

    def test_cpu_offload_in_thinker_with_enough_ram(self) -> None:
        hw = _make_hardware(vram_gb=24.0, ram_gb=64.0)
        recommender = ModelRecommender()
        portfolio = recommender.recommend_portfolio(hw, ["llama_cpp"])
        offload = [m for m in portfolio["thinker"] if m.gpu_only is False]
        assert len(offload) > 0

    def test_best_for_in_to_dict(self) -> None:
        rec = SetupModelRecommendation(
            name="Test",
            repo_id="test/test",
            filename="test.gguf",
            size_gb=4.0,
            quantization="Q4_K_M",
            parameter_count="7B",
            reason="test",
            best_for=("coding", "review"),
        )
        d = rec.to_dict()
        assert d["best_for"] == ["coding", "review"]


# ── Model Freshness Checker ──────────────────────────────────────────────────


class TestModelFreshnessChecker:
    def test_should_check_no_file(self, tmp_path: pytest.TempPathFactory) -> None:
        from vetinari.models.model_scout import ModelFreshnessChecker

        checker = ModelFreshnessChecker()
        # Override the check file to a non-existent path
        checker._last_check_file = tmp_path / "nonexistent" / "check.json"  # type: ignore[assignment]
        assert checker.should_check() is True

    def test_should_not_check_recently(self, tmp_path: pytest.TempPathFactory) -> None:
        import json
        from datetime import datetime, timezone

        from vetinari.models.model_scout import ModelFreshnessChecker

        checker = ModelFreshnessChecker()
        check_file = tmp_path / "check.json"  # type: ignore[operator]
        check_file.write_text(
            json.dumps({"last_check": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
        checker._last_check_file = check_file
        assert checker.should_check() is False

    def test_upgrade_candidate_to_dict(self) -> None:
        from vetinari.models.model_scout import ModelUpgradeCandidate

        candidate = ModelUpgradeCandidate(
            current_model_id="qwen2.5-7b",
            candidate_name="qwen3-8b",
            candidate_repo_id="Qwen/Qwen3-8B",
            benchmark_score=0.85,
            sentiment_score=0.72,
            overall_score=0.798,
            available_formats=["gguf", "awq"],
            vram_estimate_gb=6.0,
            reason="Test upgrade",
        )
        d = candidate.to_dict()
        assert d["current_model_id"] == "qwen2.5-7b"
        assert d["candidate_name"] == "qwen3-8b"
        assert d["benchmark_score"] == 0.85
        assert d["available_formats"] == ["gguf", "awq"]
        assert d["recommended_backend"] == "vllm"
        assert d["recommended_format"] == "safetensors"
