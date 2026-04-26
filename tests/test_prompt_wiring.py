"""Tests for vetinari.evaluation.prompt_wiring — prompt cache and token optimizer wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.evaluation.prompt_wiring import (
    cached_prompt_lookup,
    get_prompt_cache_stats,
    invalidate_prompt_cache,
    optimize_prompt_for_budget,
    wire_prompt_optimization,
)


class TestCachedPromptLookup:
    """Verify prompt cache lookup integration."""

    def test_cache_hit_returns_savings(self) -> None:
        mock_cache = MagicMock()
        mock_cache.get_or_cache.return_value = MagicMock(
            hit=True,
            prompt="cached prompt",
            savings_tokens=150,
        )
        with (
            patch("vetinari.optimization.prompt_cache.get_prompt_cache", return_value=mock_cache),
            patch("vetinari.optimization.prompt_cache.hash_prompt", return_value="abc123"),
        ):
            hit, prompt, savings = cached_prompt_lookup("test prompt")

        assert hit is True
        assert prompt == "cached prompt"
        assert savings == 150

    def test_graceful_degradation_when_cache_unavailable(self) -> None:
        """When PromptCache is not importable, returns the original prompt."""
        with patch.dict("sys.modules", {"vetinari.optimization.prompt_cache": None}):
            hit, prompt, savings = cached_prompt_lookup("test prompt")

        assert hit is False
        assert prompt == "test prompt"
        assert savings == 0

    def test_returns_tuple_of_correct_types(self) -> None:
        result = cached_prompt_lookup("hello world")
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        assert isinstance(result[2], int)


class TestInvalidatePromptCache:
    """Verify prompt cache invalidation."""

    def test_invalidation_does_not_raise(self) -> None:
        """Invalidation returns None and never raises, even if cache is unavailable."""
        result = invalidate_prompt_cache("some prompt")
        assert result is None

    def test_graceful_when_unavailable(self) -> None:
        with patch.dict("sys.modules", {"vetinari.optimization.prompt_cache": None}):
            result = invalidate_prompt_cache("some prompt")
        assert result is None


class TestGetPromptCacheStats:
    """Verify prompt cache stats retrieval."""

    def test_returns_dict(self) -> None:
        result = get_prompt_cache_stats()
        assert isinstance(result, dict)

    def test_graceful_when_unavailable(self) -> None:
        with patch.dict("sys.modules", {"vetinari.optimization.prompt_cache": None}):
            result = get_prompt_cache_stats()
        assert result == {}


class TestOptimizePromptForBudget:
    """Verify token optimizer wiring."""

    def test_returns_dict_with_prompt_key(self) -> None:
        result = optimize_prompt_for_budget("test prompt", task_type="general")
        assert isinstance(result, dict)
        assert "prompt" in result

    def test_fallback_when_optimizer_unavailable(self) -> None:
        with patch.dict("sys.modules", {"vetinari.token_optimizer": None}):
            result = optimize_prompt_for_budget("raw prompt", context="ctx")
        assert result["prompt"] == "raw prompt"
        assert result["context"] == "ctx"
        assert result["compressed"] is False
        assert "estimated_tokens" in result

    def test_passes_task_type_through(self) -> None:
        result = optimize_prompt_for_budget(
            "prompt",
            task_type="coding",
            task_description="Build a widget",
        )
        assert isinstance(result, dict)

    def test_fallback_estimated_tokens_is_reasonable(self) -> None:
        prompt = "a" * 400
        with patch.dict("sys.modules", {"vetinari.token_optimizer": None}):
            result = optimize_prompt_for_budget(prompt)
        assert result["estimated_tokens"] == 100  # 400 chars / 4


class TestWirePromptOptimization:
    """Verify the startup wiring function."""

    def test_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="vetinari.evaluation.prompt_wiring"):
            wire_prompt_optimization()
        assert "Prompt optimization wiring" in caplog.text

    def test_logs_status(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO", logger="vetinari.evaluation.prompt_wiring"):
            wire_prompt_optimization()
        assert "Prompt optimization wiring" in caplog.text


class TestPackageExports:
    """Verify all prompt wiring functions are importable from evaluation package."""

    def test_evaluation_package_exports(self) -> None:
        from vetinari.evaluation import (
            cached_prompt_lookup,
            get_prompt_cache_stats,
            invalidate_prompt_cache,
            optimize_prompt_for_budget,
            wire_prompt_optimization,
        )

        assert callable(cached_prompt_lookup)
        assert callable(get_prompt_cache_stats)
        assert callable(invalidate_prompt_cache)
        assert callable(optimize_prompt_for_budget)
        assert callable(wire_prompt_optimization)
