"""Tests for vetinari.models.model_pool (ModelPool).

Covers:
- B.6: Unknown model memory default uses max(2, budget * 0.25) not hardcoded 2
- B.8: Discovery race condition — self.models is never an empty partial list
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from vetinari.models.model_pool import ModelPool


class TestModelPoolMemoryDefault:
    """B.6: Unknown model size should use budget-proportional default."""

    def _make_pool(self, budget_gb: int = 32) -> ModelPool:
        from vetinari.models.model_pool import ModelPool

        return ModelPool(config={}, memory_budget_gb=budget_gb)

    def test_unknown_model_uses_budget_proportion(self):
        """Unknown model with 32 GB budget gets 8 GB default (32 * 0.25)."""
        pool = self._make_pool(budget_gb=32)

        fake_model = MagicMock()
        fake_model.id = "big-model"
        fake_model.name = "Big Model"
        fake_model.endpoint = ""
        fake_model.capabilities = []
        fake_model.context_len = 4096
        fake_model.memory_gb = 0  # unknown size

        with (
            patch("vetinari.models.model_pool.ModelPool._discover_via_llama_swap", return_value=None),
            patch(
                "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter.discover_models",
                return_value=[fake_model],
            ),
            patch("vetinari.models.model_pool._seed_discovered_models"),
        ):
            pool.discover_models()

        assert len(pool.models) == 1
        # With 32 GB budget: max(2, 32 * 0.25) = max(2, 8.0) = 8.0
        assert pool.models[0]["memory_gb"] == pytest.approx(8.0), (
            f"Unknown model with 32 GB budget should default to 8 GB, got {pool.models[0]['memory_gb']}"
        )

    def test_unknown_model_minimum_2gb(self):
        """Unknown model with 4 GB budget gets at least 2 GB (not 1 GB = 4 * 0.25)."""
        pool = self._make_pool(budget_gb=4)

        fake_model = MagicMock()
        fake_model.id = "small-unknown"
        fake_model.name = "Small Unknown"
        fake_model.endpoint = ""
        fake_model.capabilities = []
        fake_model.context_len = 2048
        fake_model.memory_gb = 0  # unknown size

        with (
            patch("vetinari.models.model_pool.ModelPool._discover_via_llama_swap", return_value=None),
            patch(
                "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter.discover_models",
                return_value=[fake_model],
            ),
            patch("vetinari.models.model_pool._seed_discovered_models"),
        ):
            pool.discover_models()

        assert len(pool.models) == 1
        # With 4 GB budget: max(2, 4 * 0.25) = max(2, 1.0) = 2.0 — minimum wins
        assert pool.models[0]["memory_gb"] >= 2.0, (
            f"Minimum memory default must be 2 GB, got {pool.models[0]['memory_gb']}"
        )


class TestModelPoolDiscoveryRace:
    """B.8: Concurrent reads during discovery must never return empty list."""

    def test_models_never_empty_during_discovery(self):
        """self.models must never be [] while discovery is in progress.

        We seed the pool with one model (last-known-good), then trigger a slow
        discovery that sleeps mid-loop.  A concurrent reader must see the old
        models, never an empty list.
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)
        # Seed an existing model list so there's a last-known-good state
        pool.models = [{"id": "seed-model", "name": "Seed", "memory_gb": 4}]
        pool._last_known_good = list(pool.models)

        seen_empty = []

        def _slow_discover(*_args, **_kwargs):
            """Discovery that takes time — concurrent reads happen here."""
            import time

            time.sleep(0.05)
            return []  # returns nothing, triggers fallback

        def _reader():
            """Reads self.models repeatedly while discovery runs."""
            import time

            for _ in range(20):
                snapshot = list(pool.models)
                if len(snapshot) == 0:
                    seen_empty.append(True)
                time.sleep(0.005)

        reader_thread = threading.Thread(target=_reader)

        with (
            patch("vetinari.models.model_pool.ModelPool._discover_via_llama_swap", return_value=None),
            patch(
                "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter.discover_models",
                side_effect=_slow_discover,
            ),
            patch("vetinari.models.model_pool._seed_discovered_models"),
        ):
            reader_thread.start()
            pool.discover_models()
            reader_thread.join()

        assert not seen_empty, "Concurrent readers saw an empty model list during discovery — race condition!"


class TestModelPoolDiscoverMergeById:
    """33G.3 Defect 1: discover_models() merge must use canonical id, not display name."""

    def test_same_name_different_id_both_survive_merge(self):
        """Two models with the same display name but different ids must both appear in
        the final model list after discover_models().

        The old code deduped by ``name``; the fixed code dedupes by ``id``.
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)

        # The static config contributes one model
        pool.config = {
            "models": [
                {
                    "id": "org/llama-8b-v2",  # different id — must survive
                    "name": "Llama 8B",  # same display name as discovered model
                    "endpoint": "http://localhost:8080",
                    "capabilities": ["general"],
                    "context_len": 4096,
                    "memory_gb": 8.0,
                }
            ]
        }

        # Discovery returns a model with the same name but a different id
        discovered_model = MagicMock()
        discovered_model.id = "org/llama-8b-v1"  # different id
        discovered_model.name = "Llama 8B"  # same display name
        discovered_model.endpoint = "http://localhost:8081"
        discovered_model.capabilities = ["general"]
        discovered_model.context_len = 4096
        discovered_model.memory_gb = 8.0

        with (
            patch("vetinari.models.model_pool.ModelPool._discover_via_llama_swap", return_value=None),
            patch(
                "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter.discover_models",
                return_value=[discovered_model],
            ),
            patch("vetinari.models.model_pool._seed_discovered_models"),
        ):
            pool.discover_models()

        model_ids = [m["id"] for m in pool.models]
        assert "org/llama-8b-v1" in model_ids, "Discovered model missing after merge"
        assert "org/llama-8b-v2" in model_ids, "Static config model missing after merge"
        assert len(pool.models) == 2, (
            f"Expected 2 distinct models (merged by id), got {len(pool.models)}: {model_ids}"
        )

    def test_same_id_not_duplicated_by_merge(self):
        """A model present in both discovery and static config (same id) must appear
        exactly once in the merged list.
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)

        # Static config has a model
        pool.config = {
            "models": [
                {
                    "id": "shared-model-id",
                    "name": "Shared Model",
                    "endpoint": "http://localhost:8080",
                    "capabilities": ["general"],
                    "context_len": 4096,
                    "memory_gb": 8.0,
                }
            ]
        }

        # Discovery returns the same model id
        discovered_model = MagicMock()
        discovered_model.id = "shared-model-id"  # same id as static config
        discovered_model.name = "Shared Model (discovered)"
        discovered_model.endpoint = "http://localhost:8081"
        discovered_model.capabilities = ["general"]
        discovered_model.context_len = 4096
        discovered_model.memory_gb = 8.0

        with (
            patch("vetinari.models.model_pool.ModelPool._discover_via_llama_swap", return_value=None),
            patch(
                "vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter.discover_models",
                return_value=[discovered_model],
            ),
            patch("vetinari.models.model_pool._seed_discovered_models"),
        ):
            pool.discover_models()

        model_ids = [m["id"] for m in pool.models]
        assert model_ids.count("shared-model-id") == 1, (
            f"Same-id model must not be duplicated, found {model_ids.count('shared-model-id')} copies"
        )


class TestModelPoolTaskScopedReliability:
    """33G.3 Defect 2a: reliability cache key must be scoped to task type, not hardwired ':general'."""

    def test_different_task_types_produce_different_cache_keys(self):
        """Scoring a coding task and a reasoning task must query the router with
        different cache keys so per-task performance data is used correctly.
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)

        captured_keys: list[str] = []

        mock_router = MagicMock()

        def _mock_get_performance_cache(key: str):
            captured_keys.append(key)
            return None  # No performance data — causes fallback to SLA

        mock_router.get_performance_cache = _mock_get_performance_cache

        coding_task = {"type": "coding", "inputs": [], "id": "t1", "description": "write code"}
        reasoning_task = {"type": "reasoning", "inputs": [], "id": "t2", "description": "reason"}
        model = {"id": "test-model", "capabilities": [], "context_len": 4096, "memory_gb": 8}

        with (
            patch(
                "vetinari.models.dynamic_model_router.get_model_router",
                return_value=mock_router,
            ),
            patch("vetinari.analytics.sla.get_sla_tracker"),
        ):
            pool._score_task_model(coding_task, model)
            pool._score_task_model(reasoning_task, model)

        assert "test-model:coding" in captured_keys, (
            f"Expected cache key 'test-model:coding', got: {captured_keys}"
        )
        assert "test-model:reasoning" in captured_keys, (
            f"Expected cache key 'test-model:reasoning', got: {captured_keys}"
        )
        # Keys must differ — not both ':general'
        assert captured_keys[0] != captured_keys[1], (
            "Both task types produced the same cache key — reliability is not task-scoped"
        )


class TestModelPoolExactCostMatch:
    """33G.3 Defect 2b: cost lookup must use exact id match, not substring match."""

    def test_substring_id_does_not_steal_cost_of_longer_id(self):
        """Model 'llama' must not receive the cost of model 'llama-70b' just because
        'llama' is a substring of 'llama-70b'.
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)

        mock_report = MagicMock()
        mock_report.total_requests = 10
        mock_report.by_model = {
            "llama-70b": 0.80,  # expensive
            "llama": 0.10,  # cheap — must not be confused with 'llama-70b'
        }

        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_tracker):
            score_llama = pool._get_cost_efficiency("llama")
            score_llama_70b = pool._get_cost_efficiency("llama-70b")

        # 'llama' has cost 0.10, max_cost = 0.80, so efficiency = 1 - (0.10/0.80) ≈ 0.875
        # 'llama-70b' has cost 0.80, efficiency = 0 (most expensive)
        assert score_llama > score_llama_70b, (
            f"'llama' (cheaper) should score higher than 'llama-70b' (more expensive). "
            f"Got llama={score_llama:.3f}, llama-70b={score_llama_70b:.3f}"
        )
        assert score_llama_70b == pytest.approx(0.0), (
            f"Most expensive model should have efficiency 0.0, got {score_llama_70b:.3f}"
        )

    def test_no_cost_data_returns_free_assumption(self):
        """When the cost report has no data, efficiency score should be 1.0
        (assume free / local model).
        """
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)

        mock_report = MagicMock()
        mock_report.total_requests = 0
        mock_report.by_model = {}

        mock_tracker = MagicMock()
        mock_tracker.get_report.return_value = mock_report

        with patch("vetinari.analytics.cost.get_cost_tracker", return_value=mock_tracker):
            score = pool._get_cost_efficiency("any-model")

        assert score == pytest.approx(1.0), (
            f"No cost data should assume free (1.0), got {score}"
        )


class TestModelPoolWarmup:
    """Session 15 warm-up should cover every discovered model, not just the first."""

    def test_warm_up_primary_schedules_all_models(self):
        """Each discovered model should get a best-effort warm-up thread."""
        from vetinari.models.model_pool import ModelPool

        pool = ModelPool(config={}, memory_budget_gb=32)
        pool.models = [
            {"id": "model-a", "name": "Model A"},
            {"id": "model-b", "name": "Model B"},
        ]

        started: list[tuple] = []

        class _InlineThread:
            def __init__(self, target=None, args=(), **kwargs):
                self._target = target
                self._args = args
                self.kwargs = kwargs

            def start(self):
                started.append(self._args)
                self._target(*self._args)

        adapter = MagicMock()

        with (
            patch("threading.Thread", _InlineThread),
            patch("vetinari.adapters.base.ProviderConfig"),
            patch("vetinari.adapters.base.ProviderType"),
            patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", return_value=adapter),
        ):
            pool.warm_up_primary()

        assert started == [("model-a",), ("model-b",)]
        assert adapter.load_model.call_args_list[0].args == ("model-a",)
        assert adapter.load_model.call_args_list[1].args == ("model-b",)
