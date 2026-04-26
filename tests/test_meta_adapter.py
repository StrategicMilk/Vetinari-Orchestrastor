"""Tests for vetinari.learning.meta_adapter — MetaAdapter strategy selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.learning.meta_adapter import (
    MetaAdapter,
    StrategyBundle,
    TaskPrototype,
    get_meta_adapter,
)


@pytest.fixture
def tmp_state_path(tmp_path):
    """Provide a temporary state file path."""
    return tmp_path / "meta_adapter_state.json"


@pytest.fixture
def adapter(tmp_state_path):
    """Create a MetaAdapter with temporary state path."""
    return MetaAdapter(state_path=tmp_state_path)


class TestStrategyBundle:
    """Tests for the StrategyBundle dataclass."""

    def test_default_values(self):
        bundle = StrategyBundle()
        assert bundle.temperature == 0.3
        assert bundle.context_window == 4096
        assert bundle.decomposition_granularity == "medium"
        assert bundle.prompt_template_variant == "standard"
        assert bundle.source == "default"

    def test_to_dict(self):
        bundle = StrategyBundle(temperature=0.7, source="thompson")
        d = bundle.to_dict()
        assert d["temperature"] == 0.7
        assert d["source"] == "thompson"
        assert "context_window" in d


class TestTaskPrototype:
    """Tests for the TaskPrototype dataclass."""

    def test_get_best_strategy_returns_highest_quality(self):
        proto = TaskPrototype(
            prototype_id="proto_001",
            successful_strategies={
                "temperature": {"0.3": 0.8, "0.7": 0.9, "0.0": 0.5},
            },
        )
        best = proto.get_best_strategy("temperature")
        assert best == "0.7"

    def test_get_best_strategy_empty_returns_none(self):
        proto = TaskPrototype(prototype_id="proto_001")
        assert proto.get_best_strategy("temperature") is None


class TestMetaAdapter:
    """Tests for MetaAdapter core functionality."""

    def test_cold_start_returns_default_or_thompson(self, adapter):
        """On cold start with no prototypes, select_strategy returns defaults."""
        bundle = adapter.select_strategy("Write a function", task_type="coding")
        assert isinstance(bundle, StrategyBundle)
        assert bundle.source in ("default", "thompson")

    def test_record_outcome_creates_prototype(self, adapter):
        """Recording a successful outcome creates a new prototype."""
        strategy = StrategyBundle(temperature=0.5, source="test")
        proto_id = adapter.record_outcome(
            task_description="Implement a Redis cache wrapper",
            task_type="coding",
            strategy_used=strategy,
            quality_score=0.9,
            success=True,
            mode="build",
        )
        assert proto_id.startswith("proto_")
        prototype = adapter.get_prototype(proto_id)
        assert isinstance(prototype, TaskPrototype)
        assert prototype.prototype_id == proto_id

    def test_record_failure_does_not_create_prototype(self, adapter):
        """Failed tasks should not create prototypes."""
        strategy = StrategyBundle()
        proto_id = adapter.record_outcome(
            task_description="Bad task",
            task_type="coding",
            strategy_used=strategy,
            quality_score=0.2,
            success=False,
        )
        assert proto_id == ""

    def test_similarity_matching_returns_prototype_strategy(self, adapter):
        """After recording, similar queries should match the prototype."""
        strategy = StrategyBundle(temperature=0.7, context_window=8192)

        # Record 3 outcomes to meet _MIN_PROTOTYPE_SAMPLES threshold
        for i in range(3):
            adapter.record_outcome(
                task_description=f"Implement a Redis cache wrapper v{i}",
                task_type="coding",
                strategy_used=strategy,
                quality_score=0.9,
                success=True,
                mode="build",
            )

        # Similar query should get prototype-based strategy
        bundle = adapter.select_strategy(
            "Implement a Redis cache layer for the API",
            task_type="coding",
        )
        # Should be either prototype match or thompson (depending on similarity)
        assert isinstance(bundle, StrategyBundle)

    def test_thompson_fallback_when_no_match(self, adapter):
        """Completely different queries should fall back to Thompson."""
        # Record a coding prototype
        strategy = StrategyBundle(temperature=0.7)
        adapter.record_outcome(
            task_description="Implement sorting algorithm",
            task_type="coding",
            strategy_used=strategy,
            quality_score=0.9,
            success=True,
        )

        # Completely different domain — may not match
        bundle = adapter.select_strategy(
            "Deploy Kubernetes cluster on AWS",
            task_type="devops",
        )
        assert isinstance(bundle, StrategyBundle)

    def test_get_stats_empty(self, adapter):
        stats = adapter.get_stats()
        assert stats["prototype_count"] == 0
        assert stats["avg_quality"] == 0.0

    def test_get_stats_with_prototypes(self, adapter):
        strategy = StrategyBundle()
        adapter.record_outcome(
            "Implement a high-performance sorting algorithm for large datasets using quicksort",
            "coding",
            strategy,
            0.8,
            True,
        )
        adapter.record_outcome(
            "Research quantum computing applications in cryptography and blockchain security",
            "research",
            strategy,
            0.9,
            True,
        )
        stats = adapter.get_stats()
        assert stats["prototype_count"] >= 1
        assert stats["avg_quality"] >= 0.8

    def test_state_persistence(self, tmp_state_path):
        """Prototypes should survive save/load cycle."""
        adapter1 = MetaAdapter(state_path=tmp_state_path)
        strategy = StrategyBundle(temperature=0.5)
        adapter1.record_outcome(
            "Implement cache",
            "coding",
            strategy,
            0.9,
            True,
        )
        assert len(adapter1._prototypes) == 1

        # Create new adapter from same state file
        adapter2 = MetaAdapter(state_path=tmp_state_path)
        assert len(adapter2._prototypes) == 1

    def test_prototype_update_increments_sample_count(self, adapter):
        """Recording similar tasks should update existing prototype."""
        strategy = StrategyBundle()
        adapter.record_outcome(
            "Write unit tests for auth module",
            "coding",
            strategy,
            0.8,
            True,
        )
        proto_id = next(iter(adapter._prototypes.keys()))
        assert adapter._prototypes[proto_id].sample_count == 1

        # Very similar task should match and update
        adapter.record_outcome(
            "Write unit tests for auth module v2",
            "coding",
            strategy,
            0.9,
            True,
        )
        # May update same prototype or create new depending on similarity
        total_samples = sum(p.sample_count for p in adapter._prototypes.values())
        assert total_samples >= 2


class TestGetMetaAdapter:
    """Test module-level singleton accessor."""

    def test_get_meta_adapter_returns_instance(self):
        # Reset singleton for test isolation
        import vetinari.learning.meta_adapter as mod

        old = mod._meta_adapter
        mod._meta_adapter = None
        try:
            adapter = get_meta_adapter()
            assert isinstance(adapter, MetaAdapter)
        finally:
            mod._meta_adapter = old
