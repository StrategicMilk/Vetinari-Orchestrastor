"""Tests for vetinari.awareness.context_graph — four-quadrant awareness model."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from vetinari.awareness.context_graph import (
    ContextEntry,
    ContextGraph,
    ContextSnapshot,
    get_context_graph,
    reset_context_graph,
)
from vetinari.types import ContextQuadrant

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the context graph singleton between tests."""
    reset_context_graph()
    yield
    reset_context_graph()


@pytest.fixture
def graph() -> ContextGraph:
    return ContextGraph()


# -- ContextEntry -------------------------------------------------------------


class TestContextEntry:
    def test_repr_shows_key_and_value(self) -> None:
        entry = ContextEntry(key="vram", value=8192, confidence=0.9)
        assert "vram" in repr(entry)
        assert "8192" in repr(entry)
        assert "0.90" in repr(entry)


# -- ContextSnapshot ----------------------------------------------------------


class TestContextSnapshot:
    def test_get_returns_value_from_quadrant(self) -> None:
        snap = ContextSnapshot(
            self_context={"loaded_models": ContextEntry(key="loaded_models", value=3)},
        )
        assert snap.get(ContextQuadrant.SELF, "loaded_models") == 3

    def test_get_returns_none_for_missing_key(self) -> None:
        snap = ContextSnapshot()
        assert snap.get(ContextQuadrant.SELF, "nonexistent") is None

    def test_get_returns_none_for_empty_quadrant(self) -> None:
        snap = ContextSnapshot()
        assert snap.get(ContextQuadrant.USER, "any_key") is None

    def test_repr_shows_counts(self) -> None:
        snap = ContextSnapshot(
            self_context={"a": ContextEntry(key="a", value=1)},
            user={"b": ContextEntry(key="b", value=2), "c": ContextEntry(key="c", value=3)},
        )
        r = repr(snap)
        assert "self=1" in r
        assert "user=2" in r
        assert "env=0" in r


# -- ContextGraph -------------------------------------------------------------


class TestContextGraphUpdateSelf:
    def test_update_self_stores_entry(self, graph: ContextGraph) -> None:
        graph.update_self("loaded_models", 3, source="model_pool")
        snap = graph.get_context()
        assert snap.get(ContextQuadrant.SELF, "loaded_models") == 3

    def test_update_self_overwrites_previous(self, graph: ContextGraph) -> None:
        graph.update_self("vram_free", 4096)
        graph.update_self("vram_free", 2048)
        snap = graph.get_context()
        assert snap.get(ContextQuadrant.SELF, "vram_free") == 2048


class TestContextGraphUpdateEnvironment:
    def test_update_environment_stores_entry(self, graph: ContextGraph) -> None:
        graph.update_environment("gpu_model", "RTX 4090", source="hwinfo")
        snap = graph.get_context()
        assert snap.get(ContextQuadrant.ENVIRONMENT, "gpu_model") == "RTX 4090"


class TestContextGraphRecordUserSignal:
    def test_record_user_signal_stores_entry(self, graph: ContextGraph) -> None:
        graph.record_user_signal("prefers_verbose", True, source="implicit_feedback")
        snap = graph.get_context()
        assert snap.get(ContextQuadrant.USER, "prefers_verbose") is True


class TestContextGraphRecordRelationship:
    def test_record_relationship_stores_entry(self, graph: ContextGraph) -> None:
        graph.record_relationship(
            "quality_drop_after_switch",
            {"from": "model_a", "to": "model_b", "drop": 0.15},
        )
        snap = graph.get_context()
        result = snap.get(ContextQuadrant.RELATIONSHIPS, "quality_drop_after_switch")
        assert result["drop"] == 0.15


class TestContextGraphGetContext:
    def test_returns_all_quadrants_by_default(self, graph: ContextGraph) -> None:
        graph.update_self("a", 1)
        graph.update_environment("b", 2)
        graph.record_user_signal("c", 3)
        graph.record_relationship("d", 4)
        snap = graph.get_context()
        assert snap.get(ContextQuadrant.SELF, "a") == 1
        assert snap.get(ContextQuadrant.ENVIRONMENT, "b") == 2
        assert snap.get(ContextQuadrant.USER, "c") == 3
        assert snap.get(ContextQuadrant.RELATIONSHIPS, "d") == 4

    def test_filters_to_requested_quadrants(self, graph: ContextGraph) -> None:
        graph.update_self("a", 1)
        graph.update_environment("b", 2)
        snap = graph.get_context(quadrants=[ContextQuadrant.SELF])
        assert snap.get(ContextQuadrant.SELF, "a") == 1
        assert snap.get(ContextQuadrant.ENVIRONMENT, "b") is None

    def test_snapshot_is_isolated_from_mutations(self, graph: ContextGraph) -> None:
        graph.update_self("x", 10)
        snap = graph.get_context()
        graph.update_self("x", 999)
        assert snap.get(ContextQuadrant.SELF, "x") == 10


class TestContextGraphConfidenceClamping:
    def test_confidence_clamped_to_0_1(self, graph: ContextGraph) -> None:
        graph.update_self("over", 1, confidence=5.0)
        graph.update_self("under", 1, confidence=-2.0)
        snap = graph.get_context()
        assert snap.self_context["over"].confidence == 1.0
        assert snap.self_context["under"].confidence == 0.0


class TestContextGraphStaleEntries:
    def test_fresh_entries_not_stale(self, graph: ContextGraph) -> None:
        graph.update_self("fresh", 1)
        stale = graph.get_stale_entries()
        assert len(stale) == 0

    def test_old_entries_flagged_stale(self, graph: ContextGraph) -> None:
        graph.update_self("old_entry", 1)
        # Manually backdate the entry
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        original = graph._quadrants[ContextQuadrant.SELF]["old_entry"]
        graph._quadrants[ContextQuadrant.SELF]["old_entry"] = replace(original, updated_at=old_time.isoformat())
        stale = graph.get_stale_entries()
        assert len(stale) == 1
        assert stale[0][1] == "old_entry"


# -- Singleton ----------------------------------------------------------------


class TestSingleton:
    def test_get_context_graph_returns_same_instance(self) -> None:
        g1 = get_context_graph()
        g2 = get_context_graph()
        assert g1 is g2

    def test_reset_creates_new_instance(self) -> None:
        g1 = get_context_graph()
        reset_context_graph()
        g2 = get_context_graph()
        assert g1 is not g2
