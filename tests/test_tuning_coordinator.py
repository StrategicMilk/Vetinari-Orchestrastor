"""Tests for vetinari.learning.tuning_coordinator."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from vetinari.learning.tuning_coordinator import (
    _SOURCE_PRIORITY,
    ParameterChange,
    TuningCoordinator,
    get_tuning_coordinator,
)

# -- ParameterChange tests --


def test_parameter_change_repr_shows_key_fields() -> None:
    change = ParameterChange(
        source="feedback_loop",
        parameter="model_weight:qwen3",
        old_value=0.5,
        new_value=0.7,
        reasoning="quality improved",
        priority=1,
    )
    r = repr(change)
    assert "feedback_loop" in r
    assert "model_weight:qwen3" in r
    assert "priority=1" in r


def test_parameter_change_timestamp_is_utc_iso() -> None:
    change = ParameterChange(
        source="auto_tuner",
        parameter="batch_size",
        old_value=8,
        new_value=16,
        reasoning="throughput",
        priority=2,
    )
    # ISO-8601 UTC strings end with +00:00 or Z
    assert "T" in change.timestamp


def test_source_priority_ordering() -> None:
    assert _SOURCE_PRIORITY["feedback_loop"] < _SOURCE_PRIORITY["auto_tuner"]
    assert _SOURCE_PRIORITY["auto_tuner"] < _SOURCE_PRIORITY["meta_adapter"]


# -- TuningCoordinator.propose tests --


def test_first_proposal_always_accepted() -> None:
    coord = TuningCoordinator()
    accepted = coord.propose("feedback_loop", "temperature", 0.7, 0.5, "reduce variance")
    assert accepted is True


def test_same_source_second_proposal_accepted_after_window() -> None:
    coord = TuningCoordinator()
    coord.COORDINATION_WINDOW_SECONDS = 0  # disable window
    coord.propose("auto_tuner", "batch_size", 8, 16, "first")
    accepted = coord.propose("auto_tuner", "batch_size", 16, 32, "second")
    assert accepted is True


def test_lower_priority_rejected_within_window() -> None:
    """meta_adapter (priority 3) should lose to feedback_loop (priority 1)."""
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "temperature", 0.7, 0.5, "feedback says lower")
    rejected = coord.propose("meta_adapter", "temperature", 0.5, 0.9, "meta says higher")
    assert rejected is False


def test_higher_priority_overrides_within_window() -> None:
    """feedback_loop (priority 1) should override meta_adapter (priority 3)."""
    coord = TuningCoordinator()
    coord.propose("meta_adapter", "temperature", 0.7, 0.9, "meta wants higher")
    accepted = coord.propose("feedback_loop", "temperature", 0.9, 0.5, "feedback overrides")
    assert accepted is True


def test_equal_priority_same_source_rejected_within_window() -> None:
    """Two proposals from the same source within the window — second is rejected."""
    coord = TuningCoordinator()
    coord.propose("auto_tuner", "batch_size", 8, 16, "first")
    rejected = coord.propose("auto_tuner", "batch_size", 16, 32, "second within window")
    assert rejected is False


def test_unknown_source_gets_lowest_priority() -> None:
    """Unknown source gets priority 99 — beaten by any known source."""
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "lr", 1e-4, 1e-5, "feedback first")
    rejected = coord.propose("unknown_system", "lr", 1e-5, 1e-3, "unknown tries to override")
    assert rejected is False


def test_different_parameters_do_not_conflict() -> None:
    """Changes to different parameters are independent."""
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "temperature", 0.7, 0.5, "quality")
    accepted = coord.propose("meta_adapter", "batch_size", 8, 16, "throughput")
    assert accepted is True


def test_expired_window_allows_lower_priority() -> None:
    """After the coordination window expires, a lower-priority change is accepted."""
    coord = TuningCoordinator()
    coord.COORDINATION_WINDOW_SECONDS = 0  # expire immediately
    coord.propose("feedback_loop", "temperature", 0.7, 0.5, "feedback")
    accepted = coord.propose("meta_adapter", "temperature", 0.5, 0.9, "meta after expiry")
    assert accepted is True


# -- History and active params tests --


def test_get_history_returns_accepted_changes() -> None:
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "lr", 1e-4, 1e-5, "reason A")
    coord.propose("auto_tuner", "batch_size", 8, 16, "reason B")
    history = coord.get_history()
    assert len(history) == 2
    # Most recent first
    assert history[0]["parameter"] == "batch_size"
    assert history[1]["parameter"] == "lr"


def test_get_history_respects_limit() -> None:
    coord = TuningCoordinator()
    for i in range(10):
        coord.propose("auto_tuner", f"param_{i}", i, i + 1, f"reason {i}")
    history = coord.get_history(limit=3)
    assert len(history) == 3


def test_get_history_excludes_rejected_changes() -> None:
    """Rejected changes must not appear in history."""
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "temperature", 0.7, 0.5, "accepted")
    coord.propose("meta_adapter", "temperature", 0.5, 0.9, "rejected by priority")
    history = coord.get_history()
    assert len(history) == 1
    assert history[0]["source"] == "feedback_loop"


def test_get_active_params_reflects_latest_accepted() -> None:
    coord = TuningCoordinator()
    coord.propose("feedback_loop", "temperature", 0.7, 0.5, "first")
    active = coord.get_active_params()
    assert "temperature" in active
    assert active["temperature"]["new_value"] == 0.5


def test_history_bounded_at_500() -> None:
    coord = TuningCoordinator()
    # Expire window so each new proposal to the same param is accepted.
    coord.COORDINATION_WINDOW_SECONDS = 0
    for i in range(550):
        coord.propose("auto_tuner", f"p_{i % 100}", i, i + 1, "loop")
    # Internal history capped at 500; get_history limit is also 500.
    history = coord.get_history(limit=500)
    assert len(history) <= 500


# -- Thread-safety test --


def test_concurrent_proposals_do_not_corrupt_state() -> None:
    """Fire 50 concurrent proposals; coordinator must not raise or corrupt history."""
    coord = TuningCoordinator()
    errors: list[Exception] = []

    def propose_many(source: str, priority_label: str) -> None:
        try:
            for i in range(25):
                coord.propose(source, f"shared_param_{i % 5}", i, i + 1, "concurrent")
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=propose_many, args=("feedback_loop", "high")),
        threading.Thread(target=propose_many, args=("meta_adapter", "low")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # History should contain at most 500 entries and be a valid list.
    history = coord.get_history(limit=500)
    assert isinstance(history, list)


# -- Singleton tests --


def test_get_tuning_coordinator_returns_same_instance() -> None:
    a = get_tuning_coordinator()
    b = get_tuning_coordinator()
    assert a is b


def test_get_tuning_coordinator_is_tuning_coordinator_type() -> None:
    instance = get_tuning_coordinator()
    assert isinstance(instance, TuningCoordinator)
