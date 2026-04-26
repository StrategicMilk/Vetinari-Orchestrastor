"""Tests for vetinari.learning.implicit_feedback — user behavior tracking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.learning.implicit_feedback import (
    FeedbackSignal,
    FeedbackSummary,
    ImplicitFeedbackCollector,
    get_implicit_feedback_collector,
    reset_implicit_feedback_collector,
)
from vetinari.types import FeedbackAction

TEST_MODEL_ID = "test-model-7b-q4"
TEST_TASK_TYPE = "code"


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the feedback collector singleton between tests."""
    reset_implicit_feedback_collector()
    yield
    reset_implicit_feedback_collector()


@pytest.fixture
def collector() -> ImplicitFeedbackCollector:
    return ImplicitFeedbackCollector()


# -- FeedbackSignal -----------------------------------------------------------


class TestFeedbackSignal:
    def test_repr_shows_key_fields(self) -> None:
        s = FeedbackSignal(
            task_id="task-1",
            model_id=TEST_MODEL_ID,
            action=FeedbackAction.EDITED,
        )
        assert "task-1" in repr(s)
        assert "edited" in repr(s)

    def test_defaults_assigned(self) -> None:
        s = FeedbackSignal()
        assert s.signal_id  # Non-empty
        assert s.action == FeedbackAction.ACCEPTED
        assert s.edit_diff is None
        assert s.metadata == {}


# -- FeedbackSummary -----------------------------------------------------------


class TestFeedbackSummary:
    def test_repr_shows_counts(self) -> None:
        s = FeedbackSummary(
            model_id=TEST_MODEL_ID,
            task_type=TEST_TASK_TYPE,
            accept_count=8,
            edit_count=2,
            acceptance_rate=0.8,
        )
        assert "80%" in repr(s)
        assert TEST_MODEL_ID in repr(s)


# -- record --------------------------------------------------------------------


class TestRecord:
    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_record_returns_signal(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        signal = collector.record(
            task_id="task-1",
            model_id=TEST_MODEL_ID,
            task_type=TEST_TASK_TYPE,
            action=FeedbackAction.ACCEPTED,
        )
        assert isinstance(signal, FeedbackSignal)
        assert signal.task_id == "task-1"
        assert signal.action == FeedbackAction.ACCEPTED

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_record_with_edit_diff(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        signal = collector.record(
            task_id="task-2",
            model_id=TEST_MODEL_ID,
            task_type=TEST_TASK_TYPE,
            action=FeedbackAction.EDITED,
            edit_diff="- old line\n+ new line",
        )
        assert signal.edit_diff == "- old line\n+ new line"

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_record_increments_stats(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        collector.record(task_id="t1", model_id=TEST_MODEL_ID, task_type=TEST_TASK_TYPE, action=FeedbackAction.ACCEPTED)
        collector.record(task_id="t2", model_id=TEST_MODEL_ID, task_type=TEST_TASK_TYPE, action=FeedbackAction.ACCEPTED)
        collector.record(task_id="t3", model_id=TEST_MODEL_ID, task_type=TEST_TASK_TYPE, action=FeedbackAction.EDITED)
        summary = collector.get_summary(TEST_MODEL_ID, TEST_TASK_TYPE)
        assert summary.accept_count == 2
        assert summary.edit_count == 1

    @patch("vetinari.learning.implicit_feedback._get_context_graph")
    def test_record_updates_context_graph(self, mock_cg_fn: MagicMock, collector: ImplicitFeedbackCollector) -> None:
        mock_graph = MagicMock()
        mock_cg_fn.return_value = mock_graph
        signal = collector.record(
            task_id="t1",
            model_id=TEST_MODEL_ID,
            task_type=TEST_TASK_TYPE,
            action=FeedbackAction.ACCEPTED,
        )
        mock_graph.record_user_signal.assert_called()
        assert signal.task_id == "t1"
        assert signal.model_id == TEST_MODEL_ID
        kwargs = mock_graph.record_user_signal.call_args.kwargs
        assert kwargs["key"] == f"acceptance_rate_{TEST_MODEL_ID}_{TEST_TASK_TYPE}"
        assert kwargs["value"] == 1.0
        assert kwargs["source"] == "implicit_feedback"


# -- get_summary ---------------------------------------------------------------


class TestGetSummary:
    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_acceptance_rate_calculated(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        for _ in range(3):
            collector.record(task_id="t", model_id="m1", task_type="code", action=FeedbackAction.ACCEPTED)
        collector.record(task_id="t", model_id="m1", task_type="code", action=FeedbackAction.REGENERATED)
        summary = collector.get_summary("m1", "code")
        assert summary.acceptance_rate == pytest.approx(0.75)
        assert summary.regenerate_count == 1

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_unknown_model_returns_zero_counts(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        summary = collector.get_summary("unknown-model", "code")
        assert summary.accept_count == 0
        assert summary.acceptance_rate == 0.0


# -- get_signals ---------------------------------------------------------------


class TestGetSignals:
    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_returns_newest_first(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        collector.record(task_id="t1", model_id="m", task_type="code", action=FeedbackAction.ACCEPTED)
        collector.record(task_id="t2", model_id="m", task_type="code", action=FeedbackAction.EDITED)
        signals = collector.get_signals()
        assert signals[0].task_id == "t2"
        assert signals[1].task_id == "t1"

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_filter_by_task_id(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        collector.record(task_id="t1", model_id="m", task_type="code", action=FeedbackAction.ACCEPTED)
        collector.record(task_id="t2", model_id="m", task_type="code", action=FeedbackAction.ACCEPTED)
        signals = collector.get_signals(task_id="t1")
        assert len(signals) == 1
        assert signals[0].task_id == "t1"

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_limit_parameter(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        for i in range(10):
            collector.record(task_id=f"t{i}", model_id="m", task_type="code", action=FeedbackAction.ACCEPTED)
        signals = collector.get_signals(limit=3)
        assert len(signals) == 3


# -- should_ask_explicit_question ----------------------------------------------


class TestShouldAskExplicitQuestion:
    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_contradiction_user_edits_high_score(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        """User edited output but Inspector scored it high — contradiction."""
        collector.record(
            task_id="t1",
            model_id="m",
            task_type="code",
            action=FeedbackAction.EDITED,
            inspector_score=0.9,
        )
        assert collector.should_ask_explicit_question("t1") is True

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_contradiction_user_accepts_low_score(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        """User accepted output but Inspector scored it low — contradiction."""
        collector.record(
            task_id="t1",
            model_id="m",
            task_type="code",
            action=FeedbackAction.ACCEPTED,
            inspector_score=0.1,
        )
        assert collector.should_ask_explicit_question("t1") is True

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_no_contradiction_when_aligned(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        """User accepted high-scoring output — no contradiction."""
        collector.record(
            task_id="t1",
            model_id="m",
            task_type="code",
            action=FeedbackAction.ACCEPTED,
            inspector_score=0.85,
        )
        assert collector.should_ask_explicit_question("t1") is False

    @patch("vetinari.learning.implicit_feedback._get_context_graph", return_value=None)
    def test_no_contradiction_without_inspector_score(self, _mock_cg, collector: ImplicitFeedbackCollector) -> None:
        """No Inspector score means no contradiction check."""
        collector.record(
            task_id="t1",
            model_id="m",
            task_type="code",
            action=FeedbackAction.EDITED,
        )
        assert collector.should_ask_explicit_question("t1") is False


# -- Singleton ----------------------------------------------------------------


class TestSingleton:
    def test_get_returns_same_instance(self) -> None:
        c1 = get_implicit_feedback_collector()
        c2 = get_implicit_feedback_collector()
        assert c1 is c2

    def test_reset_creates_new_instance(self) -> None:
        c1 = get_implicit_feedback_collector()
        reset_implicit_feedback_collector()
        c2 = get_implicit_feedback_collector()
        assert c1 is not c2
