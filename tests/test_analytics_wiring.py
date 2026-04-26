"""Tests for vetinari.analytics.wiring — the centralized analytics pipeline wiring module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import vetinari.analytics.wiring as wiring
from vetinari.types import AgentType


@pytest.fixture(autouse=True)
def reset_wiring_singletons():
    """Reset all wiring lazy singletons before each test for isolation."""
    wiring.reset_all()
    yield
    wiring.reset_all()


# ---------------------------------------------------------------------------
# record_inference_cost
# ---------------------------------------------------------------------------


class TestRecordInferenceCost:
    def test_calls_cost_tracker_record(self) -> None:
        mock_tracker = MagicMock()
        _mock_entry_cls = MagicMock()

        with (
            patch("vetinari.analytics.wiring._lazy_cost_tracker", return_value=mock_tracker),
            patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=MagicMock()),
        ):
            wiring.record_inference_cost(
                agent_type=AgentType.FOREMAN.value,
                task_id="t-001",
                provider="openai",
                model_id="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                latency_ms=320.0,
            )
            mock_tracker.record.assert_called_once()
        entry = mock_tracker.record.call_args.args[0]
        assert entry.agent == AgentType.FOREMAN.value
        assert entry.task_id == "t-001"
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.latency_ms == 320.0

    def test_calls_sla_tracker_record_latency_and_request(self) -> None:
        mock_sla = MagicMock()

        with (
            patch("vetinari.analytics.wiring._lazy_cost_tracker", return_value=MagicMock()),
            patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=mock_sla),
        ):
            wiring.record_inference_cost(
                agent_type=AgentType.WORKER.value,
                task_id="t-002",
                provider="local",
                model_id="llama3",
                input_tokens=200,
                output_tokens=80,
                latency_ms=150.0,
            )
            mock_sla.record_latency.assert_called_once_with("local:llama3", 150.0, success=True)
            mock_sla.record_request.assert_called_once_with(success=True)

    def test_cost_tracker_exception_does_not_raise(self) -> None:
        bad_tracker = MagicMock()
        bad_tracker.record.side_effect = RuntimeError("DB exploded")

        with (
            patch("vetinari.analytics.wiring._lazy_cost_tracker", return_value=bad_tracker),
            patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=MagicMock()),
        ):
            # Should not raise — exception is swallowed
            assert (
                wiring.record_inference_cost(
                    agent_type=AgentType.INSPECTOR.value,
                    task_id="t-003",
                    provider="openai",
                    model_id="gpt-4o",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=50.0,
                )
                is None
            )

    def test_sla_tracker_exception_does_not_raise(self) -> None:
        bad_sla = MagicMock()
        bad_sla.record_latency.side_effect = RuntimeError("SLA broken")

        with (
            patch("vetinari.analytics.wiring._lazy_cost_tracker", return_value=MagicMock()),
            patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=bad_sla),
        ):
            assert (
                wiring.record_inference_cost(
                    agent_type=AgentType.FOREMAN.value,
                    task_id="t-004",
                    provider="local",
                    model_id="llama3",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=50.0,
                )
                is None
            )

    def test_none_tracker_is_silently_skipped(self) -> None:
        with (
            patch("vetinari.analytics.wiring._lazy_cost_tracker", return_value=None),
            patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=None),
        ):
            # Should not raise even when trackers are None
            result = wiring.record_inference_cost(
                agent_type=AgentType.FOREMAN.value,
                task_id="t-005",
                provider="local",
                model_id="llama3",
                input_tokens=10,
                output_tokens=5,
                latency_ms=50.0,
            )
            assert result is None  # graceful degradation when trackers unavailable


# ---------------------------------------------------------------------------
# record_inference_failure
# ---------------------------------------------------------------------------


class TestRecordInferenceFailure:
    def test_calls_sla_tracker_with_failure(self) -> None:
        mock_sla = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=mock_sla):
            wiring.record_inference_failure(
                agent_type=AgentType.WORKER.value,
                provider="openai",
                model_id="gpt-4o",
                latency_ms=200.0,
            )
            mock_sla.record_latency.assert_called_once_with("openai:gpt-4o", 200.0, success=False)
            mock_sla.record_request.assert_called_once_with(success=False)

    def test_exception_does_not_raise(self) -> None:
        bad_sla = MagicMock()
        bad_sla.record_latency.side_effect = OSError("network gone")

        with patch("vetinari.analytics.wiring._lazy_sla_tracker", return_value=bad_sla):
            assert (
                wiring.record_inference_failure(
                    agent_type=AgentType.FOREMAN.value,
                    provider="local",
                    model_id="llama3",
                    latency_ms=100.0,
                )
                is None
            )


# ---------------------------------------------------------------------------
# record_task_metrics
# ---------------------------------------------------------------------------


class TestRecordTaskMetrics:
    def test_calls_detector_three_times(self) -> None:
        mock_detector = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_anomaly_detector", return_value=mock_detector):
            wiring.record_task_metrics(
                task_id="t-010",
                agent_type=AgentType.WORKER.value,
                latency_ms=500.0,
                quality_score=0.85,
                token_count=1200,
                success=True,
            )
            assert mock_detector.detect.call_count == 3
            calls = {c.args[0] for c in mock_detector.detect.call_args_list}
            assert calls == {"task_latency", "task_quality", "task_tokens"}

    def test_exception_does_not_raise(self) -> None:
        bad_detector = MagicMock()
        bad_detector.detect.side_effect = ValueError("bad value")

        with patch("vetinari.analytics.wiring._lazy_anomaly_detector", return_value=bad_detector):
            assert (
                wiring.record_task_metrics(
                    task_id="t-011",
                    agent_type=AgentType.FOREMAN.value,
                    latency_ms=100.0,
                    quality_score=0.5,
                    token_count=300,
                    success=False,
                )
                is None
            )


# ---------------------------------------------------------------------------
# record_periodic_metrics
# ---------------------------------------------------------------------------


class TestRecordPeriodicMetrics:
    def test_calls_forecaster_ingest_three_times(self) -> None:
        mock_forecaster = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_forecaster", return_value=mock_forecaster):
            wiring.record_periodic_metrics(
                request_rate=2.5,
                avg_latency_ms=300.0,
                queue_depth=4,
            )
            assert mock_forecaster.ingest.call_count == 3
            metrics = {c.args[0] for c in mock_forecaster.ingest.call_args_list}
            assert metrics == {"request_rate", "avg_latency_ms", "queue_depth"}

    def test_exception_does_not_raise(self) -> None:
        bad_forecaster = MagicMock()
        bad_forecaster.ingest.side_effect = Exception("forecaster down")

        with patch("vetinari.analytics.wiring._lazy_forecaster", return_value=bad_forecaster):
            assert wiring.record_periodic_metrics(request_rate=1.0, avg_latency_ms=100.0, queue_depth=0) is None


# ---------------------------------------------------------------------------
# record_quality_score
# ---------------------------------------------------------------------------


class TestRecordQualityScore:
    def test_calls_drift_ensemble_observe(self) -> None:
        mock_drift = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_drift_ensemble", return_value=mock_drift):
            wiring.record_quality_score(0.92)
            mock_drift.observe.assert_called_once_with(0.92)

    def test_exception_does_not_raise(self) -> None:
        bad_drift = MagicMock()
        bad_drift.observe.side_effect = ZeroDivisionError("div by zero")

        with patch("vetinari.analytics.wiring._lazy_drift_ensemble", return_value=bad_drift):
            assert wiring.record_quality_score(0.5) is None


class TestRecordQualityScoresBatch:
    def test_calls_drift_ensemble_observe_many_with_validated_scores(self) -> None:
        mock_drift = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_drift_ensemble", return_value=mock_drift):
            wiring.record_quality_scores_batch([0, 0.5, 1.0])

        mock_drift.observe_many.assert_called_once_with([0.0, 0.5, 1.0])

    @pytest.mark.parametrize("score", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.5", None])
    def test_rejects_malformed_scores_before_recording(self, score: Any) -> None:
        mock_drift = MagicMock()

        with (
            patch("vetinari.analytics.wiring._lazy_drift_ensemble", return_value=mock_drift),
            pytest.raises(ValueError, match="quality score"),
        ):
            wiring.record_quality_scores_batch([0.8, score])

        mock_drift.observe_many.assert_not_called()


# ---------------------------------------------------------------------------
# record_pipeline_event
# ---------------------------------------------------------------------------


class TestRecordPipelineEvent:
    def test_calls_value_stream_record_event(self) -> None:
        mock_vs = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_value_stream", return_value=mock_vs):
            wiring.record_pipeline_event(
                execution_id="exec-1",
                task_id="task-1",
                agent_type=AgentType.FOREMAN.value,
                timing_event="task_dispatched",
                metadata={"plan_id": "p-1"},
            )
            mock_vs.record_event.assert_called_once_with(
                execution_id="exec-1",
                task_id="task-1",
                agent_type=AgentType.FOREMAN.value,
                timing_event="task_dispatched",
                metadata={"plan_id": "p-1"},
            )

    def test_none_metadata_defaults_to_empty_dict(self) -> None:
        mock_vs = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_value_stream", return_value=mock_vs):
            wiring.record_pipeline_event(
                execution_id="exec-2",
                task_id="task-2",
                agent_type=AgentType.WORKER.value,
                timing_event="task_completed",
            )
            _, kwargs = mock_vs.record_event.call_args
            assert kwargs["metadata"] == {}

    def test_exception_does_not_raise(self) -> None:
        bad_vs = MagicMock()
        bad_vs.record_event.side_effect = RuntimeError("store full")

        with patch("vetinari.analytics.wiring._lazy_value_stream", return_value=bad_vs):
            assert (
                wiring.record_pipeline_event(
                    execution_id="exec-3",
                    task_id="task-3",
                    agent_type=AgentType.INSPECTOR.value,
                    timing_event="task_rejected",
                )
                is None
            )


# ---------------------------------------------------------------------------
# predict_cost
# ---------------------------------------------------------------------------


class TestPredictCost:
    def test_returns_estimate_from_predictor(self) -> None:
        from vetinari.analytics.cost_predictor import CostEstimate

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = CostEstimate(
            tokens=2000,
            latency_seconds=25.0,
            cost_usd=0.006,
            confidence=0.8,
        )

        with patch("vetinari.analytics.wiring._lazy_cost_predictor", return_value=mock_predictor):
            result = wiring.predict_cost(
                task_type="coding",
                complexity=0.7,
                scope_size=300,
                model_id="claude-sonnet",
            )

        assert result["tokens"] == 2000
        assert result["latency_seconds"] == 25.0
        assert result["cost_usd"] == 0.006
        assert result["confidence"] == 0.8

    def test_returns_zero_dict_on_exception(self) -> None:
        bad_predictor = MagicMock()
        bad_predictor.predict.side_effect = RuntimeError("predictor broke")

        with patch("vetinari.analytics.wiring._lazy_cost_predictor", return_value=bad_predictor):
            result = wiring.predict_cost(
                task_type="analysis",
                complexity=0.5,
                scope_size=50,
                model_id="local",
            )

        assert result == {"tokens": 0, "latency_seconds": 0.0, "cost_usd": 0.0, "confidence": 0.0}

    def test_returns_zero_dict_when_predictor_is_none(self) -> None:
        with patch("vetinari.analytics.wiring._lazy_cost_predictor", return_value=None):
            result = wiring.predict_cost("planning", 0.3, 10, "gpt-4o-mini")

        assert result["tokens"] == 0
        assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# record_actual_cost
# ---------------------------------------------------------------------------


class TestRecordActualCost:
    def test_calls_predictor_record_actual(self) -> None:
        mock_predictor = MagicMock()

        with patch("vetinari.analytics.wiring._lazy_cost_predictor", return_value=mock_predictor):
            wiring.record_actual_cost(
                task_type="coding",
                complexity=0.7,
                scope_size=300,
                model_id="claude-sonnet",
                actual_tokens=2100,
                actual_latency=26.5,
                actual_cost=0.0063,
            )
            mock_predictor.record_actual.assert_called_once_with(
                task_type="coding",
                complexity=0.7,
                scope_size=300,
                model="claude-sonnet",
                actual_tokens=2100,
                actual_latency=26.5,
                actual_cost=0.0063,
            )

    def test_exception_does_not_raise(self) -> None:
        bad_predictor = MagicMock()
        bad_predictor.record_actual.side_effect = RuntimeError("write failed")

        with patch("vetinari.analytics.wiring._lazy_cost_predictor", return_value=bad_predictor):
            assert (
                wiring.record_actual_cost(
                    task_type="review",
                    complexity=0.3,
                    scope_size=20,
                    model_id="claude-haiku",
                    actual_tokens=400,
                    actual_latency=5.0,
                    actual_cost=0.0001,
                )
                is None
            )


# ---------------------------------------------------------------------------
# reset_all
# ---------------------------------------------------------------------------


class TestResetAll:
    def test_clears_all_singletons(self) -> None:
        # Seed each private global
        wiring._cost_tracker = MagicMock()
        wiring._sla_tracker = MagicMock()
        wiring._anomaly_detector = MagicMock()
        wiring._forecaster = MagicMock()
        wiring._drift_ensemble = MagicMock()
        wiring._value_stream = MagicMock()
        wiring._cost_predictor = MagicMock()

        wiring.reset_all()

        assert wiring._cost_tracker is None
        assert wiring._sla_tracker is None
        assert wiring._anomaly_detector is None
        assert wiring._forecaster is None
        assert wiring._drift_ensemble is None
        assert wiring._value_stream is None
        assert wiring._cost_predictor is None


# ---------------------------------------------------------------------------
# __init__ re-exports
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_all_wiring_functions_are_exported(self) -> None:
        from vetinari import analytics

        expected = {
            "record_inference_cost",
            "record_inference_failure",
            "record_task_metrics",
            "record_periodic_metrics",
            "record_quality_score",
            "record_pipeline_event",
            "predict_cost",
            "record_actual_cost",
            "reset_wiring",
        }
        assert expected.issubset(set(analytics.__all__))
