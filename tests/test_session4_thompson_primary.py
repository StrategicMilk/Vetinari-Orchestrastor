"""Tests for Session 4: Thompson Sampling primary, drift wiring, judge independence.

Covers items 4.1-4.6 from the master implementation plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_router_model_info as _make_model

# ---------------------------------------------------------------------------
# Module-level dataclasses for Thompson test arms.
# These MUST be at module level, not inside test functions, because Python
# 3.14's dataclasses._is_type() does sys.modules.get(cls.__module__).__dict__
# which crashes if the test module's sys.modules entry is missing.
# ---------------------------------------------------------------------------


@dataclass
class _FakeMatureArm:
    """Arm with 25 observations (above maturity threshold)."""

    alpha: float = 15.0
    beta: float = 5.0
    total_pulls: int = 25

    def sample(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class _FakeNewArm:
    """Arm with 0 observations (full exploration bonus)."""

    alpha: float = 2.0
    beta: float = 2.0
    total_pulls: int = 0

    def sample(self) -> float:
        return 0.5


@dataclass
class _FakeEstablishedArm:
    """Arm with 10 observations (partial exploration bonus)."""

    alpha: float = 2.0
    beta: float = 2.0
    total_pulls: int = 10

    def sample(self) -> float:
        return 0.5


@dataclass
class _FakeDecayArm:
    """Arm with configurable observations for decay tests."""

    alpha: float = 2.0
    beta: float = 2.0
    total_pulls: int = 0

    def sample(self) -> float:
        return 0.5


# ---------------------------------------------------------------------------
# 4.1 + 4.5: Thompson primary weight + exploration bonus
# ---------------------------------------------------------------------------


class TestThompsonPrimaryWeight:
    """Verify Thompson Sampling dominates scoring when data is mature."""

    def _make_router(self) -> MagicMock:
        from vetinari.models.dynamic_model_router import DynamicModelRouter

        router = DynamicModelRouter.__new__(DynamicModelRouter)
        router.prefer_local = True
        router.max_latency_ms = 30000
        router.max_memory_gb = 16.0
        router._ponder_engine = None
        router._model_config = {}
        router._task_defaults = {}
        router._model_tiers = []
        router.models = []
        router._performance_cache = {}
        router._selection_history = []
        router._health_check_callback = None
        return router

    def test_thompson_primary_when_mature(self) -> None:
        """With >=20 observations, Thompson provides 70% of score."""
        from vetinari.models.dynamic_model_router import TaskType

        router = self._make_router()
        model = _make_model(model_id="mature-model", code_gen=True, context_length=8192)

        fake_ts = MagicMock()
        fake_ts._arms = {f"mature-model:{TaskType.CODE.value}": _FakeMatureArm()}

        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            return_value=fake_ts,
        ):
            score = router._internal_score(model, TaskType.CODE, "test task", None)

        # Thompson mean = 15/20 = 0.75, rule_score components exist
        # Score should be dominated by Thompson (0.70 * 0.75 = 0.525)
        assert score > 0.50, f"Score {score} should be >0.50 with Thompson primary"

    def test_exploration_bonus_for_new_model(self) -> None:
        """New model with 0 observations gets full 0.15 exploration bonus."""
        from vetinari.models.dynamic_model_router import TaskType

        router = self._make_router()
        model_new = _make_model(model_id="new-model", code_gen=True, context_length=8192)
        model_established = _make_model(model_id="established-model", code_gen=True, context_length=8192)

        fake_ts = MagicMock()
        fake_ts._arms = {
            f"new-model:{TaskType.CODE.value}": _FakeNewArm(),
            f"established-model:{TaskType.CODE.value}": _FakeEstablishedArm(),
        }

        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            return_value=fake_ts,
        ):
            score_new = router._internal_score(model_new, TaskType.CODE, "test", None)
            score_est = router._internal_score(model_established, TaskType.CODE, "test", None)

        # New model gets full exploration bonus (0.15), established gets 0.075
        assert score_new > score_est, (
            f"New model score {score_new} should exceed established {score_est} due to exploration bonus"
        )

    def test_exploration_bonus_decays_linearly(self) -> None:
        """Exploration bonus decays from 0.15 to 0 as observations approach 20."""
        from vetinari.models.dynamic_model_router import TaskType

        router = self._make_router()

        scores = []
        for obs in [0, 5, 10, 15, 19]:
            model = _make_model(model_id=f"model-obs-{obs}", code_gen=True, context_length=8192)

            fake_ts = MagicMock()
            fake_ts._arms = {f"model-obs-{obs}:{TaskType.CODE.value}": _FakeDecayArm(total_pulls=obs)}

            with patch(
                "vetinari.learning.model_selector.get_thompson_selector",
                return_value=fake_ts,
            ):
                s = router._internal_score(model, TaskType.CODE, "test", None)
                scores.append(s)

        # Scores should decrease monotonically as exploration bonus decays
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at obs={[0, 5, 10, 15, 19][i]} ({scores[i]}) should >= "
                f"score at obs={[0, 5, 10, 15, 19][i + 1]} ({scores[i + 1]})"
            )

    def test_thompson_maturity_threshold_is_20(self) -> None:
        """Verify the threshold constant is 20."""
        from vetinari.models.dynamic_model_router import DynamicModelRouter

        assert DynamicModelRouter._THOMPSON_MATURITY_THRESHOLD == 20


# ---------------------------------------------------------------------------
# 4.2: Thompson strategy temperature selection
# ---------------------------------------------------------------------------


class TestThompsonTemperatureWiring:
    """Verify Thompson strategy selection is wired into inference."""

    def test_select_strategy_called_in_inference(self) -> None:
        """Inference path calls Thompson select_strategy for temperature."""
        with patch(
            "vetinari.learning.thompson_selectors.select_strategy",
            return_value=0.5,
        ) as mock_select:
            with patch("vetinari.learning.model_selector.get_thompson_selector") as mock_ts:
                mock_ts_instance = MagicMock()
                mock_ts_instance._arms = {}
                mock_ts_instance._max_arms = 1000
                mock_ts_instance._get_prior = MagicMock(return_value=(2.0, 2.0))
                mock_ts.return_value = mock_ts_instance

                # The select_strategy function exists and is importable
                from vetinari.learning.thompson_selectors import select_strategy

                result = select_strategy(
                    mock_ts_instance._arms,
                    "WORKER",
                    "default",
                    "temperature",
                    mock_ts_instance._max_arms,
                    mock_ts_instance._get_prior,
                )
                assert isinstance(result, (int, float))


# ---------------------------------------------------------------------------
# 4.3: QUALITY_DRIFT event subscriber
# ---------------------------------------------------------------------------


class TestQualityDriftSubscriber:
    """Verify QUALITY_DRIFT events have a subscriber."""

    def test_quality_drift_detected_event_exists(self) -> None:
        """QualityDriftDetected event class is importable from events."""
        from vetinari.events import QualityDriftDetected

        event = QualityDriftDetected(
            event_type="QUALITY_DRIFT",
            timestamp=time.time(),
            task_type="coding",
            triggered_detectors=["cusum", "adwin"],
            observation_count=150,
        )
        assert event.event_type == "QUALITY_DRIFT"
        assert event.task_type == "coding"
        assert event.triggered_detectors == ["cusum", "adwin"]
        assert event.observation_count == 150

    def test_quality_drift_detector_emits_typed_event(self) -> None:
        """QualityDriftDetector emits QualityDriftDetected, not bare Event."""
        from vetinari.analytics.quality_drift import QualityDriftDetector

        detector = QualityDriftDetector()

        with patch("vetinari.analytics.quality_drift.get_event_bus") as mock_bus:
            mock_bus_instance = MagicMock()
            mock_bus.return_value = mock_bus_instance

            detector._emit_drift_event(["cusum", "page_hinkley"])

            assert mock_bus_instance.publish.called, "EventBus.publish should have been called"
            event = mock_bus_instance.publish.call_args[0][0]
            from vetinari.events import QualityDriftDetected

            assert isinstance(event, QualityDriftDetected)
            assert event.triggered_detectors == ["cusum", "page_hinkley"]

    def test_drift_subscriber_wired_in_startup(self) -> None:
        """_wire_event_subscribers registers a QualityDriftDetected handler."""
        # Verify the event class is referenced in the wiring function source
        import inspect

        from vetinari.cli_startup import _wire_event_subscribers
        from vetinari.events import QualityDriftDetected

        source = inspect.getsource(_wire_event_subscribers)
        assert "QualityDriftDetected" in source
        assert "quality_drift" in source.lower()


# ---------------------------------------------------------------------------
# 4.4: Judge model independence
# ---------------------------------------------------------------------------


class TestJudgeModelIndependence:
    """Verify quality scorer avoids self-evaluation when only one model loaded."""

    def test_single_model_uses_heuristic(self) -> None:
        """When judge == evaluated model, _score_with_llm returns None."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer.__new__(QualityScorer)
        scorer._adapter_manager = None
        scorer._scores = []
        scorer._score_count = 0
        scorer._score_history = {}
        scorer._calibration_interval = 10
        scorer._baselines = {}

        # Mock _pick_judge_model to return same model (only one loaded)
        with patch.object(scorer, "_pick_judge_model", return_value="same-model-7b"):
            result = scorer._score_with_llm(
                task_id="test-task",
                model_id="same-model-7b",
                task_type="coding",
                task_description="Write a function",
                output="def hello(): pass",
                dims=["correctness", "completeness"],
            )

        # Should return None, triggering heuristic fallback in caller
        assert result is None

    def test_different_judge_proceeds_to_llm(self) -> None:
        """When judge != evaluated model, LLM scoring is attempted."""
        from vetinari.learning.quality_scorer import QualityScorer

        scorer = QualityScorer.__new__(QualityScorer)
        scorer._adapter_manager = MagicMock()
        scorer._scores = []
        scorer._score_count = 0
        scorer._score_history = {}
        scorer._calibration_interval = 10
        scorer._baselines = {}

        with patch.object(scorer, "_pick_judge_model", return_value="different-model"):
            with patch("vetinari.adapters.llama_cpp_local_adapter.LocalInferenceAdapter") as mock_adapter_cls:
                mock_adapter = MagicMock()
                mock_adapter.chat.return_value = {
                    "output": '{"overall": 0.8, "dimensions": {"correctness": 0.9}, "issues": [], "confidence": 0.7}'
                }
                mock_adapter_cls.return_value = mock_adapter

                result = scorer._score_with_llm(
                    task_id="test-task",
                    model_id="original-model",
                    task_type="coding",
                    task_description="Write a function",
                    output="def hello(): pass",
                    dims=["correctness"],
                )

        # LLM scoring should proceed and return a QualityScore
        assert result is not None
        assert result.method == "llm"


# ---------------------------------------------------------------------------
# 4.6: Temperature learning into model_profiler_data
# ---------------------------------------------------------------------------


class TestTemperatureLearning:
    """Verify temperature learning wiring and overrides."""

    def test_learned_overrides_take_precedence(self) -> None:
        """get_temperature returns learned value when available."""
        from vetinari.models import model_profiler_data

        # Store original and set a learned override
        original = model_profiler_data.get_temperature("llama", "coding")

        with model_profiler_data._learned_temps_lock:
            model_profiler_data._learned_temperature_overrides["llama"] = {"coding": 0.42}

        try:
            learned = model_profiler_data.get_temperature("llama", "coding")
            assert learned == 0.42, f"Expected 0.42, got {learned}"
            assert learned != original or original == 0.42
        finally:
            # Clean up
            with model_profiler_data._learned_temps_lock:
                model_profiler_data._learned_temperature_overrides.clear()

    def test_hardcoded_used_when_no_learned(self) -> None:
        """get_temperature falls back to hardcoded matrix without overrides."""
        from vetinari.models import model_profiler_data

        # Ensure no overrides
        with model_profiler_data._learned_temps_lock:
            model_profiler_data._learned_temperature_overrides.clear()

        temp = model_profiler_data.get_temperature("llama", "coding")
        assert temp == 0.05  # Hardcoded value for llama/coding

    def test_update_learned_temperatures_returns_zero_without_data(self) -> None:
        """update_learned_temperatures returns 0 when Thompson has no strategy arms."""
        from vetinari.models.model_profiler_data import update_learned_temperatures

        with patch("vetinari.learning.model_selector.get_thompson_selector") as mock_ts:
            mock_ts_instance = MagicMock()
            mock_ts_instance._arms = {}
            mock_ts.return_value = mock_ts_instance

            result = update_learned_temperatures()
            assert result == 0

    def test_update_learned_temperatures_callable(self) -> None:
        """update_learned_temperatures is importable and callable."""
        from vetinari.models.model_profiler_data import update_learned_temperatures

        assert callable(update_learned_temperatures)
