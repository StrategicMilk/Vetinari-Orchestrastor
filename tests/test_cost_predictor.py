"""Tests for vetinari/analytics/cost_predictor.py."""

from __future__ import annotations

import pytest

from vetinari.analytics.cost_predictor import (
    _BASE_TOKENS,
    _CALIBRATION_THRESHOLD,
    _PER_TOKEN_COST,
    _TOKENS_PER_SECOND,
    CostEstimate,
    CostPredictor,
)


class TestCostEstimateDataclass:
    """Verify CostEstimate is a well-formed dataclass."""

    def test_fields_accessible(self):
        est = CostEstimate(tokens=1000, latency_seconds=10.0, cost_usd=0.003, confidence=0.5)
        assert est.tokens == 1000
        assert est.latency_seconds == pytest.approx(10.0)
        assert est.cost_usd == pytest.approx(0.003)
        assert est.confidence == pytest.approx(0.5)

    def test_confidence_bounds(self):
        """Confidence must be expressible in [0, 1]."""
        est = CostEstimate(tokens=100, latency_seconds=1.0, cost_usd=0.001, confidence=0.0)
        assert est.confidence == 0.0
        est2 = CostEstimate(tokens=100, latency_seconds=1.0, cost_usd=0.001, confidence=1.0)
        assert est2.confidence == 1.0


class TestPredictReturnType:
    """predict() must always return a CostEstimate."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = CostPredictor()

    def test_returns_cost_estimate_instance(self):
        result = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-sonnet")
        assert isinstance(result, CostEstimate)

    def test_tokens_positive(self):
        result = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-sonnet")
        assert result.tokens > 0

    def test_cost_positive(self):
        result = self.predictor.predict("analysis", complexity=2.0, scope_size=200, model="claude-haiku")
        assert result.cost_usd > 0.0

    def test_latency_positive(self):
        result = self.predictor.predict("planning", complexity=1.5, scope_size=50, model="gpt-4o")
        assert result.latency_seconds > 0.0

    def test_unknown_task_type_uses_default(self):
        """An unknown task_type should not raise — falls back to default base."""
        result = self.predictor.predict("unknown_xyz", complexity=1.0, scope_size=100, model="claude-sonnet")
        assert isinstance(result, CostEstimate)
        assert result.tokens > 0

    def test_unknown_model_uses_default(self):
        """An unknown model should not raise — falls back to default pricing."""
        result = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="gpt-99-turbo")
        assert isinstance(result, CostEstimate)
        assert result.cost_usd > 0.0


class TestHeuristicCalculation:
    """Verify the heuristic formula produces coherent values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = CostPredictor()

    def test_higher_complexity_means_more_tokens(self):
        low = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-sonnet")
        high = self.predictor.predict("coding", complexity=3.0, scope_size=100, model="claude-sonnet")
        assert high.tokens > low.tokens

    def test_higher_scope_means_more_tokens(self):
        small = self.predictor.predict("coding", complexity=1.0, scope_size=50, model="claude-sonnet")
        large = self.predictor.predict("coding", complexity=1.0, scope_size=500, model="claude-sonnet")
        assert large.tokens > small.tokens

    def test_cost_proportional_to_model_price(self):
        """claude-opus costs more per token than claude-haiku."""
        cheap = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-haiku")
        expensive = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-opus")
        assert expensive.cost_usd > cheap.cost_usd

    def test_latency_inversely_proportional_to_throughput(self):
        """A faster model (haiku) should produce lower latency for same token count."""
        slow = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-opus")
        fast = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-haiku")
        assert fast.latency_seconds < slow.latency_seconds

    def test_cost_equals_tokens_times_per_token_rate(self):
        """Verify cost_usd = tokens * per_token_rate for known model."""
        model = "claude-sonnet"
        result = self.predictor.predict("coding", complexity=1.0, scope_size=100, model=model)
        expected_cost = result.tokens * _PER_TOKEN_COST[model]
        assert result.cost_usd == pytest.approx(expected_cost)

    def test_latency_equals_tokens_divided_by_tps(self):
        """Verify latency_seconds = tokens / tokens_per_second for known model."""
        model = "claude-haiku"
        result = self.predictor.predict("analysis", complexity=1.0, scope_size=100, model=model)
        expected_latency = result.tokens / _TOKENS_PER_SECOND[model]
        assert result.latency_seconds == pytest.approx(expected_latency)


class TestConfidence:
    """Confidence must be 0 when no records, and grow towards 1 with records."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = CostPredictor()

    def test_zero_confidence_with_no_records(self):
        result = self.predictor.predict("coding", complexity=1.0, scope_size=100, model="claude-sonnet")
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_grows_with_records(self):
        task_type = "coding"
        for _ in range(10):
            self.predictor.record_actual(
                task_type,
                1.0,
                100,
                "claude-sonnet",
                actual_tokens=2000,
                actual_latency=25.0,
                actual_cost=0.006,
            )
        result = self.predictor.predict(task_type, complexity=1.0, scope_size=100, model="claude-sonnet")
        assert result.confidence == pytest.approx(10 / _CALIBRATION_THRESHOLD)

    def test_confidence_caps_at_one(self):
        task_type = "analysis"
        for _ in range(_CALIBRATION_THRESHOLD + 10):
            self.predictor.record_actual(
                task_type,
                1.0,
                100,
                "claude-haiku",
                actual_tokens=1500,
                actual_latency=12.5,
                actual_cost=0.0004,
            )
        result = self.predictor.predict(task_type, complexity=1.0, scope_size=100, model="claude-haiku")
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_per_task_type(self):
        """Records for task_type A do not boost confidence for task_type B."""
        for _ in range(25):
            self.predictor.record_actual(
                "coding",
                1.0,
                100,
                "claude-sonnet",
                actual_tokens=2000,
                actual_latency=25.0,
                actual_cost=0.006,
            )
        result_b = self.predictor.predict("planning", complexity=1.0, scope_size=100, model="claude-sonnet")
        assert result_b.confidence == pytest.approx(0.0)


class TestRecordActual:
    """record_actual must store records and affect future predictions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = CostPredictor()

    def test_record_increases_count(self):
        assert self.predictor.record_count() == 0
        self.predictor.record_actual(
            "coding",
            1.0,
            100,
            "claude-sonnet",
            actual_tokens=2000,
            actual_latency=25.0,
            actual_cost=0.006,
        )
        assert self.predictor.record_count() == 1

    def test_record_count_by_task_type(self):
        self.predictor.record_actual(
            "coding",
            1.0,
            100,
            "claude-sonnet",
            actual_tokens=2000,
            actual_latency=25.0,
            actual_cost=0.006,
        )
        self.predictor.record_actual(
            "analysis",
            1.0,
            100,
            "claude-haiku",
            actual_tokens=1500,
            actual_latency=12.5,
            actual_cost=0.0004,
        )
        assert self.predictor.record_count("coding") == 1
        assert self.predictor.record_count("analysis") == 1
        assert self.predictor.record_count() == 2

    def test_multiple_records_accumulate(self):
        for i in range(5):
            self.predictor.record_actual(
                "coding",
                float(i + 1),
                100 * (i + 1),
                "claude-sonnet",
                actual_tokens=1000 * (i + 1),
                actual_latency=10.0,
                actual_cost=0.003,
            )
        assert self.predictor.record_count("coding") == 5


class TestOlsCalibration:
    """After CALIBRATION_THRESHOLD records the OLS path is exercised."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor = CostPredictor()

    def _seed_records(self, task_type: str, n: int = _CALIBRATION_THRESHOLD) -> None:
        """Add n records with a known linear relationship for easy verification."""
        for i in range(n):
            complexity = 1.0 + (i % 3) * 0.5  # cycles 1.0, 1.5, 2.0
            scope = 100 + i * 10
            # Actual tokens follow a simple linear heuristic + small noise
            tokens = int(1500 * complexity * (scope / 100.0))
            self.predictor.record_actual(
                task_type,
                complexity,
                scope,
                "claude-sonnet",
                actual_tokens=tokens,
                actual_latency=tokens / 80.0,
                actual_cost=tokens * 3e-6,
            )

    def test_ols_path_returns_cost_estimate(self):
        self._seed_records("coding")
        result = self.predictor.predict("coding", complexity=2.0, scope_size=300, model="claude-sonnet")
        assert isinstance(result, CostEstimate)

    def test_ols_confidence_is_one_after_threshold(self):
        self._seed_records("coding")
        result = self.predictor.predict("coding", complexity=2.0, scope_size=300, model="claude-sonnet")
        assert result.confidence == pytest.approx(1.0)

    def test_ols_token_estimate_positive(self):
        self._seed_records("analysis")
        result = self.predictor.predict("analysis", complexity=1.5, scope_size=200, model="claude-haiku")
        assert result.tokens > 0

    def test_fallback_to_heuristic_when_few_records(self):
        """With 49 records the heuristic path is still used."""
        for _i in range(_CALIBRATION_THRESHOLD - 1):
            self.predictor.record_actual(
                "review",
                1.0,
                100,
                "claude-sonnet",
                actual_tokens=1000,
                actual_latency=12.5,
                actual_cost=0.003,
            )
        result = self.predictor.predict("review", complexity=1.0, scope_size=100, model="claude-sonnet")
        # Should still return a valid estimate (heuristic path)
        assert isinstance(result, CostEstimate)
        assert result.tokens > 0
        # Confidence just below 1.0
        assert result.confidence < 1.0


class TestThreadSafety:
    """Concurrent record_actual calls must not corrupt internal state."""

    def test_concurrent_recording(self):
        import threading

        predictor = CostPredictor()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    predictor.record_actual(
                        "coding",
                        1.0,
                        100,
                        "claude-sonnet",
                        actual_tokens=2000,
                        actual_latency=25.0,
                        actual_cost=0.006,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert predictor.record_count() == 100
