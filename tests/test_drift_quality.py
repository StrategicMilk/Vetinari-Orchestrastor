"""Tests for vetinari.analytics.quality_drift — quality drift and embedding drift detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_drift_vector as _make_vector
from vetinari.analytics.quality_drift import (
    ADWINDetector,
    CUSUMDriftDetector,
    DriftResult,
    EmbeddingDriftDetector,
    PageHinkleyDetector,
    QualityDriftDetector,
)

# ---------------------------------------------------------------------------
# DriftResult dataclass
# ---------------------------------------------------------------------------


class TestDriftResult:
    def test_default_fields(self) -> None:
        result = DriftResult()
        assert result.is_drift is False
        assert result.detectors_triggered == []
        assert result.votes == 0
        assert result.observation_count == 0

    def test_populated_fields(self) -> None:
        result = DriftResult(
            is_drift=True,
            detectors_triggered=["cusum", "adwin"],
            votes=2,
            observation_count=42,
        )
        assert result.is_drift is True
        assert result.detectors_triggered == ["cusum", "adwin"]
        assert result.votes == 2
        assert result.observation_count == 42


# ---------------------------------------------------------------------------
# CUSUMDriftDetector
# ---------------------------------------------------------------------------


class TestCUSUMDriftDetector:
    def test_no_drift_with_stable_values(self) -> None:
        detector = CUSUMDriftDetector(delta=0.01, threshold=5.0)
        # Feed stable values — should never trigger
        for _ in range(50):
            triggered = detector.observe(0.9)
        # Last call should be False for stable data
        assert not triggered

    def test_detects_drift_with_sustained_shift(self) -> None:
        detector = CUSUMDriftDetector(delta=0.01, threshold=5.0)
        # Build stable baseline
        for _ in range(20):
            detector.observe(0.9)
        # Introduce sustained large downward shift
        detected_any = False
        for _ in range(100):
            if detector.observe(0.1):
                detected_any = True
                break
        assert detected_any, "CUSUM should detect sustained shift from 0.9 to 0.1"

    def test_reset_clears_state(self) -> None:
        detector = CUSUMDriftDetector(delta=0.01, threshold=5.0)
        for _ in range(20):
            detector.observe(0.9)
        detector.reset()
        assert detector._s_pos == 0.0
        assert detector._s_neg == 0.0
        assert detector._running_mean == 0.0
        assert detector._reference_mean is None
        assert detector._count == 0

    def test_no_drift_before_min_samples(self) -> None:
        detector = CUSUMDriftDetector(delta=0.01, threshold=5.0)
        # First 9 observations should always return False (below min_samples=10)
        for _i in range(9):
            assert not detector.observe(0.0)


# ---------------------------------------------------------------------------
# PageHinkleyDetector
# ---------------------------------------------------------------------------


class TestPageHinkleyDetector:
    def test_no_drift_with_stable_values(self) -> None:
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0)
        for _ in range(50):
            triggered = detector.observe(0.85)
        assert not triggered

    def test_detects_abrupt_change(self) -> None:
        # Use a lower threshold so the test reliably detects within the iteration budget
        detector = PageHinkleyDetector(delta=0.005, threshold=10.0)
        # Build stable baseline
        for _ in range(20):
            detector.observe(0.9)
        # Introduce abrupt drop
        detected_any = False
        for _ in range(200):
            if detector.observe(0.1):
                detected_any = True
                break
        assert detected_any, "Page-Hinkley should detect abrupt shift from 0.9 to 0.1"

    def test_reset_clears_state(self) -> None:
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0)
        for _ in range(20):
            detector.observe(0.9)
        detector.reset()
        assert detector._sum == 0.0
        assert detector._mean == 0.0
        assert detector._count == 0
        assert detector._min_sum == float("inf")
        assert detector._max_sum == float("-inf")

    def test_no_drift_before_min_samples(self) -> None:
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0)
        for _i in range(9):
            assert not detector.observe(0.0)


# ---------------------------------------------------------------------------
# ADWINDetector
# ---------------------------------------------------------------------------


class TestADWINDetector:
    def test_no_drift_with_stable_values(self) -> None:
        detector = ADWINDetector(delta=0.002)
        results = [detector.observe(0.9) for _ in range(30)]
        assert not any(results), "ADWIN should not trigger on stable data"

    def test_detects_distribution_change(self) -> None:
        detector = ADWINDetector(delta=0.002)
        # Feed a stable distribution
        for _ in range(30):
            detector.observe(0.9)
        # Dramatically shift distribution
        detected_any = False
        for _ in range(100):
            if detector.observe(0.1):
                detected_any = True
                break
        assert detected_any, "ADWIN should detect a distribution change from 0.9 to 0.1"

    def test_reset_clears_state(self) -> None:
        detector = ADWINDetector(delta=0.002)
        for _ in range(20):
            detector.observe(0.9)
        detector.reset()
        assert detector._window == []
        assert detector._total == 0.0
        assert detector._count == 0

    def test_window_capped_at_max(self) -> None:
        detector = ADWINDetector(delta=0.002)
        # Feed enough stable values to hit max_window cap (500)
        for _ in range(520):
            detector.observe(0.9)
        assert len(detector._window) <= 500


# ---------------------------------------------------------------------------
# QualityDriftDetector (ensemble)
# ---------------------------------------------------------------------------


class TestQualityDriftDetector:
    def test_no_drift_when_only_one_detector_triggers(self) -> None:
        """Single detector vote should not confirm drift."""
        detector = QualityDriftDetector()
        # Patch individual detectors so only CUSUM fires
        detector._cusum.observe = MagicMock(return_value=True)
        detector._page_hinkley.observe = MagicMock(return_value=False)
        detector._adwin.observe = MagicMock(return_value=False)

        result = detector.observe(0.5)
        assert not result.is_drift
        assert result.votes == 1
        assert result.detectors_triggered == ["cusum"]

    def test_drift_confirmed_when_two_detectors_agree(self) -> None:
        """Two detector votes should confirm drift."""
        detector = QualityDriftDetector()
        detector._cusum.observe = MagicMock(return_value=True)
        detector._page_hinkley.observe = MagicMock(return_value=True)
        detector._adwin.observe = MagicMock(return_value=False)

        result = detector.observe(0.5)
        assert result.is_drift
        assert result.votes == 2
        assert "cusum" in result.detectors_triggered
        assert "page_hinkley" in result.detectors_triggered

    def test_drift_confirmed_when_all_three_agree(self) -> None:
        """All three detector votes should confirm drift."""
        detector = QualityDriftDetector()
        detector._cusum.observe = MagicMock(return_value=True)
        detector._page_hinkley.observe = MagicMock(return_value=True)
        detector._adwin.observe = MagicMock(return_value=True)

        result = detector.observe(0.5)
        assert result.is_drift
        assert result.votes == 3

    def test_observation_count_increments(self) -> None:
        detector = QualityDriftDetector()
        for _i in range(5):
            result = detector.observe(0.9)
        assert result.observation_count == 5

    def test_emits_quality_drift_event(self) -> None:
        """Drift detection should emit QUALITY_DRIFT event."""
        detector = QualityDriftDetector()
        detector._cusum.observe = MagicMock(return_value=True)
        detector._page_hinkley.observe = MagicMock(return_value=True)
        detector._adwin.observe = MagicMock(return_value=False)

        mock_bus = MagicMock()
        with patch("vetinari.analytics.quality_drift.get_event_bus", return_value=mock_bus):
            detector.observe(0.2)

        mock_bus.publish.assert_called_once()
        published_event = mock_bus.publish.call_args[0][0]
        assert published_event.event_type == "QUALITY_DRIFT"

    def test_reset_clears_observation_count(self) -> None:
        detector = QualityDriftDetector()
        for _ in range(10):
            detector.observe(0.9)
        detector.reset()
        result = detector.observe(0.9)
        assert result.observation_count == 1

    def test_no_event_when_no_drift(self) -> None:
        """No event should be emitted when fewer than 2 detectors agree."""
        detector = QualityDriftDetector()
        detector._cusum.observe = MagicMock(return_value=False)
        detector._page_hinkley.observe = MagicMock(return_value=False)
        detector._adwin.observe = MagicMock(return_value=False)

        mock_bus = MagicMock()
        with patch("vetinari.analytics.quality_drift.get_event_bus", return_value=mock_bus):
            detector.observe(0.9)

        mock_bus.publish.assert_not_called()

    def test_event_bus_failure_does_not_raise(self) -> None:
        """Event bus failure should be caught silently."""
        detector = QualityDriftDetector()
        detector._cusum.observe = MagicMock(return_value=True)
        detector._page_hinkley.observe = MagicMock(return_value=True)
        detector._adwin.observe = MagicMock(return_value=False)

        with patch("vetinari.analytics.quality_drift.get_event_bus", side_effect=RuntimeError("bus down")):
            # Should not raise
            result = detector.observe(0.2)
        assert result.is_drift


# ---------------------------------------------------------------------------
# EmbeddingDriftDetector
# ---------------------------------------------------------------------------


class TestEmbeddingDriftDetector:
    def test_no_drift_before_baseline_established(self) -> None:
        """First full batch establishes baseline — must not return drift."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.3)
        results = []
        for _ in range(20):
            results.append(detector.observe(_make_vector(0.5)))
        # The 20th observation triggers baseline establishment (not drift)
        assert not any(results)

    def test_no_drift_with_similar_vectors(self) -> None:
        """Second batch similar to baseline should not trigger drift."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.3)
        # First batch — establishes baseline
        for _ in range(20):
            detector.observe(_make_vector(0.5))
        # Second batch — nearly identical
        detected_any = False
        for _ in range(20):
            if detector.observe(_make_vector(0.51)):
                detected_any = True
                break
        assert not detected_any, "Similar vectors should not trigger embedding drift"

    def test_detects_drift_with_divergent_vectors(self) -> None:
        """Second batch orthogonal to baseline should trigger drift."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.01)
        # First batch (baseline): unit vector along first dimension
        baseline_vec = [1.0] + [0.0] * 9
        for _ in range(20):
            detector.observe(baseline_vec)
        # Second batch: unit vector along second dimension (orthogonal)
        divergent_vec = [0.0, 1.0] + [0.0] * 8
        detected = False
        for _ in range(20):
            if detector.observe(divergent_vec):
                detected = True
                break
        assert detected, "Orthogonal vectors should trigger embedding drift"

    def test_emits_new_domain_detected_event(self) -> None:
        """Drift detection should emit NEW_DOMAIN_DETECTED event."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.01)
        baseline_vec = [1.0] + [0.0] * 9
        for _ in range(20):
            detector.observe(baseline_vec)

        divergent_vec = [0.0, 1.0] + [0.0] * 8
        mock_bus = MagicMock()
        with patch("vetinari.analytics.quality_drift.get_event_bus", return_value=mock_bus):
            for _ in range(20):
                detector.observe(divergent_vec)

        mock_bus.publish.assert_called()
        published_event = mock_bus.publish.call_args[0][0]
        assert published_event.event_type == "NEW_DOMAIN_DETECTED"

    def test_event_bus_failure_does_not_raise(self) -> None:
        """Event bus failure should be caught silently."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.01)
        baseline_vec = [1.0] + [0.0] * 9
        for _ in range(20):
            detector.observe(baseline_vec)

        divergent_vec = [0.0, 1.0] + [0.0] * 8
        with patch("vetinari.analytics.quality_drift.get_event_bus", side_effect=RuntimeError("bus down")):
            # Should not raise — bus failure is swallowed
            for _ in range(20):
                detector.observe(divergent_vec)
        # Detector continues tracking observations after bus failure
        # _baseline_centroid was set during the initial 20 baseline observations
        assert detector._baseline_centroid is not None
        assert isinstance(detector._baseline_centroid, list)
        assert len(detector._baseline_centroid) == 10

    def test_reset_clears_state(self) -> None:
        detector = EmbeddingDriftDetector(cosine_threshold=0.3)
        for _ in range(20):
            detector.observe(_make_vector(0.5))
        detector.reset()
        assert detector._baseline_centroid is None
        assert detector._current_batch == []

    def test_baseline_updated_after_drift(self) -> None:
        """After drift is detected, baseline should be updated to new centroid."""
        detector = EmbeddingDriftDetector(cosine_threshold=0.01)
        baseline_vec = [1.0] + [0.0] * 9
        for _ in range(20):
            detector.observe(baseline_vec)

        old_baseline = list(detector._baseline_centroid)  # type: ignore[arg-type]

        divergent_vec = [0.0, 1.0] + [0.0] * 8
        for _ in range(20):
            detector.observe(divergent_vec)

        # Baseline should have changed
        assert detector._baseline_centroid != old_baseline


# ---------------------------------------------------------------------------
# EmbeddingDriftDetector._cosine_distance
# ---------------------------------------------------------------------------


class TestCosineDistance:
    def test_identical_vectors(self) -> None:
        dist = EmbeddingDriftDetector._cosine_distance([1.0, 0.0], [1.0, 0.0])
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors(self) -> None:
        dist = EmbeddingDriftDetector._cosine_distance([1.0, 0.0], [0.0, 1.0])
        assert dist == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors(self) -> None:
        dist = EmbeddingDriftDetector._cosine_distance([1.0, 0.0], [-1.0, 0.0])
        assert dist == pytest.approx(2.0, abs=1e-9)

    def test_zero_vector_returns_one(self) -> None:
        dist = EmbeddingDriftDetector._cosine_distance([0.0, 0.0], [1.0, 0.0])
        assert dist == 1.0

    def test_partial_overlap(self) -> None:
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        dist = EmbeddingDriftDetector._cosine_distance(a, b)
        # cos(45°) = 1/sqrt(2), distance = 1 - 1/sqrt(2) ≈ 0.293
        import math

        expected = 1.0 - (1.0 / math.sqrt(2))
        assert dist == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_quality_drift_detector(self) -> None:
        from vetinari.analytics.quality_drift import QualityDriftDetector as QDD

        assert QDD is QualityDriftDetector

    def test_import_from_analytics_package(self) -> None:
        from vetinari.analytics import (
            ADWINDetector,
            CUSUMDriftDetector,
            DriftResult,
            EmbeddingDriftDetector,
            PageHinkleyDetector,
            QualityDriftDetector,
        )

        assert callable(QualityDriftDetector)
        assert callable(EmbeddingDriftDetector)
        assert callable(DriftResult)
        assert callable(CUSUMDriftDetector)
        assert callable(PageHinkleyDetector)
        assert callable(ADWINDetector)
