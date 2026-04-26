"""Tests for the Anomaly → Circuit Breaker integration (US-201).

Covers CUSUMDetector, EnsembleAnomalyDetector, and the wiring to the
circuit breaker registry and event bus.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_ensemble_anomaly_detector
from vetinari.analytics.anomaly import (
    AnomalyResult,
    CUSUMDetector,
    EnsembleAnomalyDetector,
)
from vetinari.events import AnomalyDetected, get_event_bus, reset_event_bus
from vetinari.resilience import (
    ResilienceCircuitState,
    get_circuit_breaker_registry,
)
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_event_bus():
    """Reset the event bus singleton before/after each test."""
    reset_event_bus()
    yield
    reset_event_bus()


# ---------------------------------------------------------------------------
# CUSUMDetector
# ---------------------------------------------------------------------------


class TestCUSUMDetector:
    def test_detects_sustained_positive_shift(self):
        """A large upward spike after stable observations should be detected."""
        detector = CUSUMDetector(delta=0.5, threshold=5.0)
        # Feed stable baseline (need >10 samples first)
        for _ in range(15):
            detector.detect(10.0)
        # Now inject a sustained high spike
        detected = False
        for _ in range(30):
            if detector.detect(100.0):
                detected = True
                break
        assert detected, "CUSUM should detect a large sustained positive shift"

    def test_no_detection_with_stable_values(self):
        """Constant observations must never trigger a detection."""
        detector = CUSUMDetector(delta=0.5, threshold=5.0)
        for _ in range(50):
            result = detector.detect(42.0)
            assert result is False, "Stable values must not trigger CUSUM"

    def test_reset_clears_state(self):
        """After reset the detector behaves as if freshly constructed."""
        detector = CUSUMDetector(delta=0.5, threshold=5.0)
        # Build up enough history to be past min_samples
        for _ in range(15):
            detector.detect(10.0)
        # Inject spike to push CUSUM high
        for _ in range(10):
            detector.detect(200.0)
        detector.reset()
        # After reset, first 10 observations should not detect (below min_samples)
        for _ in range(9):
            assert detector.detect(200.0) is False

    def test_detects_negative_shift(self):
        """A large downward shift should also be detected."""
        detector = CUSUMDetector(delta=0.5, threshold=5.0)
        for _ in range(15):
            detector.detect(100.0)
        detected = False
        for _ in range(30):
            if detector.detect(0.0):
                detected = True
                break
        assert detected, "CUSUM should detect a large sustained negative shift"

    def test_resets_cusum_after_detection(self):
        """After a detection the internal accumulators reset so a second burst can be caught."""
        detector = CUSUMDetector(delta=0.5, threshold=5.0)
        for _ in range(15):
            detector.detect(10.0)
        # First burst
        first = False
        for _ in range(30):
            if detector.detect(200.0):
                first = True
                break
        assert first is True, "CUSUM should detect the first sustained burst"
        # Accumulators should have been reset; another burst can trigger again
        second = False
        for _ in range(30):
            if detector.detect(200.0):
                second = True
                break
        assert second is True, "CUSUM should detect a second burst after reset"


# ---------------------------------------------------------------------------
# EnsembleAnomalyDetector — voting logic
# ---------------------------------------------------------------------------


class TestEnsembleAnomalyDetectorVoting:
    def _prime_cusum(self, detector: EnsembleAnomalyDetector) -> None:
        """Feed stable values so CUSUM is past min_samples."""
        for _ in range(15):
            detector.observe(latency=10.0, error_rate=0.01)

    def test_two_detectors_trigger_anomaly_with_real_cusum(self):
        """Both CUSUM detectors should fire on genuinely anomalous input."""
        detector = make_ensemble_anomaly_detector()
        self._prime_cusum(detector)

        # Feed extreme values repeatedly to trigger both real CUSUM detectors
        result = None
        for _ in range(30):
            result = detector.observe(latency=999.0, error_rate=0.99)
            if result is not None:
                break

        assert result is not None
        assert result.is_anomaly is True
        assert result.method == "ensemble"
        assert result.score >= 2.0
        assert "cusum_latency" in result.reason
        assert "cusum_error_rate" in result.reason

    def test_single_detector_insufficient(self):
        """Only one detector firing must not produce an anomaly result."""
        detector = make_ensemble_anomaly_detector()
        self._prime_cusum(detector)

        # Mock only one detector to fire — tests the voting threshold
        detector._cusum_latency.detect = MagicMock(return_value=True)
        detector._cusum_error_rate.detect = MagicMock(return_value=False)

        # Supply no token_usage so the zscore path is skipped
        result = detector.observe(latency=999.0)
        assert result is None

    def test_no_detectors_fire_on_normal_data(self):
        """Normal data within baseline range should not trigger anomaly."""
        detector = make_ensemble_anomaly_detector()
        self._prime_cusum(detector)

        # Normal values close to the primed baseline
        result = detector.observe(latency=10.0, error_rate=0.01)
        assert result is None

    def test_triggered_detectors_names_in_result(self):
        """The triggered detector names appear in the anomaly reason string."""
        detector = make_ensemble_anomaly_detector()
        self._prime_cusum(detector)

        # Feed extreme values to trigger both real detectors
        result = None
        for _ in range(30):
            result = detector.observe(latency=999.0, error_rate=0.99)
            if result is not None:
                break

        assert result is not None
        assert "cusum_latency" in result.reason
        assert "cusum_error_rate" in result.reason


# ---------------------------------------------------------------------------
# EnsembleAnomalyDetector — circuit breaker wiring
# ---------------------------------------------------------------------------


class TestEnsembleCircuitBreakerWiring:
    def test_circuit_breaker_tripped_on_anomaly(self):
        """A confirmed anomaly must trip the circuit breaker via the registry."""
        mock_breaker = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_breaker

        detector = EnsembleAnomalyDetector(agent_type=AgentType.FOREMAN.value)
        detector._cusum_latency.detect = MagicMock(return_value=True)
        detector._cusum_error_rate.detect = MagicMock(return_value=True)

        # Patch at the package level since the import is done lazily via vetinari.resilience
        with patch(
            "vetinari.resilience.get_circuit_breaker_registry",
            return_value=mock_registry,
        ):
            result = detector.observe(latency=999.0, error_rate=0.99)

        assert isinstance(result, AnomalyResult)
        assert result.is_anomaly is True
        assert result.score == 2.0
        mock_registry.get.assert_called_once_with(AgentType.FOREMAN.value)
        mock_breaker.trip.assert_called_once()

    def test_circuit_breaker_failure_does_not_raise(self):
        """A broken circuit breaker registry must not propagate exceptions."""
        detector = EnsembleAnomalyDetector(agent_type=AgentType.FOREMAN.value)
        detector._cusum_latency.detect = MagicMock(return_value=True)
        detector._cusum_error_rate.detect = MagicMock(return_value=True)

        # Make the entire resilience module unavailable via the lazy import path
        with patch(
            "vetinari.resilience.circuit_breaker.CircuitBreakerRegistry.get",
            side_effect=RuntimeError("registry exploded"),
        ):
            # Should not raise; exception must be swallowed and logged
            result = detector.observe(latency=999.0, error_rate=0.99)

        # Anomaly result still returned despite circuit breaker failure
        assert isinstance(result, AnomalyResult)
        assert result.is_anomaly is True

    def test_trip_method_transitions_to_open(self):
        """CircuitBreaker.trip() must move a CLOSED breaker to OPEN state."""
        from vetinari.resilience import CircuitBreaker

        cb = CircuitBreaker("test_trip")
        assert cb.state == ResilienceCircuitState.CLOSED
        cb.trip()
        # Check the internal state directly (bypass auto-transition logic)
        assert cb._state == ResilienceCircuitState.OPEN
        assert cb.stats.trips == 1

    def test_trip_on_already_open_breaker_is_idempotent(self):
        """Calling trip() on an already-OPEN breaker must not increment trips again."""
        from vetinari.resilience import CircuitBreaker

        cb = CircuitBreaker("test_trip_idempotent")
        cb.trip()
        assert cb.stats.trips == 1
        cb.trip()  # already open — should be a no-op
        assert cb.stats.trips == 1


# ---------------------------------------------------------------------------
# EnsembleAnomalyDetector — event bus wiring
# ---------------------------------------------------------------------------


class TestEnsembleEventBusWiring:
    def test_anomaly_detected_event_emitted(self):
        """A confirmed anomaly must publish an AnomalyDetected event to the bus."""
        detector = EnsembleAnomalyDetector(agent_type=AgentType.WORKER.value)
        detector._cusum_latency.detect = MagicMock(return_value=True)
        detector._cusum_error_rate.detect = MagicMock(return_value=True)

        # Suppress circuit breaker side-effect (lazy import — patch at source)
        with patch(
            "vetinari.resilience.circuit_breaker.CircuitBreakerRegistry.get",
            return_value=MagicMock(),
        ):
            detector.observe(latency=999.0, error_rate=0.99)

        bus = get_event_bus()
        history = bus.get_history(event_type=AnomalyDetected)
        assert len(history) == 1
        event = history[0]
        assert isinstance(event, AnomalyDetected)
        assert event.agent_type == AgentType.WORKER.value
        assert event.anomaly_type == "ensemble"
        assert event.score == 2.0
        assert "cusum_latency" in event.triggered_detectors
        assert "cusum_error_rate" in event.triggered_detectors

    def test_event_bus_failure_does_not_raise(self):
        """A broken event bus must not propagate exceptions."""
        detector = EnsembleAnomalyDetector(agent_type=AgentType.INSPECTOR.value)
        detector._cusum_latency.detect = MagicMock(return_value=True)
        detector._cusum_error_rate.detect = MagicMock(return_value=True)

        with (
            patch(
                "vetinari.resilience.circuit_breaker.CircuitBreakerRegistry.get",
                return_value=MagicMock(),
            ),
            patch(
                "vetinari.events.EventBus.publish",
                side_effect=RuntimeError("bus exploded"),
            ),
        ):
            result = detector.observe(latency=999.0, error_rate=0.99)

        # Anomaly result still returned despite event bus failure
        assert isinstance(result, AnomalyResult)
        assert result.is_anomaly is True


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


class TestImports:
    def test_imports_succeed(self):
        """All public names introduced in US-201 must be importable."""
        from vetinari.analytics.anomaly import (
            CUSUMDetector,
            EnsembleAnomalyDetector,
        )
        from vetinari.events import AnomalyDetected
        from vetinari.resilience import CircuitBreaker

        # Verify the trip() method exists
        assert callable(getattr(CircuitBreaker, "trip", None))

    def test_vetinari_package_imports(self):
        """The top-level vetinari package must still import cleanly."""
        import types

        import vetinari

        assert isinstance(vetinari, types.ModuleType)
        assert hasattr(vetinari, "__version__")
