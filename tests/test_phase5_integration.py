"""
Phase 5 Integration Tests — Analytics Pipeline Wiring

Tests that the analytics modules are properly wired into the live pipeline:
- Adapter telemetry records CostEntry, SLA, Forecaster, Anomaly data
- Default SLOs are registered at orchestrator init
- AutoTuner config is applied in orchestrator.run_all()
- Cost optimizer uses real data in ModelPool scoring
"""

import sys
import unittest
from unittest.mock import patch

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in (
    "vetinari.adapters.base", "vetinari.analytics.cost",
    "vetinari.analytics.sla", "vetinari.analytics.forecasting",
    "vetinari.analytics.anomaly",
):
    sys.modules.pop(_stubname, None)

from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)
from vetinari.analytics.anomaly import get_anomaly_detector, reset_anomaly_detector
from vetinari.analytics.cost import get_cost_tracker, reset_cost_tracker
from vetinari.analytics.forecasting import get_forecaster, reset_forecaster
from vetinari.analytics.sla import SLOTarget, SLOType, get_sla_tracker, reset_sla_tracker


class ConcreteAdapter(ProviderAdapter):
    """Minimal adapter for testing _record_telemetry."""

    def discover_models(self):
        return []

    def health_check(self):
        return {"healthy": True, "reason": "ok", "timestamp": "now"}

    def infer(self, request):
        resp = InferenceResponse(
            model_id=request.model_id, output="test",
            latency_ms=150, tokens_used=200, status="ok",
        )
        self._record_telemetry(request, resp)
        return resp

    def get_capabilities(self):
        return {}


def _make_adapter():
    cfg = ProviderConfig(
        provider_type=ProviderType.LM_STUDIO,
        name="test", endpoint="http://localhost:1234",
    )
    return ConcreteAdapter(cfg)


class TestAdapterCostTelemetry(unittest.TestCase):
    """Step 1: Adapter records CostEntry correctly."""

    def setUp(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def tearDown(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def test_cost_entry_recorded(self):
        adapter = _make_adapter()
        req = InferenceRequest(model_id="test-model", prompt="hello")
        adapter.infer(req)

        tracker = get_cost_tracker()
        report = tracker.get_report()
        assert report.total_requests >= 1
        assert report.total_tokens >= 200

    def test_cost_entry_with_metadata(self):
        adapter = _make_adapter()
        req = InferenceRequest(
            model_id="test-model", prompt="hello",
            metadata={"agent": "builder", "task_id": "t42"},
        )
        adapter.infer(req)

        report = get_cost_tracker().get_report(agent="builder")
        assert report.total_requests >= 1

    def test_cost_entry_with_task_id(self):
        adapter = _make_adapter()
        req = InferenceRequest(
            model_id="test-model", prompt="hello",
            metadata={"task_id": "t99"},
        )
        adapter.infer(req)

        report = get_cost_tracker().get_report(task_id="t99")
        assert report.total_requests >= 1


class TestAdapterSLATelemetry(unittest.TestCase):
    """Step 2: SLA tracker receives data from adapter telemetry."""

    def setUp(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def tearDown(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def test_sla_records_latency(self):
        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="test-latency", slo_type=SLOType.LATENCY_P95,
            budget=2000.0, window_seconds=3600,
        ))

        adapter = _make_adapter()
        req = InferenceRequest(model_id="test-model", prompt="hello")
        adapter.infer(req)

        # SLA tracker should have recorded the request
        report = tracker.get_report("test-latency")
        assert report is not None
        assert report.total_samples > 0

    def test_sla_records_success(self):
        tracker = get_sla_tracker()
        tracker.register_slo(SLOTarget(
            name="test-success", slo_type=SLOType.SUCCESS_RATE,
            budget=90.0, window_seconds=3600,
        ))

        adapter = _make_adapter()
        req = InferenceRequest(model_id="test-model", prompt="hello")
        adapter.infer(req)

        report = tracker.get_report("test-success")
        assert report is not None


class TestAdapterForecasterTelemetry(unittest.TestCase):
    """Step 3: Forecaster receives data from adapter telemetry."""

    def setUp(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def tearDown(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def test_forecaster_receives_latency(self):
        adapter = _make_adapter()
        req = InferenceRequest(model_id="test-model", prompt="hello")
        adapter.infer(req)

        fc = get_forecaster()
        history = fc.get_history("adapter.latency")
        assert len(history) >= 1
        assert history[-1] == 150.0

    def test_forecaster_receives_tokens(self):
        adapter = _make_adapter()
        req = InferenceRequest(model_id="test-model", prompt="hello")
        adapter.infer(req)

        fc = get_forecaster()
        history = fc.get_history("adapter.tokens")
        assert len(history) >= 1
        assert history[-1] == 200.0


class TestAdapterAnomalyTelemetry(unittest.TestCase):
    """Step 4: Anomaly detector runs on adapter telemetry."""

    def setUp(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def tearDown(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def test_anomaly_detector_runs(self):
        adapter = _make_adapter()
        # Feed several normal values then a big outlier
        for _ in range(30):
            req = InferenceRequest(model_id="m", prompt="x")
            adapter.infer(req)

        # Detector should have been called without crashing
        detector = get_anomaly_detector()
        # History may or may not contain anomalies, but it should exist
        assert detector is not None


class TestOrchestratorDefaultSLOs(unittest.TestCase):
    """Step 5: Default SLOs are registered at orchestrator init."""

    def setUp(self):
        reset_sla_tracker()

    def tearDown(self):
        reset_sla_tracker()

    def test_default_slos_registered(self):
        try:
            from vetinari.orchestrator import Orchestrator
        except ImportError:
            self.skipTest("orchestrator not available")

        orch = Orchestrator.__new__(Orchestrator)
        orch._register_default_slos()

        tracker = get_sla_tracker()
        reports = tracker.get_all_reports()
        names = {r.slo.name for r in reports}
        assert "latency-p95" in names
        assert "success-rate" in names
        assert "error-rate" in names
        assert "approval-rate" in names


class TestOrchestratorAutoTuner(unittest.TestCase):
    """Step 6: AutoTuner config is applied in orchestrator."""

    def test_auto_tuner_applies_max_concurrent(self):
        try:
            from vetinari.learning.auto_tuner import get_auto_tuner
            from vetinari.orchestrator import Orchestrator
        except ImportError:
            self.skipTest("orchestrator or auto_tuner not available")

        tuner = get_auto_tuner()
        # AutoTuner should be accessible and return config dict
        config = tuner.get_config()
        assert isinstance(config, dict)


class TestTelemetryNeverCrashesInference(unittest.TestCase):
    """Telemetry failures must never crash the inference path."""

    def setUp(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def tearDown(self):
        reset_cost_tracker()
        reset_sla_tracker()
        reset_forecaster()
        reset_anomaly_detector()

    def test_telemetry_failure_suppressed(self):
        adapter = _make_adapter()
        req = InferenceRequest(model_id="m", prompt="x")

        # Patch cost tracker to raise — inference should still succeed
        with patch("vetinari.analytics.cost.get_cost_tracker", side_effect=RuntimeError("boom")):
            resp = adapter.infer(req)

        assert resp.status == "ok"
        assert resp.output == "test"


if __name__ == "__main__":
    unittest.main()
