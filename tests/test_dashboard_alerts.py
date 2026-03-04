"""
Tests for vetinari/dashboard/alerts.py  (Phase 4 Step 2)

Coverage:
    - AlertThreshold dataclass creation and serialisation
    - AlertRecord dataclass creation and serialisation
    - _resolve_metric helper
    - AlertCondition checks (gt, lt, eq)
    - Singleton behaviour
    - register / unregister / clear thresholds
    - evaluate_all: no data, condition not met, condition met
    - Suppression (alert fires once until cleared)
    - Duration-based firing
    - Alert cleared when condition no longer holds
    - get_active_alerts / get_history / get_stats
    - Dispatcher routing (log, email, webhook, unknown)
    - reset_alert_engine
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from vetinari.dashboard.alerts import (
    AlertCondition,
    AlertEngine,
    AlertRecord,
    AlertSeverity,
    AlertThreshold,
    DISPATCHERS,
    _resolve_metric,
    get_alert_engine,
    reset_alert_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_threshold(
    name="test-alert",
    metric_key="adapters.average_latency_ms",
    condition=AlertCondition.GREATER_THAN,
    threshold_value=500.0,
    severity=AlertSeverity.MEDIUM,
    channels=None,
    duration_seconds=0,
) -> AlertThreshold:
    return AlertThreshold(
        name=name,
        metric_key=metric_key,
        condition=condition,
        threshold_value=threshold_value,
        severity=severity,
        channels=channels or ["log"],
        duration_seconds=duration_seconds,
    )


def _make_snapshot_dict(latency=100.0, approval_rate=95.0, risk=0.2):
    """Return a dict shaped like MetricsSnapshot.to_dict()."""
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "uptime_ms": 1000.0,
        "adapters": {
            "total_providers": 1,
            "total_requests": 10,
            "total_successful": 9,
            "total_failed": 1,
            "average_latency_ms": latency,
            "total_tokens_used": 500,
            "providers": {},
        },
        "memory": {"backends": {}},
        "plan": {
            "total_decisions": 5,
            "approved": 4,
            "rejected": 1,
            "auto_approved": 2,
            "approval_rate": approval_rate,
            "average_risk_score": risk,
            "average_approval_time_ms": 120.0,
        },
    }


def _make_mock_api(latency=100.0, approval_rate=95.0, risk=0.2):
    """Return a MagicMock DashboardAPI whose get_latest_metrics returns a mock snapshot."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = _make_snapshot_dict(latency, approval_rate, risk)
    api = MagicMock()
    api.get_latest_metrics.return_value = snapshot
    return api


# ---------------------------------------------------------------------------
# AlertThreshold
# ---------------------------------------------------------------------------

class TestAlertThreshold(unittest.TestCase):

    def test_creation_defaults(self):
        t = _make_threshold()
        self.assertEqual(t.name, "test-alert")
        self.assertEqual(t.condition, AlertCondition.GREATER_THAN)
        self.assertEqual(t.severity, AlertSeverity.MEDIUM)
        self.assertEqual(t.channels, ["log"])
        self.assertEqual(t.duration_seconds, 0)

    def test_to_dict_keys(self):
        t = _make_threshold()
        d = t.to_dict()
        for key in ("name", "metric_key", "condition", "threshold_value",
                    "severity", "channels", "duration_seconds"):
            self.assertIn(key, d)

    def test_to_dict_values(self):
        t = _make_threshold(condition=AlertCondition.LESS_THAN,
                            severity=AlertSeverity.HIGH,
                            threshold_value=50.0)
        d = t.to_dict()
        self.assertEqual(d["condition"], "lt")
        self.assertEqual(d["severity"], "high")
        self.assertEqual(d["threshold_value"], 50.0)


# ---------------------------------------------------------------------------
# AlertRecord
# ---------------------------------------------------------------------------

class TestAlertRecord(unittest.TestCase):

    def test_creation(self):
        t = _make_threshold()
        r = AlertRecord(threshold=t, current_value=999.9)
        self.assertEqual(r.current_value, 999.9)
        self.assertIsInstance(r.trigger_time, float)

    def test_to_dict(self):
        t = _make_threshold()
        r = AlertRecord(threshold=t, current_value=123.4, trigger_time=1700000000.0)
        d = r.to_dict()
        self.assertIn("threshold", d)
        self.assertEqual(d["current_value"], 123.4)
        self.assertEqual(d["trigger_time"], 1700000000.0)


# ---------------------------------------------------------------------------
# _resolve_metric helper
# ---------------------------------------------------------------------------

class TestResolveMetric(unittest.TestCase):

    def setUp(self):
        self.snap = _make_snapshot_dict(latency=250.0, approval_rate=80.0)

    def test_resolve_nested_key(self):
        val = _resolve_metric(self.snap, "adapters.average_latency_ms")
        self.assertAlmostEqual(val, 250.0)

    def test_resolve_plan_key(self):
        val = _resolve_metric(self.snap, "plan.approval_rate")
        self.assertAlmostEqual(val, 80.0)

    def test_missing_top_key_returns_none(self):
        self.assertIsNone(_resolve_metric(self.snap, "nonexistent.key"))

    def test_missing_nested_key_returns_none(self):
        self.assertIsNone(_resolve_metric(self.snap, "adapters.does_not_exist"))

    def test_non_numeric_value_returns_none(self):
        snap = {"adapters": {"label": "hello"}}
        self.assertIsNone(_resolve_metric(snap, "adapters.label"))

    def test_integer_value_coerced_to_float(self):
        snap = {"adapters": {"total_requests": 42}}
        val = _resolve_metric(snap, "adapters.total_requests")
        self.assertIsInstance(val, float)
        self.assertEqual(val, 42.0)


# ---------------------------------------------------------------------------
# AlertEngine singleton
# ---------------------------------------------------------------------------

class TestAlertEngineSingleton(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_get_alert_engine_returns_same_instance(self):
        e1 = get_alert_engine()
        e2 = get_alert_engine()
        self.assertIs(e1, e2)

    def test_reset_creates_new_instance(self):
        e1 = get_alert_engine()
        reset_alert_engine()
        e2 = get_alert_engine()
        self.assertIsNot(e1, e2)


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------

class TestThresholdManagement(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_register_threshold(self):
        self.engine.register_threshold(_make_threshold(name="a"))
        self.assertEqual(len(self.engine.list_thresholds()), 1)

    def test_register_duplicate_name_replaces(self):
        self.engine.register_threshold(_make_threshold(name="a", threshold_value=100.0))
        self.engine.register_threshold(_make_threshold(name="a", threshold_value=200.0))
        ts = self.engine.list_thresholds()
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0].threshold_value, 200.0)

    def test_unregister_existing(self):
        self.engine.register_threshold(_make_threshold(name="b"))
        result = self.engine.unregister_threshold("b")
        self.assertTrue(result)
        self.assertEqual(len(self.engine.list_thresholds()), 0)

    def test_unregister_nonexistent_returns_false(self):
        result = self.engine.unregister_threshold("missing")
        self.assertFalse(result)

    def test_clear_thresholds(self):
        self.engine.register_threshold(_make_threshold(name="x"))
        self.engine.register_threshold(_make_threshold(name="y"))
        self.engine.clear_thresholds()
        self.assertEqual(len(self.engine.list_thresholds()), 0)


# ---------------------------------------------------------------------------
# evaluate_all – condition checks
# ---------------------------------------------------------------------------

class TestEvaluateAll(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_no_thresholds_returns_empty(self):
        fired = self.engine.evaluate_all(api=_make_mock_api())
        self.assertEqual(fired, [])

    def test_greater_than_not_triggered(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=500.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        self.assertEqual(len(fired), 0)

    def test_greater_than_triggered(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        self.assertEqual(len(fired), 1)
        self.assertAlmostEqual(fired[0].current_value, 100.0)

    def test_less_than_triggered(self):
        self.engine.register_threshold(_make_threshold(
            name="low-approval",
            metric_key="plan.approval_rate",
            condition=AlertCondition.LESS_THAN,
            threshold_value=50.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(approval_rate=30.0))
        self.assertEqual(len(fired), 1)

    def test_equals_triggered(self):
        self.engine.register_threshold(_make_threshold(
            name="exact-match",
            metric_key="plan.average_risk_score",
            condition=AlertCondition.EQUALS,
            threshold_value=0.5,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(risk=0.5))
        self.assertEqual(len(fired), 1)

    def test_missing_metric_key_skipped(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.nonexistent_field",
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api())
        self.assertEqual(len(fired), 0)

    def test_alert_fires_only_once_while_active(self):
        """Suppress re-firing while the condition remains triggered."""
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        api = _make_mock_api(latency=100.0)
        fired1 = self.engine.evaluate_all(api=api)
        fired2 = self.engine.evaluate_all(api=api)
        self.assertEqual(len(fired1), 1)
        self.assertEqual(len(fired2), 0)  # suppressed

    def test_alert_clears_when_condition_no_longer_met(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))  # fires
        self.assertEqual(len(self.engine.get_active_alerts()), 1)

        self.engine.evaluate_all(api=_make_mock_api(latency=10.0))   # clears
        self.assertEqual(len(self.engine.get_active_alerts()), 0)

    def test_alert_refires_after_clear(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        api_high = _make_mock_api(latency=100.0)
        api_low  = _make_mock_api(latency=10.0)

        self.engine.evaluate_all(api=api_high)  # fires
        self.engine.evaluate_all(api=api_low)   # clears
        fired = self.engine.evaluate_all(api=api_high)  # should fire again
        self.assertEqual(len(fired), 1)


# ---------------------------------------------------------------------------
# Duration-based alerts
# ---------------------------------------------------------------------------

class TestDurationAlerts(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_duration_not_yet_satisfied(self):
        self.engine.register_threshold(_make_threshold(
            name="dur-alert",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            duration_seconds=60,   # 60 s — won't pass in a unit test
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        self.assertEqual(len(fired), 0)
        self.assertEqual(len(self.engine.get_active_alerts()), 0)

    def test_duration_satisfied_fires(self):
        self.engine.register_threshold(_make_threshold(
            name="dur-alert",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            duration_seconds=5,
        ))
        api = _make_mock_api(latency=100.0)

        # First evaluation — starts the duration timer
        self.engine.evaluate_all(api=api)
        self.assertEqual(len(self.engine.get_active_alerts()), 0)

        # Patch the start time to be 10 s ago so duration is satisfied
        self.engine._duration_start["dur-alert"] -= 10

        fired = self.engine.evaluate_all(api=api)
        self.assertEqual(len(fired), 1)

    def test_duration_resets_when_condition_clears(self):
        self.engine.register_threshold(_make_threshold(
            name="dur-alert",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            duration_seconds=60,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        self.assertIn("dur-alert", self.engine._duration_start)

        # Condition clears — duration tracker should be removed
        self.engine.evaluate_all(api=_make_mock_api(latency=10.0))
        self.assertNotIn("dur-alert", self.engine._duration_start)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

class TestIntrospection(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_get_stats_initial(self):
        stats = self.engine.get_stats()
        self.assertEqual(stats["registered_thresholds"], 0)
        self.assertEqual(stats["active_alerts"], 0)
        self.assertEqual(stats["total_fired"], 0)

    def test_get_stats_after_fire(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        stats = self.engine.get_stats()
        self.assertEqual(stats["registered_thresholds"], 1)
        self.assertEqual(stats["active_alerts"], 1)
        self.assertEqual(stats["total_fired"], 1)

    def test_get_history_accumulates(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        api_high = _make_mock_api(latency=100.0)
        api_low  = _make_mock_api(latency=10.0)

        self.engine.evaluate_all(api=api_high)  # fire #1
        self.engine.evaluate_all(api=api_low)   # clear
        self.engine.evaluate_all(api=api_high)  # fire #2

        history = self.engine.get_history()
        self.assertEqual(len(history), 2)


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------

class TestDispatchers(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_log_dispatcher_called(self):
        with patch.dict(DISPATCHERS, {"log": MagicMock()}):
            self.engine.register_threshold(_make_threshold(
                channels=["log"],
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            ))
            self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
            DISPATCHERS["log"].assert_called_once()

    def test_email_dispatcher_called(self):
        with patch.dict(DISPATCHERS, {"email": MagicMock()}):
            self.engine.register_threshold(_make_threshold(
                name="email-alert",
                channels=["email"],
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            ))
            self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
            DISPATCHERS["email"].assert_called_once()

    def test_unknown_channel_does_not_raise(self):
        self.engine.register_threshold(_make_threshold(
            name="unknown-chan",
            channels=["slack"],   # not in DISPATCHERS
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        # Should not raise
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))

    def test_multiple_channels(self):
        mock_log = MagicMock()
        mock_email = MagicMock()
        with patch.dict(DISPATCHERS, {"log": mock_log, "email": mock_email}):
            self.engine.register_threshold(_make_threshold(
                name="multi-chan",
                channels=["log", "email"],
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            ))
            self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
            mock_log.assert_called_once()
            mock_email.assert_called_once()


if __name__ == "__main__":
    unittest.main()
