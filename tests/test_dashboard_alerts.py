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

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from vetinari.dashboard.alerts import (
    DISPATCHERS,
    AlertCondition,
    AlertRecord,
    AlertSeverity,
    AlertThreshold,
    DISPATCHERS,
    _dispatch_email,
    _dispatch_log,
    _dispatch_webhook,
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
        assert t.name == "test-alert"
        assert t.condition == AlertCondition.GREATER_THAN
        assert t.severity == AlertSeverity.MEDIUM
        assert t.channels == ["log"]
        assert t.duration_seconds == 0

    def test_to_dict_keys(self):
        t = _make_threshold()
        d = t.to_dict()
        for key in ("name", "metric_key", "condition", "threshold_value",
                    "severity", "channels", "duration_seconds"):
            assert key in d

    def test_to_dict_values(self):
        t = _make_threshold(condition=AlertCondition.LESS_THAN,
                            severity=AlertSeverity.HIGH,
                            threshold_value=50.0)
        d = t.to_dict()
        assert d["condition"] == "lt"
        assert d["severity"] == "high"
        assert d["threshold_value"] == 50.0


# ---------------------------------------------------------------------------
# AlertRecord
# ---------------------------------------------------------------------------

class TestAlertRecord(unittest.TestCase):

    def test_creation(self):
        t = _make_threshold()
        r = AlertRecord(threshold=t, current_value=999.9)
        assert r.current_value == 999.9
        assert isinstance(r.trigger_time, float)

    def test_to_dict(self):
        t = _make_threshold()
        r = AlertRecord(threshold=t, current_value=123.4, trigger_time=1700000000.0)
        d = r.to_dict()
        assert "threshold" in d
        assert d["current_value"] == 123.4
        assert d["trigger_time"] == 1700000000.0


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
        assert _resolve_metric(self.snap, "nonexistent.key") is None

    def test_missing_nested_key_returns_none(self):
        assert _resolve_metric(self.snap, "adapters.does_not_exist") is None

    def test_non_numeric_value_returns_none(self):
        snap = {"adapters": {"label": "hello"}}
        assert _resolve_metric(snap, "adapters.label") is None

    def test_integer_value_coerced_to_float(self):
        snap = {"adapters": {"total_requests": 42}}
        val = _resolve_metric(snap, "adapters.total_requests")
        assert isinstance(val, float)
        assert val == 42.0


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
        assert e1 is e2

    def test_reset_creates_new_instance(self):
        e1 = get_alert_engine()
        reset_alert_engine()
        e2 = get_alert_engine()
        assert e1 is not e2


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
        assert len(self.engine.list_thresholds()) == 1

    def test_register_duplicate_name_replaces(self):
        self.engine.register_threshold(_make_threshold(name="a", threshold_value=100.0))
        self.engine.register_threshold(_make_threshold(name="a", threshold_value=200.0))
        ts = self.engine.list_thresholds()
        assert len(ts) == 1
        assert ts[0].threshold_value == 200.0

    def test_unregister_existing(self):
        self.engine.register_threshold(_make_threshold(name="b"))
        result = self.engine.unregister_threshold("b")
        assert result
        assert len(self.engine.list_thresholds()) == 0

    def test_unregister_nonexistent_returns_false(self):
        result = self.engine.unregister_threshold("missing")
        assert not result

    def test_clear_thresholds(self):
        self.engine.register_threshold(_make_threshold(name="x"))
        self.engine.register_threshold(_make_threshold(name="y"))
        self.engine.clear_thresholds()
        assert len(self.engine.list_thresholds()) == 0


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
        assert fired == []

    def test_greater_than_not_triggered(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=500.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert len(fired) == 0

    def test_greater_than_triggered(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert len(fired) == 1
        self.assertAlmostEqual(fired[0].current_value, 100.0)

    def test_less_than_triggered(self):
        self.engine.register_threshold(_make_threshold(
            name="low-approval",
            metric_key="plan.approval_rate",
            condition=AlertCondition.LESS_THAN,
            threshold_value=50.0,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(approval_rate=30.0))
        assert len(fired) == 1

    def test_equals_triggered(self):
        self.engine.register_threshold(_make_threshold(
            name="exact-match",
            metric_key="plan.average_risk_score",
            condition=AlertCondition.EQUALS,
            threshold_value=0.5,
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api(risk=0.5))
        assert len(fired) == 1

    def test_missing_metric_key_skipped(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.nonexistent_field",
        ))
        fired = self.engine.evaluate_all(api=_make_mock_api())
        assert len(fired) == 0

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
        assert len(fired1) == 1
        assert len(fired2) == 0  # suppressed

    def test_alert_clears_when_condition_no_longer_met(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))  # fires
        assert len(self.engine.get_active_alerts()) == 1

        self.engine.evaluate_all(api=_make_mock_api(latency=10.0))   # clears
        assert len(self.engine.get_active_alerts()) == 0

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
        assert len(fired) == 1


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
        assert len(fired) == 0
        assert len(self.engine.get_active_alerts()) == 0

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
        assert len(self.engine.get_active_alerts()) == 0

        # Patch the start time to be 10 s ago so duration is satisfied
        self.engine._duration_start["dur-alert"] -= 10

        fired = self.engine.evaluate_all(api=api)
        assert len(fired) == 1

    def test_duration_resets_when_condition_clears(self):
        self.engine.register_threshold(_make_threshold(
            name="dur-alert",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            duration_seconds=60,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert "dur-alert" in self.engine._duration_start

        # Condition clears — duration tracker should be removed
        self.engine.evaluate_all(api=_make_mock_api(latency=10.0))
        assert "dur-alert" not in self.engine._duration_start


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
        assert stats["registered_thresholds"] == 0
        assert stats["active_alerts"] == 0
        assert stats["total_fired"] == 0

    def test_get_stats_after_fire(self):
        self.engine.register_threshold(_make_threshold(
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
        ))
        self.engine.evaluate_all(api=_make_mock_api(latency=100.0))
        stats = self.engine.get_stats()
        assert stats["registered_thresholds"] == 1
        assert stats["active_alerts"] == 1
        assert stats["total_fired"] == 1

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
        assert len(history) == 2


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


# ---------------------------------------------------------------------------
# _dispatch_log
# ---------------------------------------------------------------------------

class TestDispatchLog(unittest.TestCase):

    def _make_alert(self, severity=AlertSeverity.HIGH):
        t = _make_threshold(severity=severity)
        return AlertRecord(threshold=t, current_value=999.0, trigger_time=1700000000.0)

    def test_dispatch_log_high_severity(self):
        alert = self._make_alert(AlertSeverity.HIGH)
        # Should not raise
        _dispatch_log(alert)

    def test_dispatch_log_medium_severity(self):
        alert = self._make_alert(AlertSeverity.MEDIUM)
        _dispatch_log(alert)


# ---------------------------------------------------------------------------
# _dispatch_email
# ---------------------------------------------------------------------------

class TestDispatchEmail(unittest.TestCase):

    def _make_alert(self):
        t = _make_threshold(severity=AlertSeverity.HIGH)
        return AlertRecord(threshold=t, current_value=600.0, trigger_time=1700000000.0)

    def test_email_skipped_without_env_vars(self):
        env = {"VETINARI_SMTP_HOST": "", "VETINARI_ALERT_FROM": "", "VETINARI_ALERT_TO": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("VETINARI_SMTP_HOST", None)
            os.environ.pop("VETINARI_ALERT_FROM", None)
            os.environ.pop("VETINARI_ALERT_TO", None)
            # Should not raise, just log and return
            _dispatch_email(self._make_alert())

    def test_email_sent_successfully(self):
        env = {
            "VETINARI_SMTP_HOST": "smtp.test.com",
            "VETINARI_SMTP_PORT": "587",
            "VETINARI_ALERT_FROM": "from@test.com",
            "VETINARI_ALERT_TO": "to@test.com",
            "VETINARI_SMTP_USER": "user",
            "VETINARI_SMTP_PASS": "pass",
        }
        mock_smtp = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            with patch("smtplib.SMTP", return_value=mock_smtp):
                mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
                mock_smtp.__exit__ = MagicMock(return_value=False)
                _dispatch_email(self._make_alert())
                mock_smtp.sendmail.assert_called_once()

    def test_email_smtp_exception_caught(self):
        env = {
            "VETINARI_SMTP_HOST": "smtp.test.com",
            "VETINARI_SMTP_PORT": "587",
            "VETINARI_ALERT_FROM": "from@test.com",
            "VETINARI_ALERT_TO": "to@test.com",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("smtplib.SMTP", side_effect=ConnectionError("refused")):
                # Should not raise
                _dispatch_email(self._make_alert())

    def test_email_without_credentials(self):
        env = {
            "VETINARI_SMTP_HOST": "smtp.test.com",
            "VETINARI_SMTP_PORT": "587",
            "VETINARI_ALERT_FROM": "from@test.com",
            "VETINARI_ALERT_TO": "to@test.com",
        }
        mock_smtp = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("VETINARI_SMTP_USER", None)
            os.environ.pop("VETINARI_SMTP_PASS", None)
            with patch("smtplib.SMTP", return_value=mock_smtp):
                mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
                mock_smtp.__exit__ = MagicMock(return_value=False)
                _dispatch_email(self._make_alert())
                mock_smtp.login.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_webhook
# ---------------------------------------------------------------------------

class TestDispatchWebhook(unittest.TestCase):

    def _make_alert(self):
        t = _make_threshold(severity=AlertSeverity.MEDIUM)
        return AlertRecord(threshold=t, current_value=600.0, trigger_time=1700000000.0)

    def test_webhook_skipped_without_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETINARI_WEBHOOK_URL", None)
            # Should not raise
            _dispatch_webhook(self._make_alert())

    def test_webhook_posted_successfully(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()
        with patch.dict(os.environ, {"VETINARI_WEBHOOK_URL": "http://hook.test/alert"}):
            with patch("vetinari.dashboard.alerts.requests.post", return_value=mock_resp) as mock_post:
                _dispatch_webhook(self._make_alert())
                mock_post.assert_called_once()

    def test_webhook_exception_caught(self):
        with patch.dict(os.environ, {"VETINARI_WEBHOOK_URL": "http://hook.test/alert"}):
            with patch("vetinari.dashboard.alerts.requests.post", side_effect=ConnectionError("down")):
                # Should not raise
                _dispatch_webhook(self._make_alert())


# ---------------------------------------------------------------------------
# AlertEngine — _check_condition edge case
# ---------------------------------------------------------------------------

class TestCheckConditionEdge(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_unknown_condition_returns_false(self):
        """A condition value not in the enum should return False."""
        t = _make_threshold()
        # Monkey-patch a bogus condition
        t.condition = "bogus"
        result = self.engine._check_condition(t, 100.0)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Duration — elapsed not yet satisfied intermediate path
# ---------------------------------------------------------------------------

class TestDurationIntermediate(unittest.TestCase):

    def setUp(self):
        reset_alert_engine()
        self.engine = get_alert_engine()

    def tearDown(self):
        reset_alert_engine()

    def test_duration_intermediate_not_satisfied(self):
        """Second evaluation with duration not yet elapsed returns no alerts."""
        self.engine.register_threshold(_make_threshold(
            name="dur-mid",
            metric_key="adapters.average_latency_ms",
            condition=AlertCondition.GREATER_THAN,
            threshold_value=50.0,
            duration_seconds=120,
        ))
        api = _make_mock_api(latency=100.0)

        # First eval starts timer
        self.engine.evaluate_all(api=api)
        # Second eval — timer started but not enough time passed
        fired = self.engine.evaluate_all(api=api)
        self.assertEqual(len(fired), 0)


if __name__ == "__main__":
    unittest.main()
