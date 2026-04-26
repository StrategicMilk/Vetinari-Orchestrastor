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

from __future__ import annotations

import os
from email import message_from_string
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_alert_record, make_alert_threshold, make_system_snapshot
from vetinari.constants import ALERT_SEND_TIMEOUT
from vetinari.dashboard.alerts import (
    DISPATCHERS,
    AlertCondition,
    AlertRecord,
    AlertSeverity,
    AlertThreshold,
    _dispatch_email,
    _dispatch_log,
    _dispatch_webhook,
    _resolve_metric,
    get_alert_engine,
    reset_alert_engine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alert_engine():
    """Provide a fresh AlertEngine for each test, reset afterwards."""
    reset_alert_engine()
    yield get_alert_engine()
    reset_alert_engine()


def _make_mock_api(latency=100.0, approval_rate=95.0, risk=0.2):
    """Build a parameterised DashboardAPI mock for tests that need varying metric values."""
    snapshot = MagicMock()
    snapshot.to_dict.return_value = make_system_snapshot(latency=latency, approval_rate=approval_rate, risk=risk)
    api = MagicMock()
    api.get_latest_metrics.return_value = snapshot
    return api


# ---------------------------------------------------------------------------
# AlertThreshold
# ---------------------------------------------------------------------------


class TestAlertThreshold:
    def test_creation_defaults(self) -> None:
        t = make_alert_threshold()
        assert t.name == "test-alert"
        assert t.condition == AlertCondition.GREATER_THAN
        assert t.severity == AlertSeverity.MEDIUM
        assert t.channels == ["log"]
        assert t.duration_seconds == 0

    def test_to_dict_keys(self) -> None:
        t = make_alert_threshold()
        d = t.to_dict()
        for key in ("name", "metric_key", "condition", "threshold_value", "severity", "channels", "duration_seconds"):
            assert key in d

    def test_to_dict_values(self) -> None:
        t = make_alert_threshold(
            condition=AlertCondition.LESS_THAN,
            severity=AlertSeverity.HIGH,
            threshold_value=50.0,
        )
        d = t.to_dict()
        assert d["condition"] == "lt"
        assert d["severity"] == "high"
        assert d["threshold_value"] == 50.0


# ---------------------------------------------------------------------------
# AlertRecord
# ---------------------------------------------------------------------------


class TestAlertRecord:
    def test_creation(self) -> None:
        t = make_alert_threshold()
        r = AlertRecord(threshold=t, current_value=999.9)
        assert r.current_value == 999.9
        assert isinstance(r.trigger_time, float)

    def test_to_dict(self) -> None:
        t = make_alert_threshold()
        r = AlertRecord(threshold=t, current_value=123.4, trigger_time=1700000000.0)
        d = r.to_dict()
        assert "threshold" in d
        assert d["current_value"] == 123.4
        assert d["trigger_time"] == 1700000000.0


# ---------------------------------------------------------------------------
# _resolve_metric helper
# ---------------------------------------------------------------------------


class TestResolveMetric:
    def test_resolve_nested_key(self) -> None:
        snap = make_system_snapshot(latency=250.0, approval_rate=80.0)
        val = _resolve_metric(snap, "adapters.average_latency_ms")
        assert val == pytest.approx(250.0)

    def test_resolve_plan_key(self) -> None:
        snap = make_system_snapshot(latency=250.0, approval_rate=80.0)
        val = _resolve_metric(snap, "plan.approval_rate")
        assert val == pytest.approx(80.0)

    def test_missing_top_key_returns_none(self) -> None:
        snap = make_system_snapshot(latency=250.0, approval_rate=80.0)
        assert _resolve_metric(snap, "nonexistent.key") is None

    def test_missing_nested_key_returns_none(self) -> None:
        snap = make_system_snapshot(latency=250.0, approval_rate=80.0)
        assert _resolve_metric(snap, "adapters.does_not_exist") is None

    def test_non_numeric_value_returns_none(self) -> None:
        snap = {"adapters": {"label": "hello"}}
        assert _resolve_metric(snap, "adapters.label") is None

    def test_integer_value_coerced_to_float(self) -> None:
        snap = {"adapters": {"total_requests": 42}}
        val = _resolve_metric(snap, "adapters.total_requests")
        assert isinstance(val, float)
        assert val == 42.0


# ---------------------------------------------------------------------------
# AlertEngine singleton
# ---------------------------------------------------------------------------


class TestAlertEngineSingleton:
    def test_get_alert_engine_returns_same_instance(self) -> None:
        reset_alert_engine()
        try:
            e1 = get_alert_engine()
            e2 = get_alert_engine()
            assert e1 is e2
        finally:
            reset_alert_engine()

    def test_reset_creates_new_instance(self) -> None:
        reset_alert_engine()
        try:
            e1 = get_alert_engine()
            reset_alert_engine()
            e2 = get_alert_engine()
            assert e1 is not e2
        finally:
            reset_alert_engine()


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------


class TestThresholdManagement:
    def test_register_threshold(self, alert_engine) -> None:
        alert_engine.register_threshold(make_alert_threshold(name="a"))
        assert len(alert_engine.list_thresholds()) == 1

    def test_register_duplicate_name_replaces(self, alert_engine) -> None:
        alert_engine.register_threshold(make_alert_threshold(name="a", threshold_value=100.0))
        alert_engine.register_threshold(make_alert_threshold(name="a", threshold_value=200.0))
        ts = alert_engine.list_thresholds()
        assert len(ts) == 1
        assert ts[0].threshold_value == 200.0

    def test_unregister_existing(self, alert_engine) -> None:
        alert_engine.register_threshold(make_alert_threshold(name="b"))
        result = alert_engine.unregister_threshold("b")
        assert result is True
        assert len(alert_engine.list_thresholds()) == 0

    def test_unregister_nonexistent_returns_false(self, alert_engine) -> None:
        result = alert_engine.unregister_threshold("missing")
        assert not result

    def test_clear_thresholds(self, alert_engine) -> None:
        alert_engine.register_threshold(make_alert_threshold(name="x"))
        alert_engine.register_threshold(make_alert_threshold(name="y"))
        alert_engine.clear_thresholds()
        assert len(alert_engine.list_thresholds()) == 0


# ---------------------------------------------------------------------------
# evaluate_all -- condition checks
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def test_no_thresholds_returns_empty(self, alert_engine) -> None:
        fired = alert_engine.evaluate_all(api=_make_mock_api())
        assert fired == []

    def test_greater_than_not_triggered(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=500.0,
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert len(fired) == 0

    def test_greater_than_triggered(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert len(fired) == 1
        assert fired[0].current_value == pytest.approx(100.0)

    def test_less_than_triggered(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="low-approval",
                metric_key="plan.approval_rate",
                condition=AlertCondition.LESS_THAN,
                threshold_value=50.0,
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api(approval_rate=30.0))
        assert len(fired) == 1

    def test_equals_triggered(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="exact-match",
                metric_key="plan.average_risk_score",
                condition=AlertCondition.EQUALS,
                threshold_value=0.5,
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api(risk=0.5))
        assert len(fired) == 1

    def test_missing_metric_key_skipped(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.nonexistent_field",
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api())
        assert len(fired) == 0

    def test_alert_fires_only_once_while_active(self, alert_engine) -> None:
        """Suppress re-firing while the condition remains triggered."""
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        api = _make_mock_api(latency=100.0)
        fired1 = alert_engine.evaluate_all(api=api)
        fired2 = alert_engine.evaluate_all(api=api)
        assert len(fired1) == 1
        assert len(fired2) == 0  # suppressed

    def test_alert_clears_when_condition_no_longer_met(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))  # fires
        assert len(alert_engine.get_active_alerts()) == 1

        alert_engine.evaluate_all(api=_make_mock_api(latency=10.0))  # clears
        assert len(alert_engine.get_active_alerts()) == 0

    def test_alert_refires_after_clear(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        api_high = _make_mock_api(latency=100.0)
        api_low = _make_mock_api(latency=10.0)

        alert_engine.evaluate_all(api=api_high)  # fires
        alert_engine.evaluate_all(api=api_low)  # clears
        fired = alert_engine.evaluate_all(api=api_high)  # should fire again
        assert len(fired) == 1


# ---------------------------------------------------------------------------
# Duration-based alerts
# ---------------------------------------------------------------------------


class TestDurationAlerts:
    def test_duration_not_yet_satisfied(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="dur-alert",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
                duration_seconds=60,
            )
        )
        fired = alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert len(fired) == 0
        assert len(alert_engine.get_active_alerts()) == 0

    def test_duration_satisfied_fires(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="dur-alert",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
                duration_seconds=5,
            )
        )
        api = _make_mock_api(latency=100.0)

        # First evaluation -- starts the duration timer
        alert_engine.evaluate_all(api=api)
        assert len(alert_engine.get_active_alerts()) == 0

        # Patch the start time to be 10 s ago so duration is satisfied
        alert_engine._duration_start["dur-alert"] -= 10

        fired = alert_engine.evaluate_all(api=api)
        assert len(fired) == 1

    def test_duration_resets_when_condition_clears(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="dur-alert",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
                duration_seconds=60,
            )
        )
        alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert "dur-alert" in alert_engine._duration_start

        # Condition clears -- duration tracker should be removed
        alert_engine.evaluate_all(api=_make_mock_api(latency=10.0))
        assert "dur-alert" not in alert_engine._duration_start


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_get_stats_initial(self, alert_engine) -> None:
        stats = alert_engine.get_stats()
        assert stats["registered_thresholds"] == 0
        assert stats["active_alerts"] == 0
        assert stats["total_fired"] == 0

    def test_get_stats_after_fire(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        stats = alert_engine.get_stats()
        assert stats["registered_thresholds"] == 1
        assert stats["active_alerts"] == 1
        assert stats["total_fired"] == 1

    def test_get_history_accumulates(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        api_high = _make_mock_api(latency=100.0)
        api_low = _make_mock_api(latency=10.0)

        alert_engine.evaluate_all(api=api_high)  # fire #1
        alert_engine.evaluate_all(api=api_low)  # clear
        alert_engine.evaluate_all(api=api_high)  # fire #2

        history = alert_engine.get_history()
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


class TestDispatchers:
    def test_log_dispatcher_called(self, alert_engine) -> None:
        with patch.dict(DISPATCHERS, {"log": MagicMock()}):
            alert_engine.register_threshold(
                make_alert_threshold(
                    channels=["log"],
                    metric_key="adapters.average_latency_ms",
                    condition=AlertCondition.GREATER_THAN,
                    threshold_value=50.0,
                )
            )
            alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
            DISPATCHERS["log"].assert_called_once()

    def test_email_dispatcher_called(self, alert_engine) -> None:
        with patch.dict(DISPATCHERS, {"email": MagicMock()}):
            alert_engine.register_threshold(
                make_alert_threshold(
                    name="email-alert",
                    channels=["email"],
                    metric_key="adapters.average_latency_ms",
                    condition=AlertCondition.GREATER_THAN,
                    threshold_value=50.0,
                )
            )
            alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
            DISPATCHERS["email"].assert_called_once()

    def test_unknown_channel_does_not_raise(self, alert_engine) -> None:
        alert_engine.register_threshold(
            make_alert_threshold(
                name="unknown-chan",
                channels=["slack"],  # not in DISPATCHERS
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
            )
        )
        # Should not raise -- alert fires but unknown channel is silently skipped
        alerts = alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))
        assert isinstance(alerts, list)

    def test_multiple_channels(self, alert_engine) -> None:
        mock_log = MagicMock()
        mock_email = MagicMock()
        with patch.dict(DISPATCHERS, {"log": mock_log, "email": mock_email}):
            alert_engine.register_threshold(
                make_alert_threshold(
                    name="multi-chan",
                    channels=["log", "email"],
                    metric_key="adapters.average_latency_ms",
                    condition=AlertCondition.GREATER_THAN,
                    threshold_value=50.0,
                )
            )
            alerts = alert_engine.evaluate_all(api=_make_mock_api(latency=100.0))

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.threshold.name == "multi-chan"
        assert alert.current_value == pytest.approx(100.0)
        assert alert_engine.get_active_alerts() == [alert]
        assert mock_log.call_args.args == (alert,)
        assert mock_email.call_args.args == (alert,)


# ---------------------------------------------------------------------------
# _dispatch_log
# ---------------------------------------------------------------------------


class TestDispatchLog:
    def test_dispatch_log_high_severity(self) -> None:
        alert = make_alert_record(threshold=make_alert_threshold(severity=AlertSeverity.HIGH))
        _dispatch_log(alert)
        assert alert.threshold.severity == AlertSeverity.HIGH

    def test_dispatch_log_medium_severity(self) -> None:
        alert = make_alert_record(threshold=make_alert_threshold(severity=AlertSeverity.MEDIUM))
        _dispatch_log(alert)
        assert alert.threshold.severity == AlertSeverity.MEDIUM


# ---------------------------------------------------------------------------
# _dispatch_email
# ---------------------------------------------------------------------------


class TestDispatchEmail:
    def test_email_skipped_without_env_vars(self) -> None:
        env = {"VETINARI_SMTP_HOST": "", "VETINARI_ALERT_FROM": "", "VETINARI_ALERT_TO": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("VETINARI_SMTP_HOST", None)
            os.environ.pop("VETINARI_ALERT_FROM", None)
            os.environ.pop("VETINARI_ALERT_TO", None)
            with patch("smtplib.SMTP") as mock_smtp:
                _dispatch_email(make_alert_record(current_value=600.0))
                mock_smtp.assert_not_called()

    def test_email_sent_successfully(self) -> None:
        env = {
            "VETINARI_SMTP_HOST": "smtp.test.com",
            "VETINARI_SMTP_PORT": "587",
            "VETINARI_ALERT_FROM": "from@test.com",
            "VETINARI_ALERT_TO": "to@test.com",
            "VETINARI_SMTP_USER": "user",
            "VETINARI_SMTP_PASS": "pass",
        }
        mock_smtp = MagicMock()
        alert = make_alert_record(current_value=600.0)
        with patch.dict(os.environ, env, clear=False):
            with patch("smtplib.SMTP", return_value=mock_smtp) as smtp_cls:
                mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
                mock_smtp.__exit__ = MagicMock(return_value=False)
                _dispatch_email(alert)

        assert smtp_cls.call_args.args == ("smtp.test.com", 587)
        assert smtp_cls.call_args.kwargs == {"timeout": 10}
        assert mock_smtp.ehlo.call_count == 2
        assert mock_smtp.starttls.call_count == 1
        assert mock_smtp.login.call_args.args == ("user", "pass")

        from_addr, recipients, raw_message = mock_smtp.sendmail.call_args.args
        assert from_addr == "from@test.com"
        assert recipients == ["to@test.com"]
        message = message_from_string(raw_message)
        assert message["Subject"] == "[Vetinari Alert] [MEDIUM] test-alert"
        assert message["From"] == "from@test.com"
        assert message["To"] == "to@test.com"

        body = message.get_payload(decode=True).decode(message.get_content_charset() or "us-ascii")
        assert "Alert: test-alert" in body
        assert "Metric: adapters.average_latency_ms" in body
        assert "Current value: 600" in body
        assert "Threshold (gt): 500" in body
        assert f"Triggered at: {alert.trigger_time}" in body

    def test_email_smtp_exception_caught(self) -> None:
        env = {
            "VETINARI_SMTP_HOST": "smtp.test.com",
            "VETINARI_SMTP_PORT": "587",
            "VETINARI_ALERT_FROM": "from@test.com",
            "VETINARI_ALERT_TO": "to@test.com",
        }
        alert = make_alert_record(current_value=600.0)
        with patch.dict(os.environ, env, clear=False):
            with (
                patch("smtplib.SMTP", side_effect=ConnectionError("refused")) as mock_smtp,
                patch("vetinari.dashboard.alerts.logger.exception") as mock_log_exception,
            ):
                result = _dispatch_email(alert)

        assert result is None
        assert mock_smtp.call_args.args == ("smtp.test.com", 587)
        assert mock_smtp.call_args.kwargs == {"timeout": 10}
        assert mock_log_exception.call_args.args == ("EMAIL: failed to send alert '%s'", alert.threshold.name)

    def test_email_without_credentials(self) -> None:
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
                _dispatch_email(make_alert_record(current_value=600.0))
                mock_smtp.login.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_webhook
# ---------------------------------------------------------------------------


class TestDispatchWebhook:
    def test_webhook_skipped_without_url(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETINARI_WEBHOOK_URL", None)
            with patch("vetinari.dashboard.alerts.create_session") as mock_create:
                _dispatch_webhook(make_alert_record(current_value=600.0))
                mock_create.assert_not_called()

    def test_webhook_posted_successfully(self) -> None:
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        alert = make_alert_record(current_value=600.0)
        with patch.dict(os.environ, {"VETINARI_WEBHOOK_URL": "http://hook.test/alert"}):
            with patch("vetinari.dashboard.alerts.create_session", return_value=mock_session) as mock_create_session:
                _dispatch_webhook(alert)

        assert mock_create_session.call_count == 1
        assert mock_session.post.call_args.args == ("http://hook.test/alert",)
        assert mock_session.post.call_args.kwargs == {
            "json": {
                "name": alert.threshold.name,
                "metric": alert.threshold.metric_key,
                "value": alert.current_value,
                "threshold": alert.threshold.threshold_value,
                "severity": alert.threshold.severity.value,
                "timestamp": alert.trigger_time,
            },
            "timeout": ALERT_SEND_TIMEOUT,
        }
        assert mock_resp.raise_for_status.call_count == 1

    def test_webhook_exception_caught(self) -> None:
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("down")
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch.dict(os.environ, {"VETINARI_WEBHOOK_URL": "http://hook.test/alert"}):
            with patch("vetinari.dashboard.alerts.create_session", return_value=mock_session):
                _dispatch_webhook(make_alert_record(current_value=600.0))
                assert mock_session.post.call_count >= 1


# ---------------------------------------------------------------------------
# AlertEngine -- _check_condition edge case
# ---------------------------------------------------------------------------


class TestCheckConditionEdge:
    def test_unknown_condition_returns_false(self, alert_engine) -> None:
        """A condition value not in the enum should return False."""
        t = make_alert_threshold()
        # Monkey-patch a bogus condition
        t.condition = "bogus"
        result = alert_engine._check_condition(t, 100.0)
        assert not result


# ---------------------------------------------------------------------------
# Duration -- elapsed not yet satisfied intermediate path
# ---------------------------------------------------------------------------


class TestDurationIntermediate:
    def test_duration_intermediate_not_satisfied(self, alert_engine) -> None:
        """Second evaluation with duration not yet elapsed returns no alerts."""
        alert_engine.register_threshold(
            make_alert_threshold(
                name="dur-mid",
                metric_key="adapters.average_latency_ms",
                condition=AlertCondition.GREATER_THAN,
                threshold_value=50.0,
                duration_seconds=120,
            )
        )
        api = _make_mock_api(latency=100.0)

        # First eval starts timer
        alert_engine.evaluate_all(api=api)
        # Second eval -- timer started but not enough time passed
        fired = alert_engine.evaluate_all(api=api)
        assert len(fired) == 0
