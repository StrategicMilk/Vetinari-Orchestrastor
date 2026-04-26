"""Tests for vetinari.notifications.webhook — Discord/Slack/generic payload formats and retry."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from tests.factories import make_notification
from vetinari.notifications.manager import Notification, NotificationPriority
from vetinari.notifications.webhook import WebhookConfig, WebhookNotifier


def _write_webhook_config(tmp_path: Path, webhooks: list[dict]) -> Path:
    """Write a notifications.yaml to tmp_path and return its path."""
    config_path = tmp_path / "notifications.yaml"
    config_path.write_text(yaml.dump({"webhooks": webhooks}), encoding="utf-8")
    return config_path


# -- Format tests -------------------------------------------------------------


class TestPayloadFormats:
    """Each format function returns the expected top-level structure."""

    def test_format_discord_returns_embeds_key(self, tmp_path: Path) -> None:
        """_format_discord() produces a dict with 'embeds' key."""
        notifier = WebhookNotifier(config_path=tmp_path / "missing.yaml")
        result = notifier._format_discord([make_notification(action_type="build")])
        assert "embeds" in result
        assert isinstance(result["embeds"], list)

    def test_format_slack_returns_blocks_key(self, tmp_path: Path) -> None:
        """_format_slack() produces a dict with 'blocks' key."""
        notifier = WebhookNotifier(config_path=tmp_path / "missing.yaml")
        result = notifier._format_slack([make_notification(action_type="build")])
        assert "blocks" in result
        assert isinstance(result["blocks"], list)

    def test_format_generic_returns_notifications_key(self, tmp_path: Path) -> None:
        """_format_generic() produces a dict with 'notifications' key."""
        notifier = WebhookNotifier(config_path=tmp_path / "missing.yaml")
        result = notifier._format_generic([make_notification(action_type="build")])
        assert "notifications" in result
        assert len(result["notifications"]) == 1


# -- Event filtering ----------------------------------------------------------


class TestEventFiltering:
    """Webhook-specific event lists are respected."""

    def test_notification_filtered_out_by_event_list(self, tmp_path: Path) -> None:
        """A notification whose action_type is not in events is not delivered."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://example.com/hook", "format": "generic", "events": ["security_alert"]}],
        )
        notifier = WebhookNotifier(config_path=config_path)
        webhook = notifier._webhooks[0]
        notification = make_notification(action_type="build_complete")
        filtered = notifier._filter_by_events([notification], webhook)
        assert filtered == []

    def test_notification_passes_event_filter(self, tmp_path: Path) -> None:
        """A notification matching the event list is included."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://example.com/hook", "format": "generic", "events": ["build_complete"]}],
        )
        notifier = WebhookNotifier(config_path=config_path)
        webhook = notifier._webhooks[0]
        notification = make_notification(action_type="build_complete")
        filtered = notifier._filter_by_events([notification], webhook)
        assert len(filtered) == 1

    def test_none_events_means_all_pass(self, tmp_path: Path) -> None:
        """A webhook with events=None receives all notification types."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://example.com/hook", "format": "generic"}],
        )
        notifier = WebhookNotifier(config_path=config_path)
        webhook = notifier._webhooks[0]
        assert webhook.events is None
        notifications = [
            make_notification(action_type="build"),
            make_notification(action_type="security_alert"),
        ]
        filtered = notifier._filter_by_events(notifications, webhook)
        assert len(filtered) == 2

    def test_empty_events_list_means_all_pass(self, tmp_path: Path) -> None:
        """A WebhookConfig with events=[] passes all notifications (same as events=None).

        Regression test for defect #8: _filter_by_events() previously used
        ``if webhook.events is None`` which skipped the empty-list case, causing
        events=[] to block ALL notifications instead of passing them all through.
        """
        notifier = WebhookNotifier(config_path=tmp_path / "missing.yaml")
        webhook = WebhookConfig(url="http://example.com/hook", format="generic", events=[])
        notifications = [
            make_notification(action_type="build_complete"),
            make_notification(action_type="security_alert"),
        ]
        filtered = notifier._filter_by_events(notifications, webhook)
        assert len(filtered) == 2, "events=[] must pass all notifications through, same as events=None"


# -- Retry logic --------------------------------------------------------------


class TestRetryLogic:
    """_send_with_retry() retries on failure and records health stats."""

    def test_retry_exhausted_records_failure(self, tmp_path: Path) -> None:
        """When all retry attempts fail, the webhook failure count is incremented."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://fail.example.com/hook", "format": "generic"}],
        )
        notifier = WebhookNotifier(config_path=config_path)

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("connection refused")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            # Patch sleep to avoid delays in tests
            with patch("time.sleep"):
                success = notifier._send_with_retry("http://fail.example.com/hook", {"test": True})

        assert success is False
        health = notifier.get_health()
        assert health["http://fail.example.com/hook"]["failures"] >= 1

    def test_success_records_success_count(self, tmp_path: Path) -> None:
        """A successful delivery increments the success counter."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://ok.example.com/hook", "format": "generic"}],
        )
        notifier = WebhookNotifier(config_path=config_path)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            success = notifier._send_with_retry("http://ok.example.com/hook", {"test": True})

        assert success is True
        health = notifier.get_health()
        assert health["http://ok.example.com/hook"]["successes"] == 1


# -- get_health ---------------------------------------------------------------


class TestGetHealth:
    """get_health() returns per-webhook statistics."""

    def test_get_health_returns_dict_per_webhook(self, tmp_path: Path) -> None:
        """Each configured webhook URL appears in get_health() with numeric counters."""
        config_path = _write_webhook_config(
            tmp_path,
            [{"url": "http://example.com/a"}, {"url": "http://example.com/b"}],
        )
        notifier = WebhookNotifier(config_path=config_path)
        health = notifier.get_health()
        assert "http://example.com/a" in health
        assert "http://example.com/b" in health
        for stats in health.values():
            assert "successes" in stats
            assert "failures" in stats
