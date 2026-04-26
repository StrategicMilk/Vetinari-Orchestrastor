"""Tests for vetinari.notifications.manager — priority-based channel dispatcher."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from vetinari.notifications.manager import Notification, NotificationManager
from vetinari.types import NotificationPriority

# -- Helpers ------------------------------------------------------------------


def _make_manager() -> NotificationManager:
    """Create a fresh NotificationManager for each test."""
    return NotificationManager(autostart=False)


def _recording_handler(received: list[list[Notification]]):
    """Return a channel handler that records every call's notification list."""

    def handler(notifications: list[Notification]) -> None:
        received.append(list(notifications))

    return handler


# -- register / unregister ----------------------------------------------------


class TestChannelRegistration:
    """register_channel() and unregister_channel() manage the channel map."""

    def test_register_channel_appears_in_registered(self) -> None:
        """After register_channel, the name shows up in registered_channels."""
        manager = _make_manager()
        manager.register_channel("dashboard", lambda n: None)
        assert "dashboard" in manager.registered_channels

    def test_unregister_channel_removes_it(self) -> None:
        """After unregister_channel, the name is absent from registered_channels."""
        manager = _make_manager()
        manager.register_channel("desktop", lambda n: None)
        manager.unregister_channel("desktop")
        assert "desktop" not in manager.registered_channels

    def test_unregister_nonexistent_is_safe(self) -> None:
        """unregister_channel on an unknown name does not raise and leaves state unchanged."""
        manager = _make_manager()
        channels_before = set(manager.registered_channels)
        manager.unregister_channel("phantom_channel")
        assert set(manager.registered_channels) == channels_before


# -- Priority routing ---------------------------------------------------------


class TestPriorityRouting:
    """notify() routes to the correct channels per priority tier."""

    def test_critical_dispatches_to_all_channels(self) -> None:
        """CRITICAL notifications go to dashboard, desktop, and webhook."""
        manager = _make_manager()
        received: dict[str, list] = {"dashboard": [], "desktop": [], "webhook": []}
        for name in received:
            manager.register_channel(name, _recording_handler(received[name]))

        manager.notify("Alert", "System critical", NotificationPriority.CRITICAL)

        assert len(received["dashboard"]) == 1
        assert len(received["desktop"]) == 1
        assert len(received["webhook"]) == 1

    def test_high_dispatches_to_desktop_and_dashboard(self) -> None:
        """HIGH notifications go to dashboard and desktop, NOT webhook."""
        manager = _make_manager()
        received: dict[str, list] = {"dashboard": [], "desktop": [], "webhook": []}
        for name in received:
            manager.register_channel(name, _recording_handler(received[name]))

        manager.notify("Approval needed", "Please review", NotificationPriority.HIGH)

        assert len(received["dashboard"]) == 1
        assert len(received["desktop"]) == 1
        assert received["webhook"] == []

    def test_medium_dispatches_to_dashboard_only_immediately(self) -> None:
        """MEDIUM notifications reach dashboard immediately; webhook is batched."""
        manager = _make_manager()
        received: dict[str, list] = {"dashboard": [], "webhook": []}
        for name in received:
            manager.register_channel(name, _recording_handler(received[name]))

        manager.notify("Task done", "Completed successfully", NotificationPriority.MEDIUM)

        assert len(received["dashboard"]) == 1
        assert received["webhook"] == []  # Batched, not immediate

    def test_low_dispatches_to_dashboard_badge_only(self) -> None:
        """LOW notifications only target dashboard_badge."""
        manager = _make_manager()
        received: dict[str, list] = {
            "dashboard": [],
            "desktop": [],
            "dashboard_badge": [],
        }
        for name in received:
            manager.register_channel(name, _recording_handler(received[name]))

        manager.notify("Metric update", "All nominal", NotificationPriority.LOW)

        assert len(received["dashboard_badge"]) == 1
        assert received["dashboard"] == []
        assert received["desktop"] == []


# -- flush_batches ------------------------------------------------------------


class TestFlushBatches:
    """flush_batches() delivers notifications that were manually buffered."""

    def test_flush_delivers_buffered_notifications(self) -> None:
        """flush_batches() delivers whatever is in the batch buffer to registered channels."""
        manager = _make_manager()
        received: list[list[Notification]] = []
        manager.register_channel("webhook", _recording_handler(received))

        # Manually inject into batch buffer to simulate batching behavior
        import uuid
        from datetime import datetime, timezone

        notif = Notification(
            notification_id=f"ntf_{uuid.uuid4().hex[:12]}",
            title="Batched",
            body="Body",
            priority=NotificationPriority.MEDIUM,
            action_type="build",
            metadata={},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manager._batch_buffer["webhook"].append(notif)

        flushed = manager.flush_batches()
        assert flushed >= 1
        assert len(received) >= 1

    def test_flush_clears_buffer(self) -> None:
        """After flush_batches(), the batch buffer is empty."""
        manager = _make_manager()
        manager.register_channel("webhook", lambda n: None)

        import uuid
        from datetime import datetime, timezone

        notif = Notification(
            notification_id=f"ntf_{uuid.uuid4().hex[:12]}",
            title="X",
            body="Y",
            priority=NotificationPriority.MEDIUM,
            action_type="evt",
            metadata={},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manager._batch_buffer["webhook"].append(notif)
        manager.flush_batches()
        # Buffer should be empty after flush
        assert dict(manager._batch_buffer) == {}

    def test_background_worker_flushes_medium_webhook_notifications(self) -> None:
        """The live flush worker eventually delivers batched MEDIUM webhooks."""
        manager = NotificationManager(batch_flush_interval_s=0.01)
        received: list[list[Notification]] = []
        manager.register_channel("webhook", _recording_handler(received))
        try:
            manager.notify("Task done", "Completed successfully", NotificationPriority.MEDIUM)
            deadline = time.monotonic() + 1.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            manager.stop()

        assert len(received) == 1
        assert received[0][0].title == "Task done"


# -- Semantic batching --------------------------------------------------------


class TestSemanticBatching:
    """3+ same action_type notifications are collapsed into a single summary."""

    def test_three_same_type_collapsed_to_summary(self) -> None:
        """_semantic_batch returns one summary notification for 3+ same action_type."""
        manager = _make_manager()
        received: list[list[Notification]] = []
        manager.register_channel("webhook", _recording_handler(received))

        # Manually build 3 same-type notifications and inject into buffer
        import uuid
        from datetime import datetime, timezone

        notifications = [
            Notification(
                notification_id=f"ntf_{uuid.uuid4().hex[:12]}",
                title=f"Prompt opt {i}",
                body="Details",
                priority=NotificationPriority.MEDIUM,
                action_type="prompt_optimization",
                metadata={},
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            for i in range(3)
        ]
        manager._batch_buffer["webhook"].extend(notifications)

        manager.flush_batches()

        # Should be a single summary notification in the webhook delivery
        assert len(received) == 1
        batch = received[0]
        assert len(batch) == 1  # 3 collapsed into 1 summary
        assert batch[0].metadata.get("batch_count") == 3


# -- get_delivery_log ---------------------------------------------------------


class TestDeliveryLog:
    """get_delivery_log() returns records of dispatched notifications."""

    def test_delivery_log_records_notifications(self) -> None:
        """notify() populates the delivery log with entries."""
        manager = _make_manager()
        manager.register_channel("dashboard", lambda n: None)
        manager.notify("Test", "Body", NotificationPriority.HIGH)
        log = manager.get_delivery_log()
        assert len(log) >= 1
        entry = log[0]
        assert "notification_id" in entry
        assert "priority" in entry
