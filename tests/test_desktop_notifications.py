"""Tests for vetinari.notifications.desktop — graceful degradation and OS toast delivery."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_notification
from vetinari.notifications.manager import Notification, NotificationPriority

# -- Graceful degradation when desktop-notifier is absent ---------------------


class TestGracefulDegradation:
    """DesktopNotificationChannel works safely without desktop-notifier installed."""

    def test_is_available_false_when_library_missing(self) -> None:
        """is_available returns False when desktop_notifier cannot be imported."""
        import importlib

        module_name = "vetinari.notifications.desktop"
        original_module = sys.modules.get(module_name)

        # Temporarily hide the desktop_notifier module
        with patch.dict(sys.modules, {"desktop_notifier": None}):
            sys.modules.pop(module_name, None)
            desktop_mod = importlib.import_module(module_name)
            from vetinari.notifications.desktop import DesktopNotificationChannel

            channel = DesktopNotificationChannel()
            assert channel.is_available is False

        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module
        else:
            importlib.import_module(module_name)

    def test_deliver_does_not_raise_when_unavailable(self) -> None:
        """deliver() is a no-op (no exception) when the backend is absent."""
        import vetinari.notifications.desktop as desktop_mod

        saved = desktop_mod._HAS_DESKTOP_NOTIFIER
        desktop_mod._HAS_DESKTOP_NOTIFIER = False
        try:
            from vetinari.notifications.desktop import DesktopNotificationChannel

            channel = DesktopNotificationChannel()
            # Must not raise — graceful degradation
            result = channel.deliver([make_notification()])
            assert result is None or result == []
        finally:
            desktop_mod._HAS_DESKTOP_NOTIFIER = saved

    def test_native_backend_constructor_failure_degrades_to_unavailable(self) -> None:
        """Constructor failures from optional OS backends must not escape."""
        import vetinari.notifications.desktop as desktop_mod

        saved = desktop_mod._HAS_DESKTOP_NOTIFIER
        desktop_mod._HAS_DESKTOP_NOTIFIER = True
        try:
            with patch.object(desktop_mod, "_NativeNotifier", side_effect=PermissionError("registry denied")):
                channel = desktop_mod.DesktopNotificationChannel()
                assert channel.is_available is False
                assert channel.deliver([make_notification()]) is None
        finally:
            desktop_mod._HAS_DESKTOP_NOTIFIER = saved


# -- Delivery when backend is present -----------------------------------------


class TestDeliveryWithBackend:
    """deliver() calls notifier.send_sync() when desktop-notifier is installed."""

    def test_deliver_calls_send_sync(self) -> None:
        """deliver() invokes _notifier.send_sync() for each notification."""
        mock_notifier = MagicMock()

        import vetinari.notifications.desktop as desktop_mod

        saved_flag = desktop_mod._HAS_DESKTOP_NOTIFIER
        desktop_mod._HAS_DESKTOP_NOTIFIER = True

        # Patch the NativeNotifier constructor at the place it's cached
        with patch.object(
            desktop_mod,
            "_NativeNotifier",
            return_value=mock_notifier,
            create=True,
        ):
            try:
                from vetinari.notifications.desktop import DesktopNotificationChannel

                channel = DesktopNotificationChannel()
                channel._notifier = mock_notifier  # Ensure it uses our mock
                notification = make_notification(title="Alert", body="Something happened")
                channel.deliver([notification])
                mock_notifier.send_sync.assert_called_once_with(
                    title="Alert",
                    message="Something happened",
                )
            finally:
                desktop_mod._HAS_DESKTOP_NOTIFIER = saved_flag

    def test_is_available_true_when_notifier_set(self) -> None:
        """is_available is True when _notifier is not None."""
        import vetinari.notifications.desktop as desktop_mod
        from vetinari.notifications.desktop import DesktopNotificationChannel

        channel = DesktopNotificationChannel.__new__(DesktopNotificationChannel)
        channel._notifier = MagicMock()
        channel._warned = False
        saved_flag = desktop_mod._HAS_DESKTOP_NOTIFIER
        desktop_mod._HAS_DESKTOP_NOTIFIER = True
        try:
            assert channel.is_available is True
        finally:
            desktop_mod._HAS_DESKTOP_NOTIFIER = saved_flag
