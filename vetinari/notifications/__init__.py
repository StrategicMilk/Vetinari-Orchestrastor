"""Notification subsystem — multi-channel dispatch for Vetinari events.

Routes notifications to the correct channels based on priority tier:
CRITICAL -> all channels, HIGH -> desktop + dashboard, MEDIUM -> dashboard
+ batched webhook, LOW -> badge + daily digest.

Channels register themselves with the NotificationManager singleton.
"""

from __future__ import annotations

from vetinari.notifications.manager import NotificationManager, get_notification_manager

__all__ = [
    "NotificationManager",
    "get_notification_manager",
]
