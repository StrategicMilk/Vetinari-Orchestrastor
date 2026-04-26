"""Webhook Notifications — Discord, Slack, and generic HTTP delivery.

Delivers notifications to configured webhook endpoints with format-specific
payloads (Discord rich embed, Slack block kit, generic JSON POST).
Includes retry with exponential backoff and per-webhook event filtering.

Configuration loaded from ``config/notifications.yaml``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vetinari.config_paths import resolve_config_path
from vetinari.constants import MAX_RETRIES, RETRY_BASE_DELAY

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = resolve_config_path("notifications.yaml")

# Valid webhook payload format identifiers
_VALID_FORMATS: frozenset[str] = frozenset({"discord", "slack", "generic"})

# Valid notification event names that webhooks may subscribe to
_VALID_EVENTS: frozenset[str] = frozenset({
    "task_completed",
    "training_completed",
    "approval_needed",
    "error",
    "quality_alert",
    "daily_digest",
    "trust_promotion",
    "cost_alert",
    "security_alert",
    "build_complete",
})

if TYPE_CHECKING:
    from vetinari.notifications.manager import Notification


@dataclass(frozen=True)
class WebhookConfig:
    """Configuration for a single webhook endpoint.

    Args:
        url: The webhook URL.
        format: Payload format (``"discord"``, ``"slack"``, or ``"generic"``).
        events: List of event types to deliver (empty = all events).
        enabled: Whether this webhook is active.
    """

    url: str
    format: str = "generic"  # discord | slack | generic
    events: list[str] | None = None  # None = all events
    enabled: bool = True

    def __repr__(self) -> str:
        return "WebhookConfig(...)"


class WebhookNotifier:
    """Webhook notification channel with format-specific payloads and retry.

    Loads webhook configurations from YAML and delivers notifications
    via HTTP POST with exponential backoff on failure.

    Side effects in __init__:
      - Reads ``config/notifications.yaml`` from disk
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._webhooks: list[WebhookConfig] = []
        self._lock = threading.Lock()
        self._health: dict[str, dict[str, int]] = {}  # url -> {successes, failures}
        self._load_config()

    def _load_config(self) -> None:
        """Load webhook configurations from YAML file."""
        if not self._config_path.exists():
            logger.info(
                "Webhook config not found at %s — no webhooks configured",
                self._config_path,
            )
            return

        try:
            raw = self._config_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            webhook_defs = data.get("webhooks", [])
            for entry in webhook_defs:
                if not isinstance(entry, dict) or "url" not in entry:
                    continue

                fmt = entry.get("format", "generic")
                if fmt not in _VALID_FORMATS:
                    logger.warning(
                        "Webhook for %s has unknown format %r — skipping. Valid formats: %s",
                        entry["url"],
                        fmt,
                        ", ".join(sorted(_VALID_FORMATS)),
                    )
                    continue

                raw_events: list[str] | None = entry.get("events")
                if raw_events is not None:
                    unknown = [e for e in raw_events if e not in _VALID_EVENTS]
                    if unknown:
                        logger.warning(
                            "Webhook for %s has unknown event names %r — skipping. Valid events: %s",
                            entry["url"],
                            unknown,
                            ", ".join(sorted(_VALID_EVENTS)),
                        )
                        continue

                self._webhooks.append(
                    WebhookConfig(
                        url=entry["url"],
                        format=fmt,
                        events=raw_events,
                        enabled=entry.get("enabled", True),
                    )
                )
                self._health[entry["url"]] = {"successes": 0, "failures": 0}

            logger.info("Loaded %d webhook configurations", len(self._webhooks))
        except Exception:
            logger.warning(
                "Failed to load webhook config from %s — no webhooks will be active",
                self._config_path,
            )

    def deliver(self, notifications: list[Notification]) -> None:
        """Deliver notifications to all configured webhooks.

        Each webhook receives only events matching its event filter.
        Delivery uses retry with exponential backoff.

        Args:
            notifications: List of Notification objects to deliver.
        """
        for webhook in self._webhooks:
            if not webhook.enabled:
                continue

            # Filter notifications by webhook's event subscription
            filtered = self._filter_by_events(notifications, webhook)
            if not filtered:
                continue

            payload = self._format_payload(filtered, webhook.format)
            self._send_with_retry(webhook.url, payload)

    def _filter_by_events(
        self,
        notifications: list[Notification],
        webhook: WebhookConfig,
    ) -> list[Notification]:
        """Filter notifications to only those matching the webhook's event list.

        Both ``events=None`` and ``events=[]`` mean "all events" — the empty
        list is treated as "no filter" rather than "no events allowed", which
        matches the documented contract on ``WebhookConfig``.
        """
        if not webhook.events:  # None or [] both mean "all events"
            return notifications
        return [n for n in notifications if n.action_type in webhook.events]

    def _format_payload(self, notifications: list[Notification], fmt: str) -> dict[str, Any]:
        """Format notification payload for the target platform.

        Args:
            notifications: Notifications to format.
            fmt: Target format (``"discord"``, ``"slack"``, or ``"generic"``).

        Returns:
            Platform-specific payload dict.
        """
        if fmt == "discord":
            return self._format_discord(notifications)
        if fmt == "slack":
            return self._format_slack(notifications)
        return self._format_generic(notifications)

    def _format_discord(self, notifications: list[Notification]) -> dict[str, Any]:
        """Format as Discord rich embed."""
        embeds = []
        for n in notifications[:10]:  # Discord limit: 10 embeds per message
            color_map = {"critical": 0xFF0000, "high": 0xFF8C00, "medium": 0x3498DB, "low": 0x95A5A6}
            embeds.append({
                "title": n.title,
                "description": n.body,
                "color": color_map.get(n.priority.value, 0x95A5A6),
                "footer": {"text": f"Vetinari | {n.action_type}"},
                "timestamp": n.created_at,
            })
        return {"embeds": embeds}

    def _format_slack(self, notifications: list[Notification]) -> dict[str, Any]:
        """Format as Slack block kit message."""
        blocks = []
        for n in notifications:
            emoji_map = {
                "critical": ":rotating_light:",
                "high": ":warning:",
                "medium": ":information_source:",
                "low": ":memo:",
            }
            emoji = emoji_map.get(n.priority.value, ":memo:")
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} *{n.title}*\n{n.body}",
                },
            })
        return {"blocks": blocks}

    def _format_generic(self, notifications: list[Notification]) -> dict[str, Any]:
        """Format as generic JSON POST body."""
        return {
            "source": "vetinari",
            "notifications": [
                {
                    "id": n.notification_id,
                    "title": n.title,
                    "body": n.body,
                    "priority": n.priority.value,
                    "action_type": n.action_type,
                    "metadata": n.metadata,
                    "created_at": n.created_at,
                }
                for n in notifications
            ],
        }

    def _send_with_retry(self, url: str, payload: dict[str, Any]) -> bool:
        """Send payload to webhook URL with exponential backoff retry.

        Args:
            url: The webhook endpoint URL.
            payload: The formatted payload dict.

        Returns:
            True if delivery succeeded, False after all retries exhausted.
        """
        for attempt in range(MAX_RETRIES):
            try:
                import httpx  # noqa: VET070 — optional dep [notifications]

                with httpx.Client(timeout=10.0) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()

                with self._lock:
                    if url in self._health:
                        self._health[url]["successes"] += 1
                return True

            except ImportError:
                logger.warning("httpx not installed — webhook delivery disabled. Install with: pip install httpx")  # noqa: VET301 — user guidance string
                return False

            except Exception:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Webhook delivery to %s failed (attempt %d/%d) — retrying in %.1fs",
                    url[:50],
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)

        with self._lock:
            if url in self._health:
                self._health[url]["failures"] += 1

        logger.warning(
            "Webhook delivery to %s failed after %d retries — notification dropped",
            url[:50],
            MAX_RETRIES,
        )
        return False

    def get_health(self) -> dict[str, dict[str, int]]:
        """Return per-webhook success/failure counts.

        Returns:
            Dict mapping webhook URLs to their delivery health stats.
        """
        with self._lock:
            return dict(self._health)


def create_webhook_channel(config_path: Path | None = None) -> WebhookNotifier | None:
    """Create and register the webhook notification channel.

    Args:
        config_path: Optional override for YAML config path.

    Returns:
        The WebhookNotifier instance, or None if no webhooks configured.
    """
    notifier = WebhookNotifier(config_path=config_path)
    if not notifier._webhooks:
        return None

    try:
        from vetinari.notifications.manager import get_notification_manager

        get_notification_manager().register_channel("webhook", notifier.deliver)
    except Exception:
        logger.warning("Failed to register webhook channel with notification manager")

    return notifier
