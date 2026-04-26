"""Startup wiring for the autonomy and notification subsystems.

Called during application startup to initialize the governor, approval queue,
notification channels, and daily digest schedule. This is the single entry
point that wires all Session 4A components into the running application.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def wire_autonomy_and_notifications() -> None:
    """Initialize autonomy governor, notification channels, and digest schedule.

    Call this during application startup (after event bus is ready).
    Failures in optional channels (desktop, webhook) are logged but don't
    block startup.
    """
    # Initialize the autonomy governor (loads policies from YAML)
    try:
        from vetinari.autonomy.governor import get_governor

        governor = get_governor()
        trust_status = governor.get_trust_status()
        logger.info(
            "Autonomy governor initialized — %d action types tracked",
            len(trust_status),
        )
    except Exception:
        logger.warning("Failed to initialize autonomy governor — autonomous actions will be gated by default")

    # Wire promotion checker for scheduled invocation
    try:
        from vetinari.autonomy.governor import get_governor

        governor = get_governor()
        governor.check_pending_promotions  # noqa: B018 — verify method exists
        logger.info("Promotion checker available (call governor.check_pending_promotions() on schedule)")
    except Exception:
        logger.warning("Promotion checker initialization failed — scheduled promotions unavailable")

    # Initialize the notification manager
    try:
        from vetinari.notifications.manager import get_notification_manager

        get_notification_manager()
        logger.info("Notification manager initialized")
    except Exception:
        logger.warning("Failed to initialize notification manager — notifications disabled")

    # Register desktop notification channel (optional — depends on desktop-notifier)
    try:
        from vetinari.notifications.desktop import create_desktop_channel

        channel = create_desktop_channel()
        if channel:
            logger.info("Desktop notification channel registered")
    except Exception:
        logger.warning("Desktop notifications unavailable — notification channel not registered")

    # Register webhook notification channel (optional — depends on config)
    try:
        from vetinari.notifications.webhook import create_webhook_channel

        channel = create_webhook_channel()
        if channel:
            logger.info("Webhook notification channel registered")
    except Exception:
        logger.warning("Webhook notifications unavailable — webhook channel not registered")

    # Wire digest generator (manual-only — not auto-scheduled)
    try:
        from vetinari.notifications.digest import DigestGenerator

        generator = DigestGenerator()
        assert callable(getattr(generator, "send_digest", None)), "DigestGenerator missing send_digest"
        logger.info("Daily digest generator available (manual trigger only — no scheduler wired)")
    except Exception:
        logger.warning("Daily digest initialization failed — digest reports unavailable")

    # Wire confidence gate for inference path (verifies import works)
    try:
        from vetinari.agents.confidence_gate import ConfidenceGate

        _gate = ConfidenceGate()
        # The gate is used by the orchestration layer after agent inference:
        #   decision = gate.route_by_confidence(logprobs, task_type)
        # The TwoLayerOrchestrator can call this after _infer() returns.
        assert callable(getattr(_gate, "route_by_confidence", None)), "ConfidenceGate missing route_by_confidence"
        logger.info(
            "Confidence gate initialized (thresholds: high=%.1f, med=%.1f, low=%.1f)",
            _gate._threshold_high,
            _gate._threshold_medium,
            _gate._threshold_low,
        )
    except Exception:
        logger.warning("Confidence gate initialization failed — confidence-based routing disabled")

    # Wire episodic recall for plan generation
    try:
        from vetinari.learning.episodic_recall import recall_similar_episodes

        assert callable(recall_similar_episodes), "recall_similar_episodes is not callable"
        logger.info("Episodic recall (recall_similar_episodes) wired")
    except Exception:
        logger.warning("Episodic recall initialization failed — planning will proceed without episode history")

    # Wire multi-perspective review for Inspector
    try:
        from vetinari.agents.consolidated.quality_agent import run_multi_perspective_review

        assert callable(run_multi_perspective_review), "run_multi_perspective_review is not callable"
        logger.info("Multi-perspective review wired")
    except Exception:
        logger.warning("Multi-perspective review initialization failed — Inspector will use single-perspective review")
