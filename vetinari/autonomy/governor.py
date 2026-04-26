"""Autonomy Governor — five-level policy engine for autonomous action gating.

Central policy engine that decides whether an action can proceed autonomously,
needs human approval, or should be blocked. Every autonomous action in the
system routes through ``governor.request_permission()`` before executing.

Levels: L0 (Manual) -> L1 (Suggest) -> L2 (Act & Report) -> L3 (Act & Log)
-> L4 (Full Auto). Policy is loaded from ``config/autonomy_policies.yaml``.

This module also houses the Progressive Trust Engine: per-action-type
success tracking that triggers auto-promotion proposals after sustained
reliability and auto-demotes after consecutive failures.

Global autonomy mode (CONSERVATIVE / BALANCED / AGGRESSIVE) sets risk-tier
defaults. Domain care levels provide per-domain overrides on top of the mode.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from vetinari.types import AutonomyLevel, AutonomyMode, DomainCareLevel, PermissionDecision

logger = logging.getLogger(__name__)

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "autonomy_policies.yaml"

# Trust engine thresholds
_PROMOTION_SUCCESS_RATE = 0.95  # 95% success rate required to suggest promotion
_PROMOTION_MIN_ACTIONS = 50  # Minimum actions before promotion suggestion
_DEMOTION_CONSECUTIVE_FAILURES = 3  # Auto-demote after this many consecutive failures

# Auto-promotion veto window: pending promotions apply after this delay unless vetoed
_VETO_WINDOW_HOURS = 1

# -- Risk tier defaults per autonomy mode ----------------------------------------
# Maps (mode, risk_tier) -> AutonomyLevel
_MODE_DEFAULTS: dict[AutonomyMode, dict[str, AutonomyLevel]] = {
    AutonomyMode.CONSERVATIVE: {
        "risky": AutonomyLevel.L1_SUGGEST,
        "medium": AutonomyLevel.L2_ACT_REPORT,
        "safe": AutonomyLevel.L3_ACT_LOG,
    },
    AutonomyMode.BALANCED: {
        "risky": AutonomyLevel.L2_ACT_REPORT,
        "medium": AutonomyLevel.L3_ACT_LOG,
        "safe": AutonomyLevel.L4_FULL_AUTO,
    },
    AutonomyMode.AGGRESSIVE: {
        "risky": AutonomyLevel.L3_ACT_LOG,
        "medium": AutonomyLevel.L4_FULL_AUTO,
        "safe": AutonomyLevel.L4_FULL_AUTO,
    },
}

# Confidence bands and their corresponding autonomy levels per mode
_CONFIDENCE_BANDS = [
    ("high", 0.85),  # >= 0.85
    ("medium", 0.6),  # >= 0.6
    ("low", 0.4),  # >= 0.4
    ("very_low", 0.0),  # < 0.4
]

# Maps (mode, band) -> AutonomyLevel for confidence-based routing
_MODE_CONFIDENCE_LEVELS: dict[AutonomyMode, dict[str, AutonomyLevel]] = {
    AutonomyMode.CONSERVATIVE: {
        "high": AutonomyLevel.L3_ACT_LOG,
        "medium": AutonomyLevel.L2_ACT_REPORT,
        "low": AutonomyLevel.L1_SUGGEST,
        "very_low": AutonomyLevel.L0_MANUAL,
    },
    AutonomyMode.BALANCED: {
        "high": AutonomyLevel.L4_FULL_AUTO,
        "medium": AutonomyLevel.L3_ACT_LOG,
        "low": AutonomyLevel.L2_ACT_REPORT,
        "very_low": AutonomyLevel.L1_SUGGEST,
    },
    AutonomyMode.AGGRESSIVE: {
        "high": AutonomyLevel.L4_FULL_AUTO,
        "medium": AutonomyLevel.L4_FULL_AUTO,
        "low": AutonomyLevel.L3_ACT_LOG,
        "very_low": AutonomyLevel.L2_ACT_REPORT,
    },
}

_LEVEL_ORDER = [
    AutonomyLevel.L0_MANUAL,
    AutonomyLevel.L1_SUGGEST,
    AutonomyLevel.L2_ACT_REPORT,
    AutonomyLevel.L3_ACT_LOG,
    AutonomyLevel.L4_FULL_AUTO,
]


@dataclass
class ActionPolicy:
    """Policy configuration for a single action type.

    Args:
        level: The autonomy level controlling human involvement.
        max_change_pct: Maximum allowed change percentage (0-100) for bounded actions.
        rollback_on_regression: Whether to auto-rollback if quality regresses.
    """

    level: AutonomyLevel
    max_change_pct: float = 100.0  # No limit by default
    rollback_on_regression: bool = False


@dataclass(frozen=True)
class TrustRecord:
    """Per-action-type trust tracking for the progressive trust engine.

    Tracks success/failure history to support promotion suggestions
    and automatic demotions.
    """

    total_actions: int = 0
    successful_actions: int = 0
    consecutive_failures: int = 0
    was_demoted: bool = False  # True if last level change was a demotion

    def __repr__(self) -> str:
        return (
            f"TrustRecord(total={self.total_actions}, "
            f"successful={self.successful_actions}, "
            f"consecutive_failures={self.consecutive_failures})"
        )

    @property
    def success_rate(self) -> float:
        """Rolling success rate as a fraction 0.0-1.0."""
        if self.total_actions == 0:
            return 0.0
        return self.successful_actions / self.total_actions

    @property
    def eligible_for_promotion(self) -> bool:
        """Whether this action type meets promotion criteria."""
        return self.total_actions >= _PROMOTION_MIN_ACTIONS and self.success_rate >= _PROMOTION_SUCCESS_RATE


@dataclass
class PendingPromotion:
    """A pending auto-promotion awaiting the veto window expiry.

    Created by ``auto_promote()`` when trust criteria are met.
    Applied automatically by ``check_pending_promotions()`` after
    ``_VETO_WINDOW_HOURS`` unless cancelled by ``veto_promotion()``.

    Attributes:
        action_type: The action type proposed for promotion.
        current_level: The current autonomy level.
        new_level: The proposed promoted level.
        proposed_at: ISO-8601 timestamp when the proposal was created.
        veto_deadline: ISO-8601 timestamp after which the promotion auto-applies.
    """

    action_type: str
    current_level: AutonomyLevel
    new_level: AutonomyLevel
    proposed_at: str  # ISO-8601
    veto_deadline: str  # ISO-8601

    def __repr__(self) -> str:
        return "PendingPromotion(...)"


@dataclass
class PermissionResult:
    """Rich result from a permission request including action tracking data.

    Returned by ``request_permission_full()`` when the caller needs more
    than just the bare decision — especially when the decision is DEFER and
    the action has been enqueued for human approval.

    Attributes:
        decision: The permission decision (APPROVE, DENY, or DEFER).
        action_type: The action type that was evaluated.
        action_id: Unique ID of the enqueued approval request (only set when DEFER).
        level: The autonomy level that produced this decision.
        policy: The full ActionPolicy that was applied.
    """

    decision: PermissionDecision
    action_type: str
    action_id: str | None
    level: AutonomyLevel
    policy: ActionPolicy

    def __repr__(self) -> str:
        return "PermissionResult(...)"


@dataclass
class PromotionSuggestion:
    """A suggestion to promote an action type to a higher autonomy level.

    Not applied automatically — requires human confirmation.
    """

    action_type: str
    current_level: AutonomyLevel
    suggested_level: AutonomyLevel
    success_rate: float
    total_actions: int

    def __repr__(self) -> str:
        return "PromotionSuggestion(...)"


class AutonomyGovernor:
    """Five-level autonomy policy engine with progressive trust.

    Gates every autonomous action through ``request_permission()``, which
    consults the per-action-type policy to decide approve/deny/defer.

    Supports confidence-based routing: when ``confidence`` is passed to
    ``request_permission_full()``, the effective level is derived from both
    the policy ceiling and the confidence band under the active autonomy mode.

    Domain care levels provide per-domain overrides (e.g. force all deployment
    actions to require review regardless of global mode or confidence).

    Side effects in __init__:
      - Loads policy from YAML file at ``policy_path``
      - Initializes trust tracking dicts (no I/O, in-memory only)
    """

    def __init__(self, policy_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._policy_path = policy_path or _DEFAULT_POLICY_PATH
        self._policies: dict[str, ActionPolicy] = {}
        self._trust_records: dict[str, TrustRecord] = defaultdict(TrustRecord)
        self._default_level = AutonomyLevel.L1_SUGGEST  # Conservative default
        self._autonomy_mode = AutonomyMode.BALANCED  # Global mode default
        self._domain_care_levels: dict[str, DomainCareLevel] = {}
        # Pending auto-promotions awaiting veto-window expiry (action_type -> PendingPromotion)
        self._pending_promotions: dict[str, PendingPromotion] = {}
        # Action types blocked from promotion until the veto is lifted (permanent veto set)
        self._vetoed_actions: set[str] = set()
        self._load_policies()

    def _load_policies(self) -> None:
        """Load action policies from the YAML configuration file."""
        if not self._policy_path.exists():
            logger.warning(
                "Autonomy policy file not found at %s — using default L1 for all actions",
                self._policy_path,
            )
            return

        try:
            raw = self._policy_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
            if not isinstance(data, dict):
                logger.warning("Autonomy policy file has invalid structure — expected top-level dict")
                return

            actions = data.get("actions", {})
            for action_type, config in actions.items():
                if not isinstance(config, dict):
                    logger.warning("Skipping invalid policy entry for %s — expected dict", action_type)
                    continue
                level_str = config.get("level", "L1")
                try:
                    level = AutonomyLevel(level_str)
                except ValueError:
                    logger.warning(
                        "Unknown autonomy level %r for action %s — defaulting to L1",
                        level_str,
                        action_type,
                    )
                    level = AutonomyLevel.L1_SUGGEST

                self._policies[action_type] = ActionPolicy(
                    level=level,
                    max_change_pct=float(config.get("max_change_pct", 100.0)),
                    rollback_on_regression=bool(config.get("rollback_on_regression", False)),
                )

            logger.info("Loaded autonomy policies for %d action types", len(self._policies))

            # Load global defaults if present
            defaults = data.get("defaults", {})
            default_level_str = defaults.get("level")
            if default_level_str:
                import contextlib

                with contextlib.suppress(ValueError):
                    self._default_level = AutonomyLevel(default_level_str)

            # Load global autonomy mode
            mode_str = data.get("global_autonomy_mode")
            if mode_str:
                try:
                    self._autonomy_mode = AutonomyMode(mode_str)
                except ValueError:
                    logger.warning(
                        "Unknown autonomy mode %r in policy file — defaulting to BALANCED",
                        mode_str,
                    )

            # Load domain care levels
            domain_levels = data.get("domain_care_levels", {})
            for domain, care_str in domain_levels.items():
                try:
                    self._domain_care_levels[domain] = DomainCareLevel(care_str)
                except ValueError:
                    logger.warning(
                        "Unknown domain care level %r for domain %s — skipping",
                        care_str,
                        domain,
                    )

        except Exception:
            logger.warning(
                "Failed to load autonomy policies from %s — using defaults",
                self._policy_path,
            )

    def get_policy(self, action_type: str) -> ActionPolicy:
        """Get the policy for an action type, falling back to default level.

        Args:
            action_type: The action type identifier (e.g. ``"parameter_tuning"``).

        Returns:
            ActionPolicy for this action type.
        """
        return self._policies.get(action_type, ActionPolicy(level=self._default_level))

    def request_permission(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """Check whether an action is permitted under the current autonomy policy.

        This is the main gate — every autonomous action routes through here.

        IMPORTANT — DEFER is a blocking decision, not a soft suggestion.
        Callers MUST treat DEFER the same as DENY and halt execution until a
        human approves the action via the approval queue.  Only APPROVE may
        proceed.  Pattern callers must use:

            decision = governor.request_permission(...)
            if decision != PermissionDecision.APPROVE:
                # block — covers both DENY and DEFER

        Args:
            action_type: The action type identifier (e.g. ``"prompt_optimization"``).
            details: Optional metadata about the specific action instance.

        Returns:
            APPROVE if the action can proceed autonomously.  DENY if blocked
            outright.  DEFER if the action has been queued for human approval —
            callers MUST NOT proceed when DEFER is returned.
        """
        policy = self.get_policy(action_type)
        level = policy.level
        decision = self._level_to_decision(level)

        # Enforce max_change_pct constraint
        if details and policy.max_change_pct < 100.0:
            change_pct = details.get("change_pct", 0.0)
            if change_pct > policy.max_change_pct:
                logger.info(
                    "Action %s change_pct=%.1f exceeds max=%.1f — deferring to human",
                    action_type,
                    change_pct,
                    policy.max_change_pct,
                )
                decision = PermissionDecision.DEFER

        logger.info(
            "Permission request: action=%s level=%s decision=%s",
            action_type,
            level.value,
            decision.value,
        )
        return decision

    def request_permission_full(
        self,
        action_type: str,
        details: dict[str, Any] | None = None,
        confidence: float = 0.0,
        domain: str | None = None,
    ) -> PermissionResult:
        """Check permission and enqueue in the approval queue when deferred.

        Unlike ``request_permission()`` which returns only the decision, this
        method performs the full gate operation: evaluates the policy, applies
        confidence-based routing and domain care level overrides, logs every
        decision in the audit trail, and — when the decision is DEFER —
        automatically enqueues the action in the approval queue.

        Confidence routing: when ``confidence > 0``, the effective level is
        ``min(policy_level, confidence_level_for_mode)``. This allows high-confidence
        actions to proceed at higher levels and low-confidence to be gated.

        Domain overrides: if the domain has a ``DomainCareLevel.REVIEW`` override,
        the effective level is capped at L1 regardless of confidence or mode.

        Args:
            action_type: The action type identifier (e.g. ``"prompt_optimization"``).
            details: Optional metadata about the specific action instance.
            confidence: Agent confidence score for this action (0.0-1.0).
            domain: Optional domain name for per-domain care level override.

        Returns:
            PermissionResult with the decision, action_id (if deferred), and
            the level and policy that produced the decision.
        """
        policy = self.get_policy(action_type)
        level = policy.level

        # Apply domain care level override before confidence routing
        if domain is not None:
            care = self._domain_care_levels.get(domain)
            if care == DomainCareLevel.REVIEW:
                # Force to L1 — must be reviewed regardless of confidence
                level = AutonomyLevel.L1_SUGGEST
            # DomainCareLevel.AUTO falls through to normal confidence routing

        # Apply confidence-based routing when confidence is non-zero and no domain forced L1
        if confidence > 0.0 and level != AutonomyLevel.L1_SUGGEST:
            confidence_level = self._confidence_to_level(confidence)
            # Effective level is min(policy ceiling, confidence-derived level)
            level = self._min_level(level, confidence_level)

        # Enforce max_change_pct constraint
        if details and policy.max_change_pct < 100.0:
            change_pct = details.get("change_pct", 0.0)
            if change_pct > policy.max_change_pct:
                logger.info(
                    "Action %s change_pct=%.1f exceeds max=%.1f — deferring to human",
                    action_type,
                    change_pct,
                    policy.max_change_pct,
                )
                level = AutonomyLevel.L1_SUGGEST

        decision = self._level_to_decision(level)
        action_id: str | None = None

        # Late import to avoid circular dependency at module load time
        from vetinari.autonomy.approval_queue import get_approval_queue

        queue = get_approval_queue()

        if decision == PermissionDecision.DEFER:
            action_id = queue.enqueue(action_type, details=details, confidence=confidence)
            logger.info(
                "Action %s deferred — enqueued as %s (confidence=%.2f)",
                action_type,
                action_id,
                confidence,
            )

        # Audit every decision regardless of outcome
        queue.log_decision(
            action_type=action_type,
            autonomy_level=level,
            decision=decision,
            confidence=confidence,
            details=details,
        )

        logger.info(
            "Permission request (full): action=%s level=%s decision=%s action_id=%s",
            action_type,
            level.value,
            decision.value,
            action_id or "n/a",
        )
        return PermissionResult(
            decision=decision,
            action_type=action_type,
            action_id=action_id,
            level=level,
            policy=policy,
        )

    def _level_to_decision(self, level: AutonomyLevel) -> PermissionDecision:
        """Map an autonomy level to a permission decision.

        L0 (Manual) -> DENY (must be triggered by human).
        L1 (Suggest) -> DEFER (route to approval queue).
        L2-L4 -> APPROVE (proceed with varying reporting levels).
        """
        if level == AutonomyLevel.L0_MANUAL:
            return PermissionDecision.DENY
        if level == AutonomyLevel.L1_SUGGEST:
            return PermissionDecision.DEFER
        return PermissionDecision.APPROVE

    @staticmethod
    def _confidence_to_band(confidence: float) -> str:
        """Map a confidence score (0.0-1.0) to a named risk band.

        Args:
            confidence: Agent confidence score.

        Returns:
            One of ``"high"``, ``"medium"``, ``"low"``, or ``"very_low"``.
        """
        for band, threshold in _CONFIDENCE_BANDS:
            if confidence >= threshold:
                return band
        return "very_low"

    def _confidence_to_level(self, confidence: float) -> AutonomyLevel:
        """Map confidence to an autonomy level under the active mode.

        Args:
            confidence: Agent confidence score (0.0-1.0).

        Returns:
            AutonomyLevel appropriate for this confidence under current mode.
        """
        band = self._confidence_to_band(confidence)
        return _MODE_CONFIDENCE_LEVELS[self._autonomy_mode][band]

    def _min_level(self, a: AutonomyLevel, b: AutonomyLevel) -> AutonomyLevel:
        """Return the lower of two autonomy levels (more conservative).

        Args:
            a: First autonomy level.
            b: Second autonomy level.

        Returns:
            The lower level (earlier in the L0->L4 ordering).
        """
        return a if _LEVEL_ORDER.index(a) <= _LEVEL_ORDER.index(b) else b

    # -- Global Autonomy Mode ---------------------------------------------------

    def get_autonomy_mode(self) -> AutonomyMode:
        """Return the current global autonomy mode.

        Returns:
            The active AutonomyMode (CONSERVATIVE, BALANCED, or AGGRESSIVE).
        """
        with self._lock:
            return self._autonomy_mode

    def set_autonomy_mode(self, mode: AutonomyMode) -> None:
        """Set the global autonomy mode.

        Args:
            mode: The new AutonomyMode to apply.
        """
        with self._lock:
            self._autonomy_mode = mode
        logger.info("Autonomy mode set to %s", mode.value)

    def get_mode_default(self, risk_tier: str) -> AutonomyLevel:
        """Return the default autonomy level for a risk tier under the active mode.

        Args:
            risk_tier: One of ``"risky"``, ``"medium"``, or ``"safe"``.

        Returns:
            AutonomyLevel for the tier, or L1 for unknown tiers.
        """
        with self._lock:
            return _MODE_DEFAULTS.get(self._autonomy_mode, {}).get(risk_tier, AutonomyLevel.L1_SUGGEST)

    # -- Domain Care Levels -----------------------------------------------------

    def get_domain_care_level(self, domain: str) -> DomainCareLevel | None:
        """Return the care level for a domain, or None if not configured.

        Args:
            domain: Domain identifier (e.g. ``"code-generation"``).

        Returns:
            DomainCareLevel if configured, None otherwise.
        """
        return self._domain_care_levels.get(domain)

    # -- Progressive Trust Engine -----------------------------------------------

    def record_outcome(self, action_type: str, *, success: bool) -> None:
        """Record the outcome of an autonomous action for trust tracking.

        Updates per-action-type success/failure counts. Triggers automatic
        demotion after ``_DEMOTION_CONSECUTIVE_FAILURES`` consecutive failures.
        Triggers auto-promotion proposal when trust criteria are met.

        Args:
            action_type: The action type that was executed.
            success: Whether the action completed successfully.
        """
        with self._lock:
            record = self._trust_records[action_type]
            if success:
                updated = dataclasses.replace(
                    record,
                    total_actions=record.total_actions + 1,
                    successful_actions=record.successful_actions + 1,
                    consecutive_failures=0,
                )
                self._trust_records[action_type] = updated
                # Trigger auto-promotion proposal if trust criteria now met
                if updated.eligible_for_promotion:
                    self.auto_promote(action_type)
            else:
                new_failures = record.consecutive_failures + 1
                updated = dataclasses.replace(
                    record,
                    total_actions=record.total_actions + 1,
                    consecutive_failures=new_failures,
                )
                self._trust_records[action_type] = updated
                policy = self.get_policy(action_type)
                # Immediate rollback for regression-sensitive actions (first failure)
                if (
                    policy.rollback_on_regression and new_failures >= 1
                ) or new_failures >= _DEMOTION_CONSECUTIVE_FAILURES:
                    self._auto_demote(action_type)

    def auto_promote(self, action_type: str) -> PendingPromotion | None:
        """Create a pending auto-promotion for an action type.

        Called by ``record_outcome()`` when trust criteria are met. Creates a
        ``PendingPromotion`` with a veto deadline ``_VETO_WINDOW_HOURS`` from
        now. Does nothing if a pending promotion already exists for this type.

        Args:
            action_type: The action type to propose for promotion.

        Returns:
            The created PendingPromotion, or None if one already exists.
        """
        if action_type in self._pending_promotions:
            return None

        policy = self.get_policy(action_type)
        promoted = self._promote_one_level(policy.level)
        if promoted == policy.level:
            return None  # Already at highest level

        now = datetime.now(timezone.utc)
        deadline = now + timedelta(hours=_VETO_WINDOW_HOURS)
        pending = PendingPromotion(
            action_type=action_type,
            current_level=policy.level,
            new_level=promoted,
            proposed_at=now.isoformat(),
            veto_deadline=deadline.isoformat(),
        )
        self._pending_promotions[action_type] = pending
        logger.info(
            "Auto-promotion proposed for %s: %s -> %s (veto deadline %s)",
            action_type,
            policy.level.value,
            promoted.value,
            deadline.isoformat(),
        )
        return pending

    def get_pending_promotions(self) -> dict[str, PendingPromotion]:
        """Return a copy of all pending auto-promotions.

        Returns:
            Dict mapping action_type to PendingPromotion (copy, not live reference).
        """
        with self._lock:
            return dict(self._pending_promotions)

    def _auto_demote(self, action_type: str) -> None:
        """Immediately drop an action type one autonomy level.

        Demotion is instant and automatic — no human confirmation needed.
        This is the safety-asymmetric design: demotions are fast, promotions
        require human sign-off.
        """
        policy = self.get_policy(action_type)
        current = policy.level
        demoted = self._demote_one_level(current)
        if demoted == current:
            return  # Already at lowest level

        self._policies[action_type] = ActionPolicy(
            level=demoted,
            max_change_pct=policy.max_change_pct,
            rollback_on_regression=policy.rollback_on_regression,
        )
        self._trust_records[action_type] = dataclasses.replace(
            self._trust_records[action_type],
            was_demoted=True,
            consecutive_failures=0,
        )
        logger.warning(
            "Auto-demoted action %s from %s to %s after %d consecutive failures",
            action_type,
            current.value,
            demoted.value,
            _DEMOTION_CONSECUTIVE_FAILURES,
        )

    def _demote_one_level(self, level: AutonomyLevel) -> AutonomyLevel:
        """Return the autonomy level one step below the given level."""
        idx = _LEVEL_ORDER.index(level)
        if idx == 0:
            return level
        return _LEVEL_ORDER[idx - 1]

    def _promote_one_level(self, level: AutonomyLevel) -> AutonomyLevel:
        """Return the autonomy level one step above the given level."""
        idx = _LEVEL_ORDER.index(level)
        if idx >= len(_LEVEL_ORDER) - 1:
            return level
        return _LEVEL_ORDER[idx + 1]

    def suggest_promotions(self) -> list[PromotionSuggestion]:
        """Return promotion suggestions for action types that meet criteria.

        Checks all tracked action types for 95%+ success rate over 50+ actions.
        These are *suggestions* — they require human confirmation to apply.
        Vetoed action types are excluded.

        Returns:
            List of PromotionSuggestion for action types eligible for promotion.
        """
        suggestions: list[PromotionSuggestion] = []
        with self._lock:
            for action_type, record in self._trust_records.items():
                if not record.eligible_for_promotion:
                    continue
                if action_type in self._vetoed_actions:
                    continue
                policy = self.get_policy(action_type)
                promoted = self._promote_one_level(policy.level)
                if promoted == policy.level:
                    continue  # Already at highest level
                suggestions.append(
                    PromotionSuggestion(
                        action_type=action_type,
                        current_level=policy.level,
                        suggested_level=promoted,
                        success_rate=record.success_rate,
                        total_actions=record.total_actions,
                    )
                )
        return suggestions

    def apply_promotion(self, action_type: str) -> bool:
        """Apply a human-confirmed promotion for an action type.

        Args:
            action_type: The action type to promote.

        Returns:
            True if promotion was applied, False if not eligible, vetoed, or already at max.
        """
        with self._lock:
            if action_type in self._vetoed_actions:
                logger.info("Promotion for %s blocked by veto", action_type)
                return False
            record = self._trust_records.get(action_type)
            if record is None or not record.eligible_for_promotion:
                return False
            policy = self.get_policy(action_type)
            promoted = self._promote_one_level(policy.level)
            if promoted == policy.level:
                return False
            self._policies[action_type] = ActionPolicy(
                level=promoted,
                max_change_pct=policy.max_change_pct,
                rollback_on_regression=policy.rollback_on_regression,
            )
            # Reset counters for the new level
            self._trust_records[action_type] = dataclasses.replace(
                record,
                was_demoted=False,
                total_actions=0,
                successful_actions=0,
                consecutive_failures=0,
            )
            logger.info(
                "Promoted action %s from %s to %s (human-confirmed)",
                action_type,
                policy.level.value,
                promoted.value,
            )
            return True

    def get_trust_status(self) -> dict[str, dict[str, Any]]:
        """Return trust tracking data for all action types.

        Returns:
            Dict mapping action_type to trust metrics (total, successes, rate, etc.).
        """
        with self._lock:
            return {
                action_type: {
                    "total_actions": record.total_actions,
                    "successful_actions": record.successful_actions,
                    "success_rate": round(record.success_rate, 3),
                    "consecutive_failures": record.consecutive_failures,
                    "eligible_for_promotion": record.eligible_for_promotion,
                    "current_level": self.get_policy(action_type).level.value,
                }
                for action_type, record in self._trust_records.items()
            }

    def check_pending_promotions(self) -> list[str]:
        """Check pending promotions and apply those whose veto window has expired.

        Called periodically by a scheduler (e.g. every hour). Applies any pending
        promotions whose veto deadline has passed. Returns the list of applied
        action types. Suggestions still within the veto window are not applied.

        Returns:
            List of action_type strings where promotions were applied.
        """
        now = datetime.now(timezone.utc)
        applied: list[str] = []

        with self._lock:
            expired = [
                action_type
                for action_type, pending in self._pending_promotions.items()
                if datetime.fromisoformat(pending.veto_deadline) <= now
            ]

            for action_type in expired:
                pending = self._pending_promotions.pop(action_type)
                old_policy = self.get_policy(action_type)
                self._policies[action_type] = ActionPolicy(
                    level=pending.new_level,
                    max_change_pct=old_policy.max_change_pct,
                    rollback_on_regression=old_policy.rollback_on_regression,
                )
                applied.append(action_type)
                logger.info(
                    "Auto-promotion applied for %s: %s -> %s (veto window expired)",
                    action_type,
                    pending.current_level.value,
                    pending.new_level.value,
                )

        return applied

    def veto_promotion(self, action_type: str) -> bool:
        """Veto promotion for an action type — cancels pending and blocks future.

        Cancels any pending auto-promotion for the action type AND adds it to
        the permanent veto set. The veto persists until clear_veto() is called.

        Args:
            action_type: The action type to veto.

        Returns:
            True (veto always succeeds).
        """
        with self._lock:
            # Cancel any pending auto-promotion (trust history preserved)
            self._pending_promotions.pop(action_type, None)

            # Add to permanent veto set
            self._vetoed_actions.add(action_type)

        logger.info("Vetoed promotion for action type %s", action_type)
        return True

    def clear_veto(self, action_type: str) -> bool:
        """Remove a promotion veto for an action type.

        Args:
            action_type: The action type to un-veto.

        Returns:
            True if a veto was cleared, False if no veto existed.
        """
        with self._lock:
            if action_type in self._vetoed_actions:
                self._vetoed_actions.discard(action_type)
                return True
            return False

    def get_vetoed_actions(self) -> frozenset[str]:
        """Return the set of action types currently vetoed from promotion.

        Returns:
            Frozen set of action type strings with active vetoes.
        """
        with self._lock:
            return frozenset(self._vetoed_actions)


# -- Singleton ----------------------------------------------------------------

_governor: AutonomyGovernor | None = None
_governor_lock = threading.Lock()


def get_governor(policy_path: Path | None = None) -> AutonomyGovernor:
    """Get or create the singleton AutonomyGovernor.

    Args:
        policy_path: Optional override for policy YAML path (used in tests).

    Returns:
        The singleton AutonomyGovernor instance.
    """
    global _governor
    if _governor is None:
        with _governor_lock:
            if _governor is None:
                _governor = AutonomyGovernor(policy_path=policy_path)
    return _governor


def reset_governor() -> None:
    """Reset the singleton governor for test isolation."""
    global _governor
    with _governor_lock:
        _governor = None
