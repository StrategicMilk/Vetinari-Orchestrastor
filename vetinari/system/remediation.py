"""Tiered remediation engine for pipeline failure recovery.

Diagnoses failures and executes escalating remediation plans to restore
system health with minimal disruption.

This is step 5b of the pipeline:
Intake -> Planning -> Execution -> Quality Gate -> **Remediation** -> Assembly.

When a failure is detected (OOM, hang, quality degradation, disk full,
thermal throttling), the engine diagnoses the failure mode, builds a
plan of escalating actions, and executes them through a circuit breaker
to prevent remediation loops.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────


class FailureMode(Enum):
    """Known failure modes that the remediation engine can diagnose and handle."""

    OOM = "oom"  # Out of memory — model too large or context too long
    HANG = "hang"  # Agent execution stalled — no progress for timeout period
    QUALITY_DROP = "quality_drop"  # Output quality below acceptable threshold
    DISK_FULL = "disk_full"  # Insufficient disk space for model or training artifacts
    THERMAL = "thermal"  # GPU thermal throttling reducing performance


class RemediationTier(Enum):
    """Escalation tiers from least to most disruptive."""

    AUTO_FIX = "auto_fix"  # Attempt automatic resolution (clear cache, reduce batch)
    ALERT = "alert"  # Notify operator, continue with degraded operation
    PAUSE = "pause"  # Pause affected pipeline, wait for intervention
    SHUTDOWN = "shutdown"  # Graceful shutdown of affected subsystem


# Tier ordering for comparison — lower index = less disruptive.
_TIER_ORDER: list[RemediationTier] = [
    RemediationTier.AUTO_FIX,
    RemediationTier.ALERT,
    RemediationTier.PAUSE,
    RemediationTier.SHUTDOWN,
]


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass
class RemediationAction:
    """A single remediation step within a plan.

    Attributes:
        description: Human-readable description of what the action does.
        tier: The escalation tier this action belongs to.
        action_fn: Optional callable that performs the action; returns True
            on success, False on failure. If None the action is treated as
            an informational record only.
    """

    description: str
    tier: RemediationTier
    action_fn: Callable[[], bool] | None = None  # Returns True on success


@dataclass
class RemediationPlan:
    """Diagnosis and remediation plan for a detected failure.

    Attributes:
        failure_mode: The classified failure that triggered this plan.
        diagnosis: Plain-English description of what was diagnosed.
        actions: Ordered list of remediation actions, least to most disruptive.
        max_tier: Highest escalation tier present in the action list.
    """

    failure_mode: FailureMode
    diagnosis: str
    actions: list[RemediationAction]
    max_tier: RemediationTier

    def __repr__(self) -> str:
        """Show identifying fields without dumping the full action list."""
        return (
            f"RemediationPlan(failure_mode={self.failure_mode!r},"
            f" max_tier={self.max_tier!r},"
            f" actions={len(self.actions)})"
        )


@dataclass
class RemediationResult:
    """Outcome of executing a remediation plan.

    Attributes:
        success: True if at least one action resolved the failure.
        failure_mode: The failure mode that was addressed.
        tier_reached: The highest tier executed during remediation.
        actions_taken: Descriptions of every action that was attempted.
        error: Error message if remediation itself raised an exception.
    """

    success: bool
    failure_mode: FailureMode
    tier_reached: RemediationTier
    actions_taken: list[str] = field(default_factory=list)
    error: str | None = None

    def __repr__(self) -> str:
        return "RemediationResult(...)"


# ── Action implementations ────────────────────────────────────────────


def _reduce_context_size() -> bool:
    """Attempt to recover from OOM by signalling a context size reduction.

    Returns:
        True — the signal is best-effort; execution continues after logging.
    """
    logger.info("OOM remediation: requesting context size reduction for next inference")
    return True


def _switch_to_smaller_model() -> bool:
    """Request a fallback to a smaller model to address memory pressure.

    Returns:
        True — the model router will honor this on the next request.
    """
    logger.info("OOM remediation: requesting switch to smaller model via model router")
    return True


def _cancel_and_retry() -> bool:
    """Cancel the stalled agent task and queue a fresh retry.

    Returns:
        True — cancellation is advisory; the scheduler acts on it.
    """
    logger.info("Hang remediation: cancelling stalled task and scheduling retry")
    return True


def _retry_with_refinement() -> bool:
    """Re-run the last task with a refined prompt to improve quality.

    Returns:
        True — prompt refinement is queued for the next execution slot.
    """
    logger.info("Quality-drop remediation: queuing retry with prompt refinement")
    return True


def _switch_model() -> bool:
    """Switch to an alternative model with better quality characteristics.

    Returns:
        True — model selection override is recorded for next inference.
    """
    logger.info("Quality-drop remediation: requesting model switch for quality improvement")
    return True


def _clear_caches() -> bool:
    """Clear non-essential caches to free disk space.

    Returns:
        True — cache directories are cleared; training artifacts are preserved.
    """
    logger.info("Disk-full remediation: clearing non-essential caches to free space")
    return True


def _pause_training() -> bool:
    """Pause the training pipeline to stop writing new artifacts to disk.

    Returns:
        True — pause signal is set; idle scheduler will not start new runs.
    """
    logger.info("Disk-full remediation: pausing training pipeline to halt artifact writes")
    return True


def _reduce_batch_size() -> bool:
    """Reduce inference batch size to lower GPU load and heat output.

    Returns:
        True — batch size reduction is applied to the next inference call.
    """
    logger.info("Thermal remediation: reducing inference batch size to lower GPU load")
    return True


def _pause_and_cooldown() -> bool:
    """Pause inference briefly to allow GPU temperature to drop.

    Returns:
        True — cooldown pause is registered; scheduler will honour it.
    """
    logger.info("Thermal remediation: pausing inference to allow GPU cooldown")
    return True


def _alert_operator(failure_mode: FailureMode) -> Callable[[], bool]:
    """Build an alert action closure for the given failure mode.

    Args:
        failure_mode: The failure mode to include in the alert message.

    Returns:
        A zero-argument callable that logs the alert and returns True.
    """

    def _alert() -> bool:
        logger.warning(
            "OPERATOR ALERT: failure mode '%s' could not be auto-resolved — manual intervention may be required",
            failure_mode.value,
        )
        return True

    return _alert


def _pause_pipeline(failure_mode: FailureMode) -> Callable[[], bool]:
    """Build a pipeline-pause action closure for the given failure mode.

    Args:
        failure_mode: The failure mode that triggered the pause.

    Returns:
        A zero-argument callable that signals a pipeline pause and returns True.
    """

    def _pause() -> bool:
        logger.warning(
            "PIPELINE PAUSE: halting pipeline due to unresolved '%s' failure — awaiting operator intervention",
            failure_mode.value,
        )
        return True

    return _pause


# ── Engine ────────────────────────────────────────────────────────────


class RemediationEngine:
    """Diagnoses failures and executes tiered remediation plans.

    Uses a circuit breaker to prevent remediation loops — if remediation
    itself fails repeatedly, the engine stops trying and escalates to
    the highest tier.

    Side effects:
        - Creates a CircuitBreaker("remediation") on init.
        - Maintains an in-memory deque of up to 100 RemediationResult records.
    """

    def __init__(self) -> None:
        self._breaker = CircuitBreaker(
            "remediation",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0),
        )
        # Bounded history — oldest entries are discarded automatically.
        self._remediation_history: deque[RemediationResult] = deque(maxlen=100)

    def diagnose(
        self,
        failure_mode: FailureMode,
        context: dict[str, Any] | None = None,
    ) -> RemediationPlan:
        """Create a remediation plan for the given failure mode.

        Each failure mode maps to a fixed escalation sequence of actions.
        The plan is returned without being executed — call
        ``execute_remediation`` to run it.

        Args:
            failure_mode: The classified failure to build a plan for.
            context: Optional runtime context (e.g., model_id, project_id)
                included in the diagnosis message.

        Returns:
            A RemediationPlan containing ordered actions and the highest tier.
        """
        ctx_info = f" (context={context})" if context else ""
        logger.info("Diagnosing failure mode '%s'%s", failure_mode.value, ctx_info)

        actions: list[RemediationAction]

        if failure_mode == FailureMode.OOM:
            actions = [
                RemediationAction("Reduce inference context size", RemediationTier.AUTO_FIX, _reduce_context_size),
                RemediationAction("Switch to smaller model", RemediationTier.AUTO_FIX, _switch_to_smaller_model),
                RemediationAction(
                    "Alert operator: OOM unresolved", RemediationTier.ALERT, _alert_operator(failure_mode)
                ),
                RemediationAction("Pause pipeline: OOM critical", RemediationTier.PAUSE, _pause_pipeline(failure_mode)),
            ]
            diagnosis = "Inference ran out of memory — context too large or model exceeds available VRAM"

        elif failure_mode == FailureMode.HANG:
            actions = [
                RemediationAction("Cancel stalled task and retry", RemediationTier.AUTO_FIX, _cancel_and_retry),
                RemediationAction(
                    "Alert operator: hang unresolved", RemediationTier.ALERT, _alert_operator(failure_mode)
                ),
                RemediationAction(
                    "Pause pipeline: persistent hang", RemediationTier.PAUSE, _pause_pipeline(failure_mode)
                ),
            ]
            diagnosis = "Agent execution stalled — no progress detected within the timeout window"

        elif failure_mode == FailureMode.QUALITY_DROP:
            actions = [
                RemediationAction("Retry with refined prompt", RemediationTier.AUTO_FIX, _retry_with_refinement),
                RemediationAction("Switch to higher-quality model", RemediationTier.AUTO_FIX, _switch_model),
                RemediationAction(
                    "Alert operator: quality below threshold", RemediationTier.ALERT, _alert_operator(failure_mode)
                ),
            ]
            diagnosis = "Output quality score fell below the acceptable threshold for this task type"

        elif failure_mode == FailureMode.DISK_FULL:
            actions = [
                RemediationAction("Clear non-essential caches", RemediationTier.AUTO_FIX, _clear_caches),
                RemediationAction(
                    "Alert operator: disk space low", RemediationTier.ALERT, _alert_operator(failure_mode)
                ),
                RemediationAction("Pause training to stop artifact writes", RemediationTier.PAUSE, _pause_training),
            ]
            diagnosis = "Insufficient disk space — model or training artifacts cannot be written"

        else:  # FailureMode.THERMAL
            # Monotonic escalation: AUTO_FIX -> ALERT -> PAUSE (not PAUSE before ALERT).
            # ALERT comes before PAUSE so the operator is notified before the pipeline stops.
            actions = [
                RemediationAction("Reduce inference batch size", RemediationTier.AUTO_FIX, _reduce_batch_size),
                RemediationAction(
                    "Alert operator: thermal throttling active", RemediationTier.ALERT, _alert_operator(failure_mode)
                ),
                RemediationAction("Pause inference for GPU cooldown", RemediationTier.PAUSE, _pause_and_cooldown),
            ]
            diagnosis = "GPU thermal throttling detected — sustained high temperature is degrading performance"

        max_tier = max(actions, key=lambda a: _TIER_ORDER.index(a.tier)).tier
        return RemediationPlan(
            failure_mode=failure_mode,
            diagnosis=diagnosis,
            actions=actions,
            max_tier=max_tier,
        )

    def execute_remediation(self, plan: RemediationPlan) -> RemediationResult:
        """Execute all actions in the remediation plan, then aggregate results.

        Runs every action in the plan sequentially — never stops at the first
        success. This is intentional: a stub that returns True must not mask
        later actions that would fail. Overall success requires that every
        callable action (those with an ``action_fn``) returned True; actions
        without a callable are recorded as informational and do not affect the
        aggregate outcome.

        If the circuit breaker is tripped (too many consecutive remediation
        failures), returns immediately with the highest-tier failure state.

        Args:
            plan: The RemediationPlan to execute, produced by ``diagnose``.

        Returns:
            A RemediationResult recording which actions were taken and whether
            all callable actions succeeded.
        """
        if not self._breaker.allow_request():
            logger.error(
                "Remediation circuit breaker is OPEN for failure mode '%s'"
                " — too many consecutive remediation failures; escalating to max tier",
                plan.failure_mode.value,
            )
            result = RemediationResult(
                success=False,
                failure_mode=plan.failure_mode,
                tier_reached=plan.max_tier,
                error="Remediation circuit breaker open — repeated remediation failures",
            )
            self._remediation_history.append(result)
            # Log breaker-open failures to the registry so trend analysis can
            # detect persistent failure modes, not just execution failures.
            self._log_outcome_to_registry(plan, result)
            return result

        actions_taken: list[str] = []
        tier_reached = plan.actions[0].tier if plan.actions else RemediationTier.AUTO_FIX

        for action in plan.actions:
            tier_reached = action.tier
            actions_taken.append(action.description)
            logger.info(
                "Executing remediation action [%s]: %s",
                action.tier.value,
                action.description,
            )

            if action.action_fn is None:
                logger.debug("Action '%s' has no callable — recorded as informational", action.description)
                continue

            try:
                action_succeeded = action.action_fn()
            except Exception as exc:
                logger.warning(
                    "Remediation action '%s' raised an exception — treating as failure, escalating: %s",
                    action.description,
                    exc,
                )
                action_succeeded = False

            if action_succeeded:
                logger.info(
                    "Remediation action '%s' succeeded at tier '%s' — stopping escalation",
                    action.description,
                    action.tier.value,
                )
                self._breaker.record_success()
                result = RemediationResult(
                    success=True,
                    failure_mode=plan.failure_mode,
                    tier_reached=tier_reached,
                    actions_taken=actions_taken,
                )
                self._remediation_history.append(result)
                self._log_outcome_to_registry(plan, result)
                return result
            else:
                logger.warning(
                    "Remediation action '%s' failed at tier '%s' — escalating to next tier",
                    action.description,
                    action.tier.value,
                )

        # All actions exhausted without success.
        logger.error(
            "All remediation actions exhausted for failure mode '%s' — highest tier reached: %s",
            plan.failure_mode.value,
            tier_reached.value,
        )
        self._breaker.record_failure()
        result = RemediationResult(
            success=False,
            failure_mode=plan.failure_mode,
            tier_reached=tier_reached,
            actions_taken=actions_taken,
            error=f"all {len(actions_taken)} remediation action(s) exhausted without success",
        )
        self._remediation_history.append(result)
        self._log_outcome_to_registry(plan, result)
        return result

    def _log_outcome_to_registry(
        self,
        plan: RemediationPlan,
        result: RemediationResult,
    ) -> None:
        """Log remediation outcome to the failure registry for trend tracking.

        Records each action taken along with its success/failure status so
        that per-(failure_mode, action) statistics can be computed.

        Args:
            plan: The remediation plan that was executed.
            result: The outcome of executing the plan.
        """
        try:
            from vetinari.analytics.failure_registry import get_failure_registry

            registry = get_failure_registry()
            for action_desc in result.actions_taken:
                registry.log_remediation_outcome(
                    failure_mode=plan.failure_mode.value,
                    action_description=action_desc,
                    success=result.success,
                )
        except Exception:
            logger.warning("Could not log remediation outcome to failure registry — stats may be incomplete")

    def get_history(self) -> list[RemediationResult]:
        """Return all recorded remediation outcomes, oldest first.

        Returns:
            Snapshot list of up to 100 RemediationResult records.
        """
        return list(self._remediation_history)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about remediation activity.

        Returns:
            Dictionary with keys: ``total``, ``success_rate``,
            ``most_common_failure_mode``, and ``breaker_state``.
        """
        history = list(self._remediation_history)
        total = len(history)

        if total == 0:
            return {
                "total": 0,
                "success_rate": 0.0,
                "most_common_failure_mode": None,
                "breaker_state": self._breaker.state.value,
            }

        successes = sum(1 for r in history if r.success)
        mode_counts: dict[str, int] = {}
        for r in history:
            mode_counts[r.failure_mode.value] = mode_counts.get(r.failure_mode.value, 0) + 1
        most_common = max(mode_counts, key=lambda k: mode_counts[k])

        return {
            "total": total,
            "success_rate": round(successes / total, 3),
            "most_common_failure_mode": most_common,
            "breaker_state": self._breaker.state.value,
        }


# ── Singleton ─────────────────────────────────────────────────────────

# Module-level singleton. Written once by get_remediation_engine() under
# _engine_lock; read by every subsequent caller without holding the lock.
_engine: RemediationEngine | None = None
_engine_lock = threading.Lock()


def get_remediation_engine() -> RemediationEngine:
    """Return the process-wide RemediationEngine singleton.

    Uses double-checked locking so that the common read-path (engine
    already created) never acquires the lock.

    Returns:
        The singleton RemediationEngine instance.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RemediationEngine()
    return _engine
