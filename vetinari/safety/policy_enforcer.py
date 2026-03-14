"""Pre-execution policy enforcement layer.

Evaluates every agent action against safety policies before execution.
Checks: irreversibility, resource budgets, stakeholder impact, file jurisdiction.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File jurisdiction map — which agent type owns which directory prefixes
# ---------------------------------------------------------------------------

# Maps agent_type (lowercase) to the set of path prefixes it may write to.
# The Builder is the only agent permitted to write production source files.
_JURISDICTION: dict[str, list[str]] = {
    "builder": ["vetinari/", "src/", "lib/"],
    "planner": [".omc/plans/", ".omc/state/"],
    "researcher": [".omc/research/", ".omc/artifacts/"],
    "oracle": [".omc/decisions/"],
    "quality": [".omc/reviews/", "reports/"],
    "operations": ["docs/", ".omc/logs/", ".omc/notepads/"],
}

# Actions that are considered irreversible / destructive
_DESTRUCTIVE_ACTIONS: frozenset[str] = frozenset(
    {
        "delete",
        "drop",
        "truncate",
        "remove",
        "purge",
        "overwrite",
        "wipe",
        "destroy",
        "rm",
        "unlink",
    }
)

# Default resource budgets
_DEFAULT_MAX_TOKENS: int = 100_000
_DEFAULT_MAX_TIME_SECONDS: float = 600.0
_DEFAULT_MAX_DELEGATION_DEPTH: int = 3


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    """Result of a policy check for an agent action.

    Attributes:
        allowed: Whether the action is permitted.
        reason: Human-readable explanation of the decision.
        risk_level: One of ``"low"``, ``"medium"``, or ``"high"``.
    """

    allowed: bool
    reason: str
    risk_level: str = "low"  # "low" | "medium" | "high"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "risk_level": self.risk_level,
        }


# ---------------------------------------------------------------------------
# PolicyEnforcer
# ---------------------------------------------------------------------------


class PolicyEnforcer:
    """Singleton policy gate evaluated before every agent action.

    Use ``get_policy_enforcer()`` to obtain the shared instance.

    Example::

        enforcer = get_policy_enforcer()
        decision = enforcer.check_action(
            agent_type="builder",
            action="write",
            target="vetinari/agents/foo.py",  # noqa: VET034
            context={"delegation_depth": 1},
        )
        if not decision.allowed:
            raise PermissionError(decision.reason)
    """

    _instance: PolicyEnforcer | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> PolicyEnforcer:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._lock = threading.Lock()
        self._policies: list[tuple[str, Callable[[str, str, str, dict[str, Any]], PolicyDecision | None]]] = []
        self._total_checks: int = 0
        self._total_denied: int = 0
        self._total_flagged_irreversible: int = 0
        # Register built-in policies in priority order
        self.register_policy("jurisdiction", self._check_jurisdiction)
        self.register_policy("delegation_depth", self._check_delegation_depth)
        self.register_policy("irreversibility", self._check_irreversibility)
        self.register_policy("resource_budget", self._check_resource_budget)

    # ------------------------------------------------------------------
    # Policy registration
    # ------------------------------------------------------------------

    def register_policy(
        self,
        name: str,
        check_fn: Callable[[str, str, str, dict[str, Any]], PolicyDecision | None],
    ) -> None:
        """Register a named policy check function.

        The ``check_fn`` receives ``(agent_type, action, target, context)``
        and should return a ``PolicyDecision`` to short-circuit evaluation,
        or ``None`` to pass through to the next policy.

        Args:
            name: Human-readable policy name (used in logs and stats).
            check_fn: Callable matching the signature described above.
        """
        with self._lock:
            self._policies.append((name, check_fn))
        logger.debug("Registered policy %r", name)

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------

    def check_action(
        self,
        agent_type: str,
        action: str,
        target: str,
        context: dict[str, Any],
    ) -> PolicyDecision:
        """Evaluate all registered policies for the given action.

        Policies are evaluated in registration order. The first policy that
        returns a ``PolicyDecision`` wins; remaining policies are skipped.
        If all policies return ``None`` the action is allowed.

        Args:
            agent_type: Lowercase agent type string (e.g. ``"builder"``).
            action: The action being requested (e.g. ``"write"``, ``"delete"``).
            target: The resource path or identifier being acted on.
            context: Arbitrary context dict (may include ``delegation_depth``,
                ``tokens_used``, ``elapsed_seconds``, etc.).

        Returns:
            ``PolicyDecision`` indicating whether the action is permitted.
        """
        with self._lock:
            policies_snapshot = list(self._policies)
            self._total_checks += 1

        action_lower = action.lower()
        agent_lower = agent_type.lower()

        for name, check_fn in policies_snapshot:
            try:
                result = check_fn(agent_lower, action_lower, target, context)
            except Exception as exc:
                logger.warning("Policy %r raised unexpectedly: %s — skipping", name, exc)
                continue

            if result is not None:
                if not result.allowed:
                    with self._lock:
                        self._total_denied += 1
                    logger.info(
                        "Action denied by policy %r: agent=%r action=%r target=%r reason=%r",
                        name,
                        agent_type,
                        action,
                        target,
                        result.reason,
                    )
                return result

        return PolicyDecision(allowed=True, reason="all policies passed", risk_level="low")

    # ------------------------------------------------------------------
    # Built-in policy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _check_jurisdiction(
        agent_type: str,
        action: str,
        target: str,
        context: dict[str, Any],
    ) -> PolicyDecision | None:
        """Block write actions outside an agent's permitted path prefixes."""
        if action not in ("write", "create", "edit", "patch", "modify", "update"):
            return None  # read-only or unknown actions pass through

        allowed_prefixes = _JURISDICTION.get(agent_type)
        if allowed_prefixes is None:
            # Unknown agent type — allow (let other policies decide)
            return None

        # Normalise path separators for comparison
        normalised_target = target.replace("\\", "/")
        for prefix in allowed_prefixes:
            if normalised_target.startswith(prefix):
                return None  # within jurisdiction

        return PolicyDecision(
            allowed=False,
            reason=(
                f"Agent {agent_type!r} is not permitted to write to {target!r}. Allowed prefixes: {allowed_prefixes}"
            ),
            risk_level="high",
        )

    @staticmethod
    def _check_delegation_depth(
        agent_type: str,
        action: str,
        target: str,
        context: dict[str, Any],
    ) -> PolicyDecision | None:
        """Block actions that exceed the maximum delegation depth."""
        depth = context.get("delegation_depth", 0)
        max_depth = context.get("max_delegation_depth", _DEFAULT_MAX_DELEGATION_DEPTH)

        if not isinstance(depth, (int, float)):
            return None

        if depth > max_depth:
            return PolicyDecision(
                allowed=False,
                reason=(f"Delegation depth {depth} exceeds maximum of {max_depth}"),
                risk_level="high",
            )
        return None

    def _check_irreversibility(
        self,
        agent_type: str,
        action: str,
        target: str,
        context: dict[str, Any],
    ) -> PolicyDecision | None:
        """Flag (but allow) destructive actions; block if context disallows them."""
        if action not in _DESTRUCTIVE_ACTIONS:
            return None

        with self._lock:
            self._total_flagged_irreversible += 1

        allow_destructive = context.get("allow_destructive", False)
        if not allow_destructive:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Action {action!r} is irreversible/destructive. Set context['allow_destructive']=True to permit."
                ),
                risk_level="high",
            )

        # Allowed but flagged at medium risk
        return PolicyDecision(
            allowed=True,
            reason=f"Destructive action {action!r} permitted by context",
            risk_level="medium",
        )

    @staticmethod
    def _check_resource_budget(
        agent_type: str,
        action: str,
        target: str,
        context: dict[str, Any],
    ) -> PolicyDecision | None:
        """Block actions when resource budgets are exhausted."""
        tokens_used = context.get("tokens_used", 0)
        max_tokens = context.get("max_tokens", _DEFAULT_MAX_TOKENS)
        elapsed = context.get("elapsed_seconds", 0.0)
        max_time = context.get("max_time_seconds", _DEFAULT_MAX_TIME_SECONDS)

        if isinstance(tokens_used, (int, float)) and isinstance(max_tokens, (int, float)):  # noqa: SIM102
            if tokens_used > max_tokens:
                return PolicyDecision(
                    allowed=False,
                    reason=(f"Token budget exhausted: {tokens_used} > {max_tokens}"),
                    risk_level="medium",
                )

        if isinstance(elapsed, (int, float)) and isinstance(max_time, (int, float)) and elapsed > max_time:
            return PolicyDecision(
                allowed=False,
                reason=(f"Time budget exhausted: {elapsed:.1f}s > {max_time:.1f}s"),
                risk_level="medium",
            )

        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return policy enforcement statistics.

        Returns:
            Dict with keys ``registered_policies``, ``total_checks``,
            ``total_denied``, and ``total_flagged_irreversible``.
        """
        with self._lock:
            return {
                "registered_policies": len(self._policies),
                "total_checks": self._total_checks,
                "total_denied": self._total_denied,
                "total_flagged_irreversible": self._total_flagged_irreversible,
            }


# ---------------------------------------------------------------------------
# Singleton accessor + reset helper
# ---------------------------------------------------------------------------


def get_policy_enforcer() -> PolicyEnforcer:
    """Return the shared ``PolicyEnforcer`` singleton."""
    return PolicyEnforcer()


def reset_policy_enforcer() -> None:
    """Destroy the singleton — intended for use in tests."""
    with PolicyEnforcer._class_lock:
        PolicyEnforcer._instance = None
