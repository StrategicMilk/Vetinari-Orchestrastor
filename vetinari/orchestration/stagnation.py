"""Stagnation Detection for Agent Orchestration.

Monitors task execution progress and flags runs that are stuck
(repeated outputs, excessive errors, or blown time budgets).

Supports both a global mode (single detector for the whole run) and a
scoped mode (per-department / per-task tracking via detect_scoped).

Usage::

    from vetinari.orchestration.stagnation import StagnationDetector

    detector = StagnationDetector(
        max_repeated_outputs=3,
        error_threshold=10,
        time_budget=300.0,
    )
    # Global tracking
    detector.record_output("same thing")
    detector.record_output("same thing")
    detector.record_output("same thing")
    assert detector.is_stagnant()

    # Scoped tracking
    detector.detect_scoped("dept-A", "output")
    stagnant = detector.get_stagnant_scopes()
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _ScopeState:
    """Per-scope stagnation tracking state.

    Holds the last output string, consecutive repeat count, cumulative
    error count, and a bounded history of all outputs seen for one scope
    (task_id, dept label, etc.).
    """

    last_output: str | None = None
    repeat_count: int = 0
    error_count: int = 0
    output_history: deque[str] = field(default_factory=lambda: deque(maxlen=1000))

    def __repr__(self) -> str:
        return "_ScopeState(...)"


class StagnationDetector:
    """Monitor task execution for signs of stagnation.

    Stagnation is declared when **any** of the following conditions is met:

    * The same output string has been recorded *max_repeated_outputs*
      times in a row.
    * The cumulative error count reaches *error_threshold*.
    * Wall-clock time since the detector was started exceeds
      *time_budget* seconds.

    Args:
        max_repeated_outputs: Consecutive identical outputs before
            stagnation is declared.
        error_threshold: Total errors before stagnation is declared.
        time_budget: Maximum allowed execution time in seconds.
            ``None`` disables the time check.

    Example:
        >>> d = StagnationDetector(max_repeated_outputs=2, error_threshold=5)
        >>> d.record_output("x")
        >>> d.record_output("x")
        >>> d.is_stagnant()
        True
    """

    def __init__(
        self,
        max_repeated_outputs: int = 3,
        error_threshold: int = 10,
        time_budget: float | None = None,
        scope: str = "",
    ) -> None:
        """Initialise the stagnation detector.

        Args:
            max_repeated_outputs: Consecutive identical outputs before
                stagnation is declared.  Must be >= 2.
            error_threshold: Total errors before stagnation is declared.
            time_budget: Maximum allowed wall-clock time in seconds.
                ``None`` disables the time check.
            scope: Optional label grouping this detector within a plan
                (e.g. task_id or plan_id) for structured log messages.
        """
        if max_repeated_outputs < 2:
            raise ValueError("max_repeated_outputs must be >= 2 (stagnation requires at least one repetition)")
        if error_threshold < 1:
            raise ValueError("error_threshold must be >= 1")
        if time_budget is not None and time_budget <= 0:
            raise ValueError("time_budget must be > 0 when set")

        self.max_repeated_outputs: int = max_repeated_outputs
        self.error_threshold: int = error_threshold
        self.time_budget: float | None = time_budget
        self.scope: str = scope  # Identifies which plan/task is being monitored

        self._start_time: float = time.monotonic()
        self.error_count: int = 0
        self._last_output: str | None = None
        self.repeat_count: int = 0
        self._output_history: deque[str] = deque(maxlen=1000)

        # Per-scope tracking: scope_name -> _ScopeState
        self.per_scope_state: dict[str, _ScopeState] = {}

    # -- public properties ---------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since the detector was created."""
        return time.monotonic() - self._start_time

    @property
    def output_history(self) -> list[str]:
        """Chronological list of all recorded outputs."""
        return list(self._output_history)

    # -- recording -----------------------------------------------------------

    def record_output(self, output: str) -> None:
        """Record an agent output and update the repeat counter.

        Args:
            output: The output string produced by the agent.
        """
        self._output_history.append(output)
        if output == self._last_output:
            self.repeat_count += 1
        else:
            self._last_output = output
            self.repeat_count = 1

    def record_error(self) -> None:
        """Increment the error counter."""
        self.error_count += 1

    # -- queries -------------------------------------------------------------

    def is_stagnant(self) -> bool:
        """Return True if any stagnation condition is met.

        Returns:
            True when the detector considers execution stalled.
        """
        if self.repeat_count >= self.max_repeated_outputs:
            return True
        if self.error_count >= self.error_threshold:
            return True
        return bool(self.time_budget is not None and self.elapsed >= self.time_budget)

    def stagnation_reasons(self) -> list[str]:
        """Return human-readable descriptions of all active stagnation triggers.

        Returns:
            A list of reason strings (empty when not stagnant).
        """
        reasons: list[str] = []
        if self.repeat_count >= self.max_repeated_outputs:
            reasons.append(
                f"Repeated output detected: same output seen "
                f"{self.repeat_count} times (threshold: {self.max_repeated_outputs})",
            )
        if self.error_count >= self.error_threshold:
            reasons.append(f"Error threshold exceeded: {self.error_count} errors (threshold: {self.error_threshold})")
        if self.time_budget is not None and self.elapsed >= self.time_budget:
            reasons.append(f"Time budget exceeded: {self.elapsed:.1f}s elapsed (budget: {self.time_budget:.1f}s)")
        return reasons

    def reset(self) -> None:
        """Reset global counters and restart the clock.

        Scoped state in ``per_scope_state`` is intentionally preserved so
        that per-department history survives a global reset.  Call
        ``reset_scope()`` to clear individual scopes.
        """
        self._start_time = time.monotonic()
        self.error_count = 0
        self._last_output = None
        self.repeat_count = 0
        self._output_history.clear()

    # -- scoped API ----------------------------------------------------------

    def detect_scoped(self, scope: str, output: str, *, error: bool = False) -> bool:
        """Record an output (and optionally an error) for a named scope.

        Each scope maintains its own counters independently from the global
        counters and from every other scope.  The global time_budget, when set,
        is checked against the detector's overall elapsed time.

        Args:
            scope: Identifier for this scope (e.g. task_id, dept label).
            output: The output string produced by the agent for this scope.
            error: If True, also increment the scope's error count.

        Returns:
            True if this scope is now stagnant, False otherwise.
        """
        state = self.per_scope_state.setdefault(scope, _ScopeState())

        # Update repeat tracker
        if output == state.last_output:
            state.repeat_count += 1
        else:
            state.last_output = output
            state.repeat_count = 1
        state.output_history.append(output)

        if error:
            state.error_count += 1

        return self._is_scope_stagnant(state)

    def _is_scope_stagnant(self, state: _ScopeState) -> bool:
        """Return True if the given scope state meets any stagnation condition.

        Args:
            state: The _ScopeState to evaluate.

        Returns:
            True when the scope is considered stagnant.
        """
        if state.repeat_count >= self.max_repeated_outputs:
            return True
        if state.error_count >= self.error_threshold:
            return True
        return bool(self.time_budget is not None and self.elapsed >= self.time_budget)

    def get_stagnant_scopes(self) -> list[str]:
        """Return sorted list of scope names that are currently stagnant.

        Returns:
            Sorted list of stagnant scope identifiers.
        """
        return sorted(scope for scope, state in self.per_scope_state.items() if self._is_scope_stagnant(state))

    def reset_scope(self, scope: str) -> None:
        """Clear stagnation state for a single scope.

        Has no effect if the scope has never been seen.

        Args:
            scope: The scope identifier to reset.
        """
        self.per_scope_state.pop(scope, None)
