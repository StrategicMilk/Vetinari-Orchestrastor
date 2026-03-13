"""Circuit Breaker and Stagnation Detection for Agent Orchestration.

=================================================================

Provides two resilience primitives for the orchestration layer:

- **CircuitBreaker**: Prevents cascading failures by tracking consecutive
  errors and temporarily halting calls to a failing subsystem.
- **StagnationDetector**: Monitors task execution progress and flags runs
  that are stuck (repeated outputs, excessive errors, or blown time budgets).

Usage::

    from vetinari.orchestration.circuit_breaker import CircuitBreaker, StagnationDetector

    cb = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    if cb.allow_request():
        try:
            result = call_agent(...)
            cb.record_success()
        except Exception:
            cb.record_failure()

    detector = StagnationDetector(
        max_repeated_outputs=3,
        error_threshold=10,
        time_budget=300.0,
    )
    detector.record_output("same thing")
    detector.record_output("same thing")
    detector.record_output("same thing")
    assert detector.is_stagnant()
"""

from __future__ import annotations

import time
from enum import Enum


class CircuitState(Enum):
    """Possible states of the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Detect and react to sustained agent failures.

    The breaker starts **CLOSED** (healthy).  After *failure_threshold*
    consecutive failures it transitions to **OPEN**, rejecting every
    request.  Once *reset_timeout* seconds have elapsed the breaker
    moves to **HALF_OPEN** and allows a single probe request.  A success
    resets to CLOSED; a failure re-opens the breaker.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        reset_timeout: Seconds to wait in OPEN before probing.
        max_retries: Maximum retries the caller should attempt before
            giving up entirely.  Exposed as an advisory attribute —
            enforcement is the caller's responsibility.

    Example:
        >>> cb = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)
        >>> cb.state
        <CircuitState.CLOSED: 'closed'>
        >>> for _ in range(3):
        ...     cb.record_failure()
        >>> cb.state
        <CircuitState.OPEN: 'open'>
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if reset_timeout < 0:
            raise ValueError("reset_timeout must be >= 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.failure_threshold: int = failure_threshold
        self.reset_timeout: float = reset_timeout
        self.max_retries: int = max_retries

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: float | None = None
        self._total_failures: int = 0
        self._total_successes: int = 0

    # -- public properties ---------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, advancing from OPEN to HALF_OPEN if the timeout has elapsed."""
        if self._state is CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Successes recorded since the last failure (or since creation)."""
        return self._success_count

    @property
    def total_failures(self) -> int:
        """Lifetime failure count."""
        return self._total_failures

    @property
    def total_successes(self) -> int:
        """Lifetime success count."""
        return self._total_successes

    @property
    def error_rate(self) -> float:
        """Lifetime error rate as a fraction in [0.0, 1.0].

        Returns 0.0 when no calls have been recorded.
        """
        total = self._total_failures + self._total_successes
        if total == 0:
            return 0.0
        return self._total_failures / total

    # -- public methods ------------------------------------------------------

    def allow_request(self) -> bool:
        """Return whether a request should be attempted.

        Returns:
            True if the circuit is CLOSED or HALF_OPEN, False if OPEN.
        """
        current = self.state  # triggers timeout check
        return current is not CircuitState.OPEN

    def record_failure(self) -> None:
        """Record a failed call and potentially open the circuit."""
        self._failure_count += 1
        self._total_failures += 1
        self._success_count = 0
        self._last_failure_time = time.monotonic()

        if self._state is CircuitState.HALF_OPEN:
            # Probe failed — reopen immediately.
            self._state = CircuitState.OPEN
        elif self._state is CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call, potentially closing the circuit."""
        self._success_count += 1
        self._total_successes += 1
        self._failure_count = 0

        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED

    def reset(self) -> None:
        """Force the breaker back to CLOSED with zero counters."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


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
    ) -> None:
        if max_repeated_outputs < 1:
            raise ValueError("max_repeated_outputs must be >= 1")
        if error_threshold < 1:
            raise ValueError("error_threshold must be >= 1")
        if time_budget is not None and time_budget <= 0:
            raise ValueError("time_budget must be > 0 when set")

        self.max_repeated_outputs: int = max_repeated_outputs
        self.error_threshold: int = error_threshold
        self.time_budget: float | None = time_budget

        self._start_time: float = time.monotonic()
        self._error_count: int = 0
        self._last_output: str | None = None
        self._repeat_count: int = 0
        self._output_history: list[str] = []

    # -- public properties ---------------------------------------------------

    @property
    def error_count(self) -> int:
        """Total recorded errors."""
        return self._error_count

    @property
    def repeat_count(self) -> int:
        """Current consecutive identical-output count."""
        return self._repeat_count

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
            self._repeat_count += 1
        else:
            self._last_output = output
            self._repeat_count = 1

    def record_error(self) -> None:
        """Increment the error counter."""
        self._error_count += 1

    # -- queries -------------------------------------------------------------

    def is_stagnant(self) -> bool:
        """Return True if any stagnation condition is met.

        Returns:
            True when the detector considers execution stalled.
        """
        if self._repeat_count >= self.max_repeated_outputs:
            return True
        if self._error_count >= self.error_threshold:
            return True
        return bool(self.time_budget is not None and self.elapsed >= self.time_budget)

    def stagnation_reasons(self) -> list[str]:
        """Return human-readable descriptions of all active stagnation triggers.

        Returns:
            A list of reason strings (empty when not stagnant).
        """
        reasons: list[str] = []
        if self._repeat_count >= self.max_repeated_outputs:
            reasons.append(
                f"Repeated output detected: same output seen "
                f"{self._repeat_count} times (threshold: {self.max_repeated_outputs})"
            )
        if self._error_count >= self.error_threshold:
            reasons.append(f"Error threshold exceeded: {self._error_count} errors (threshold: {self.error_threshold})")
        if self.time_budget is not None and self.elapsed >= self.time_budget:
            reasons.append(f"Time budget exceeded: {self.elapsed:.1f}s elapsed (budget: {self.time_budget:.1f}s)")
        return reasons

    def reset(self) -> None:
        """Reset all internal counters and restart the clock."""
        self._start_time = time.monotonic()
        self._error_count = 0
        self._last_output = None
        self._repeat_count = 0
        self._output_history.clear()
