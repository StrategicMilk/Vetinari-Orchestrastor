"""Circuit Breaker — production-grade, thread-safe, per-agent breakers.

Per-agent circuit breakers that prevent cascading failures when LLM
backends or downstream services become unreliable.  Includes a singleton
registry, exponential backoff, and dashboard-friendly stats.

Three states:
  CLOSED   -- normal operation; failures are counted
  OPEN     -- tripped; calls are rejected immediately
  HALF_OPEN -- recovery probe; one call allowed through

Config per agent:
  failure_threshold  = 3     (consecutive failures to trip)
  recovery_timeout   = 60s   (time in OPEN before moving to HALF_OPEN)
  half_open_max_calls = 1    (probe calls allowed in HALF_OPEN)

Exponential backoff is applied to retries before the breaker trips.

For the lightweight, non-threaded breaker used inside the orchestration
graph layer, see :mod:`vetinari.orchestration.circuit_breaker`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""

    failure_threshold: int = 3
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 1
    # Exponential backoff for retries before tripping
    backoff_base: float = 1.0  # seconds
    backoff_max: float = 30.0  # seconds
    backoff_factor: float = 2.0


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for a circuit breaker."""

    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change: float = 0.0
    trips: int = 0  # total times breaker has tripped

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "trips": self.trips,
        }


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, breaker_name: str, remaining: float):
        self.breaker_name = breaker_name
        self.remaining = remaining
        super().__init__(f"Circuit breaker '{breaker_name}' is OPEN — retry in {remaining:.1f}s")


class CircuitBreaker:
    """Per-agent circuit breaker.

    Usage::

        cb = CircuitBreaker("planner")
        result = cb.call(lambda: llm_infer(prompt))

    Or as a guard::

        if cb.allow_request():
            try:
                result = llm_infer(prompt)
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # fallback logic
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        self._half_open_calls = 0
        self._opened_at: float = 0.0

    # ── Public properties ─────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Current state (may auto-transition from OPEN → HALF_OPEN)."""
        with self._lock:
            self._maybe_transition()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    # ── Core API ──────────────────────────────────────────────────────

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_successes += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe succeeded → close the circuit
                self._transition(CircuitState.CLOSED)
                logger.info("Circuit breaker '%s' recovered → CLOSED", self.name)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_failures += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed → re-open
                self._trip()
                logger.warning("Circuit breaker '%s' probe failed → re-OPEN", self.name)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._trip()
                    logger.warning(
                        "Circuit breaker '%s' tripped after %d consecutive failures → OPEN",
                        self.name,
                        self._stats.consecutive_failures,
                    )

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``fn`` through the circuit breaker.

        Raises CircuitBreakerOpen if the circuit is open and not ready
        for a probe.

        Returns:
            The T result.

        Raises:
            CircuitBreakerOpen: If the operation fails.
        """
        if not self.allow_request():
            remaining = self._time_until_half_open()
            self._stats.total_rejections += 1
            raise CircuitBreakerOpen(self.name, remaining)

        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            self._transition(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._stats.consecutive_successes = 0
            logger.info("Circuit breaker '%s' manually reset → CLOSED", self.name)

    def get_backoff_delay(self) -> float:
        """Return the current exponential backoff delay in seconds.

        Returns:
            The computed value.
        """
        n = self._stats.consecutive_failures
        if n == 0:
            return 0.0
        delay = self.config.backoff_base * (self.config.backoff_factor ** (n - 1))
        return min(delay, self.config.backoff_max)

    # ── Internal ──────────────────────────────────────────────────────

    def _trip(self) -> None:
        """Trip the breaker → OPEN. Must hold lock."""
        self._transition(CircuitState.OPEN)
        self._opened_at = time.monotonic()
        self._stats.trips += 1

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state. Must hold lock."""
        if self._state != new_state:
            self._state = new_state
            self._stats.last_state_change = time.monotonic()
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0

    def _maybe_transition(self) -> None:
        """Auto-transition OPEN → HALF_OPEN if recovery timeout elapsed. Must hold lock."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.config.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
                logger.info(
                    "Circuit breaker '%s' recovery timeout elapsed → HALF_OPEN",
                    self.name,
                )

    def _time_until_half_open(self) -> float:
        """Seconds until OPEN → HALF_OPEN transition."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.config.recovery_timeout - elapsed)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for dashboards."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": self._stats.to_dict(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "half_open_max_calls": self.config.half_open_max_calls,
            },
            "time_until_half_open": self._time_until_half_open(),
        }


# ── Per-agent defaults ────────────────────────────────────────────────

_AGENT_BREAKER_CONFIGS: dict[str, CircuitBreakerConfig] = {
    "PLANNER": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60),
    "CONSOLIDATED_RESEARCHER": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=90),
    "CONSOLIDATED_ORACLE": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60),
    "BUILDER": CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120),
    "QUALITY": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60),
    "OPERATIONS": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60),
}


class CircuitBreakerRegistry:
    """Singleton registry of per-agent circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, agent_type: str) -> CircuitBreaker:
        """Get or create the circuit breaker for an agent type.

        Returns:
            The CircuitBreaker result.
        """
        with self._lock:
            if agent_type not in self._breakers:
                config = _AGENT_BREAKER_CONFIGS.get(agent_type, CircuitBreakerConfig())
                self._breakers[agent_type] = CircuitBreaker(agent_type, config)
            return self._breakers[agent_type]

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Return all registered breakers.

        Returns:
            The result string.
        """
        with self._lock:
            return dict(self._breakers)

    def get_summary(self) -> dict[str, Any]:
        """Return a dashboard-friendly summary of all breakers.

        Returns:
            The result string.
        """
        with self._lock:
            return {name: cb.to_dict() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()


# ── Singleton ─────────────────────────────────────────────────────────

_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry.

    Returns:
        The result string.
    """
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry
