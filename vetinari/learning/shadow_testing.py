"""Shadow Testing Framework — safe A/B testing for system improvements.

Forks the current config, runs a candidate alongside production on the same
workload, compares metrics, and only promotes if ALL metrics pass thresholds.
Includes 24-hour auto-rollback on degradation.

This is the safety gate between learning subsystem proposals and production
deployment. All improvements (prompt evolution, parameter tuning, training)
flow through shadow testing before promotion.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from vetinari.constants import get_user_dir

logger = logging.getLogger(__name__)

# Promotion thresholds — ALL must pass for a candidate to be promoted
_MIN_QUALITY_IMPROVEMENT = -0.02  # Allow up to 2% quality regression
_MAX_LATENCY_RATIO = 1.3  # Candidate can be at most 30% slower than production
_MAX_ERROR_RATE_INCREASE = 0.05  # Allow at most 5 percentage points more errors
_ROLLBACK_WINDOW_HOURS = 24  # Auto-rollback if degradation detected within this window
_ROLLBACK_QUALITY_MARGIN = 0.05  # Roll back if quality falls 5 pts below pre-promotion baseline


@dataclass
class ShadowMetrics:
    """Collected metrics for one side (production or candidate) of a shadow test."""

    quality_scores: list[float] = field(default_factory=list)  # noqa: VET220 — capped to 500 at append site; list kept for json.dump(asdict()) compatibility
    latency_ms_values: list[float] = field(default_factory=list)  # noqa: VET220 — capped to 500 at append site; list kept for json.dump(asdict()) compatibility
    error_count: int = 0
    total_runs: int = 0

    def __repr__(self) -> str:
        return f"ShadowMetrics(total_runs={self.total_runs!r}, error_count={self.error_count!r})"

    @property
    def avg_quality(self) -> float:
        """Mean quality score across all runs (0.0 if no data yet)."""
        return sum(self.quality_scores) / max(len(self.quality_scores), 1)

    @property
    def avg_latency(self) -> float:
        """Mean latency in milliseconds (0.0 if no data yet)."""
        return sum(self.latency_ms_values) / max(len(self.latency_ms_values), 1)

    @property
    def error_rate(self) -> float:
        """Fraction of runs that produced errors (0.0 if no runs yet)."""
        return self.error_count / max(self.total_runs, 1)


@dataclass
class ShadowTest:
    """A shadow test comparing production vs candidate configuration.

    Tracks paired metric observations for both variants until enough data
    accumulates for a statistically reasonable promotion decision.
    """

    test_id: str
    description: str
    production_config: dict[str, Any]
    candidate_config: dict[str, Any]
    production_metrics: ShadowMetrics = field(default_factory=ShadowMetrics)
    candidate_metrics: ShadowMetrics = field(default_factory=ShadowMetrics)
    # Status lifecycle: running -> promoted | rejected | rolled_back
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: str | None = None
    min_samples: int = 10  # Minimum observations before a promotion decision is allowed

    def __repr__(self) -> str:
        return f"ShadowTest(test_id={self.test_id!r}, status={self.status!r})"


class ShadowTestRunner:
    """Manages shadow tests for safe improvement deployment.

    Coordinates config forking, parallel metric collection, promotion
    decisions, and auto-rollback within 24 hours of promotion.

    All public methods are thread-safe via a single reentrant lock.

    Usage::

        runner = get_shadow_test_runner()
        test_id = runner.create_test("Prompt evolution v3", prod_cfg, cand_cfg)

        # Record observations from both variants as they run
        runner.record_production(test_id, quality=0.8, latency_ms=200)
        runner.record_candidate(test_id, quality=0.85, latency_ms=210)

        # Evaluate once enough samples have accumulated
        decision = runner.evaluate(test_id)

        # Periodically check for rollback if the test was promoted
        runner.check_rollback(test_id, current_quality=recent_avg)
    """

    def __init__(self) -> None:
        # Maps test_id -> ShadowTest; guarded by _lock for all mutations
        self._tests: dict[str, ShadowTest] = {}
        self._lock = threading.Lock()
        self._load_state()

    def create_test(
        self,
        description: str,
        production_config: dict[str, Any],
        candidate_config: dict[str, Any],
        min_samples: int = 10,
    ) -> str:
        """Create a new shadow test and persist it immediately.

        Args:
            description: Human-readable description of the improvement being tested.
            production_config: Snapshot of the current production configuration.
            candidate_config: Proposed candidate configuration to evaluate.
            min_samples: Minimum observations per variant before evaluate() will decide.

        Returns:
            Unique test ID (e.g. ``shadow_a3f9b1c2``).
        """
        test_id = f"shadow_{uuid.uuid4().hex[:8]}"
        test = ShadowTest(
            test_id=test_id,
            description=description,
            production_config=production_config,
            candidate_config=candidate_config,
            min_samples=min_samples,
        )

        with self._lock:
            self._tests[test_id] = test
            self._save_state()

        logger.info("[ShadowTest] Created test %s: %s", test_id, description)
        return test_id

    def record_production(
        self,
        test_id: str,
        quality: float,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record one production-variant observation for a running test.

        Silently ignores observations for tests that have already concluded
        (promoted, rejected, or rolled back) so callers need not check status
        before recording.

        Args:
            test_id: Shadow test ID returned by :meth:`create_test`.
            quality: Quality score for this run, 0.0-1.0.
            latency_ms: Wall-clock inference latency in milliseconds.
            error: Whether this run ended in an error condition.
        """
        with self._lock:
            test = self._tests.get(test_id)
            if test and test.status == "running":
                test.production_metrics.quality_scores.append(quality)
                if len(test.production_metrics.quality_scores) > 500:
                    test.production_metrics.quality_scores = test.production_metrics.quality_scores[-500:]
                test.production_metrics.latency_ms_values.append(latency_ms)
                if len(test.production_metrics.latency_ms_values) > 500:
                    test.production_metrics.latency_ms_values = test.production_metrics.latency_ms_values[-500:]
                test.production_metrics.total_runs += 1
                if error:
                    test.production_metrics.error_count += 1

    def record_candidate(
        self,
        test_id: str,
        quality: float,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record one candidate-variant observation for a running test.

        Silently ignores observations for tests that have already concluded
        so callers need not check status before recording.

        Args:
            test_id: Shadow test ID returned by :meth:`create_test`.
            quality: Quality score for this run, 0.0-1.0.
            latency_ms: Wall-clock inference latency in milliseconds.
            error: Whether this run ended in an error condition.
        """
        with self._lock:
            test = self._tests.get(test_id)
            if test and test.status == "running":
                test.candidate_metrics.quality_scores.append(quality)
                if len(test.candidate_metrics.quality_scores) > 500:
                    test.candidate_metrics.quality_scores = test.candidate_metrics.quality_scores[-500:]
                test.candidate_metrics.latency_ms_values.append(latency_ms)
                if len(test.candidate_metrics.latency_ms_values) > 500:
                    test.candidate_metrics.latency_ms_values = test.candidate_metrics.latency_ms_values[-500:]
                test.candidate_metrics.total_runs += 1
                if error:
                    test.candidate_metrics.error_count += 1

    def evaluate(self, test_id: str) -> dict[str, Any]:
        """Decide whether the candidate variant should be promoted to production.

        ALL three thresholds must pass for promotion:
        - Quality regression no worse than ``_MIN_QUALITY_IMPROVEMENT``
        - Latency ratio no higher than ``_MAX_LATENCY_RATIO``
        - Error-rate increase no higher than ``_MAX_ERROR_RATE_INCREASE``

        Mutates test status to ``"promoted"`` or ``"rejected"`` and persists state.

        Args:
            test_id: Shadow test ID returned by :meth:`create_test`.

        Returns:
            Dict containing ``decision`` (one of ``"promote"``, ``"reject"``,
            ``"insufficient_data"``, ``"not_found"``, or the current status if
            already concluded) plus supporting metric values and reasoning.
        """
        with self._lock:
            test = self._tests.get(test_id)
            if not test:
                return {"decision": "not_found", "reasoning": f"Test {test_id} not found"}

            if test.status != "running":
                return {"decision": test.status, "reasoning": f"Test already {test.status}"}

            prod = test.production_metrics
            cand = test.candidate_metrics

            if prod.total_runs < test.min_samples or cand.total_runs < test.min_samples:
                return {
                    "decision": "insufficient_data",
                    "production_runs": prod.total_runs,
                    "candidate_runs": cand.total_runs,
                    "min_samples": test.min_samples,
                }

            quality_delta = cand.avg_quality - prod.avg_quality
            latency_ratio = cand.avg_latency / max(prod.avg_latency, 1.0)
            error_rate_delta = cand.error_rate - prod.error_rate

            reasons: list[str] = []
            if quality_delta < _MIN_QUALITY_IMPROVEMENT:
                reasons.append(
                    f"Quality regression: delta {quality_delta:+.3f} worse than floor {_MIN_QUALITY_IMPROVEMENT:+.3f}"
                )
            if latency_ratio > _MAX_LATENCY_RATIO:
                reasons.append(
                    f"Latency regression: candidate is {latency_ratio:.2f}x slower, limit is {_MAX_LATENCY_RATIO:.2f}x"
                )
            if error_rate_delta > _MAX_ERROR_RATE_INCREASE:
                reasons.append(
                    f"Error rate increase: {error_rate_delta:+.3f} exceeds limit {_MAX_ERROR_RATE_INCREASE:+.3f}"
                )

            if reasons:
                test.status = "rejected"
                self._save_state()
                logger.info(
                    "[ShadowTest] Rejected %s — %d threshold(s) failed: %s",
                    test_id,
                    len(reasons),
                    "; ".join(reasons),
                )
                return {
                    "decision": "reject",
                    "quality_delta": round(quality_delta, 4),
                    "latency_ratio": round(latency_ratio, 3),
                    "error_rate_delta": round(error_rate_delta, 4),
                    "reasons": reasons,
                }

            test.status = "promoted"
            test.promoted_at = datetime.now(timezone.utc).isoformat()
            self._save_state()

            logger.info(
                "[ShadowTest] Promoted %s: quality_delta=%+.3f, latency_ratio=%.2f, error_rate_delta=%+.3f",
                test_id,
                quality_delta,
                latency_ratio,
                error_rate_delta,
            )
            return {
                "decision": "promote",
                "quality_delta": round(quality_delta, 4),
                "latency_ratio": round(latency_ratio, 3),
                "error_rate_delta": round(error_rate_delta, 4),
                "candidate_config": test.candidate_config,
            }

    def check_rollback(self, test_id: str, current_quality: float) -> bool:
        """Check whether a promoted test should be rolled back due to degradation.

        Should be called periodically (e.g. every 15 minutes) after promotion,
        passing the most recent rolling-average quality score. If the quality
        has fallen more than ``_ROLLBACK_QUALITY_MARGIN`` below the pre-promotion
        production baseline AND the promotion is still within the rollback window,
        the test is marked ``"rolled_back"`` and ``True`` is returned.

        Callers are responsible for reverting the actual configuration — this
        method only updates the test status and persists it.

        Args:
            test_id: Shadow test ID returned by :meth:`create_test`.
            current_quality: Recent production quality score (0.0-1.0).

        Returns:
            ``True`` if rollback was triggered; ``False`` otherwise.
        """
        with self._lock:
            test = self._tests.get(test_id)
            if not test or test.status != "promoted" or not test.promoted_at:
                return False

            promoted_time = datetime.fromisoformat(test.promoted_at)
            elapsed_hours = (datetime.now(timezone.utc) - promoted_time).total_seconds() / 3600
            if elapsed_hours > _ROLLBACK_WINDOW_HOURS:
                # Promotion is now permanent — past the rollback window
                return False

            baseline_quality = test.production_metrics.avg_quality
            if current_quality < baseline_quality - _ROLLBACK_QUALITY_MARGIN:
                test.status = "rolled_back"
                self._save_state()
                logger.warning(
                    "[ShadowTest] Rolling back %s: current_quality=%.3f dropped below "
                    "baseline=%.3f - %.3f margin (promoted %.1fh ago)",
                    test_id,
                    current_quality,
                    baseline_quality,
                    _ROLLBACK_QUALITY_MARGIN,
                    elapsed_hours,
                )
                return True

        return False

    def get_active_tests(self) -> list[dict[str, Any]]:
        """Return summary dicts for all currently-running shadow tests.

        Returns:
            List of dicts, each containing ``test_id``, ``description``,
            ``status``, ``production_runs``, ``candidate_runs``, and
            ``created_at``.
        """
        with self._lock:
            return [
                {
                    "test_id": t.test_id,
                    "description": t.description,
                    "status": t.status,
                    "production_runs": t.production_metrics.total_runs,
                    "candidate_runs": t.candidate_metrics.total_runs,
                    "created_at": t.created_at,
                }
                for t in self._tests.values()
                if t.status == "running"
            ]

    def _load_state(self) -> None:
        """Restore shadow test state from the JSON persistence file on disk.

        Failures are logged at WARNING level and the runner starts empty rather
        than aborting startup — a missing or corrupt state file is recoverable.
        """
        state_file = get_user_dir() / "shadow_tests.json"
        if not state_file.exists():
            return
        try:
            with state_file.open(encoding="utf-8") as f:
                data = json.load(f)
            for test_data in data.get("tests", []):
                prod_m = test_data.pop("production_metrics", {})
                cand_m = test_data.pop("candidate_metrics", {})
                test = ShadowTest(**test_data)
                test.production_metrics = ShadowMetrics(**prod_m) if prod_m else ShadowMetrics()
                test.candidate_metrics = ShadowMetrics(**cand_m) if cand_m else ShadowMetrics()
                self._tests[test.test_id] = test
            logger.info("[ShadowTest] Loaded %d test(s) from %s", len(self._tests), state_file)
        except Exception as exc:
            logger.warning(
                "[ShadowTest] Could not load state from %s — starting empty: %s",
                state_file,
                exc,
            )

    def _save_state(self) -> None:
        """Persist shadow test state to the JSON file on disk.

        Called under _lock. Failures are logged at WARNING — a save failure
        means the in-memory state continues to work but will not survive restart.
        """
        from dataclasses import asdict

        state_file = get_user_dir() / "shadow_tests.json"
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"tests": [asdict(t) for t in self._tests.values()]}
            with state_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(
                "[ShadowTest] Could not save state to %s — in-memory state intact: %s",
                state_file,
                exc,
            )


# Module-level singleton; guarded by double-checked locking.
# Written by: get_shadow_test_runner()
# Read by: learning subsystem, training pipeline, prompt evolution
_shadow_runner: ShadowTestRunner | None = None
_shadow_runner_lock = threading.Lock()


def get_shadow_test_runner() -> ShadowTestRunner:
    """Return the singleton ShadowTestRunner instance (thread-safe).

    Uses double-checked locking so only the first caller pays initialisation
    cost (state file load). All subsequent callers return the cached instance.

    Returns:
        The shared :class:`ShadowTestRunner` instance.
    """
    global _shadow_runner
    if _shadow_runner is None:
        with _shadow_runner_lock:
            if _shadow_runner is None:
                _shadow_runner = ShadowTestRunner()
    return _shadow_runner
