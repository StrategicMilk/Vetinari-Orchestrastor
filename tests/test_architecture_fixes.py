"""Tests for Department 4.5 Architecture Fixes (US-087).

Covers:
- Enforcement composite errors (collect-all instead of fail-on-first)
- Event bus eviction logging with counter
- Audit log absolute path resolution
- Feedback loop init retry with warnings
- Cost optimizer silent failure warnings
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.events import _HISTORY_MAX_LENGTH, Event, EventBus

# ---------------------------------------------------------------------------
# Enforcement composite errors
# ---------------------------------------------------------------------------


class TestEnforcementCompositeErrors:
    """enforce_all() collects all violations before raising."""

    def test_single_violation_raises_original(self):
        """Single violation raises the original exception type, not composite."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import QualityGateFailed
        from vetinari.types import AgentType

        with pytest.raises(QualityGateFailed):
            enforce_all(AgentType.WORKER, quality_score=0.0)

    def test_multiple_violations_raises_composite(self):
        """Multiple violations raise CompositeEnforcementError."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import CompositeEnforcementError
        from vetinari.types import AgentType

        with pytest.raises(CompositeEnforcementError) as exc_info:
            enforce_all(
                AgentType.WORKER,
                quality_score=0.0,
                current_depth=999,
            )
        assert len(exc_info.value.violations) == 2
        assert "2 enforcement violation(s)" in str(exc_info.value)

    def test_no_violations_passes_silently(self):
        """No violations means no exception."""
        from vetinari.enforcement import enforce_all
        from vetinari.types import AgentType

        # High quality, low depth — should pass
        result = enforce_all(AgentType.WORKER, quality_score=1.0, current_depth=0)
        assert result is None  # enforce_all() returns None when all checks pass

    def test_composite_error_has_violations_list(self):
        """CompositeEnforcementError stores the list of violations."""
        from vetinari.exceptions import CompositeEnforcementError

        v1 = ValueError("first")
        v2 = ValueError("second")
        err = CompositeEnforcementError([v1, v2])
        assert err.violations == [v1, v2]
        assert "2 enforcement violation(s)" in str(err)


# ---------------------------------------------------------------------------
# Event bus eviction logging
# ---------------------------------------------------------------------------


class TestEventBusEviction:
    """EventBus logs on eviction with counter."""

    def test_eviction_count_starts_at_zero(self):
        """Fresh event bus has zero evictions."""
        bus = EventBus()
        assert bus.eviction_count == 0

    def test_eviction_count_increments(self):
        """Publishing beyond maxlen increments eviction counter."""
        bus = EventBus()
        # Fill the deque to capacity
        import time

        for _ in range(_HISTORY_MAX_LENGTH):
            bus.publish(Event(event_type="test", timestamp=time.time()))
        assert bus.eviction_count == 0

        # One more triggers eviction
        bus.publish(Event(event_type="overflow", timestamp=time.time()))
        assert bus.eviction_count == 1

    def test_eviction_logs_warning(self, caplog):
        """First eviction logs a warning message."""
        import time

        bus = EventBus()
        for _ in range(_HISTORY_MAX_LENGTH):
            bus.publish(Event(event_type="fill", timestamp=time.time()))

        with caplog.at_level(logging.WARNING, logger="vetinari.events"):
            bus.publish(Event(event_type="overflow", timestamp=time.time()))

        assert any("evicting oldest event" in rec.message for rec in caplog.records)

    def test_clear_resets_eviction_count(self):
        """clear() resets the eviction counter."""
        import time

        bus = EventBus()
        for _ in range(_HISTORY_MAX_LENGTH + 5):
            bus.publish(Event(event_type="test", timestamp=time.time()))
        assert bus.eviction_count > 0
        bus.clear()
        assert bus.eviction_count == 0


# ---------------------------------------------------------------------------
# Audit log absolute path resolution
# ---------------------------------------------------------------------------


class TestAuditPathResolution:
    """audit.py resolves absolute paths."""

    def test_relative_path_resolved_to_absolute(self):
        """Relative audit_dir is resolved to an absolute path."""
        from vetinari.audit import AuditLogger

        logger_inst = AuditLogger(audit_dir="logs/audit")
        assert logger_inst.file_path.is_absolute()

    def test_absolute_path_stays_absolute(self, tmp_path):
        """Absolute audit_dir remains absolute."""
        from vetinari.audit import AuditLogger

        logger_inst = AuditLogger(audit_dir=str(tmp_path / "audit"))
        assert logger_inst.file_path.is_absolute()
        assert str(tmp_path) in str(logger_inst.file_path)


# ---------------------------------------------------------------------------
# Feedback loop init retry
# ---------------------------------------------------------------------------


class TestFeedbackLoopResilience:
    """feedback_loop.py warns + retries on init failure."""

    def test_retries_memory_init(self, caplog):
        """_get_memory retries on failure and logs warning."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        with (
            patch("vetinari.learning.feedback_loop.FeedbackLoop._MAX_INIT_RETRIES", 2),
            patch(
                "vetinari.memory.get_memory_store",
                side_effect=RuntimeError("connection failed"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = fl._get_memory()
        assert result is None
        warning_msgs = [r for r in caplog.records if "Failed to initialize memory store" in r.message]
        assert len(warning_msgs) >= 1

    def test_retries_router_init(self, caplog):
        """_get_router retries on failure and logs warning."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        with (
            patch("vetinari.learning.feedback_loop.FeedbackLoop._MAX_INIT_RETRIES", 2),
            patch(
                "vetinari.models.dynamic_model_router.get_model_router",
                side_effect=RuntimeError("no router"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = fl._get_router()
        assert result is None
        warning_msgs = [r for r in caplog.records if "Failed to initialize model router" in r.message]
        assert len(warning_msgs) >= 1

    def test_successful_init_no_retry(self):
        """Successful init on first attempt does not retry."""
        from vetinari.learning.feedback_loop import FeedbackLoop

        fl = FeedbackLoop()
        mock_store = MagicMock()
        with patch("vetinari.memory.get_memory_store", return_value=mock_store):
            result = fl._get_memory()
        assert result is mock_store


# ---------------------------------------------------------------------------
# Cost optimizer silent failure warning
# ---------------------------------------------------------------------------


class TestCostOptimizerWarnings:
    """cost_optimizer.py warns on failure instead of silent None."""

    def test_warns_when_tracker_unavailable(self, caplog):
        """Logs warning when CostTracker is unavailable."""
        from vetinari.learning.cost_optimizer import CostOptimizer

        opt = CostOptimizer()
        opt._cost_tracker = None  # Ensure tracker is not set

        with (
            patch.object(opt, "_get_cost_tracker", return_value=None),
            caplog.at_level(logging.WARNING),
        ):
            efficiencies = opt._get_efficiencies("coding", ["model-a"])

        assert len(efficiencies) == 1
        warning_msgs = [r for r in caplog.records if "CostTracker unavailable" in r.message]
        assert len(warning_msgs) >= 1

    def test_warns_on_cost_data_failure(self, caplog):
        """Logs warning when cost data retrieval fails."""
        from vetinari.learning.cost_optimizer import CostOptimizer

        opt = CostOptimizer()
        mock_tracker = MagicMock()
        mock_tracker.get_report.side_effect = RuntimeError("db error")

        with (
            patch.object(opt, "_get_cost_tracker", return_value=mock_tracker),
            caplog.at_level(logging.WARNING),
        ):
            efficiencies = opt._get_efficiencies("coding", ["model-a"])

        assert len(efficiencies) == 1
        warning_msgs = [r for r in caplog.records if "Failed to get cost data" in r.message]
        assert len(warning_msgs) >= 1
