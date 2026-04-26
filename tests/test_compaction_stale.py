"""Tests for should_clear_stale_tool_results in vetinari.context.compaction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vetinari.context.compaction import should_clear_stale_tool_results


class TestStaleToolResults:
    """Stale detection based on elapsed time since last pipeline stage."""

    def test_stale_when_over_1_hour(self) -> None:
        """Returns True when last_stage_time is more than 1 hour ago."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        assert should_clear_stale_tool_results(two_hours_ago) is True

    def test_not_stale_when_under_1_hour(self) -> None:
        """Returns False when last_stage_time is 30 minutes ago."""
        thirty_min_ago = datetime.now(timezone.utc) - timedelta(minutes=30)
        assert should_clear_stale_tool_results(thirty_min_ago) is False

    def test_not_stale_when_none(self) -> None:
        """Returns False when last_stage_time is None (no stage completed yet)."""
        assert should_clear_stale_tool_results(None) is False

    def test_boundary_exactly_1_hour(self) -> None:
        """Returns True when elapsed time is exactly 3600 seconds (at threshold)."""
        exactly_1_hour_ago = datetime.now(timezone.utc) - timedelta(seconds=3600)
        assert should_clear_stale_tool_results(exactly_1_hour_ago) is True

    def test_just_under_1_hour(self) -> None:
        """Returns False when elapsed time is just under the 1-hour threshold."""
        just_under = datetime.now(timezone.utc) - timedelta(seconds=3599)
        assert should_clear_stale_tool_results(just_under) is False

    def test_naive_datetime_treated_as_utc(self) -> None:
        """A naive datetime (no tzinfo) is assumed to be UTC and handled correctly."""
        # Create a naive datetime 2 hours ago
        two_hours_ago_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
        assert two_hours_ago_naive.tzinfo is None  # confirm it's naive
        assert should_clear_stale_tool_results(two_hours_ago_naive) is True

    def test_recent_naive_datetime_not_stale(self) -> None:
        """A naive datetime 10 minutes ago is treated as UTC and not stale."""
        recent_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
        assert should_clear_stale_tool_results(recent_naive) is False
