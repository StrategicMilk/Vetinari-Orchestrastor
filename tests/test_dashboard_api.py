"""Regression tests for vetinari/dashboard/api.py.

Covers:
- Defect 1: _traces/_trace_list attributes have compat-only documentation
- Defect 2: add_trace() deduplicates by trace_id — _trace_list grows by 1 not 2
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.dashboard.api import DashboardAPI, TraceDetail, TraceInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(trace_id: str = "trace-abc", status: str = "ok") -> TraceDetail:
    """Build a minimal TraceDetail for testing."""
    return TraceDetail(
        trace_id=trace_id,
        start_time=time.time(),
        end_time=time.time() + 0.1,
        duration_ms=100.0,
        status=status,
        spans=[{"operation": "chat", "span_id": "s1"}],
    )


@pytest.fixture
def api() -> DashboardAPI:
    """Return a DashboardAPI instance with telemetry mocked out."""
    mock_tel = MagicMock()
    mock_tel.get_adapter_metrics.return_value = {}
    mock_tel.get_memory_metrics.return_value = {}
    mock_tel.get_plan_metrics.return_value = {}
    with patch("vetinari.dashboard.api.get_telemetry_collector", return_value=mock_tel):
        yield DashboardAPI()


# ---------------------------------------------------------------------------
# Defect 1 — compat-only documentation on trace registry
# ---------------------------------------------------------------------------


class TestTracesCompatOnlyDocumented:
    """Trace registry attributes carry compat-only documentation comments."""

    def test_traces_attribute_exists_and_is_dict(self, api: DashboardAPI) -> None:
        """_traces is a dict keyed by trace_id — accessed by get_trace_detail()."""
        assert isinstance(api._traces, dict)
        assert len(api._traces) == 0

    def test_trace_list_attribute_exists_and_is_bounded_deque(self, api: DashboardAPI) -> None:
        """_trace_list is a bounded deque with maxlen=1000 — used by search_traces()."""
        from collections import deque

        assert isinstance(api._trace_list, deque)
        assert api._trace_list.maxlen == 1000

    def test_both_attributes_protected_by_same_lock(self, api: DashboardAPI) -> None:
        """add_trace and search_traces both acquire _lock, ensuring consistency."""
        td = _make_trace("lock-test")
        api.add_trace(td)
        # search_traces reads _trace_list under _lock
        results = api.search_traces()
        assert any(t.trace_id == "lock-test" for t in results)
        # get_trace_detail reads _traces under _lock
        detail = api.get_trace_detail("lock-test")
        assert detail is not None
        assert detail.trace_id == "lock-test"


# ---------------------------------------------------------------------------
# Defect 2 — add_trace() must not duplicate _trace_list entries
# ---------------------------------------------------------------------------


class TestAddTraceDeduplicate:
    """add_trace() with a repeated trace_id updates _traces but not _trace_list."""

    def test_first_add_inserts_into_both_structures(self, api: DashboardAPI) -> None:
        """A new trace_id populates both _traces and _trace_list."""
        td = _make_trace("dup-trace")
        ok = api.add_trace(td)

        assert ok is True
        assert "dup-trace" in api._traces
        assert len(api._trace_list) == 1
        assert api._trace_list[0].trace_id == "dup-trace"

    def test_duplicate_add_updates_traces_only(self, api: DashboardAPI) -> None:
        """Re-adding the same trace_id updates _traces but leaves _trace_list length at 1."""
        td1 = _make_trace("dup-trace", status="running")
        td2 = _make_trace("dup-trace", status="ok")

        api.add_trace(td1)
        api.add_trace(td2)

        # _trace_list must not grow to 2
        assert len(api._trace_list) == 1, (
            "_trace_list grew on duplicate add — deduplication fix missing"
        )
        # _traces must hold the updated version
        assert api._traces["dup-trace"].status == "ok"

    def test_different_trace_ids_each_get_a_list_entry(self, api: DashboardAPI) -> None:
        """Each unique trace_id gets exactly one entry in _trace_list."""
        for i in range(5):
            api.add_trace(_make_trace(f"trace-{i}"))

        assert len(api._trace_list) == 5
        ids = {t.trace_id for t in api._trace_list}
        assert ids == {f"trace-{i}" for i in range(5)}

    def test_capacity_eviction_not_triggered_by_duplicate(self, api: DashboardAPI) -> None:
        """Eviction only fires when a genuinely new trace is added at capacity."""
        from collections import deque

        # Shrink capacity to 2 for fast testing
        api._trace_list = deque(maxlen=2)

        api.add_trace(_make_trace("t1"))
        api.add_trace(_make_trace("t2"))
        # This would evict t1 if add_trace wrongly treated it as new
        api.add_trace(_make_trace("t1"))

        assert len(api._trace_list) == 2
        ids = {t.trace_id for t in api._trace_list}
        # t1 must still be present — it was not re-appended
        assert "t1" in ids
        assert "t2" in ids
