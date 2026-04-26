"""Mounted governance tests for manufacturing/kaizen/SPC/workflow read routes.

Proves that all manufacturing GET routes return bounded 503 on subsystem
failure, not raw 500 or silently empty 200 responses that hide errors.

Most tests go through the full Litestar HTTP stack via TestClient.  For routes
whose handlers call module-level singleton helpers (_get_kaizen_log,
_get_gate_runner) that are defined in litestar_manufacturing_api, Litestar's
TestClient caches handler references at app-creation time so post-creation
patches on the module namespace do not reach the running handler.  Those tests
use direct handler invocation via asyncio.run(h.fn(...)) instead, which
bypasses the HTTP stack but still exercises the handler's error-path logic and
the litestar_error_response envelope.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is only built once; each test
    creates its own TestClient context so connection state does not leak.

    Returns:
        A Litestar application instance.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=True)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


@pytest.fixture(autouse=True)
def reset_manufacturing_singletons():
    """Clear cached singleton instances before each test.

    The manufacturing API module caches ImprovementLog and WorkflowGateRunner
    in module-level variables to avoid re-creating them per request. Tests that
    patch _get_kaizen_log or _get_gate_runner need these variables cleared so
    the patched version is called instead of returning the cached instance.

    Yields:
        None  -  setup/teardown only.
    """
    import vetinari.web.litestar_manufacturing_api as mfg_mod

    old_kaizen = mfg_mod._kaizen_log_instance
    old_gate = mfg_mod._gate_runner_instance
    mfg_mod._kaizen_log_instance = None
    mfg_mod._gate_runner_instance = None
    yield
    mfg_mod._kaizen_log_instance = old_kaizen
    mfg_mod._gate_runner_instance = old_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert that *response* is a bounded 503 with ``status: error`` envelope.

    Works with both TestClient HTTP responses (which expose ``.json()``) and
    Litestar Response objects returned by direct handler invocation (which
    expose ``.content`` as a dict).

    Args:
        response: HTTP response from the TestClient, or a Litestar Response
            returned directly by a handler under test.
    """
    body_repr = repr(getattr(response, "text", getattr(response, "content", None)))[:400]
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {body_repr}"
    # TestClient responses have .json(); direct Litestar Responses have .content (dict).
    body = response.json() if callable(getattr(response, "json", None)) else response.content
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


def _get_handler(handlers: list, path: str) -> object:
    """Find a Litestar route handler by its registered path.

    Used by direct-invocation tests to locate the right handler from the list
    returned by create_manufacturing_handlers() without going through the HTTP
    stack.

    Args:
        handlers: List of Litestar route handler objects.
        path: URL path to find (e.g. ``/api/v1/kaizen/report``).

    Returns:
        The matching route handler object.

    Raises:
        ValueError: When no handler is registered for *path*.
    """
    for h in handlers:
        if hasattr(h, "paths") and path in h.paths:
            return h
    raise ValueError(f"No handler registered for path {path!r}")


# ---------------------------------------------------------------------------
# GET /api/v1/bottleneck
# ---------------------------------------------------------------------------


class TestApiBottleneck:
    """GET /api/v1/bottleneck  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_bottleneck_identifier to raise; endpoint must return 503."""
        with patch(
            "vetinari.orchestration.bottleneck.get_bottleneck_identifier",
            side_effect=RuntimeError("bottleneck subsystem down"),
        ):
            response = client.get("/api/v1/bottleneck")
        _assert_503_error(response)

    def test_get_status_raises_returns_503(self, client: object) -> None:
        """When get_status() raises after successful import, return 503."""
        mock_identifier = MagicMock()
        mock_identifier.get_status.side_effect = RuntimeError("status read failed")
        with patch(
            "vetinari.orchestration.bottleneck.get_bottleneck_identifier",
            return_value=mock_identifier,
        ):
            response = client.get("/api/v1/bottleneck")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/value-stream/aggregate
# ---------------------------------------------------------------------------


class TestApiValueStreamAggregate:
    """GET /api/v1/value-stream/aggregate  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_value_stream_analyzer to raise; endpoint must return 503."""
        with patch(
            "vetinari.analytics.value_stream.get_value_stream_analyzer",
            side_effect=RuntimeError("value stream subsystem down"),
        ):
            response = client.get("/api/v1/value-stream/aggregate")
        _assert_503_error(response)

    def test_get_aggregate_report_raises_returns_503(self, client: object) -> None:
        """When get_aggregate_report() raises, return 503."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_aggregate_report.side_effect = RuntimeError("aggregate failed")
        with patch(
            "vetinari.analytics.value_stream.get_value_stream_analyzer",
            return_value=mock_analyzer,
        ):
            response = client.get("/api/v1/value-stream/aggregate")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/models/recommendations
# ---------------------------------------------------------------------------


class TestApiModelRecommendations:
    """GET /api/v1/models/recommendations  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_model_scout to raise; endpoint must return 503."""
        with patch(
            "vetinari.models.model_scout.get_model_scout",
            side_effect=RuntimeError("model scout unavailable"),
        ):
            response = client.get("/api/v1/models/recommendations")
        _assert_503_error(response)

    def test_get_recommendations_raises_returns_503(self, client: object) -> None:
        """When get_recommendations() raises, return 503."""
        mock_scout = MagicMock()
        mock_scout.get_recommendations.side_effect = RuntimeError("recommendations failed")
        with patch(
            "vetinari.models.model_scout.get_model_scout",
            return_value=mock_scout,
        ):
            response = client.get("/api/v1/models/recommendations")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/kaizen/report
# ---------------------------------------------------------------------------


class TestApiKaizenReport:
    """GET /api/v1/kaizen/report  -  subsystem failure must return 503.

    Uses direct handler invocation because _get_kaizen_log is a module-level
    singleton function and Litestar's TestClient caches handler references at
    app-creation time, preventing post-creation patches from reaching the
    running handler.
    """

    def test_subsystem_failure_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """Patch _get_kaizen_log to raise; handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        with patch.object(m, "_get_kaizen_log", side_effect=RuntimeError("kaizen log unavailable")):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/report")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)

    def test_get_weekly_report_raises_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """When get_weekly_report() raises, handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        mock_log = MagicMock()
        mock_log.get_weekly_report.side_effect = RuntimeError("weekly report failed")
        with patch.object(m, "_get_kaizen_log", return_value=mock_log):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/report")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)


# ---------------------------------------------------------------------------
# GET /api/v1/kaizen/improvements
# ---------------------------------------------------------------------------


class TestApiKaizenImprovements:
    """GET /api/v1/kaizen/improvements  -  subsystem failure must return 503.

    Uses direct handler invocation for the same reason as TestApiKaizenReport.
    """

    def test_subsystem_failure_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """Patch _get_kaizen_log to raise; handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        with patch.object(m, "_get_kaizen_log", side_effect=RuntimeError("kaizen log unavailable")):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/improvements")
            # Pass status=None explicitly to bypass the Litestar Parameter default object.
            result = asyncio.run(handler.fn(status=None))
        _assert_503_error(result)

    def test_get_improvements_raises_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """When get_improvements_by_status() raises, handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        mock_log = MagicMock()
        mock_log.get_improvements_by_status.side_effect = RuntimeError("improvements read failed")
        with patch.object(m, "_get_kaizen_log", return_value=mock_log):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/improvements")
            # Pass status=None explicitly to bypass the Litestar Parameter default object.
            result = asyncio.run(handler.fn(status=None))
        _assert_503_error(result)


# ---------------------------------------------------------------------------
# GET /api/v1/kaizen/defect-trends
# ---------------------------------------------------------------------------


class TestApiKaizenDefectTrends:
    """GET /api/v1/kaizen/defect-trends  -  subsystem failure must return 503.

    Uses direct handler invocation for the same reason as TestApiKaizenReport.
    """

    def test_subsystem_failure_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """Patch _get_kaizen_log to raise; handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        with patch.object(m, "_get_kaizen_log", side_effect=RuntimeError("kaizen log unavailable")):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/defect-trends")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)

    def test_get_weekly_defect_counts_raises_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """When get_weekly_defect_counts() raises, handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        mock_log = MagicMock()
        mock_log.get_weekly_defect_counts.side_effect = RuntimeError("defect counts failed")
        with patch.object(m, "_get_kaizen_log", return_value=mock_log):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/kaizen/defect-trends")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)


# ---------------------------------------------------------------------------
# GET /api/v1/constraints/violations/stats
# ---------------------------------------------------------------------------


class TestApiConstraintViolationStats:
    """GET /api/v1/constraints/violations/stats  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_constraint_registry to raise; endpoint must return 503."""
        with patch(
            "vetinari.constraints.registry.get_constraint_registry",
            side_effect=RuntimeError("constraint registry unavailable"),
        ):
            response = client.get("/api/v1/constraints/violations/stats")
        _assert_503_error(response)

    def test_get_violation_stats_raises_returns_503(self, client: object) -> None:
        """When get_violation_stats() raises, return 503."""
        mock_registry = MagicMock()
        mock_registry.get_violation_stats.side_effect = RuntimeError("stats read failed")
        with patch(
            "vetinari.constraints.registry.get_constraint_registry",
            return_value=mock_registry,
        ):
            response = client.get("/api/v1/constraints/violations/stats")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/benchmarks/runs/{run_id}/results
# ---------------------------------------------------------------------------


class TestApiBenchmarkRunResults:
    """GET /api/v1/benchmarks/runs/{run_id}/results  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch MetricStore to raise on instantiation; endpoint must return 503."""
        with patch(
            "vetinari.benchmarks.runner.MetricStore",
            side_effect=RuntimeError("metric store unavailable"),
        ):
            response = client.get("/api/v1/benchmarks/runs/run-xyz-001/results")
        _assert_503_error(response)

    def test_missing_run_returns_404(self, client: object) -> None:
        """When run_id has no results, endpoint must return 404."""
        mock_store = MagicMock()
        mock_store.load_results.return_value = []
        with patch("vetinari.benchmarks.runner.MetricStore", return_value=mock_store):
            response = client.get("/api/v1/benchmarks/runs/nonexistent-run/results")
        assert response.status_code == 404, (
            f"Missing run must return 404, got {response.status_code}. Body: {response.text[:400]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/spc/chart/{metric_name}
# ---------------------------------------------------------------------------


class TestApiSpcChart:
    """GET /api/v1/spc/chart/{metric_name}  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_spc_monitor to raise; endpoint must return 503."""
        with patch(
            "vetinari.workflow.spc.get_spc_monitor",
            side_effect=RuntimeError("SPC monitor unavailable"),
        ):
            response = client.get("/api/v1/spc/chart/latency_ms")
        _assert_503_error(response)

    def test_missing_metric_returns_404(self, client: object) -> None:
        """When metric has no chart data, endpoint must return 404."""
        mock_monitor = MagicMock()
        mock_monitor.get_chart.return_value = None
        with patch("vetinari.workflow.spc.get_spc_monitor", return_value=mock_monitor):
            response = client.get("/api/v1/spc/chart/nonexistent_metric")
        assert response.status_code == 404, (
            f"Missing metric must return 404, got {response.status_code}. Body: {response.text[:400]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/spc/chart/{metric_name}/cpk
# ---------------------------------------------------------------------------


class TestApiSpcCpk:
    """GET /api/v1/spc/chart/{metric_name}/cpk  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_spc_monitor to raise; cpk endpoint must return 503."""
        with patch(
            "vetinari.workflow.spc.get_spc_monitor",
            side_effect=RuntimeError("SPC monitor unavailable"),
        ):
            response = client.get("/api/v1/spc/chart/latency_ms/cpk")
        _assert_503_error(response)

    def test_missing_metric_returns_404(self, client: object) -> None:
        """When metric has no chart, cpk endpoint must return 404."""
        mock_monitor = MagicMock()
        mock_monitor.get_chart.return_value = None
        with patch("vetinari.workflow.spc.get_spc_monitor", return_value=mock_monitor):
            response = client.get("/api/v1/spc/chart/nonexistent_metric/cpk")
        assert response.status_code == 404, (
            f"Missing metric cpk must return 404, got {response.status_code}. Body: {response.text[:400]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/spc/alerts
# ---------------------------------------------------------------------------


class TestApiSpcAlerts:
    """GET /api/v1/spc/alerts  -  subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_spc_monitor to raise; alerts endpoint must return 503."""
        with patch(
            "vetinari.workflow.spc.get_spc_monitor",
            side_effect=RuntimeError("SPC monitor unavailable"),
        ):
            response = client.get("/api/v1/spc/alerts")
        _assert_503_error(response)

    def test_get_alerts_raises_returns_503(self, client: object) -> None:
        """When get_alerts() raises after import, return 503."""
        mock_monitor = MagicMock()
        mock_monitor.get_alerts.side_effect = RuntimeError("alerts read failed")
        with patch("vetinari.workflow.spc.get_spc_monitor", return_value=mock_monitor):
            response = client.get("/api/v1/spc/alerts")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# GET /api/v1/workflow/gates
# ---------------------------------------------------------------------------


class TestApiWorkflowGates:
    """GET /api/v1/workflow/gates  -  subsystem failure must return 503.

    Uses direct handler invocation because the handler calls get_gate_runner
    via a local import inside the try block.  The local-import path creates a
    new name binding on each call, so patching the function at module level in
    vetinari.workflow.quality_gates works correctly here; however, the
    module-scope TestClient caches the handler, so to be safe and consistent
    with the other singleton tests, direct invocation is used.
    """

    def test_subsystem_failure_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """Patch get_gate_runner to raise; handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        with patch(
            "vetinari.workflow.quality_gates.get_gate_runner", side_effect=RuntimeError("gate runner unavailable")
        ):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/workflow/gates")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)

    def test_gates_attribute_raises_returns_503(self, reset_manufacturing_singletons: object) -> None:
        """When accessing runner.gates raises, handler must return a 503 Response."""
        import vetinari.web.litestar_manufacturing_api as m

        mock_runner = MagicMock()
        # Make .gates raise when iterated
        type(mock_runner).gates = property(lambda self: (_ for _ in ()).throw(RuntimeError("gates broken")))
        with patch("vetinari.workflow.quality_gates.get_gate_runner", return_value=mock_runner):
            handlers = m.create_manufacturing_handlers()
            handler = _get_handler(handlers, "/api/v1/workflow/gates")
            result = asyncio.run(handler.fn())
        _assert_503_error(result)
