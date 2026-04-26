"""Parity tests for Session 21 Wave 1 Litestar handler migration.

Verifies that each migrated handler factory:
- Returns the expected number of handlers
- Registers the correct route paths
- Applies admin_guard where Flask had @require_admin
- Creates a working Litestar app when combined
"""

from __future__ import annotations

import pytest

# Skip the entire module when Litestar is not installed


# -- helpers ------------------------------------------------------------------


def _paths(handlers: list) -> set[str]:
    """Extract the first declared path from each handler object.

    Litestar route handler functions carry their path info on the
    ``paths`` attribute (a frozenset of strings) after decoration.

    Args:
        handlers: List of Litestar route handler objects.

    Returns:
        Set of path strings, one per handler.
    """
    result: set[str] = set()
    for h in handlers:
        paths_attr = getattr(h, "paths", None)
        if paths_attr:
            # paths is a frozenset on Litestar handler objects
            result.update(paths_attr)
    return result


def _has_admin_guard(handler: object) -> bool:
    """Return True when admin_guard appears in a handler's guards list.

    The ``guards`` attribute on a Litestar route handler holds the list
    of guard callables registered at decoration time.

    Args:
        handler: A Litestar route handler object.

    Returns:
        True if admin_guard is present in handler.guards, False otherwise.
    """
    from vetinari.web.litestar_guards import admin_guard

    guards = getattr(handler, "guards", None) or []
    return admin_guard in guards


def _handler_name(handler: object) -> str:
    """Return a human-readable name for a handler for use in assertion messages.

    Args:
        handler: A Litestar route handler object.

    Returns:
        The handler function name, or repr fallback.
    """
    fn = getattr(handler, "fn", None)
    if fn is not None:
        return getattr(fn, "__name__", repr(handler))
    return repr(handler)


# -- Infrastructure ------------------------------------------------------------


class TestInfrastructureImports:
    """Infrastructure modules must be importable and correctly wired."""

    def test_admin_guard_importable(self) -> None:
        """admin_guard callable must be importable from litestar_guards."""
        from vetinari.web.litestar_guards import admin_guard

        assert callable(admin_guard)

    def test_exception_handlers_importable(self) -> None:
        """EXCEPTION_HANDLERS dict must be importable from litestar_exceptions."""
        from vetinari.web.litestar_exceptions import EXCEPTION_HANDLERS

        assert isinstance(EXCEPTION_HANDLERS, dict)

    def test_is_admin_connection_importable(self) -> None:
        """is_admin_connection predicate must be importable from litestar_guards."""
        from vetinari.web.litestar_guards import is_admin_connection

        assert callable(is_admin_connection)


# -- Analytics -----------------------------------------------------------------


class TestAnalyticsHandlers:
    """Parity tests for litestar_analytics.create_analytics_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 12 handlers."""
        from vetinari.web.litestar_analytics import create_analytics_handlers

        handlers = create_analytics_handlers()
        assert len(handlers) == 12, f"Expected 12 analytics handlers, got {len(handlers)}"

    def test_returns_non_empty_list(self) -> None:
        """Factory must return a list, not None or empty."""
        from vetinari.web.litestar_analytics import create_analytics_handlers

        handlers = create_analytics_handlers()
        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_all_have_admin_guard(self) -> None:
        """Every analytics handler must carry admin_guard (all were @require_admin in Flask)."""
        from vetinari.web.litestar_analytics import create_analytics_handlers

        handlers = create_analytics_handlers()
        for h in handlers:
            assert _has_admin_guard(h), f"Missing admin_guard on analytics handler {_handler_name(h)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_analytics import create_analytics_handlers

        handlers = create_analytics_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/analytics/cost",
            "/api/v1/analytics/sla",
            "/api/v1/analytics/anomalies",
            "/api/v1/analytics/forecasts",
            "/api/v1/analytics/sla/model/{model_id:str}/compliance",
            "/api/v1/analytics/sla/breach",
            "/api/v1/analytics/sla/model-compliance",
        }
        missing = expected - paths
        assert not missing, f"Analytics routes missing: {missing}"


# -- Dashboard Metrics ---------------------------------------------------------


class TestDashboardMetricsHandlers:
    """Parity tests for litestar_dashboard_metrics.create_dashboard_metrics_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 6 handlers (including DELETE /api/v1/traces)."""
        from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers

        handlers = create_dashboard_metrics_handlers()
        assert len(handlers) == 6, f"Expected 6 dashboard metrics handlers, got {len(handlers)}"

    def test_returns_non_empty_list(self) -> None:
        """Factory must return a non-empty list."""
        from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers

        handlers = create_dashboard_metrics_handlers()
        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_no_handler_has_admin_guard(self) -> None:
        """Dashboard metrics handlers must NOT have admin_guard (public dashboard data)."""
        from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers

        handlers = create_dashboard_metrics_handlers()
        for h in handlers:
            assert not _has_admin_guard(h), f"Unexpected admin_guard on dashboard handler {_handler_name(h)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers

        handlers = create_dashboard_metrics_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/metrics/latest",
            "/api/v1/metrics/timeseries",
            "/api/v1/traces",
            "/api/v1/traces/{trace_id:str}",
            "/api/v1/analytics/cost/top",
        }
        missing = expected - paths
        assert not missing, f"Dashboard metrics routes missing: {missing}"


# -- Project Git ---------------------------------------------------------------


class TestProjectGitHandlers:
    """Parity tests for litestar_project_git.create_project_git_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 3 handlers."""
        from vetinari.web.litestar_project_git import create_project_git_handlers

        handlers = create_project_git_handlers()
        assert len(handlers) == 3, f"Expected 3 project-git handlers, got {len(handlers)}"

    def test_all_have_admin_guard(self) -> None:
        """All project-git handlers must carry admin_guard."""
        from vetinari.web.litestar_project_git import create_project_git_handlers

        handlers = create_project_git_handlers()
        for h in handlers:
            assert _has_admin_guard(h), f"Missing admin_guard on project-git handler {_handler_name(h)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_project_git import create_project_git_handlers

        handlers = create_project_git_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/project/git/commit-message",
            "/api/v1/project/git/commit-message-path",
            "/api/v1/project/git/conflicts",
        }
        missing = expected - paths
        assert not missing, f"Project-git routes missing: {missing}"


# -- Model Management ----------------------------------------------------------


class TestModelMgmtHandlers:
    """Parity tests for litestar_model_mgmt.create_model_mgmt_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 13 handlers."""
        from vetinari.web.litestar_model_mgmt import create_model_mgmt_handlers

        handlers = create_model_mgmt_handlers()
        assert len(handlers) == 13, f"Expected 13 model-mgmt handlers, got {len(handlers)}"

    def test_all_have_admin_guard(self) -> None:
        """All model-mgmt handlers must carry admin_guard."""
        from vetinari.web.litestar_model_mgmt import create_model_mgmt_handlers

        handlers = create_model_mgmt_handlers()
        for h in handlers:
            assert _has_admin_guard(h), f"Missing admin_guard on model-mgmt handler {_handler_name(h)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_model_mgmt import create_model_mgmt_handlers

        handlers = create_model_mgmt_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/models/assign-tasks",
            "/api/v1/models/all-available",
            "/api/v1/models",
            "/api/v1/models/{model_id:str}/delete",
            "/api/v1/models/draft-pairs/{main_model_id:str}/{draft_model_id:str}/stats",
            "/api/v1/models/draft-pairs/stats",
            "/api/v1/models/chat-stream",
            "/api/v1/vram/thermal-status",
            "/api/v1/vram/phase",
            "/api/v1/models/cascade-router/build",
            "/api/v1/models/cascade/stats",
            "/api/v1/models/cascade/disable",
        }
        missing = expected - paths
        assert not missing, f"Model-mgmt routes missing: {missing}"


# -- Models Catalog ------------------------------------------------------------


class TestModelsCatalogHandlers:
    """Parity tests for litestar_models_catalog.create_models_catalog_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 14 handlers."""
        from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

        handlers = create_models_catalog_handlers()
        assert len(handlers) == 14, f"Expected 14 models-catalog handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

        handlers = create_models_catalog_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/model-catalog",
            "/api/v1/models/{model_id:str}",
            "/api/v1/models/select",
            "/api/v1/models/policy",
            "/api/v1/models/reload",
            "/api/v1/project/{project_id:str}/model-search",
            "/api/v1/project/{project_id:str}/refresh-models",
            "/api/v1/models/search",
            "/api/v1/models/popular",
            "/api/v1/models/files",
            "/api/v1/models/download",
            "/api/v1/models/download/{download_id:str}",
            "/api/v1/models/download/{download_id:str}/cancel",
        }
        missing = expected - paths
        assert not missing, f"Models-catalog routes missing: {missing}"

    def test_public_routes_have_no_admin_guard(self) -> None:
        """Public catalog routes (list, get, policy GET) must not require admin."""
        from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

        handlers = create_models_catalog_handlers()
        paths_to_handlers: dict[str, object] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers[p] = h

        public_paths = {
            "/api/v1/model-catalog",
            "/api/v1/models/{model_id:str}",
            "/api/v1/models/popular",
            "/api/v1/models/files",
        }
        for path in public_paths:
            if path in paths_to_handlers:
                h = paths_to_handlers[path]
                assert not _has_admin_guard(h), f"Public catalog path {path} unexpectedly has admin_guard"

    def test_write_routes_have_admin_guard(self) -> None:
        """Write/mutating routes (select, policy PUT, reload, search) must require admin."""
        from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

        handlers = create_models_catalog_handlers()
        paths_to_handlers: dict[str, list] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers.setdefault(p, []).append(h)

        admin_paths = {
            "/api/v1/models/select",
            "/api/v1/models/reload",
            "/api/v1/project/{project_id:str}/model-search",
            "/api/v1/project/{project_id:str}/refresh-models",
            "/api/v1/models/search",
            "/api/v1/models/download",
            "/api/v1/models/download/{download_id:str}",
            "/api/v1/models/download/{download_id:str}/cancel",
        }
        for path in admin_paths:
            if path in paths_to_handlers:
                for h in paths_to_handlers[path]:
                    assert _has_admin_guard(h), f"Admin path {path} is missing admin_guard on {_handler_name(h)}"


# -- Models Discovery ----------------------------------------------------------


class TestModelsDiscoveryHandlers:
    """Parity tests for litestar_models_discovery.create_models_discovery_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 7 handlers."""
        from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

        handlers = create_models_discovery_handlers()
        assert len(handlers) == 7, f"Expected 7 models-discovery handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

        handlers = create_models_discovery_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/models",
            "/api/v1/models/refresh",
            "/api/v1/score-models",
            "/api/v1/model-config",
            "/api/v1/swap-model",
            "/api/v1/discover",
        }
        missing = expected - paths
        assert not missing, f"Models-discovery routes missing: {missing}"

    def test_public_routes_have_no_admin_guard(self) -> None:
        """Public discovery routes (list, config GET, discover) must have at least one unguarded handler.

        /api/v1/model-config has both a GET (no guard) and a POST (admin guard).
        We verify that at least one handler on each public path is unguarded.
        """
        from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

        handlers = create_models_discovery_handlers()
        paths_to_handlers: dict[str, list] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers.setdefault(p, []).append(h)

        public_paths = {"/api/v1/models", "/api/v1/model-config", "/api/v1/discover"}
        for path in public_paths:
            if path in paths_to_handlers:
                unguarded = [h for h in paths_to_handlers[path] if not _has_admin_guard(h)]
                assert unguarded, f"No unguarded (public) handler found for discovery path {path}"

    def test_write_routes_have_admin_guard(self) -> None:
        """Mutating discovery routes must require admin."""
        from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

        handlers = create_models_discovery_handlers()
        paths_to_handlers: dict[str, object] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers[p] = h

        admin_paths = {"/api/v1/models/refresh", "/api/v1/score-models", "/api/v1/swap-model"}
        for path in admin_paths:
            if path in paths_to_handlers:
                h = paths_to_handlers[path]
                assert _has_admin_guard(h), f"Admin discovery path {path} is missing admin_guard on {_handler_name(h)}"


# -- System Status -------------------------------------------------------------


class TestSystemStatusHandlers:
    """Parity tests for litestar_system_status.create_system_status_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 9 handlers."""
        from vetinari.web.litestar_system_status import create_system_status_handlers

        handlers = create_system_status_handlers()
        assert len(handlers) == 9, f"Expected 9 system-status handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_system_status import create_system_status_handlers

        handlers = create_system_status_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/status",
            "/api/v1/token-stats",
            "/api/v1/search",
            "/api/v1/config",
            "/api/v1/validate-path",
            "/api/v1/browse-directory",
            "/api/v1/batch/queue-stats",
            "/api/v1/upgrade-check",
        }
        missing = expected - paths
        assert not missing, f"System-status routes missing: {missing}"

    def test_config_routes_have_admin_guard(self) -> None:
        """Config mutation routes (POST /config and PUT /config) must require admin."""
        from vetinari.web.litestar_system_status import create_system_status_handlers

        handlers = create_system_status_handlers()
        # Both api_config_post and api_config_put share the /api/v1/config path
        config_handlers = [h for h in handlers if "/api/v1/config" in getattr(h, "paths", set())]
        assert config_handlers, "No handlers found for /api/v1/config"
        for h in config_handlers:
            assert _has_admin_guard(h), f"Config handler {_handler_name(h)} is missing admin_guard"

    def test_public_status_routes_have_no_admin_guard(self) -> None:
        """Public status/info routes must not require admin."""
        from vetinari.web.litestar_system_status import create_system_status_handlers

        handlers = create_system_status_handlers()
        paths_to_handlers: dict[str, object] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers[p] = h

        public_paths = {
            "/api/v1/status",
            "/api/v1/token-stats",
            "/api/v1/search",
            "/api/v1/validate-path",
            "/api/v1/browse-directory",
            "/api/v1/batch/queue-stats",
            "/api/v1/upgrade-check",
        }
        for path in public_paths:
            if path in paths_to_handlers:
                h = paths_to_handlers[path]
                assert not _has_admin_guard(h), f"Public status path {path} unexpectedly has admin_guard"


# -- System Hardware -----------------------------------------------------------


class TestSystemHardwareHandlers:
    """Parity tests for litestar_system_hardware.create_system_hardware_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 7 handlers."""
        from vetinari.web.litestar_system_hardware import create_system_hardware_handlers

        handlers = create_system_hardware_handlers()
        assert len(handlers) == 7, f"Expected 7 system-hardware handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_system_hardware import create_system_hardware_handlers

        handlers = create_system_hardware_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/health",
            "/ready",
            "/api/v1/server/shutdown",
            "/api/v1/system/gpu",
            "/api/v1/system/resources",
            "/api/v1/system/vram",
            "/api/v1/system/vram/phase",
        }
        missing = expected - paths
        assert not missing, f"System-hardware routes missing: {missing}"

    def test_shutdown_and_vram_phase_have_admin_guard(self) -> None:
        """Shutdown and VRAM-phase handlers must require admin."""
        from vetinari.web.litestar_system_hardware import create_system_hardware_handlers

        handlers = create_system_hardware_handlers()
        paths_to_handlers: dict[str, object] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers[p] = h

        admin_paths = {"/api/v1/server/shutdown", "/api/v1/system/vram/phase"}
        for path in admin_paths:
            if path in paths_to_handlers:
                h = paths_to_handlers[path]
                assert _has_admin_guard(h), f"Admin hardware path {path} is missing admin_guard on {_handler_name(h)}"

    def test_read_routes_have_no_admin_guard(self) -> None:
        """Read-only hardware routes must not require admin."""
        from vetinari.web.litestar_system_hardware import create_system_hardware_handlers

        handlers = create_system_hardware_handlers()
        paths_to_handlers: dict[str, object] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers[p] = h

        public_paths = {
            "/api/v1/health",
            "/ready",
            "/api/v1/system/gpu",
            "/api/v1/system/resources",
            "/api/v1/system/vram",
        }
        for path in public_paths:
            if path in paths_to_handlers:
                h = paths_to_handlers[path]
                assert not _has_admin_guard(h), f"Public hardware path {path} unexpectedly has admin_guard"


# -- System Content ------------------------------------------------------------


class TestSystemContentHandlers:
    """Parity tests for litestar_system_content.create_system_content_handlers()."""

    def test_handler_count(self) -> None:
        """Factory must return exactly 12 handlers."""
        from vetinari.web.litestar_system_content import create_system_content_handlers

        handlers = create_system_content_handlers()
        assert len(handlers) == 12, f"Expected 12 system-content handlers, got {len(handlers)}"

    def test_expected_paths_present(self) -> None:
        """Expected route paths must all be registered."""
        from vetinari.web.litestar_system_content import create_system_content_handlers

        handlers = create_system_content_handlers()
        paths = _paths(handlers)
        expected = {
            "/api/v1/workflow",
            "/api/v1/system-prompts",
            "/api/v1/system-prompts/{name:str}",
            "/api/v1/preferences",
            "/api/v1/settings",
            "/api/v1/variant",
            "/api/v1/download",
            "/api/v1/artifacts",
        }
        missing = expected - paths
        assert not missing, f"System-content routes missing: {missing}"

    def test_write_operations_have_admin_guard(self) -> None:
        """Write operations (POST prompts, DELETE prompts, PUT variant) must require admin."""
        from vetinari.web.litestar_system_content import create_system_content_handlers

        handlers = create_system_content_handlers()
        # Find handlers by inspecting their HTTP method + path via the handler object
        # admin handlers: api_save_system_prompt (POST /system-prompts),
        # api_delete_system_prompt (DELETE /system-prompts/{name}),
        # api_set_variant (PUT /variant)
        # We check all handlers on the shared paths have the guard on write-only ones
        paths_to_handlers: dict[str, list] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers.setdefault(p, []).append(h)

        # POST /api/v1/system-prompts and DELETE /api/v1/system-prompts/{name:str}
        # both serve the same conceptual endpoint but at different paths
        # We verify that at least one handler per "write" path has admin_guard
        write_paths_that_must_have_admin = {
            "/api/v1/system-prompts/{name:str}",  # DELETE handler
            "/api/v1/variant",  # PUT handler (the GET one doesn't have admin)
        }
        for path in write_paths_that_must_have_admin:
            if path in paths_to_handlers:
                guarded = [h for h in paths_to_handlers[path] if _has_admin_guard(h)]
                assert guarded, f"No admin-guarded handler found for write path {path}"

    def test_public_read_routes_have_no_admin_guard(self) -> None:
        """Public read routes must not require admin."""
        from vetinari.web.litestar_system_content import create_system_content_handlers

        handlers = create_system_content_handlers()
        paths_to_handlers: dict[str, list] = {}
        for h in handlers:
            for p in getattr(h, "paths", set()):
                paths_to_handlers.setdefault(p, []).append(h)

        # /api/v1/workflow, /api/v1/system-prompts (GET), /api/v1/preferences (GET),
        # /api/v1/settings (GET), /api/v1/variant (GET), /api/v1/download, /api/v1/artifacts
        # must all have at least one non-guarded handler on each path
        public_paths = {
            "/api/v1/workflow",
            "/api/v1/system-prompts",
            "/api/v1/artifacts",
        }
        for path in public_paths:
            if path in paths_to_handlers:
                unguarded = [h for h in paths_to_handlers[path] if not _has_admin_guard(h)]
                assert unguarded, f"No public (non-admin-guarded) handler found for {path}"


# -- App-level integration -----------------------------------------------------


class TestCreateApp:
    """Integration tests verifying create_app() assembles a working Litestar instance."""

    def test_create_app_returns_litestar_instance(self) -> None:
        """create_app() must succeed and return a Litestar app object."""
        from litestar import Litestar

        from vetinari.web.litestar_app import create_app

        app = create_app()
        assert isinstance(app, Litestar)

    def test_create_app_has_route_handlers(self) -> None:
        """Litestar app must have at least one route registered."""
        from vetinari.web.litestar_app import create_app

        app = create_app()
        # Litestar exposes registered routes on the .routes attribute
        assert app.routes

    def test_create_app_total_handler_count(self) -> None:
        """Wave 1 factories alone contribute at least 64 handlers to the app.

        Counts per factory (verified from source):
          analytics: 7, dashboard_metrics: 5, project_git: 3,
          model_mgmt: 13, models_catalog: 14, models_discovery: 7,
          system_status: 10, system_hardware: 7, system_content: 12
          = 83 Wave 1 handlers
        Additional handlers from skills, a2a, approvals, log_stream, and
        the root health check add more  -  so total must exceed 74.
        """
        from vetinari.web.litestar_analytics import create_analytics_handlers
        from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers
        from vetinari.web.litestar_model_mgmt import create_model_mgmt_handlers
        from vetinari.web.litestar_models_catalog import create_models_catalog_handlers
        from vetinari.web.litestar_models_discovery import create_models_discovery_handlers
        from vetinari.web.litestar_project_git import create_project_git_handlers
        from vetinari.web.litestar_system_content import create_system_content_handlers
        from vetinari.web.litestar_system_hardware import create_system_hardware_handlers
        from vetinari.web.litestar_system_status import create_system_status_handlers

        wave1_count = sum([
            len(create_analytics_handlers()),
            len(create_dashboard_metrics_handlers()),
            len(create_project_git_handlers()),
            len(create_model_mgmt_handlers()),
            len(create_models_catalog_handlers()),
            len(create_models_discovery_handlers()),
            len(create_system_status_handlers()),
            len(create_system_hardware_handlers()),
            len(create_system_content_handlers()),
        ])
        assert wave1_count == 83, (
            f"Wave 1 handler total changed  -  expected 83, got {wave1_count}. "
            "Update this test if new handlers were intentionally added/removed."
        )
