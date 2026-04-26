"""Parity tests for Wave 2 Litestar handler migration.

Verifies that each migrated handler factory for skills_api, training_routes,
and training_api:
- Returns the expected number of handlers
- Registers the correct route paths
- Applies admin_guard where Flask had @require_admin
- Module-level utility functions in training_api are importable
"""

from __future__ import annotations

import pytest

# Skip the entire module when Litestar is not installed


# -- helpers -------------------------------------------------------------------


def _paths(handlers: list) -> set[str]:
    """Extract all declared paths from a list of handler objects.

    Args:
        handlers: List of Litestar route handler objects.

    Returns:
        Set of path strings across all handlers.
    """
    result: set[str] = set()
    for h in handlers:
        paths_attr = getattr(h, "paths", None)
        if paths_attr:
            result.update(paths_attr)
    return result


def _has_admin_guard(handler: object) -> bool:
    """Return True when admin_guard appears in a handler's guards list.

    Args:
        handler: A Litestar route handler object.

    Returns:
        True if admin_guard is present in handler.guards.
    """
    from vetinari.web.litestar_guards import admin_guard

    guards = getattr(handler, "guards", None) or []
    return admin_guard in guards


def _handler_by_path(handlers: list, path: str) -> object | None:
    """Find the first handler whose paths set contains *path*.

    Args:
        handlers: List of Litestar route handler objects.
        path: The exact path string to look for.

    Returns:
        The matching handler, or None.
    """
    for h in handlers:
        if path in (getattr(h, "paths", None) or set()):
            return h
    return None


# -- litestar_skills_api -------------------------------------------------------


class TestSkillsApiHandlers:
    """create_skills_api_handlers() covers all skill catalog and registry routes."""

    def test_returns_ten_handlers(self) -> None:
        """Factory must return exactly 10 handlers."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        assert len(handlers) == 10, f"expected 10, got {len(handlers)}"

    def test_catalog_list_path_registered(self) -> None:
        """GET /api/v1/skills/catalog must be registered."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        assert "/api/v1/skills/catalog" in _paths(handlers)

    def test_capabilities_path_registered(self) -> None:
        """GET /api/v1/skills/capabilities must be registered."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        assert "/api/v1/skills/capabilities" in _paths(handlers)

    def test_catalog_by_agent_path_registered(self) -> None:
        """GET /api/v1/skills/catalog/{agent} must be registered."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        assert "/api/v1/skills/catalog/{agent:str}" in _paths(handlers)

    def test_no_admin_guard_on_catalog_list(self) -> None:
        """Catalog list is a public endpoint  -  must not have admin_guard."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        h = _handler_by_path(handlers, "/api/v1/skills/catalog")
        assert h is not None
        assert not _has_admin_guard(h)

    def test_no_admin_guard_on_capabilities(self) -> None:
        """Capabilities endpoint is public  -  must not have admin_guard."""
        from vetinari.web.litestar_skills_api import create_skills_api_handlers

        handlers = create_skills_api_handlers()
        h = _handler_by_path(handlers, "/api/v1/skills/capabilities")
        assert h is not None
        assert not _has_admin_guard(h)


# -- litestar_training_routes --------------------------------------------------


class TestTrainingRoutesHandlers:
    """create_training_routes_handlers() covers the 6 routes from training_routes.py."""

    def test_returns_six_handlers(self) -> None:
        """Factory must return exactly 6 handlers."""
        from vetinari.web.litestar_training_routes import create_training_routes_handlers

        handlers = create_training_routes_handlers()
        assert len(handlers) == 6, f"expected 6, got {len(handlers)}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/generate-image",
            "/api/image-status",
            "/api/sd-status",
            "/api/training/stats",
            "/api/training/export",
            "/api/training/start",
        ],
    )
    def test_path_registered(self, path: str) -> None:
        """Each Flask route path must appear in the Litestar surface."""
        from vetinari.web.litestar_training_routes import create_training_routes_handlers

        handlers = create_training_routes_handlers()
        assert path in _paths(handlers), f"path not registered: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/generate-image",
            "/api/training/export",
            "/api/training/start",
        ],
    )
    def test_admin_guard_on_admin_endpoints(self, path: str) -> None:
        """Admin endpoints from Flask @require_admin must have admin_guard."""
        from vetinari.web.litestar_training_routes import create_training_routes_handlers

        handlers = create_training_routes_handlers()
        h = _handler_by_path(handlers, path)
        assert h is not None, f"handler not found for {path}"
        assert _has_admin_guard(h), f"admin_guard missing on {path}"

    @pytest.mark.parametrize(
        "path",
        ["/api/image-status", "/api/sd-status", "/api/training/stats"],
    )
    def test_no_admin_guard_on_public_endpoints(self, path: str) -> None:
        """Public endpoints must not have admin_guard."""
        from vetinari.web.litestar_training_routes import create_training_routes_handlers

        handlers = create_training_routes_handlers()
        h = _handler_by_path(handlers, path)
        assert h is not None, f"handler not found for {path}"
        assert not _has_admin_guard(h), f"admin_guard should not be on public path {path}"


# -- litestar_training_api -----------------------------------------------------


class TestTrainingApiHandlers:
    """create_training_api_handlers() covers all 22 routes from training_api.py.

    The combined factory lives in litestar_training_api_part2 and delegates
    to _create_training_api_handlers_part1 in litestar_training_api.
    """

    def test_returns_twenty_two_handlers(self) -> None:
        """Factory must return exactly 22 handlers."""
        from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

        handlers = create_training_api_handlers()
        assert len(handlers) == 22, f"expected 22, got {len(handlers)}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/training/status",
            "/api/v1/training/start",
            "/api/v1/training/pause",
            "/api/v1/training/resume",
            "/api/v1/training/stop",
            "/api/v1/training/data/stats",
            "/api/v1/training/data/seed",
            "/api/v1/training/data/seed/stream",
            "/api/v1/training/curriculum",
            "/api/v1/training/curriculum/next",
            "/api/v1/training/history",
            "/api/v1/training/jobs",
            "/api/v1/training/summary",
            "/api/v1/training/quality",
            "/api/v1/training/models",
            "/api/v1/training/adapters",
            "/api/v1/training/adapters/deployed",
        ],
    )
    def test_path_registered(self, path: str) -> None:
        """Each Flask route path must appear in the Litestar surface."""
        from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

        handlers = create_training_api_handlers()
        assert path in _paths(handlers), f"path not registered: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/training/start",
            "/api/v1/training/pause",
            "/api/v1/training/resume",
            "/api/v1/training/stop",
            "/api/v1/training/data/seed",
        ],
    )
    def test_admin_guard_on_admin_endpoints(self, path: str) -> None:
        """Endpoints that had @require_admin in Flask must have admin_guard."""
        from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

        handlers = create_training_api_handlers()
        h = _handler_by_path(handlers, path)
        assert h is not None, f"handler not found for {path}"
        assert _has_admin_guard(h), f"admin_guard missing on {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/training/status",
            "/api/v1/training/data/stats",
            "/api/v1/training/data/seed/stream",
            "/api/v1/training/curriculum",
            "/api/v1/training/curriculum/next",
            "/api/v1/training/history",
            "/api/v1/training/jobs",
            "/api/v1/training/summary",
            "/api/v1/training/quality",
            "/api/v1/training/models",
            "/api/v1/training/adapters",
            "/api/v1/training/adapters/deployed",
        ],
    )
    def test_no_admin_guard_on_public_endpoints(self, path: str) -> None:
        """Public read endpoints must not have admin_guard."""
        from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

        handlers = create_training_api_handlers()
        h = _handler_by_path(handlers, path)
        assert h is not None, f"handler not found for {path}"
        assert not _has_admin_guard(h), f"admin_guard should not be on public path {path}"

    def test_sse_stream_handler_present(self) -> None:
        """The seed/stream SSE handler must exist and not have admin_guard."""
        from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

        handlers = create_training_api_handlers()
        h = _handler_by_path(handlers, "/api/v1/training/data/seed/stream")
        assert h is not None
        assert not _has_admin_guard(h)


# -- Module-level utility functions --------------------------------------------


class TestTrainingApiUtilityFunctions:
    """Module-level utility functions must be importable and return correct shapes."""

    def test_get_training_status_returns_dict(self) -> None:
        """get_training_status() must return a dict with required keys."""
        from vetinari.web.litestar_training_api import get_training_status

        result = get_training_status()
        assert isinstance(result, dict)
        for key in ("status", "current_job", "last_run", "records_collected", "curriculum_phase", "next_activity"):
            assert key in result, f"missing key: {key}"

    def test_get_training_history_returns_list(self) -> None:
        """get_training_history() must return a list."""
        from vetinari.web.litestar_training_api import get_training_history

        result = get_training_history()
        assert isinstance(result, list)

    def test_get_training_history_limit_respected(self) -> None:
        """get_training_history(limit=N) must return at most N entries."""
        from vetinari.web.litestar_training_api import get_training_history

        result = get_training_history(limit=5)
        assert len(result) <= 5

    def test_get_quality_comparison_returns_dict(self) -> None:
        """get_quality_comparison() must return a dict with required keys."""
        from vetinari.web.litestar_training_api import get_quality_comparison

        result = get_quality_comparison()
        assert isinstance(result, dict)
        for key in ("baseline_quality", "candidate_quality", "quality_delta", "decision", "latency_ratio"):
            assert key in result, f"missing key: {key}"

    def test_get_quality_comparison_sentinel_keys_present(self) -> None:
        """get_quality_comparison() sentinel must contain all expected keys with numeric values."""
        from vetinari.web.litestar_training_api import get_quality_comparison

        result = get_quality_comparison()
        # When quality gate is unavailable the sentinel has decision="no_data"
        # and numeric 0.0 defaults  -  verify shape regardless of gate availability.
        assert isinstance(result.get("baseline_quality"), float)
        assert isinstance(result.get("latency_ratio"), float)
        assert isinstance(result.get("decision"), str)
