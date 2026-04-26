"""Litestar ASGI application factory for Vetinari.

All Flask route families have been migrated to native Litestar handlers
(328 handlers, 309 routes).  This module is now the sole runtime entry point
for the web layer — Flask and the WSGI bridge have been removed.

``create_app()`` is called by:
- ``vetinari.cli_commands.cmd_serve`` — blocking uvicorn for ``vetinari serve``
- ``vetinari.cli_commands.cmd_start`` — background dashboard thread
- ``get_app()`` — module-level singleton for ``uvicorn ... --factory``

Decision: full Litestar cutover, WSGI bridge removed (ADR-0066 follow-up).
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ── Graceful shutdown state ──────────────────────────────────────────
_shutdown_event = threading.Event()
_shutdown_event_lock = threading.Lock()


def _handle_shutdown_signal(signum: int, _frame: object) -> None:
    """Handle SIGTERM and SIGINT for graceful server shutdown.

    Sets the module-level shutdown event so background workers and
    lifespan hooks can detect the request and perform cleanup.

    Args:
        signum: The signal number received (SIGTERM or SIGINT).
        _frame: Current stack frame (unused).
    """
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — initiating graceful shutdown", sig_name)
    with _shutdown_event_lock:
        _shutdown_event.set()


def _register_shutdown_handlers() -> None:
    """Register SIGTERM and SIGINT handlers for graceful shutdown.

    Safe to call from non-main threads — signal handlers can only be
    registered from the main thread, so this function checks first.
    """
    try:
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)
        signal.signal(signal.SIGINT, _handle_shutdown_signal)
        logger.debug("Graceful shutdown handlers registered (SIGTERM, SIGINT)")
    except (OSError, ValueError) as exc:
        # ValueError raised when called from a non-main thread (e.g. in tests)
        logger.warning("Could not register shutdown signal handlers: %s", exc)


def is_shutting_down() -> bool:
    """Return True if a shutdown signal has been received.

    Returns:
        True when SIGTERM or SIGINT has been received.
    """
    return _shutdown_event.is_set()


# Optional Litestar imports — graceful fallback when not installed
try:
    from litestar import Litestar, MediaType, Request, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# ── A2A transport singleton (shared across requests) ─────────────────
# Lazy-initialised by _get_a2a_transport(); shared by both POST handlers so
# tasks submitted via a2a.taskSend remain visible to a2a.taskStatus calls.
_a2a_transport: Any = None
_a2a_transport_lock = threading.Lock()

try:
    from vetinari.web.csrf import CSRFMiddleware

    _CSRF_AVAILABLE = True
except ImportError:
    _CSRF_AVAILABLE = False

try:
    from vetinari.web.litestar_middleware import (
        CORSMiddleware,
        JsonDepthGuardMiddleware,
        RequestIdMiddleware,
        SecurityHeadersMiddleware,
        UserActivityMiddleware,
    )

    _SECURITY_MIDDLEWARE_AVAILABLE = True
except ImportError:
    _SECURITY_MIDDLEWARE_AVAILABLE = False

try:
    from vetinari.web.litestar_exceptions import EXCEPTION_HANDLERS as _EXCEPTION_HANDLERS

    _EXCEPTIONS_AVAILABLE = True
except ImportError:
    _EXCEPTION_HANDLERS = {}
    _EXCEPTIONS_AVAILABLE = False


# ── Native Litestar route handlers ──────────────────────────────────


def _create_health_handler():
    """Create the health check endpoint handler.

    Returns:
        Litestar route handler function.
    """
    if not _LITESTAR_AVAILABLE:
        return None

    @get("/health", media_type=MediaType.JSON)
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            JSON response with status.
        """
        return {"status": "ok", "server": "litestar"}

    return health_check


def _get_a2a_transport() -> Any:
    """Return the module-level A2A transport singleton.

    Creates the executor and transport on first call using double-checked
    locking so concurrent requests during startup do not race.  Both POST
    handlers share this single instance so that tasks submitted via
    ``a2a.taskSend`` remain visible to subsequent ``a2a.taskStatus`` calls
    within the same server process.

    Returns:
        The shared :class:`~vetinari.a2a.transport.A2ATransport` instance.
    """
    global _a2a_transport
    if _a2a_transport is None:
        with _a2a_transport_lock:
            if _a2a_transport is None:
                from vetinari.a2a.executor import VetinariA2AExecutor
                from vetinari.a2a.transport import A2ATransport

                executor = VetinariA2AExecutor(recover_on_init=False)
                _a2a_transport = A2ATransport(executor=executor)
                logger.info("A2A transport singleton created")
    return _a2a_transport


# JSON-RPC fields required for a valid request — used to return 400 on malformed input
# before handing off to the transport layer.
_JSONRPC_REQUIRED_FIELDS = frozenset({"jsonrpc", "method", "id"})

# Methods that create a new resource → 201 Created.  All others → 200 OK.
_A2A_METHODS_RETURNING_201 = frozenset({"a2a.taskSend"})


def _create_a2a_handlers():
    """Create native Litestar handlers for the A2A protocol endpoints.

    These handlers expose the Vetinari A2A protocol stack over HTTP,
    allowing external agents to discover agent cards and submit tasks.

    HTTP status code semantics:
    - ``a2a.taskSend``  → 201 Created (new resource)
    - ``a2a.getAgentCard``, ``a2a.taskStatus`` → 200 OK (read/query)
    - Non-dict body or missing ``jsonrpc``/``method``/``id`` → 400 Bad Request
    - Transport-level JSON-RPC error responses → 200 OK (error encoded in body per
      JSON-RPC 2.0 spec; HTTP-level 400 is only for malformed input that cannot even
      be parsed as a JSON-RPC request)

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    @get("/api/v1/a2a/cards", media_type=MediaType.JSON, guards=[admin_guard])
    async def get_agent_cards() -> list[dict]:
        """Return all A2A agent cards.

        Returns:
            JSON array of agent card dicts.
        """
        from vetinari.a2a.agent_cards import get_all_cards

        return [card.to_dict() for card in get_all_cards()]

    @post("/api/v1/a2a", media_type=MediaType.JSON, guards=[admin_guard])
    async def handle_a2a_request(data: dict) -> Response:
        """Handle incoming A2A JSON-RPC requests.

        Validates that the body is a dict containing the required JSON-RPC
        fields (``jsonrpc``, ``method``, ``id``) before delegating to the
        shared transport singleton.  Returns 400 for malformed input, 201
        for task-creation methods, and 200 for everything else.

        Args:
            data: The JSON-RPC request payload parsed from the request body.

        Returns:
            Litestar ``Response`` with the JSON-RPC response dict and the
            appropriate HTTP status code.
        """
        if not isinstance(data, dict) or not _JSONRPC_REQUIRED_FIELDS.issubset(data):
            missing = _JSONRPC_REQUIRED_FIELDS - set(data) if isinstance(data, dict) else _JSONRPC_REQUIRED_FIELDS
            logger.warning(
                "A2A /api/v1/a2a received malformed request — missing fields: %s",
                sorted(missing),
            )
            error_body = {
                "jsonrpc": "2.0",
                "id": data.get("id") if isinstance(data, dict) else None,
                "error": {
                    "code": -32600,
                    "message": f"Invalid JSON-RPC request — missing required fields: {sorted(missing)}",
                },
            }
            return Response(content=error_body, status_code=400)

        transport = _get_a2a_transport()
        response_body = await asyncio.to_thread(transport.handle_request, data)
        method = data.get("method", "")
        if not isinstance(method, str):
            method = ""
        status_code = 201 if method in _A2A_METHODS_RETURNING_201 else 200
        return Response(content=response_body, status_code=status_code)

    @post("/api/v1/a2a/raw", media_type=MediaType.JSON, guards=[admin_guard])
    async def handle_a2a_request_raw(request: Request) -> Response:
        """Handle incoming A2A JSON-RPC requests sent as a raw byte body.

        Accepts clients that send the full JSON-RPC request as raw bytes
        rather than pre-parsed JSON.  Reads the request body directly,
        parses it, then delegates to the shared transport singleton.

        Returns 400 when the body is not valid JSON or is missing required
        JSON-RPC fields.  Returns 201 for task-creation methods and 200 for
        all other methods.

        Args:
            request: The incoming Litestar ``Request`` object.

        Returns:
            Litestar ``Response`` with the JSON-RPC response dict and the
            appropriate HTTP status code.
        """
        raw_body = await request.body()
        try:
            parsed = json.loads(raw_body)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "A2A /api/v1/a2a/raw received body that is not valid JSON — rejecting: %s",
                exc,
            )
            error_body = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: request body is not valid JSON"},
            }
            return Response(content=error_body, status_code=400)

        if not isinstance(parsed, dict) or not _JSONRPC_REQUIRED_FIELDS.issubset(parsed):
            missing = _JSONRPC_REQUIRED_FIELDS - set(parsed) if isinstance(parsed, dict) else _JSONRPC_REQUIRED_FIELDS
            logger.warning(
                "A2A /api/v1/a2a/raw received malformed JSON-RPC request — missing fields: %s",
                sorted(missing),
            )
            error_body = {
                "jsonrpc": "2.0",
                "id": parsed.get("id") if isinstance(parsed, dict) else None,
                "error": {
                    "code": -32600,
                    "message": f"Invalid JSON-RPC request — missing required fields: {sorted(missing)}",
                },
            }
            return Response(content=error_body, status_code=400)

        transport = _get_a2a_transport()
        response_body = await asyncio.to_thread(transport.handle_request, parsed)
        method = parsed.get("method", "")
        if not isinstance(method, str):
            method = ""
        status_code = 201 if method in _A2A_METHODS_RETURNING_201 else 200
        return Response(content=response_body, status_code=status_code)

    return [get_agent_cards, handle_a2a_request, handle_a2a_request_raw]


@asynccontextmanager
async def _lifespan(app: Any):
    """Delegate mounted app startup/shutdown to the shared lifespan module."""
    from vetinari.web.lifespan import vetinari_lifespan

    async with vetinari_lifespan(app):
        yield


# ── Application factory ─────────────────────────────────────────────


def create_app(
    debug: bool = False,
) -> Any:
    """Create the Litestar ASGI application.

    Assembles all native Litestar route handlers, wires security middleware,
    and registers the lifespan hook for subsystem initialisation.

    Args:
        debug: Enable Litestar debug mode (extra validation and tracebacks).

    Returns:
        Litestar application instance.

    Raises:
        RuntimeError: If Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        raise RuntimeError(
            "Litestar is not installed. Install with: pip install 'litestar>=2.12' 'uvicorn>=0.30'",  # noqa: VET301 — user guidance string
        )

    # Register signal handlers so the server can shut down cleanly on SIGTERM/SIGINT
    _register_shutdown_handlers()

    route_handlers: list[Any] = []

    # Health endpoint
    health = _create_health_handler()
    if health:
        route_handlers.append(health)

    # A2A protocol endpoints
    route_handlers.extend(_create_a2a_handlers())

    # Approvals and trust API (approval queue + governor endpoints)
    from vetinari.web.approvals_api import create_approvals_handlers

    route_handlers.extend(create_approvals_handlers())

    # Wave 1 migrated Flask blueprints (Session 21)
    from vetinari.web.litestar_analytics import create_analytics_handlers

    route_handlers.extend(create_analytics_handlers())
    from vetinari.web.litestar_dashboard_metrics import create_dashboard_metrics_handlers

    route_handlers.extend(create_dashboard_metrics_handlers())
    from vetinari.web.litestar_project_git import create_project_git_handlers

    route_handlers.extend(create_project_git_handlers())
    from vetinari.web.litestar_model_mgmt import create_model_mgmt_handlers

    route_handlers.extend(create_model_mgmt_handlers())
    from vetinari.web.litestar_models_catalog import create_models_catalog_handlers

    route_handlers.extend(create_models_catalog_handlers())
    from vetinari.web.litestar_models_discovery import create_models_discovery_handlers

    route_handlers.extend(create_models_discovery_handlers())
    from vetinari.web.litestar_system_status import create_system_status_handlers

    route_handlers.extend(create_system_status_handlers())
    from vetinari.web.litestar_system_hardware import create_system_hardware_handlers

    route_handlers.extend(create_system_hardware_handlers())
    from vetinari.web.litestar_system_content import create_system_content_handlers

    route_handlers.extend(create_system_content_handlers())
    from vetinari.web.litestar_log_stream import create_log_stream_handlers

    route_handlers.extend(create_log_stream_handlers())

    # Wave 2 migrated Flask blueprints (Session 22)
    from vetinari.web.litestar_admin_routes import create_admin_handlers

    route_handlers.extend(create_admin_handlers())
    from vetinari.web.litestar_adr_routes import create_adr_routes_handlers

    route_handlers.extend(create_adr_routes_handlers())
    from vetinari.web.litestar_agents_api import create_agents_api_handlers

    route_handlers.extend(create_agents_api_handlers())
    from vetinari.web.litestar_analytics_routes import create_analytics_routes_handlers

    route_handlers.extend(create_analytics_routes_handlers())
    from vetinari.web.litestar_audit_api import create_audit_api_handlers

    route_handlers.extend(create_audit_api_handlers())
    from vetinari.web.litestar_autonomy_api import create_autonomy_api_handlers

    route_handlers.extend(create_autonomy_api_handlers())
    from vetinari.web.litestar_chat_api import create_chat_api_handlers

    route_handlers.extend(create_chat_api_handlers())
    from vetinari.web.litestar_cost_analysis_api import create_cost_analysis_api_handlers

    route_handlers.extend(create_cost_analysis_api_handlers())
    from vetinari.web.litestar_dashboard_api import create_dashboard_api_handlers

    route_handlers.extend(create_dashboard_api_handlers())
    from vetinari.web.litestar_decisions_api import create_decisions_api_handlers

    route_handlers.extend(create_decisions_api_handlers())
    from vetinari.web.litestar_decomposition_routes import create_decomposition_routes_handlers

    route_handlers.extend(create_decomposition_routes_handlers())
    from vetinari.web.litestar_learning_api import create_learning_api_handlers

    route_handlers.extend(create_learning_api_handlers())
    from vetinari.web.litestar_manufacturing_api import create_manufacturing_handlers

    route_handlers.extend(create_manufacturing_handlers())
    from vetinari.web.litestar_mcp_transport import create_mcp_transport_handlers

    route_handlers.extend(create_mcp_transport_handlers())
    from vetinari.web.litestar_memory_api import create_memory_handlers

    route_handlers.extend(create_memory_handlers())
    from vetinari.web.litestar_plan_api import create_plan_api_handlers

    route_handlers.extend(create_plan_api_handlers())
    from vetinari.web.litestar_plans_api import create_plans_api_handlers

    route_handlers.extend(create_plans_api_handlers())
    from vetinari.web.litestar_ponder_routes import create_ponder_routes_handlers

    route_handlers.extend(create_ponder_routes_handlers())
    from vetinari.web.litestar_projects_api import create_projects_api_handlers

    route_handlers.extend(create_projects_api_handlers())
    from vetinari.web.litestar_projects_execution import create_projects_execution_handlers

    route_handlers.extend(create_projects_execution_handlers())
    from vetinari.web.litestar_receipts_api import create_receipts_api_handlers

    route_handlers.extend(create_receipts_api_handlers())
    from vetinari.web.litestar_replay_api import create_replay_api_handlers

    route_handlers.extend(create_replay_api_handlers())
    from vetinari.web.litestar_sse_replay_api import create_sse_replay_handlers

    route_handlers.extend(create_sse_replay_handlers())
    from vetinari.web.litestar_rules_routes import create_rules_routes_handlers

    route_handlers.extend(create_rules_routes_handlers())
    from vetinari.web.litestar_sandbox_api import create_sandbox_handlers

    route_handlers.extend(create_sandbox_handlers())
    from vetinari.web.litestar_search_api import create_search_handlers

    route_handlers.extend(create_search_handlers())
    from vetinari.web.litestar_skills_api import create_skills_api_handlers

    route_handlers.extend(create_skills_api_handlers())
    from vetinari.web.litestar_subtasks_api import create_subtasks_api_handlers

    route_handlers.extend(create_subtasks_api_handlers())
    from vetinari.web.litestar_tasks_api import create_tasks_api_handlers

    route_handlers.extend(create_tasks_api_handlers())
    from vetinari.web.litestar_training_api_part2 import create_training_api_handlers

    route_handlers.extend(create_training_api_handlers())
    from vetinari.web.litestar_training_experiments_api import create_training_experiments_handlers

    route_handlers.extend(create_training_experiments_handlers())
    from vetinari.web.litestar_training_routes import create_training_routes_handlers

    route_handlers.extend(create_training_routes_handlers())
    from vetinari.web.litestar_visualization import create_visualization_handlers

    route_handlers.extend(create_visualization_handlers())
    from vetinari.web.litestar_milestones_api import create_milestones_handlers

    route_handlers.extend(create_milestones_handlers())

    # Build middleware list (order matters — outermost first)
    middleware: list[Any] = []

    # Security middleware ported from Flask hooks (web_ui.py).
    # ASGIMiddleware subclasses require instances, not classes.
    if _SECURITY_MIDDLEWARE_AVAILABLE:
        middleware.extend([
            UserActivityMiddleware(),  # records idle-detector activity
            RequestIdMiddleware(),  # injects X-Request-ID into log context
            SecurityHeadersMiddleware(),  # adds X-Frame-Options, CSP, etc.
            CORSMiddleware(),  # restricts CORS to localhost origins
            JsonDepthGuardMiddleware(),  # rejects JSON bodies with depth > 5
        ])

    # CSRF protection — custom header validation for mutation requests
    if _CSRF_AVAILABLE:
        middleware.append(CSRFMiddleware)

    app = Litestar(
        route_handlers=route_handlers,
        middleware=middleware,
        exception_handlers=_EXCEPTION_HANDLERS,
        lifespan=[_lifespan],
        logging_config=None,
        debug=debug,
    )

    logger.info(
        "Litestar app created with %d route handlers",
        len(route_handlers),
    )
    return app


# ── Module-level app (for uvicorn) ──────────────────────────────────

# Lazy creation to avoid import-time Litestar dependency
_app: Any = None
_app_lock = threading.Lock()


def get_app() -> Any:
    """Return the module-level Litestar app singleton.

    Creates the app on first call.  Used by uvicorn:
    ``uvicorn vetinari.web.litestar_app:get_app --factory``

    Returns:
        Litestar application instance.
    """
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                _app = create_app()
    return _app
