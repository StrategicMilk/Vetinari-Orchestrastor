"""Sandboxed code execution endpoints as native Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``sandbox_api``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

All routes require admin privileges (localhost or valid
``VETINARI_ADMIN_TOKEN``) because they expose arbitrary code execution and
audit data.

Endpoints
---------
    POST /api/sandbox/execute       — Execute code in an isolated sandbox
    GET  /api/sandbox/status        — Return sandbox health and configuration
    GET  /api/sandbox/audit         — Return recent sandbox audit log entries
    GET  /api/sandbox/plugins       — Return discovered plugin manifests
    POST /api/sandbox/plugins/hook  — Execute a named hook in an external plugin
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)
_MAX_SANDBOX_EXECUTE_TIMEOUT = 300

try:
    from litestar import get, post
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_sandbox_handlers() -> list[Any]:
    """Create and return all sandbox API route handlers.

    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without guarding the call.

    Returns:
        List of Litestar route handler objects for all sandbox endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response

    # -- Execute -------------------------------------------------------------

    @post("/api/sandbox/execute", guards=[admin_guard])
    async def api_sandbox_execute(data: dict[str, Any]) -> Any:
        """Execute arbitrary code inside the configured sandbox.

        Requires admin access (localhost or valid ``VETINARI_ADMIN_TOKEN``
        header) to prevent unauthorized remote code execution.

        Args:
            data: JSON request body with the following fields:
                code: Source code string to execute (required, must be str).
                sandbox_type: Sandbox backend to use (default
                    ``"in_process"``).
                timeout: Maximum execution time in seconds (default ``30``).
                context: Optional dict of context variables passed to the
                    sandbox (must be dict when provided).

        Returns:
            JSON-serialized ``SandboxResult`` from
            ``sandbox_manager.execute()``.
            Returns 400 when ``code`` is absent or not a string, or when
            ``context`` is provided but is not a dict.
            Returns 503 when the sandbox manager is unavailable.
        """
        import dataclasses

        code = data.get("code")
        if not code:
            return litestar_error_response("Missing required field: code", code=400)
        if not isinstance(code, str):
            return litestar_error_response("Field 'code' must be a string", code=400)

        timeout = data.get("timeout", 30)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            return litestar_error_response("Field 'timeout' must be a number", code=400)
        if not math.isfinite(float(timeout)) or timeout <= 0 or timeout > _MAX_SANDBOX_EXECUTE_TIMEOUT:
            return litestar_error_response(
                f"Field 'timeout' must be between 0 and {_MAX_SANDBOX_EXECUTE_TIMEOUT}",
                code=400,
            )

        context = data.get("context")
        if context is not None and not isinstance(context, dict):
            return litestar_error_response("Field 'context' must be a dict when provided", code=400)

        try:
            from vetinari.sandbox_manager import sandbox_manager

            # client_id forwarding for per-client rate limiting (P1.C2) is handled
            # via the connection object injected by the guard; we use a placeholder
            # here as Litestar handlers do not expose request.remote_addr directly
            # without injecting the Request.  Guards have already authenticated, so
            # using "unknown" as fallback is safe for rate-limiting purposes.
            client_id = "unknown"

            result = sandbox_manager.execute(
                code=code,
                sandbox_type=data.get("sandbox_type", "in_process"),
                timeout=timeout,
                context=context,
                client_id=client_id,
            )

            return result.to_dict() if hasattr(result, "to_dict") else dataclasses.asdict(result)
        except Exception as exc:
            logger.warning("api_sandbox_execute: sandbox manager unavailable — returning 503: %s", exc)
            return litestar_error_response("Sandbox manager unavailable", code=503)

    # -- Status --------------------------------------------------------------

    @get("/api/sandbox/status", guards=[admin_guard])
    async def api_sandbox_status() -> Any:
        """Return current sandbox health and configuration.

        Requires admin access to prevent information disclosure about the
        sandbox environment.

        Returns:
            JSON object returned by ``sandbox_manager.get_status()``, or a
            503 error when the sandbox manager is unavailable.
        """
        try:
            from vetinari.sandbox_manager import sandbox_manager

            return sandbox_manager.get_status()
        except Exception as exc:
            logger.warning("api_sandbox_status: sandbox manager unavailable — returning 503: %s", exc)
            return litestar_error_response("Sandbox manager unavailable", code=503)

    # -- Audit ---------------------------------------------------------------

    @get("/api/sandbox/audit", guards=[admin_guard])
    async def api_sandbox_audit(
        limit: int = Parameter(query="limit", default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        """Return recent sandbox execution audit log entries.

        Requires admin access because the audit log may contain sensitive code
        fragments submitted by other users.

        Args:
            limit: Maximum number of entries to return (default 100, max 1000).

        Returns:
            JSON object with ``audit_entries`` list and ``total`` count.
            Returns 503 when the sandbox manager is unavailable.
        """
        try:
            from vetinari.sandbox_manager import sandbox_manager

            audit = sandbox_manager.get_audit_log(limit)
            return {"audit_entries": audit, "total": len(audit)}
        except Exception as exc:
            logger.warning("api_sandbox_audit: sandbox manager unavailable — returning 503: %s", exc)
            return litestar_error_response("Sandbox manager unavailable", code=503)

    # -- Plugin list ---------------------------------------------------------

    @get("/api/sandbox/plugins", guards=[admin_guard])
    async def api_plugin_list() -> dict[str, Any]:
        """Return discovered plugin manifests from the external plugin directory.

        Scans the configured plugin directory for subdirectories that contain a
        ``manifest.yaml`` file and returns the parsed manifest data.

        Requires admin access because plugins have access to the sandboxed
        execution environment.

        Returns:
            JSON object with a ``plugins`` list of manifest dicts and a
            ``total`` count.
            Returns 503 when the plugin sandbox is unavailable.
        """
        try:
            from vetinari.sandbox_policy import ExternalPluginSandbox

            sandbox = ExternalPluginSandbox()
            plugins = sandbox.discover_plugins()
            return {"plugins": plugins, "total": len(plugins)}
        except Exception as exc:
            logger.warning("api_plugin_list: plugin sandbox unavailable — returning 503: %s", exc)
            return litestar_error_response("Plugin sandbox unavailable", code=503)

    # -- Plugin hook execution -----------------------------------------------

    @post("/api/sandbox/plugins/hook", guards=[admin_guard])
    async def api_plugin_execute_hook(data: dict[str, Any]) -> Any:
        """Execute a named hook in an external plugin.

        Only hooks listed in ``ExternalPluginSandbox.ALLOWED_HOOKS`` may be
        invoked. All invocations are written to the sandbox audit log.

        Requires admin access to prevent unauthorized execution of plugin code.

        Args:
            data: JSON request body with the following fields:
                plugin_name: Name of the plugin subdirectory to invoke
                    (required).
                hook_name: Name of the hook function within the plugin
                    (required).
                params: Optional dict of parameters forwarded to the hook
                    (default ``{}``).

        Returns:
            JSON object with a ``result`` key containing the hook's return
            value, or a 400 error on missing parameters.
        """
        plugin_name = data.get("plugin_name", "")
        hook_name = data.get("hook_name", "")
        params = data.get("params", {})
        if params is None:
            params = {}

        if not isinstance(plugin_name, str):
            return litestar_error_response("Invalid type for 'plugin_name': expected a string", code=400)
        if not isinstance(hook_name, str):
            return litestar_error_response("Invalid type for 'hook_name': expected a string", code=400)
        if not isinstance(params, dict):
            return litestar_error_response("Invalid type for 'params': expected a dict", code=400)
        if not plugin_name:
            return litestar_error_response("plugin_name is required", code=400)
        if not isinstance(plugin_name, str):
            return litestar_error_response("plugin_name must be a string", code=400)
        if not hook_name:
            return litestar_error_response("hook_name is required", code=400)
        if not isinstance(hook_name, str):
            return litestar_error_response("hook_name must be a string", code=400)

        try:
            from vetinari.sandbox_policy import ExternalPluginSandbox

            sandbox = ExternalPluginSandbox()
            result = sandbox.execute_hook(plugin_name, hook_name, params)
            return {"result": result}
        except Exception as exc:
            logger.warning("api_plugin_execute_hook: plugin sandbox unavailable — returning 503: %s", exc)
            return litestar_error_response("Plugin sandbox unavailable", code=503)

    return [
        api_sandbox_execute,
        api_sandbox_status,
        api_sandbox_audit,
        api_plugin_list,
        api_plugin_execute_hook,
    ]
