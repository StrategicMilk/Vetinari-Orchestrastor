"""Admin and credentials API endpoints as native Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``admin_routes``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    GET    /api/admin/credentials                      — List credentials and health
    POST   /api/admin/credentials/<source_type>        — Store or update a credential
    POST   /api/admin/credentials/<source_type>/rotate — Rotate active token
    DELETE /api/admin/credentials/<source_type>        — Remove a credential
    GET    /api/admin/credentials/health               — Credential manager health
    GET    /api/admin/permissions                      — Current user admin status
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import Request, delete, get, post
    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_admin_handlers() -> list[Any]:
    """Create and return all admin API route handlers.

    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without guarding the call.

    Returns:
        List of Litestar route handler objects for all admin endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    import re as _re

    from vetinari.web.litestar_guards import admin_guard, is_admin_connection
    from vetinari.web.responses import litestar_error_response

    # Pattern for safe credential source_type identifiers — letters, digits,
    # underscores, and hyphens only.  Rejects path traversal payloads.
    _SOURCE_TYPE_RE = _re.compile(r"[A-Za-z0-9_-]{1,128}")

    def _validate_source_type(source_type: str) -> str | None:
        """Return an error message if source_type is unsafe, else None."""
        if not _SOURCE_TYPE_RE.fullmatch(source_type):
            return "source_type must contain only letters, digits, underscores, or hyphens (max 128 chars)"
        return None

    # -- List credentials ----------------------------------------------------

    @get("/api/admin/credentials", guards=[admin_guard])
    async def api_admin_list_credentials() -> dict[str, Any]:
        """List all stored credentials and their health status.

        Requires admin privileges. Returns each credential's source type and
        the overall health check result from the credential manager.

        Returns:
            JSON response with ``status``, ``credentials``, and ``health``
            fields.
        """
        from vetinari.credentials import get_credential_manager

        manager = get_credential_manager()
        credentials = manager.list()
        health = manager.health()
        return {"status": "ok", "credentials": credentials, "health": health}

    # -- Set credential ------------------------------------------------------

    @post("/api/admin/credentials/{source_type:str}", guards=[admin_guard])
    async def api_admin_set_credential(
        source_type: str, data: dict[str, Any]
    ) -> Any:
        """Store or update a credential for a named source type.

        Requires admin privileges. Reads ``token``, ``credential_type``,
        ``rotation_days``, and ``note`` from the JSON request body.

        Args:
            source_type: Identifier for the credential source (e.g.
                ``"openai"``).
            data: JSON request body with token and optional metadata.

        Returns:
            JSON response confirming the credential was stored, or 400 when
            the token is absent.
        """
        _err = _validate_source_type(source_type)
        if _err:
            return litestar_error_response(_err, code=400)

        from vetinari.credentials import get_credential_manager

        token = data.get("token", "")
        if not isinstance(token, str):
            return litestar_error_response("'token' must be a string", code=422)
        credential_type = data.get("credential_type", "bearer")
        if not isinstance(credential_type, str):
            return litestar_error_response("'credential_type' must be a string", code=422)
        rotation_days = data.get("rotation_days", 30)
        # bool is a subclass of int in Python — must reject it before the int check
        if isinstance(rotation_days, bool):
            return litestar_error_response("'rotation_days' must be an integer, not a boolean", code=422)
        if not isinstance(rotation_days, int):
            return litestar_error_response("'rotation_days' must be an integer", code=422)
        note = data.get("note", "")
        if not isinstance(note, str):
            return litestar_error_response("'note' must be a string", code=422)

        if not token:
            return litestar_error_response("Token is required", code=400)

        get_credential_manager().set_credential(
            source_type=source_type,
            token=token,
            credential_type=credential_type,
            rotation_days=rotation_days,
            note=note,
        )

        return {"status": "ok", "message": f"Credential set for {source_type}"}

    # -- Rotate credential ---------------------------------------------------

    @post("/api/admin/credentials/{source_type:str}/rotate", guards=[admin_guard])
    async def api_admin_rotate_credential(
        source_type: str, data: dict[str, Any]
    ) -> Any:
        """Rotate the active token for an existing credential.

        Requires admin privileges. Reads the new ``token`` from the JSON body.
        Returns 404 if no credential exists for the given source type.

        Args:
            source_type: Identifier for the credential source to rotate.
            data: JSON request body with the new ``token`` (required).

        Returns:
            JSON response confirming rotation, or 404 if credential not found,
            or 400 when the new token is absent.
        """
        _err = _validate_source_type(source_type)
        if _err:
            return litestar_error_response(_err, code=400)

        from vetinari.credentials import get_credential_manager

        new_token = data.get("token", "")
        if not isinstance(new_token, str):
            return litestar_error_response("'token' must be a string", code=422)

        if not new_token:
            return litestar_error_response("New token is required", code=400)

        success = get_credential_manager().rotate(source_type, new_token)

        if success:
            return {"status": "ok", "message": f"Credential rotated for {source_type}"}
        return litestar_error_response("Credential not found", code=404)

    # -- Delete credential ---------------------------------------------------

    @delete("/api/admin/credentials/{source_type:str}", guards=[admin_guard], status_code=200)
    async def api_admin_delete_credential(source_type: str) -> dict[str, Any]:
        """Remove a stored credential by source type.

        Requires admin privileges. Permanently deletes the credential from the
        vault.

        Args:
            source_type: Identifier for the credential source to remove.

        Returns:
            JSON response confirming removal.
        """
        _err = _validate_source_type(source_type)
        if _err:
            return litestar_error_response(_err, code=400)  # type: ignore[return-value]

        from vetinari.credentials import get_credential_manager

        get_credential_manager().vault.remove_credential(source_type)
        return {"status": "ok", "message": f"Credential removed for {source_type}"}

    # -- Credentials health --------------------------------------------------

    @get("/api/admin/credentials/health", guards=[admin_guard])
    async def api_admin_credentials_health() -> dict[str, Any]:
        """Return the health status of the credential manager.

        Requires admin privileges. Checks expiry, rotation schedules, and
        vault connectivity for all stored credentials.

        Returns:
            JSON response with ``status`` and ``health`` detail from the
            manager.
        """
        from vetinari.credentials import get_credential_manager

        health = get_credential_manager().health()
        return {"status": "ok", "health": health}

    # -- Permissions ---------------------------------------------------------

    @get("/api/admin/permissions", guards=[admin_guard])
    async def api_admin_permissions(request: Request) -> dict[str, Any]:
        """Return the current user's admin permission status.

        Args:
            request: The active request carrying the connection context.

        Returns:
            JSON response with an ``admin`` boolean flag and HTTP 200.
        """
        return {"admin": is_admin_connection(request)}

    return [
        api_admin_list_credentials,
        api_admin_set_credential,
        api_admin_rotate_credential,
        api_admin_delete_credential,
        api_admin_credentials_health,
        api_admin_permissions,
    ]
