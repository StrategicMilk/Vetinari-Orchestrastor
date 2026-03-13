"""Vetinari Web Blueprints — modular route groups extracted from web_ui.py.

Each blueprint is a self-contained route group with lazy imports to
avoid circular dependencies.
"""

from __future__ import annotations

import hmac
import os
from functools import wraps

from flask import jsonify, request


def is_admin_user() -> bool:
    """Check if the current request comes from an admin user.

    Uses constant-time comparison (hmac.compare_digest) to prevent timing
    attacks when a VETINARI_ADMIN_TOKEN is configured.

    For local deployments (no token set), only requests from the loopback
    address are accepted — X-Forwarded-For is intentionally ignored unless
    VETINARI_TRUSTED_PROXY=true is set, to prevent IP-spoofing (P1.H8).
    """
    admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
    if admin_token:
        auth_header = request.headers.get("Authorization", "")
        req_token = request.headers.get("X-Admin-Token", "")
        provided = req_token or auth_header.replace("Bearer ", "")
        # Use hmac.compare_digest to prevent timing-based token oracle (P1.C1/P1.H10)
        return hmac.compare_digest(provided, admin_token)

    # No token configured — fall back to IP-based localhost check.
    # Use request.remote_addr (the actual TCP peer) unless a trusted proxy
    # is explicitly configured, to prevent X-Forwarded-For spoofing (P1.H8).
    trusted_proxy = os.environ.get("VETINARI_TRUSTED_PROXY", "").lower() in ("1", "true", "yes")
    if trusted_proxy:
        # Only read X-Forwarded-For when the operator has opted in
        forwarded = request.headers.get("X-Forwarded-For", "")
        remote = (forwarded.split(",")[0].strip() if forwarded else request.remote_addr) or ""
    else:
        remote = request.remote_addr or ""

    return remote in ("127.0.0.1", "::1", "localhost")


def require_admin(f):
    """Decorator that rejects non-admin requests with 403 before the route runs.

    Usage::

        @bp.route('/api/some-mutating-route', methods=['POST'])
        @require_admin
        def my_route():
            ...
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_admin_user():
            return jsonify({"error": "Admin privileges required"}), 403
        return f(*args, **kwargs)

    return decorated


def validate_json_fields(data: dict, required: list) -> tuple:
    """Validate that all required fields are present and non-empty in a JSON body.

    Returns (True, None) on success, or (False, error_response_tuple) on failure.
    Callers should check: ok, err = validate_json_fields(...); if not ok: return err
    """
    if data is None:
        return False, (jsonify({"error": "Request body must be JSON"}), 400)
    for field in required:
        if not data.get(field):
            return False, (jsonify({"error": f"'{field}' is required"}), 400)
    return True, None
