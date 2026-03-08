"""
Vetinari Web Blueprints — modular route groups extracted from web_ui.py.

Each blueprint is a self-contained route group with lazy imports to
avoid circular dependencies.
"""

import os
from flask import request


def is_admin_user() -> bool:
    """
    Check if the current request comes from an admin user.
    For local deployments, localhost requests are always admin.
    For network deployments, check VETINARI_ADMIN_TOKEN env var.
    """
    admin_token = os.environ.get("VETINARI_ADMIN_TOKEN", "")
    if admin_token:
        auth_header = request.headers.get("Authorization", "")
        req_token = request.headers.get("X-Admin-Token", "")
        provided = req_token or auth_header.replace("Bearer ", "")
        return provided == admin_token
    remote = request.remote_addr or ""
    return remote in ("127.0.0.1", "::1", "localhost")
