"""Flask Blueprint for admin routes.

Backward-compatibility shim: routes are now split into two modules:
  - config_routes.py      -- credentials, rules, model-relay, ponder
  - system_mgmt_routes.py -- agents, memory, decisions, sandbox, search, ADR,
                             image generation, training

``admin_bp`` is kept as an alias for ``config_bp`` so existing callers that do
``from vetinari.web.admin_bp import admin_bp`` continue to work.
``system_mgmt_bp`` must be registered separately.
"""

from vetinari.web.config_routes import config_bp as admin_bp  # noqa: F401
from vetinari.web.config_routes import config_bp  # noqa: F401
from vetinari.web.system_mgmt_routes import system_mgmt_bp  # noqa: F401

__all__ = ["admin_bp", "config_bp", "system_mgmt_bp"]
