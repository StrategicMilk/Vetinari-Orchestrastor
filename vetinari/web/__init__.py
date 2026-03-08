"""Web UI blueprints and shared state for Vetinari.

Blueprints
----------
    projects_bp   -- project CRUD and execution routes
    plans_bp      -- plan, subtask, decomposition routes
    admin_bp      -- admin, agent, memory, sandbox routes
    preferences_bp -- user preferences routes
    learning_bp   -- learning pipeline API (Thompson, quality, training)
    analytics_bp  -- analytics API (cost, SLA, anomaly, forecast)
"""

from vetinari.web.preferences import preferences_bp  # noqa: F401
from vetinari.web.variant_system import VariantLevel, VariantManager  # noqa: F401
from vetinari.web.learning_api import learning_bp  # noqa: F401
from vetinari.web.analytics_api import analytics_bp  # noqa: F401
from vetinari.web.log_stream import log_stream_bp  # noqa: F401
