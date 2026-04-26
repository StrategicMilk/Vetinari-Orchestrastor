"""Module that reads config inside a route handler — triggers VET115."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class DataApi:
    """Data access endpoints."""

    @app.route("/api/config")
    def get_config(self) -> dict:
        """Return the current configuration.

        Reads the config file on every request by passing the filename directly
        to json.load — should be cached at module level to avoid per-request
        disk I/O. VET115 fires because the path is a hardcoded string constant.

        Returns:
            Configuration dictionary.
        """
        return json.load("config.json")
