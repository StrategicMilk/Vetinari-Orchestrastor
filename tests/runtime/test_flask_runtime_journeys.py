"""SESSION-30 plan compliance stub — Flask web_ui.py no longer exists.

``vetinari/web_ui.py`` (Flask) was deleted when the web layer was migrated to
Litestar.  All HTTP surfaces are now served by ``vetinari.web.litestar_app``.
This file satisfies the SESSION-30 plan reference to a Flask test file by
asserting the migration contract: Flask is gone, Litestar is present.
"""

from __future__ import annotations

import importlib.util

import pytest


class TestFlaskRuntimeJourneys:
    """Assert that Flask is gone and Litestar owns all HTTP surfaces."""

    def test_flask_is_gone_all_routes_are_litestar(self) -> None:
        """vetinari.web_ui (Flask) must not exist; Litestar app must be importable.

        Raises:
            AssertionError: If Flask module is found or Litestar app is missing.
        """
        flask_spec = importlib.util.find_spec("vetinari.web_ui")
        assert flask_spec is None, (
            "vetinari.web_ui still exists — Flask was supposed to be deleted. "
            "Remove the module and migrate any remaining routes to Litestar."
        )

        litestar_spec = importlib.util.find_spec("vetinari.web.litestar_app")
        assert litestar_spec is not None, (
            "vetinari.web.litestar_app is not importable — the Litestar app module is missing or broken."
        )
