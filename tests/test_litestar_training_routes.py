"""Behavioral tests for the training-routes handlers returned by create_training_routes_handlers().

Tests call the underlying handler coroutines directly via ``h.fn()`` so that
the behaviour of ``api_image_status`` and ``api_sd_status`` is exercised
without requiring a full HTTP stack or a live Litestar app.  External
dependencies (DiffusionEngine) are patched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_handlers() -> dict[str, object]:
    """Build a name -> handler coroutine-fn dict from create_training_routes_handlers.

    Returns:
        Mapping of handler_name to the underlying async function (``h.fn``).
    """
    from vetinari.web.litestar_training_routes import create_training_routes_handlers

    return {h.handler_name: h.fn for h in create_training_routes_handlers()}


def _run(coro):
    """Run a coroutine synchronously and return its result.

    Args:
        coro: An awaitable coroutine.

    Returns:
        The value returned by the coroutine.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# api_image_status tests
# ---------------------------------------------------------------------------


class TestApiImageStatus:
    """Behavioral tests for the api_image_status route handler."""

    def test_api_image_status_no_models(self):
        """status == 'unavailable' and 'detail' key present when no models found."""
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.discover_models.return_value = []

        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            return_value=mock_engine,
        ):
            result = _run(handlers["api_image_status"]())

        assert result["status"] == "unavailable"
        assert "detail" in result

    def test_api_image_status_unavailable_libs(self):
        """status == 'unavailable' and 'error' key present when DiffusionEngine raises ImportError."""
        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            side_effect=ImportError("torch not installed"),
        ):
            result = _run(handlers["api_image_status"]())

        assert result["status"] == "unavailable"
        assert "error" in result

    def test_api_image_status_available(self):
        """status == 'available' and model_count == 1 when one model is discovered."""
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.discover_models.return_value = [
            {"id": "sdxl", "format": "hf", "path": "/models/sdxl"}
        ]

        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            return_value=mock_engine,
        ):
            result = _run(handlers["api_image_status"]())

        assert result["status"] == "available"
        assert result["model_count"] == 1


# ---------------------------------------------------------------------------
# api_sd_status tests
# ---------------------------------------------------------------------------


class TestApiSdStatus:
    """Behavioral tests for the api_sd_status route handler."""

    def test_api_sd_status_connected(self):
        """status == 'connected' when libs are available and models exist."""
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.has_models.return_value = True

        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            return_value=mock_engine,
        ):
            result = _run(handlers["api_sd_status"]())

        assert result["status"] == "connected"

    def test_api_sd_status_disconnected_no_models(self):
        """status == 'disconnected' when libs are available but no models are found."""
        mock_engine = MagicMock()
        mock_engine.is_available.return_value = True
        mock_engine.has_models.return_value = False

        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            return_value=mock_engine,
        ):
            result = _run(handlers["api_sd_status"]())

        assert result["status"] == "disconnected"

    def test_api_sd_status_unavailable(self):
        """status == 'unavailable' when DiffusionEngine raises ImportError."""
        handlers = _get_handlers()
        with patch(
            "vetinari.image.diffusion_engine.DiffusionEngine",
            side_effect=ImportError("diffusers not installed"),
        ):
            result = _run(handlers["api_sd_status"]())

        assert result["status"] == "unavailable"
