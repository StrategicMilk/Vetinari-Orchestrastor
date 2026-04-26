"""Tests for the Litestar chat API handler surface."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def _handler_by_name(name: str) -> Any:
    from vetinari.web.litestar_chat_api import create_chat_api_handlers

    for handler in create_chat_api_handlers():
        if handler.handler_name == name:
            return handler
    raise AssertionError(f"Handler {name!r} was not registered")


def test_chat_api_registers_current_litestar_routes() -> None:
    """Handler factory exposes the current chat API route set."""
    from vetinari.web.litestar_chat_api import create_chat_api_handlers

    routes = {
        next(iter(handler.paths))
        for handler in create_chat_api_handlers()
    }

    assert routes == {
        "/api/v1/chat/attachments",
        "/api/v1/chat/attachments/{attachment_id:str}",
        "/api/v1/chat/export/{project_id:str}",
        "/api/v1/chat/feedback",
        "/api/v1/chat/retry/{project_id:str}/{task_id:str}",
    }


def test_chat_feedback_rejects_invalid_rating() -> None:
    """Feedback handler rejects ratings outside the up/down contract."""
    handler = _handler_by_name("submit_feedback")

    response = asyncio.run(
        handler.fn(
            {
                "project_id": "project-1",
                "task_id": "task-1",
                "rating": "maybe",
            }
        )
    )

    assert response.status_code == 400
