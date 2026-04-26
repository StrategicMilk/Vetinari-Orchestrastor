"""Shared fixtures for the Vetinari LLM evaluation test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_plan_response() -> dict:
    """Return a well-formed plan dict suitable for evaluation testing.

    Returns:
        A dictionary with ``tasks``, ``dependencies``, and ``goal`` keys
        representing a realistic planner agent output.
    """
    return {
        "goal": "Implement a REST API endpoint for user registration",
        "tasks": [
            {"id": "task-1", "name": "Define user schema"},
            {"id": "task-2", "name": "Create database migration"},
            {"id": "task-3", "name": "Implement request handler"},
            {"id": "task-4", "name": "Write unit tests"},
        ],
        "dependencies": {
            "task-2": ["task-1"],
            "task-3": ["task-2"],
            "task-4": ["task-3"],
        },
    }


@pytest.fixture
def mock_code_response() -> str:
    """Return a valid Python function string suitable for evaluation testing.

    Returns:
        A string containing a syntactically correct Python module with
        imports and a function implementation.
    """
    return """
import logging
from typing import Any

logger = logging.getLogger(__name__)


def process_user_data(user_id: str, data: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Process and validate user data.

    Args:
        user_id: The unique identifier of the user.
        data: Raw data dictionary to process.

    Returns:
        Processed and validated user data dictionary.
    \"\"\"
    if not user_id:
        raise ValueError("user_id must not be empty")
    result = {
        "id": user_id,
        "processed": True,
        **data,
    }
    logger.info("Processed data for user %s", user_id)
    return result
"""


@pytest.fixture
def eval_threshold() -> float:
    """Return the default minimum passing score for evaluation tests.

    Returns:
        Float threshold value; a result with score >= this value is passing.
    """
    return 0.5
