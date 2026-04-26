"""Module with print() in production code.

IMPORTANT — test runner note:
VET035 only fires when is_in_vetinari(filepath) is True. The test runner must
either pass a path under vetinari/ or monkeypatch is_in_vetinari to return True
for this fixture's path.
"""
import logging

logger = logging.getLogger(__name__)


def show_result(result: str) -> None:
    """Display a result.

    Args:
        result: The result to display.
    """
    print(result)
