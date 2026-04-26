"""Module with commented-out code block.

IMPORTANT — test runner note:
VET036 skips files under tests/. This fixture must be copied to a tmp_path
outside tests/ before calling check_file() on it. The test handles this copy.

The three consecutive comment lines below form valid Python (ast.parse succeeds
on "x = 1\ny = 2\nz = x + y"), triggering VET036.
"""
import logging

logger = logging.getLogger(__name__)


def process() -> None:
    """Process data."""
    logger.info("Processing")
# x = 1
# y = 2
# z = x + y
