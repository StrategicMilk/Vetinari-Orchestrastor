"""Module with TODO without issue reference.

IMPORTANT — test runner note:
VET030 skips files under tests/. This fixture must be copied to a tmp_path
outside tests/ before calling check_file() on it. The test handles this copy.
"""
import logging

logger = logging.getLogger(__name__)


# TODO: implement better error handling
def process() -> None:
    """Process something."""
    logger.info("Processing")
