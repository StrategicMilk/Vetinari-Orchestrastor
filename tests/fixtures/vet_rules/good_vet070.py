"""Module with only stdlib imports."""
from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)


def process() -> None:
    """Process something with standard library only."""
    path = pathlib.Path("output.txt")
    logger.info("Processing path %s", path)
