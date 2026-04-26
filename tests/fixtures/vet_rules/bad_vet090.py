"""Module with public function missing docstring."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def public_function(value: str) -> str:
    # No docstring here — only a comment
    result = value.upper()
    logger.info("Processed %s", value)
    return result
