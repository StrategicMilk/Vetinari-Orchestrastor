"""Module with import inside a hot-path function body — triggers VET130."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def infer(prompt: str) -> str:
    """Run inference on a prompt.

    Has a local import inside the function body — repeated sys.modules lookups
    per call. The import should be at module level.

    Args:
        prompt: Input prompt string.

    Returns:
        Serialized output string.
    """
    import json

    return json.dumps({"prompt": prompt})
