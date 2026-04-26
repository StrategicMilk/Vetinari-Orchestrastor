"""Module with module-level imports only — clean for VET130."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def infer(prompt: str) -> str:
    """Run inference on a prompt.

    All imports are at module level so repeated calls incur no sys.modules
    lookup overhead.

    Args:
        prompt: Input prompt string.

    Returns:
        Serialized output string.
    """
    return json.dumps({"prompt": prompt})
