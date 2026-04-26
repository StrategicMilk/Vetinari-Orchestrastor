"""Module with an undeclared third-party import."""
from __future__ import annotations

import nonexistent_package_xyz


def process() -> None:
    """Process something."""
    nonexistent_package_xyz.run()
