"""Backward-compat shim. Canonical: vetinari.models.vram_manager.

All new code should import from vetinari.models.vram_manager directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.models.vram_manager")
_sys.modules[__name__] = _canonical
