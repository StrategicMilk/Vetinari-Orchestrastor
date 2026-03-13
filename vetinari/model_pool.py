"""Backward-compat shim. Canonical: vetinari.models.model_pool.

All new code should import from vetinari.models.model_pool directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.models.model_pool")
_sys.modules[__name__] = _canonical
