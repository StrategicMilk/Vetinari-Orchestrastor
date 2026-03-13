"""Backward-compat shim. Canonical: vetinari.models.dynamic_model_router.

All new code should import from vetinari.models.dynamic_model_router directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.models.dynamic_model_router")
_sys.modules[__name__] = _canonical
