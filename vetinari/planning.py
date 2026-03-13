"""Backward-compat shim. Canonical: vetinari.planning.planning.

All new code should import from vetinari.planning.planning directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.planning")
_sys.modules[__name__] = _canonical
