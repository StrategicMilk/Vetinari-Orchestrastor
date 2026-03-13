"""Backward-compat shim. Canonical: vetinari.models.ponder.

All new code should import from vetinari.models.ponder directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.models.ponder")
_sys.modules[__name__] = _canonical
