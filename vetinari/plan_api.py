"""Backward-compat shim. Canonical: vetinari.planning.plan_api.

All new code should import from vetinari.planning.plan_api directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.plan_api")
_sys.modules[__name__] = _canonical
