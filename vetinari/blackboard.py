"""Backward-compat shim. Canonical: vetinari.memory.blackboard.

All new code should import from vetinari.memory.blackboard directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.memory.blackboard")
_sys.modules[__name__] = _canonical
