"""Backward-compat shim. Canonical: vetinari.memory.shared_memory.

All new code should import from vetinari.memory.shared_memory directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.memory.shared_memory")
_sys.modules[__name__] = _canonical
