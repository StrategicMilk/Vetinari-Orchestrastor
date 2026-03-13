"""Backward-compat shim. Canonical: vetinari.planning.subtask_tree.

All new code should import from vetinari.planning.subtask_tree directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.subtask_tree")
_sys.modules[__name__] = _canonical
