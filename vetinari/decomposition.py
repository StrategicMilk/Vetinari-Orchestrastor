"""Backward-compat shim. Canonical: vetinari.planning.decomposition

All new code should import from vetinari.planning.decomposition directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.decomposition")
_sys.modules[__name__] = _canonical
