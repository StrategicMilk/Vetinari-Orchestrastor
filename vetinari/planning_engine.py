"""Backward-compat shim. Canonical: vetinari.planning.planning_engine

All new code should import from vetinari.planning.planning_engine directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.planning_engine")
_sys.modules[__name__] = _canonical
