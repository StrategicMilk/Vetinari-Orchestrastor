"""Backward-compat shim. Canonical: vetinari.planning.plan_mode

All new code should import from vetinari.planning.plan_mode directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.plan_mode")
_sys.modules[__name__] = _canonical
