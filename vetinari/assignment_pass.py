"""Backward-compat shim. Canonical: vetinari.planning.assignment_pass

All new code should import from vetinari.planning.assignment_pass directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.planning.assignment_pass")
_sys.modules[__name__] = _canonical
