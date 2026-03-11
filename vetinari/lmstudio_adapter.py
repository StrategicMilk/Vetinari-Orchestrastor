"""Backward-compat shim. Canonical: vetinari.adapters.lmstudio_adapter

All new code should import from vetinari.adapters.lmstudio_adapter directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.adapters.lmstudio_adapter")
_sys.modules[__name__] = _canonical
