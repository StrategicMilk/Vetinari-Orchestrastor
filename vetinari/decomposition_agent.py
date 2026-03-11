"""Backward-compat shim. Canonical: vetinari.agents.decomposition_agent

All new code should import from vetinari.agents.decomposition_agent directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.agents.decomposition_agent")
_sys.modules[__name__] = _canonical
