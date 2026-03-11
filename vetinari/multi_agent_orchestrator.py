"""Backward-compat shim. Canonical: vetinari.agents.multi_agent_orchestrator

All new code should import from vetinari.agents.multi_agent_orchestrator directly.
"""
import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.agents.multi_agent_orchestrator")
_sys.modules[__name__] = _canonical
