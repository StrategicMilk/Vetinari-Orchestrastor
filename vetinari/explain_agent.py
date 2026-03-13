"""Backward-compat shim. Canonical: vetinari.agents.explain_agent.

All new code should import from vetinari.agents.explain_agent directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.agents.explain_agent")
_sys.modules[__name__] = _canonical
