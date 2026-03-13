"""Backward-compat shim. Canonical: vetinari.agents.agent_affinity.

All new code should import from vetinari.agents.agent_affinity directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.agents.agent_affinity")
_sys.modules[__name__] = _canonical
