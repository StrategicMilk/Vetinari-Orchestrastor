"""Backward-compat shim. Canonical: vetinari.validation.goal_verifier.

All new code should import from vetinari.validation.goal_verifier directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.validation.goal_verifier")
_sys.modules[__name__] = _canonical
