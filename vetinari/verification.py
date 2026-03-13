"""Backward-compat shim. Canonical: vetinari.validation.verification.

All new code should import from vetinari.validation.verification directly.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_canonical = _importlib.import_module("vetinari.validation.verification")
_sys.modules[__name__] = _canonical
