"""Tests for vetinari.assignment_pass — backward-compat shim."""

from __future__ import annotations

import vetinari.assignment_pass as shim_module


class TestAssignmentPassShim:
    """Tests for the backward-compatibility shim module."""

    def test_shim_imports_canonical_module(self):
        """The shim should re-export from vetinari.planning.assignment_pass."""
        import vetinari.planning.assignment_pass as canonical

        # The shim replaces itself with the canonical module in sys.modules
        assert shim_module is canonical
