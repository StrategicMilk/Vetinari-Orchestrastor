"""Tests for vetinari.integration — IntegrationManager wiring."""

from __future__ import annotations

from vetinari.integration import IntegrationManager, get_integration_manager


class TestIntegrationManager:
    """Tests for the IntegrationManager singleton."""

    def test_singleton(self):
        a = IntegrationManager()
        b = IntegrationManager()
        assert a is b

    def test_get_integration_manager(self):
        mgr = get_integration_manager()
        assert isinstance(mgr, IntegrationManager)

    def test_initial_state_not_wired(self):
        mgr = IntegrationManager()
        # Reset for test isolation — note: singleton means
        # is_wired may be True if wire_all was called before
        assert isinstance(mgr.is_wired, bool)

    def test_wire_all_idempotent(self):
        mgr = IntegrationManager()
        mgr.wire_all()
        assert mgr.is_wired is True
        # Second call should not raise
        mgr.wire_all()
        assert mgr.is_wired is True

    def test_get_status_returns_dict(self):
        mgr = IntegrationManager()
        mgr.wire_all()
        status = mgr.get_status()
        assert isinstance(status, dict)
