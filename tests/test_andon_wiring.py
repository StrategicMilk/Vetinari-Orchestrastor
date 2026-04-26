"""Tests for AndonSystem wiring into TwoLayerOrchestrator."""

from __future__ import annotations

import pytest

from vetinari.orchestration.two_layer import TwoLayerOrchestrator, init_two_layer_orchestrator
from vetinari.workflow import AndonSystem, get_andon_system, reset_andon_system


class TestAndonWiring:
    """Tests for AndonSystem integration with TwoLayerOrchestrator."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Reset the global AndonSystem singleton before each test.

        TwoLayerOrchestrator now uses the singleton via get_andon_system(),
        so tests must start with a fresh instance to avoid state leakage.
        """
        reset_andon_system()
        yield
        reset_andon_system()

    def test_orchestrator_has_andon(self) -> None:
        """TwoLayerOrchestrator must expose a _andon attribute of type AndonSystem."""
        orch = TwoLayerOrchestrator()
        assert hasattr(orch, "_andon")
        assert isinstance(orch._andon, AndonSystem)

    def test_andon_critical_halts_pipeline(self) -> None:
        """Raising a critical Andon signal must cause is_paused() to return True."""
        orch = TwoLayerOrchestrator()
        assert not orch.is_paused()
        orch._andon.raise_signal(source="test", severity="critical", message="critical failure")
        assert orch.is_paused()

    def test_andon_warning_no_halt(self) -> None:
        """Raising a warning Andon signal must NOT cause is_paused() to return True.

        A warning logs but does not pause — only the local _paused flag and the
        AndonSystem's own paused state (set only on critical/emergency) determine
        the result of is_paused().
        """
        orch = TwoLayerOrchestrator()
        orch._andon.raise_signal(source="test", severity="warning", message="degraded quality")
        # Warning sets neither orch._paused nor AndonSystem._paused
        assert not orch._paused
        assert not orch._andon.is_paused()
        assert not orch.is_paused()

    def test_resume_after_halt(self) -> None:
        """Pipeline must become un-paused after resume() is called following a halt."""
        orch = TwoLayerOrchestrator()
        orch._andon.raise_signal(source="test", severity="critical", message="halt me")
        assert orch.is_paused()

        result = orch.resume()

        assert result is True
        assert not orch.is_paused()

    def test_resume_when_not_paused(self) -> None:
        """resume() must return False when the pipeline is not paused."""
        orch = TwoLayerOrchestrator()
        result = orch.resume()
        assert result is False

    def test_andon_property(self) -> None:
        """The andon property must return the same AndonSystem instance."""
        orch = TwoLayerOrchestrator()
        assert orch.andon is orch._andon
        assert isinstance(orch.andon, AndonSystem)

    def test_callback_registered(self) -> None:
        """__init__ must register _on_andon_trigger as a callback on the AndonSystem."""
        orch = TwoLayerOrchestrator()
        # AndonSystem._callbacks is a list of registered callables
        assert any(cb == orch._on_andon_trigger for cb in orch._andon._callbacks)

    def test_reinit_does_not_duplicate_callbacks(self) -> None:
        """init_two_layer_orchestrator must deregister the old orchestrator's
        Andon callback before creating a new one — reinit cycles must not
        accumulate callbacks on the shared AndonSystem.
        """
        reset_andon_system()
        orch1 = init_two_layer_orchestrator()
        # After first init: exactly one _on_andon_trigger registered on the shared AndonSystem
        andon = get_andon_system()
        initial = sum(1 for cb in andon._callbacks if getattr(cb, "__name__", "") == "_on_andon_trigger")
        assert initial == 1
        # After second init: still exactly one (not 2)
        orch2 = init_two_layer_orchestrator()
        assert orch2 is not orch1
        after_reinit = sum(
            1 for cb in get_andon_system()._callbacks if getattr(cb, "__name__", "") == "_on_andon_trigger"
        )
        assert after_reinit == 1, f"Andon callbacks accumulated across reinit — expected 1, got {after_reinit}"

    def test_deregister_callback_removes_only_once(self) -> None:
        """AndonSystem.deregister_callback returns True the first time and
        False on subsequent calls — does not raise if not registered.
        """
        andon = AndonSystem()

        def cb(sig: object) -> None:
            pass  # noqa: VET031 - empty body is intentional test double behavior

        andon.register_callback(cb)
        assert andon.deregister_callback(cb) is True
        assert andon.deregister_callback(cb) is False

    def test_shutdown_deregisters_callback(self) -> None:
        """TwoLayerOrchestrator.shutdown removes the orchestrator's Andon callback."""
        reset_andon_system()
        orch = TwoLayerOrchestrator()
        assert any(cb == orch._on_andon_trigger for cb in orch._andon._callbacks)
        orch.shutdown()
        assert not any(cb == orch._on_andon_trigger for cb in orch._andon._callbacks)
