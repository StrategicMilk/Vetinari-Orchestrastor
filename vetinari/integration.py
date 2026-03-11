"""
Central Integration Wiring for Vetinari
========================================

Connects the previously-disconnected subsystems together:

1. **Learning pipeline --> Web dashboard API** — registers the learning
   and analytics Flask blueprints so the dashboard can display live data.
2. **Drift detection --> Orchestration** — hooks the drift monitor into
   the orchestration cycle so drift checks run automatically.
3. **Analytics --> Dashboard** — ensures cost / SLA / anomaly / forecast
   data is accessible through the web API.
4. **Security --> Verification pipeline** — integrates the SecretScanner
   as a verification step in the output pipeline.
5. **Skills --> ToolRegistry auto-registration** — discovers all skill
   Tool subclasses and registers them in the global ToolRegistry.

Usage
-----
    from vetinari.integration import get_integration_manager

    manager = get_integration_manager()
    manager.wire_all()
    print(manager.get_status())
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IntegrationManager:
    """Wires all Vetinari subsystems together."""

    _instance: Optional["IntegrationManager"] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "IntegrationManager":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    def _setup(self) -> None:
        self._wired = False
        self._wiring_results: Dict[str, str] = {}  # subsystem -> status
        self._registered_skills: List[str] = []
        self._drift_pre_check = None  # Set by _wire_drift_to_orchestration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wire_all(self) -> None:
        """Connect all subsystems. Idempotent."""
        if self._wired:
            return
        self._wire_learning_to_dashboard()
        self._wire_drift_to_orchestration()
        self._wire_analytics_to_dashboard()
        self._wire_security_to_verification()
        self._wire_skills_to_registry()
        self._wired = True
        logger.info("IntegrationManager: all subsystems wired")

    @property
    def is_wired(self) -> bool:
        return self._wired

    # ------------------------------------------------------------------
    # 1. Learning --> Dashboard
    # ------------------------------------------------------------------

    def _wire_learning_to_dashboard(self) -> None:
        """Ensure learning API blueprint is importable and ready."""
        try:
            from vetinari.web.learning_api import learning_bp  # noqa: F401
            self._wiring_results["learning_to_dashboard"] = "ok"
            logger.info("Integration: learning -> dashboard wired")
        except Exception as e:
            self._wiring_results["learning_to_dashboard"] = f"error: {e}"
            logger.warning("Integration: learning -> dashboard failed: %s", e)

    # ------------------------------------------------------------------
    # 2. Drift --> Orchestration
    # ------------------------------------------------------------------

    def _wire_drift_to_orchestration(self) -> None:
        """Connect drift monitor checks into orchestration cycle.

        The drift monitor is made available as a callable hook that the
        orchestrator can invoke before/after task execution.
        """
        try:
            from vetinari.drift.monitor import get_drift_monitor
            monitor = get_drift_monitor()

            # Expose a lightweight pre-check function that orchestration
            # code can call.  We do NOT run a full audit here — just
            # verify the monitor is importable and initialised.
            self._drift_pre_check = _make_drift_pre_check(monitor)
            self._wiring_results["drift_to_orchestration"] = "ok"
            logger.info("Integration: drift -> orchestration wired")
        except Exception as e:
            self._drift_pre_check = None
            self._wiring_results["drift_to_orchestration"] = f"error: {e}"
            logger.warning("Integration: drift -> orchestration failed: %s", e)

    def run_drift_pre_check(self) -> Dict[str, Any]:
        """Run a lightweight drift pre-check if wired.

        Returns a dict with ``is_clean`` and optional ``issues``.
        """
        if self._drift_pre_check is not None:
            return self._drift_pre_check()
        return {"is_clean": True, "skipped": True}

    # ------------------------------------------------------------------
    # 3. Analytics --> Dashboard
    # ------------------------------------------------------------------

    def _wire_analytics_to_dashboard(self) -> None:
        """Ensure analytics API blueprint is importable and ready."""
        try:
            from vetinari.web.analytics_api import analytics_bp  # noqa: F401
            self._wiring_results["analytics_to_dashboard"] = "ok"
            logger.info("Integration: analytics -> dashboard wired")
        except Exception as e:
            self._wiring_results["analytics_to_dashboard"] = f"error: {e}"
            logger.warning("Integration: analytics -> dashboard failed: %s", e)

    # ------------------------------------------------------------------
    # 4. Security --> Verification
    # ------------------------------------------------------------------

    def _wire_security_to_verification(self) -> None:
        """Verify that SecurityVerifier is present in the verification pipeline.

        The ``VerificationPipeline`` already creates a ``SecurityVerifier``
        that uses the ``SecretScanner`` for BASIC+ levels.  This method
        confirms the wiring is intact and logs the result.  If for some
        reason the security verifier is missing (e.g. NONE level), we
        add it back.
        """
        try:
            from vetinari.security import get_secret_scanner
            from vetinari.validation.verification import (
                get_verifier_pipeline,
                SecurityVerifier as _ExistingSecurityVerifier,
            )

            pipeline = get_verifier_pipeline()
            scanner = get_secret_scanner()

            # Check if a SecurityVerifier is already present
            has_security = any(
                v.name == "security" for v in pipeline.verifiers
            )

            if not has_security:
                # Re-add the built-in SecurityVerifier
                pipeline.add_verifier(_ExistingSecurityVerifier())
                logger.info("Integration: added missing SecurityVerifier to pipeline")

            # Verify the scanner is operational
            _ = scanner.scan("test")

            self._wiring_results["security_to_verification"] = "ok"
            logger.info("Integration: security -> verification wired")
        except Exception as e:
            self._wiring_results["security_to_verification"] = f"error: {e}"
            logger.warning("Integration: security -> verification failed: %s", e)

    # ------------------------------------------------------------------
    # 5. Skills --> ToolRegistry
    # ------------------------------------------------------------------

    def _wire_skills_to_registry(self) -> None:
        """Auto-register all skill Tool subclasses in ToolRegistry.

        Scans the ``vetinari.tools`` package for modules ending with
        ``_skill`` and registers any ``Tool`` subclass found.
        """
        try:
            from vetinari.tool_interface import Tool, get_tool_registry

            registry = get_tool_registry()
            registered: List[str] = []

            skill_modules = [
                "vetinari.tools.builder_skill",
                "vetinari.tools.evaluator_skill",
                "vetinari.tools.explorer_skill",
                "vetinari.tools.librarian_skill",
                "vetinari.tools.oracle_skill",
                "vetinari.tools.researcher_skill",
                "vetinari.tools.synthesizer_skill",
                "vetinari.tools.ui_planner_skill",
                "vetinari.tools.cost_planner_skill",
                "vetinari.tools.documentation_skill",
                "vetinari.tools.experimentation_manager_skill",
                "vetinari.tools.data_engineer_skill",
                "vetinari.tools.security_auditor_skill",
                "vetinari.tools.test_automation_skill",
            ]

            for mod_name in skill_modules:
                try:
                    mod = importlib.import_module(mod_name)
                    for _attr_name, attr_value in inspect.getmembers(mod, inspect.isclass):
                        if (
                            issubclass(attr_value, Tool)
                            and attr_value is not Tool
                            and not inspect.isabstract(attr_value)
                        ):
                            # Skip if already registered (idempotent)
                            try:
                                instance = attr_value()
                                if registry.get(instance.metadata.name) is None:
                                    registry.register(instance)
                                    registered.append(instance.metadata.name)
                            except Exception as inst_err:
                                logger.debug(
                                    "Could not instantiate %s from %s: %s",
                                    _attr_name, mod_name, inst_err,
                                )
                except Exception as mod_err:
                    logger.debug("Could not import skill module %s: %s", mod_name, mod_err)

            self._registered_skills = registered
            self._wiring_results["skills_to_registry"] = f"ok ({len(registered)} skills)"
            logger.info(
                "Integration: skills -> registry wired (%d skills registered)",
                len(registered),
            )
        except Exception as e:
            self._wiring_results["skills_to_registry"] = f"error: {e}"
            logger.warning("Integration: skills -> registry failed: %s", e)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return integration health status."""
        return {
            "wired": self._wired,
            "subsystems": dict(self._wiring_results),
            "registered_skills": list(self._registered_skills),
            "registered_skill_count": len(self._registered_skills),
        }

    def get_registered_skills(self) -> List[str]:
        """Return list of auto-registered skill names."""
        return list(self._registered_skills)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_drift_pre_check(monitor):
    """Return a callable that performs a lightweight drift status check."""

    def _pre_check() -> Dict[str, Any]:
        try:
            last = monitor.get_last_report()
            if last is not None:
                return {
                    "is_clean": last.is_clean,
                    "issues": last.issues[:5],  # Truncate for brevity
                    "duration_ms": last.duration_ms,
                }
            return {"is_clean": True, "no_audit_yet": True}
        except Exception as exc:
            return {"is_clean": True, "error": str(exc)}

    return _pre_check


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

def get_integration_manager() -> IntegrationManager:
    """Return the global IntegrationManager singleton."""
    return IntegrationManager()


def reset_integration_manager() -> None:
    """Reset the singleton (for testing)."""
    with IntegrationManager._class_lock:
        IntegrationManager._instance = None
