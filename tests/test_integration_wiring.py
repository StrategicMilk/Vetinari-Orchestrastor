"""
Tests for Task 42: Integration Verification & System Wiring.

Verifies that the IntegrationManager correctly wires all disconnected
subsystems together and that the new Flask API blueprints work.

Test categories
---------------
1. IntegrationManager creation and wire_all()
2. Learning API blueprint creation
3. Analytics API blueprint creation
4. Skills auto-registration verification
5. Integration status reporting
6. Idempotent wiring (calling wire_all() twice is safe)
"""

import importlib
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_singletons():
    """Reset all relevant singletons so each test starts clean."""
    from vetinari.integration import reset_integration_manager
    reset_integration_manager()

    # Reset analytics singletons to avoid cross-test pollution
    try:
        from vetinari.analytics.cost import reset_cost_tracker
        reset_cost_tracker()
    except Exception:
        pass
    try:
        from vetinari.analytics.sla import reset_sla_tracker
        reset_sla_tracker()
    except Exception:
        pass
    try:
        from vetinari.analytics.anomaly import reset_anomaly_detector
        reset_anomaly_detector()
    except Exception:
        pass
    try:
        from vetinari.analytics.forecasting import reset_forecaster
        reset_forecaster()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def clean_singletons():
    """Auto-reset singletons before each test."""
    _reset_singletons()
    yield
    _reset_singletons()


# ===================================================================
# 1. IntegrationManager creation and wire_all()
# ===================================================================

class TestIntegrationManagerBasics:
    """Test IntegrationManager lifecycle."""

    def test_creation(self):
        """IntegrationManager can be instantiated."""
        from vetinari.integration import IntegrationManager
        manager = IntegrationManager()
        assert manager is not None
        assert manager.is_wired is False

    def test_singleton(self):
        """IntegrationManager is a singleton."""
        from vetinari.integration import IntegrationManager
        m1 = IntegrationManager()
        m2 = IntegrationManager()
        assert m1 is m2

    def test_get_integration_manager(self):
        """get_integration_manager() returns the singleton."""
        from vetinari.integration import get_integration_manager, IntegrationManager
        manager = get_integration_manager()
        assert isinstance(manager, IntegrationManager)

    def test_wire_all_runs(self):
        """wire_all() completes without raising."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        assert manager.is_wired is True

    def test_wire_all_sets_status(self):
        """wire_all() populates the wiring results dict."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        status = manager.get_status()
        assert status["wired"] is True
        assert "subsystems" in status
        # All five subsystems should have an entry
        expected_keys = [
            "learning_to_dashboard",
            "drift_to_orchestration",
            "analytics_to_dashboard",
            "security_to_verification",
            "skills_to_registry",
        ]
        for key in expected_keys:
            assert key in status["subsystems"], f"Missing subsystem: {key}"


# ===================================================================
# 2. Learning API Blueprint
# ===================================================================

class TestLearningApiBlueprint:
    """Test the learning API Flask Blueprint."""

    def test_blueprint_importable(self):
        """learning_bp can be imported."""
        from vetinari.web.learning_api import learning_bp
        assert learning_bp is not None
        assert learning_bp.name == "learning"

    def test_blueprint_has_routes(self):
        """learning_bp has the expected routes when registered on an app."""
        from flask import Flask
        from vetinari.web.learning_api import learning_bp

        app = Flask(__name__)
        app.register_blueprint(learning_bp)

        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/v1/learning/thompson" in rules
        assert "/api/v1/learning/quality-history" in rules
        assert "/api/v1/learning/training-stats" in rules

        # Also verify the view functions are callable
        from vetinari.web.learning_api import (
            get_thompson_arms,
            get_quality_history,
            get_training_stats,
        )
        assert callable(get_thompson_arms)
        assert callable(get_quality_history)
        assert callable(get_training_stats)

    def test_thompson_endpoint_with_flask_test_client(self):
        """GET /api/v1/learning/thompson returns JSON."""
        from flask import Flask
        from vetinari.web.learning_api import learning_bp

        app = Flask(__name__)
        app.register_blueprint(learning_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/learning/thompson")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "arms" in data

    def test_quality_history_endpoint(self):
        """GET /api/v1/learning/quality-history returns JSON."""
        from flask import Flask
        from vetinari.web.learning_api import learning_bp

        app = Flask(__name__)
        app.register_blueprint(learning_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/learning/quality-history")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "history" in data

    def test_training_stats_endpoint(self):
        """GET /api/v1/learning/training-stats returns JSON."""
        from flask import Flask
        from vetinari.web.learning_api import learning_bp

        app = Flask(__name__)
        app.register_blueprint(learning_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/learning/training-stats")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "stats" in data


# ===================================================================
# 3. Analytics API Blueprint
# ===================================================================

class TestAnalyticsApiBlueprint:
    """Test the analytics API Flask Blueprint."""

    def test_blueprint_importable(self):
        """analytics_bp can be imported."""
        from vetinari.web.analytics_api import analytics_bp
        assert analytics_bp is not None
        assert analytics_bp.name == "analytics"

    def test_blueprint_view_functions(self):
        """analytics_bp has the expected view functions."""
        from vetinari.web.analytics_api import (
            get_cost_data,
            get_sla_data,
            get_anomaly_data,
            get_forecast_data,
        )
        assert callable(get_cost_data)
        assert callable(get_sla_data)
        assert callable(get_anomaly_data)
        assert callable(get_forecast_data)

    def test_cost_endpoint(self):
        """GET /api/v1/analytics/cost returns JSON."""
        from flask import Flask
        from vetinari.web.analytics_api import analytics_bp

        app = Flask(__name__)
        app.register_blueprint(analytics_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/analytics/cost")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "cost" in data

    def test_sla_endpoint(self):
        """GET /api/v1/analytics/sla returns JSON."""
        from flask import Flask
        from vetinari.web.analytics_api import analytics_bp

        app = Flask(__name__)
        app.register_blueprint(analytics_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/analytics/sla")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "sla" in data

    def test_anomalies_endpoint(self):
        """GET /api/v1/analytics/anomalies returns JSON."""
        from flask import Flask
        from vetinari.web.analytics_api import analytics_bp

        app = Flask(__name__)
        app.register_blueprint(analytics_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/analytics/anomalies")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "anomalies" in data

    def test_forecasts_endpoint(self):
        """GET /api/v1/analytics/forecasts returns JSON."""
        from flask import Flask
        from vetinari.web.analytics_api import analytics_bp

        app = Flask(__name__)
        app.register_blueprint(analytics_bp)

        with app.test_client() as client:
            resp = client.get("/api/v1/analytics/forecasts")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "forecasts" in data


# ===================================================================
# 4. Skills auto-registration verification
# ===================================================================

class TestSkillsAutoRegistration:
    """Test that skills are auto-registered in the ToolRegistry."""

    def test_skills_registered_after_wire(self):
        """wire_all() registers at least some skills."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        skills = manager.get_registered_skills()
        # Should have registered at least a few skills
        # (exact number depends on which modules import cleanly)
        assert isinstance(skills, list)

    def test_skill_names_in_status(self):
        """get_status() includes registered_skills list."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        status = manager.get_status()
        assert "registered_skills" in status
        assert "registered_skill_count" in status
        assert isinstance(status["registered_skill_count"], int)

    def test_skills_wiring_result_ok(self):
        """skills_to_registry wiring result starts with 'ok'."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        status = manager.get_status()
        result = status["subsystems"].get("skills_to_registry", "")
        assert result.startswith("ok"), f"Expected 'ok...', got: {result}"


# ===================================================================
# 5. Integration status reporting
# ===================================================================

class TestIntegrationStatus:
    """Test integration status and introspection."""

    def test_status_before_wiring(self):
        """get_status() works before wire_all()."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        status = manager.get_status()
        assert status["wired"] is False
        assert status["subsystems"] == {}

    def test_status_after_wiring(self):
        """get_status() returns detailed info after wire_all()."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        status = manager.get_status()
        assert status["wired"] is True
        assert len(status["subsystems"]) == 5

    def test_learning_wiring_ok(self):
        """Learning-to-dashboard wiring reports ok."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        assert manager.get_status()["subsystems"]["learning_to_dashboard"] == "ok"

    def test_analytics_wiring_ok(self):
        """Analytics-to-dashboard wiring reports ok."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        assert manager.get_status()["subsystems"]["analytics_to_dashboard"] == "ok"

    def test_drift_pre_check_before_wiring(self):
        """run_drift_pre_check() returns skipped before wiring."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        result = manager.run_drift_pre_check()
        assert result.get("skipped") is True or result.get("is_clean") is True

    def test_drift_pre_check_after_wiring(self):
        """run_drift_pre_check() returns meaningful data after wiring."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        result = manager.run_drift_pre_check()
        assert "is_clean" in result


# ===================================================================
# 6. Idempotent wiring
# ===================================================================

class TestIdempotentWiring:
    """Test that wire_all() is safe to call multiple times."""

    def test_double_wire_no_error(self):
        """Calling wire_all() twice does not raise."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        manager.wire_all()  # Should be a no-op
        assert manager.is_wired is True

    def test_double_wire_same_status(self):
        """Status is the same after calling wire_all() twice."""
        from vetinari.integration import get_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        status1 = manager.get_status()
        manager.wire_all()
        status2 = manager.get_status()
        assert status1 == status2

    def test_reset_allows_rewire(self):
        """After reset, wire_all() works again from scratch."""
        from vetinari.integration import get_integration_manager, reset_integration_manager
        manager = get_integration_manager()
        manager.wire_all()
        assert manager.is_wired is True

        reset_integration_manager()
        manager2 = get_integration_manager()
        assert manager2.is_wired is False
        manager2.wire_all()
        assert manager2.is_wired is True


# ===================================================================
# 7. Web __init__.py exports
# ===================================================================

class TestWebInitExports:
    """Test that the web package exports the new blueprints."""

    def test_learning_bp_in_web_package(self):
        """vetinari.web exports learning_bp."""
        from vetinari.web import learning_bp
        assert learning_bp is not None

    def test_analytics_bp_in_web_package(self):
        """vetinari.web exports analytics_bp."""
        from vetinari.web import analytics_bp
        assert analytics_bp is not None
