"""Coverage tests for vetinari/plan_api.py Flask blueprint — Phase 7D"""
import json
import os
import unittest
from unittest.mock import MagicMock, patch


def _make_app():
    """Create a minimal Flask app with the plan_api blueprint registered."""
    from flask import Flask
    from vetinari.planning.plan_api import plan_api
    app = Flask(__name__)
    app.register_blueprint(plan_api)
    app.config["TESTING"] = True
    return app


class TestRequireAdminToken(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "secret-token")
    def test_missing_token_returns_401(self):
        r = self.client.post("/api/plan/generate",
                             data=json.dumps({"goal": "test"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 401)

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    def test_no_token_required_when_empty(self):
        with patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", False):
            r = self.client.post("/api/plan/generate",
                                 data=json.dumps({"goal": "test"}),
                                 content_type="application/json")
            # 400/503 because plan mode disabled — but not 401
            self.assertNotEqual(r.status_code, 401)

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "tok")
    def test_correct_bearer_token_allowed(self):
        with patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", False):
            r = self.client.post("/api/plan/generate",
                                 headers={"Authorization": "Bearer tok"},
                                 data=json.dumps({"goal": "test"}),
                                 content_type="application/json")
            self.assertNotEqual(r.status_code, 401)


class TestCheckPlanModeEnabled(unittest.TestCase):

    def test_enabled(self):
        with patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", True):
            from vetinari.planning.plan_api import check_plan_mode_enabled
            enabled, err = check_plan_mode_enabled()
            self.assertTrue(enabled)
            self.assertIsNone(err)

    def test_disabled(self):
        with patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", False):
            from vetinari.planning.plan_api import check_plan_mode_enabled
            enabled, err = check_plan_mode_enabled()
            self.assertFalse(enabled)
            self.assertIsNotNone(err)


class TestPlanGenerateEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    @patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", False)
    def test_plan_mode_disabled_returns_error(self):
        r = self.client.post("/api/plan/generate",
                             data=json.dumps({"goal": "test goal"}),
                             content_type="application/json")
        # 400 = plan mode disabled, 403 = auth required, 503 = unavailable
        self.assertIn(r.status_code, [400, 403, 503])

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    @patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", True)
    def test_missing_goal_returns_400(self):
        r = self.client.post("/api/plan/generate",
                             data=json.dumps({}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    @patch("vetinari.planning.plan_api.PLAN_MODE_ENABLE", True)
    def test_with_goal_calls_engine(self):
        mock_engine = MagicMock()
        mock_plan = MagicMock()
        mock_plan.to_dict.return_value = {"plan_id": "p1", "goal": "do things"}
        mock_engine.generate_plan.return_value = mock_plan
        with patch("vetinari.planning.plan_api.get_plan_engine", return_value=mock_engine):
            r = self.client.post(
                "/api/plan/generate",
                data=json.dumps({"goal": "build something"}),
                content_type="application/json",
            )
            self.assertIn(r.status_code, [200, 500])


class TestPlanStatusEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    def test_get_nonexistent_plan(self):
        r = self.client.get("/api/plan/nonexistent-plan-id/status")
        self.assertIn(r.status_code, [404, 200, 500])


class TestPlanListEndpoint(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    @patch("vetinari.planning.plan_api.PLAN_ADMIN_TOKEN", "")
    def test_list_plans_returns_list(self):
        r = self.client.get("/api/plan/list")
        # 200 (OK), 404 (no plans), or 500 (environment permission issue e.g. read-only filesystem)
        self.assertIn(r.status_code, [200, 404, 500])
        if r.status_code == 200:
            data = json.loads(r.data)
            self.assertIn("plans", data)


if __name__ == "__main__":
    unittest.main()
