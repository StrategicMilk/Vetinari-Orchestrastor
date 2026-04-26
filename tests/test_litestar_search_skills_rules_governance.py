"""Mounted request-level governance tests for search, skills, and rules routes.

Proves that:
- Exception bubbling is bounded to 500 error responses, not raw 500 frames.
- Absolute file_path values are rejected with 400 from the symbols endpoint.
- Absolute paths are stripped from the skills catalog response.
- validate_output rejects empty bodies (missing ``output`` key) with 400.
- check_trust_elevation returns 404 for missing skills, not 200 with inline error.
- POST /api/v1/rules/global rejects empty body with 400.
- model_id in rules endpoints never carries a leading slash.

All requests go through the full Litestar HTTP stack via TestClient.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# CSRF header used for non-GET requests to satisfy middleware
# ---------------------------------------------------------------------------

_CSRF = {"X-Requested-With": "XMLHttpRequest"}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


_ADMIN_TOKEN = "test-governance-admin-token"


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed and admin token set.

    Scoped to module so the Litestar app is built once; each test gets its
    own TestClient so connection state does not leak between tests.

    Returns:
        A Litestar application instance.
    """
    import os

    os.environ["VETINARI_ADMIN_TOKEN"] = _ADMIN_TOKEN
    try:
        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            from vetinari.web.litestar_app import create_app

            return create_app(debug=True)
    finally:
        os.environ.pop("VETINARI_ADMIN_TOKEN", None)


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def admin_client(app):
    """TestClient with admin token header for admin-guarded routes.

    Sets VETINARI_ADMIN_TOKEN env var and includes the token in headers so
    the admin_guard passes in Litestar's threaded execution context.

    Yields:
        A live TestClient with admin access granted.
    """
    import os

    from litestar.testing import TestClient

    os.environ["VETINARI_ADMIN_TOKEN"] = _ADMIN_TOKEN
    with TestClient(app) as tc:
        tc.headers["X-Admin-Token"] = _ADMIN_TOKEN
        yield tc
    os.environ.pop("VETINARI_ADMIN_TOKEN", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_500_error(response: object) -> None:
    """Assert the response is a 500 with status='error' envelope.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 500, f"Expected 500, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


def _assert_400_error(response: object) -> None:
    """Assert the response is a 400.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 400, f"Expected 400, got {response.status_code}. Body: {response.text[:400]}"


def _assert_404_error(response: object) -> None:
    """Assert the response is a 404.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 404, f"Expected 404, got {response.status_code}. Body: {response.text[:400]}"


# ===========================================================================
# Search routes  -  litestar_search_api.py
# ===========================================================================


class TestCodeSearchExceptionBounding:
    """GET /api/code-search  -  adapter failure must be bounded to 500."""

    def test_adapter_raise_returns_500(self, client: object) -> None:
        """When get_adapter raises, return 500 not a raw exception frame."""
        mock_registry = MagicMock()
        mock_registry.get_adapter.side_effect = RuntimeError("index corrupted")
        with patch("vetinari.code_search.code_search_registry", mock_registry):
            response = client.get("/api/code-search?q=hello")
        _assert_500_error(response)

    def test_search_raise_returns_500(self, client: object) -> None:
        """When adapter.search raises, return 500 not a raw exception frame."""
        mock_adapter = MagicMock()
        mock_adapter.search.side_effect = RuntimeError("backend gone")
        mock_registry = MagicMock()
        mock_registry.get_adapter.return_value = mock_adapter
        with patch("vetinari.code_search.code_search_registry", mock_registry):
            response = client.get("/api/code-search?q=hello")
        _assert_500_error(response)


class TestSearchStatusExceptionBounding:
    """GET /api/search/status  -  registry failure must be bounded to 500."""

    def test_import_failure_returns_500(self, client: object) -> None:
        """When code_search import fails, return 500 not an unhandled exception."""
        with patch.dict(
            "sys.modules",
            {"vetinari.code_search": None},
        ):
            response = client.get("/api/search/status")
        _assert_500_error(response)


class TestStructuralMapExceptionBounding:
    """GET /api/structural-map  -  helper failure must be bounded to 500."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_structural_map raises, return 500 not a raw exception frame."""
        # Use a real temp dir so path validation passes
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # The route checks that project_path is inside PROJECTS_DIR.
            # bypass that validation by patching it to the temp dir.
            with (
                patch("vetinari.constants.PROJECTS_DIR", Path(tmpdir).parent),
                patch(
                    "vetinari.code_search.get_structural_map",
                    side_effect=RuntimeError("repo map failed"),
                ),
            ):
                response = client.get(f"/api/structural-map?project_path={tmpdir}")
        _assert_500_error(response)


class TestRepoMapFindUsagesExceptionBounding:
    """GET /api/repo-map/usages  -  repo map failure must be bounded to 500."""

    def test_get_repo_map_raise_returns_500(self, client: object) -> None:
        """When get_repo_map raises, return 500 not a raw exception frame."""
        with patch(
            "vetinari.repo_map.get_repo_map",
            side_effect=RuntimeError("repo map unavailable"),
        ):
            response = client.get("/api/repo-map/usages?name=MyClass")
        _assert_500_error(response)

    def test_find_usages_raise_returns_500(self, client: object) -> None:
        """When find_usages raises, return 500 not a raw exception frame."""
        mock_rm = MagicMock()
        mock_rm.find_usages.side_effect = RuntimeError("index lost")
        with patch("vetinari.repo_map.get_repo_map", return_value=mock_rm):
            response = client.get("/api/repo-map/usages?name=MyClass")
        _assert_500_error(response)


class TestRepoMapFileSymbolsAbsolutePath:
    """GET /api/repo-map/symbols  -  absolute file_path values must be rejected."""

    @pytest.mark.parametrize(
        "bad_path",
        [
            "/etc/passwd",
            "/home/user/secret.py",
            "C:\\secrets\\file.py",
            "C:/secrets/file.py",
            "D:/another/path.py",
        ],
    )
    def test_absolute_path_rejected_400(self, client: object, bad_path: str) -> None:
        """Absolute paths in file_path must return 400, not echo the path to the index."""
        response = client.get(f"/api/repo-map/symbols?file_path={bad_path}")
        _assert_400_error(response)
        body = response.json()
        # Body must explain the constraint, not just be empty
        error_text = str(body).lower()
        assert "relative" in error_text or "project" in error_text, (
            f"Expected 'relative' or 'project' in error body, got: {body}"
        )

    def test_relative_path_allowed(self, client: object) -> None:
        """A relative path must not be rejected by the absolute path guard."""
        mock_rm = MagicMock()
        mock_rm.get_file_symbols.return_value = []
        with patch("vetinari.repo_map.get_repo_map", return_value=mock_rm):
            response = client.get("/api/repo-map/symbols?file_path=src/main.py")
        assert response.status_code == 200, (
            f"Expected 200 for relative path, got {response.status_code}: {response.text[:200]}"
        )

    def test_exception_returns_500(self, client: object) -> None:
        """When the repo map raises for a valid path, return 500."""
        mock_rm = MagicMock()
        mock_rm.get_file_symbols.side_effect = RuntimeError("index error")
        with patch("vetinari.repo_map.get_repo_map", return_value=mock_rm):
            response = client.get("/api/repo-map/symbols?file_path=src/main.py")
        _assert_500_error(response)


class TestRepoMapImportGraphExceptionBounding:
    """GET /api/repo-map/import-graph  -  failure must be bounded to 500."""

    def test_get_import_graph_raise_returns_500(self, client: object) -> None:
        """When get_import_graph raises, return 500 not a raw exception frame."""
        mock_rm = MagicMock()
        mock_rm.get_import_graph.side_effect = RuntimeError("graph unavailable")
        with patch("vetinari.repo_map.get_repo_map", return_value=mock_rm):
            response = client.get("/api/repo-map/import-graph")
        _assert_500_error(response)


# ===========================================================================
# Skills routes  -  litestar_skills_api.py
# ===========================================================================


class TestSkillsCatalogAbsolutePathStripping:
    """GET /api/v1/skills/catalog  -  absolute paths must be stripped from entries."""

    def test_absolute_file_path_stripped(self, client: object) -> None:
        """Catalog entries with absolute file_path values have them replaced with basename."""
        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {
            "skill_id": "test-skill",
            "name": "Test Skill",
            "file_path": "/home/user/skills/test_skill.md",
        }
        mock_catalog = {"test-skill": mock_entry}
        with patch(
            "vetinari.skills.catalog_loader._ensure_loaded",
            return_value=mock_catalog,
        ):
            response = client.get("/api/v1/skills/catalog")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        body = response.json()
        assert isinstance(body, list), f"Expected list response, got {type(body)}"
        assert len(body) == 1
        entry = body[0]
        # Must NOT contain the absolute path prefix
        assert "/home/user/skills" not in str(entry.get("file_path", "")), (
            f"Absolute path leaked into response: {entry}"
        )
        # Must contain just the filename
        assert entry.get("file_path") == "test_skill.md", f"Expected basename only, got {entry.get('file_path')!r}"

    def test_windows_absolute_path_stripped(self, client: object) -> None:
        """Catalog entries with Windows absolute paths have them replaced with basename."""
        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {
            "skill_id": "win-skill",
            "name": "Windows Skill",
            "file_path": "C:\\Users\\user\\skills\\win_skill.md",
        }
        mock_catalog = {"win-skill": mock_entry}
        with patch(
            "vetinari.skills.catalog_loader._ensure_loaded",
            return_value=mock_catalog,
        ):
            response = client.get("/api/v1/skills/catalog")
        assert response.status_code == 200
        body = response.json()
        entry = body[0]
        # The Windows prefix must not appear
        assert "C:\\" not in str(entry.get("file_path", "")), f"Windows absolute path leaked: {entry}"

    def test_catalog_exception_returns_500(self, client: object) -> None:
        """When _ensure_loaded raises, return 500."""
        with patch(
            "vetinari.skills.catalog_loader._ensure_loaded",
            side_effect=RuntimeError("catalog broken"),
        ):
            response = client.get("/api/v1/skills/catalog")
        _assert_500_error(response)


class TestSkillsCapabilityExceptionBounding:
    """GET /api/v1/skills/capabilities  -  failure must be bounded to 500."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_catalog_by_capability raises, return 500."""
        with patch(
            "vetinari.skills.catalog_loader.get_catalog_by_capability",
            side_effect=RuntimeError("catalog error"),
        ):
            response = client.get("/api/v1/skills/capabilities?capability=code_review")
        _assert_500_error(response)


class TestSkillsTagExceptionBounding:
    """GET /api/v1/skills/tags  -  failure must be bounded to 500."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_catalog_by_tag raises, return 500."""
        with patch(
            "vetinari.skills.catalog_loader.get_catalog_by_tag",
            side_effect=RuntimeError("catalog error"),
        ):
            response = client.get("/api/v1/skills/tags?tag=quality")
        _assert_500_error(response)


class TestSkillsCatalogForAgentExceptionBounding:
    """GET /api/v1/skills/catalog/{agent}  -  failure must be bounded to 500."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_catalog_by_agent raises, return 500."""
        with patch(
            "vetinari.skills.catalog_loader.get_catalog_by_agent",
            side_effect=RuntimeError("catalog error"),
        ):
            response = client.get("/api/v1/skills/catalog/foreman")
        _assert_500_error(response)


class TestSkillsSummariesExceptionBounding:
    """GET /api/v1/skills/summaries  -  failure must be bounded to 500."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When list_skill_summaries raises, return 500."""
        with patch(
            "vetinari.skills.skill_registry_convenience.list_skill_summaries",
            side_effect=RuntimeError("registry down"),
        ):
            response = client.get("/api/v1/skills/summaries")
        _assert_500_error(response)


class TestSkillSummaryExceptionBounding:
    """GET /api/v1/skills/{skill_id}/summary  -  failure and missing skill semantics."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_skill_summary raises, return 500."""
        with patch(
            "vetinari.skills.skill_registry_convenience.get_skill_summary",
            side_effect=RuntimeError("registry down"),
        ):
            response = client.get("/api/v1/skills/foreman/summary")
        _assert_500_error(response)

    def test_missing_skill_returns_404(self, client: object) -> None:
        """When the skill is not found (None returned), endpoint returns 404."""
        with patch(
            "vetinari.skills.skill_registry_convenience.get_skill_summary",
            return_value=None,
        ):
            response = client.get("/api/v1/skills/nonexistent-skill/summary")
        _assert_404_error(response)


class TestSkillTrustElevation404Consistency:
    """GET /api/v1/skills/{skill_id}/trust  -  missing skill must return 404."""

    def test_missing_skill_returns_404_not_200(self, client: object) -> None:
        """verify_trust_elevation returning error dict must yield 404, not 200."""
        with patch(
            "vetinari.skills.skill_registry_convenience.verify_skill_trust_elevation",
            return_value={"overall_pass": False, "error": "Skill missing-skill not found"},
        ):
            response = client.get("/api/v1/skills/missing-skill/trust")
        _assert_404_error(response)

    def test_present_skill_returns_200(self, client: object) -> None:
        """A found skill with gate results must return 200."""
        with patch(
            "vetinari.skills.skill_registry_convenience.verify_skill_trust_elevation",
            return_value={
                "overall_pass": True,
                "gate_results": {"g1_static": True, "g2_semantic": True, "g3_behavioral": True, "g4_permissions": True},
                "current_tier": "t3_trusted",
            },
        ):
            response = client.get("/api/v1/skills/foreman/trust")
        assert response.status_code == 200, (
            f"Expected 200 for present skill, got {response.status_code}: {response.text[:200]}"
        )
        body = response.json()
        assert body["gate_results"] == {
            "g1_static": True,
            "g2_semantic": True,
            "g3_behavioral": True,
            "g4_permissions": True,
        }, f"Expected gate_results payload in response, got {body}"
        assert body["current_tier"] == "t3_trusted", f"Expected current_tier='t3_trusted' in response, got {body}"

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When verify_skill_trust_elevation raises, return 500."""
        with patch(
            "vetinari.skills.skill_registry_convenience.verify_skill_trust_elevation",
            side_effect=RuntimeError("registry error"),
        ):
            response = client.get("/api/v1/skills/foreman/trust")
        _assert_500_error(response)


class TestSkillValidateOutputEmptyBody:
    """POST /api/v1/skills/{skill_id}/validate  -  empty body must be rejected."""

    def test_empty_body_returns_400(self, client: object) -> None:
        """POST with {} (no output field) must return 400, not a pass result."""
        response = client.post(
            "/api/v1/skills/foreman/validate",
            json={},
            headers=_CSRF,
        )
        _assert_400_error(response)
        body = response.json()
        assert "output" in str(body).lower(), f"Expected error mentioning 'output', got: {body}"

    def test_with_output_field_proceeds(self, client: object) -> None:
        """POST with output field present must proceed to validation."""
        with patch(
            "vetinari.skills.skill_registry_convenience.validate_skill_output",
            return_value=(True, []),
        ):
            response = client.post(
                "/api/v1/skills/foreman/validate",
                json={"output": "some result"},
                headers=_CSRF,
            )
        assert response.status_code == 201, (
            f"Expected 201 when output is provided, got {response.status_code}: {response.text[:200]}"
        )
        body = response.json()
        assert body["passed"] is True, f"Expected passed=True in response, got {body}"
        assert body["failures"] == [], f"Expected no validation failures in response, got {body}"

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When validate_skill_output raises, return 500."""
        with patch(
            "vetinari.skills.skill_registry_convenience.validate_skill_output",
            side_effect=RuntimeError("validator crashed"),
        ):
            response = client.post(
                "/api/v1/skills/foreman/validate",
                json={"output": "some result"},
                headers=_CSRF,
            )
        _assert_500_error(response)


class TestSkillValidationDetailExceptionBounding:
    """GET /api/v1/skills/{skill_id}/validation-detail  -  failure must be bounded."""

    def test_helper_raise_returns_500(self, client: object) -> None:
        """When get_skill_validation_detail raises, return 500."""
        with patch(
            "vetinari.skills.skill_registry.get_skill_validation_detail",
            side_effect=RuntimeError("registry down"),
        ):
            response = client.get("/api/v1/skills/foreman/validation-detail")
        _assert_500_error(response)

    def test_missing_skill_returns_404(self, client: object) -> None:
        """When None is returned for an unknown skill, endpoint returns 404."""
        with patch(
            "vetinari.skills.skill_registry.get_skill_validation_detail",
            return_value=None,
        ):
            response = client.get("/api/v1/skills/unknown-skill/validation-detail")
        _assert_404_error(response)


# ===========================================================================
# Rules routes  -  litestar_rules_routes.py
# ===========================================================================


class TestRulesGetExceptionBounding:
    """GET /api/v1/rules  -  rules manager failure must be bounded to 500."""

    def test_rules_manager_raise_returns_500(self, client: object) -> None:
        """When get_rules_manager raises, return 500."""
        with patch(
            "vetinari.rules_manager.get_rules_manager",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.get("/api/v1/rules")
        _assert_500_error(response)


class TestRulesGlobalExceptionBounding:
    """GET /api/v1/rules/global  -  rules manager failure must be bounded to 500."""

    def test_rules_manager_raise_returns_500(self, client: object) -> None:
        """When get_rules_manager raises, return 500."""
        with patch(
            "vetinari.rules_manager.get_rules_manager",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.get("/api/v1/rules/global")
        _assert_500_error(response)


class TestRulesGlobalPostEmptyBody:
    """POST /api/v1/rules/global  -  empty body must be rejected with 400."""

    def test_empty_body_returns_400(self, admin_client: object) -> None:
        """POST with {} (no rules field) must return 400, not report success."""
        response = admin_client.post(
            "/api/v1/rules/global",
            json={},
            headers=_CSRF,
        )
        _assert_400_error(response)
        body = response.json()
        assert "rules" in str(body).lower(), f"Expected error mentioning 'rules', got: {body}"

    def test_with_rules_field_succeeds(self, admin_client: object) -> None:
        """POST with a rules field present must save and return status saved."""
        mock_rm = MagicMock()
        mock_rm.get_global_rules.return_value = ["rule 1"]
        with patch("vetinari.rules_manager.get_rules_manager", return_value=mock_rm):
            response = admin_client.post(
                "/api/v1/rules/global",
                json={"rules": ["rule 1"]},
                headers=_CSRF,
            )
        assert response.status_code == 201, (
            f"Expected 201 when rules provided, got {response.status_code}: {response.text[:200]}"
        )
        body = response.json()
        assert body.get("status") == "saved", f"Expected status='saved', got {body}"

    def test_exception_returns_500(self, admin_client: object) -> None:
        """When get_rules_manager raises during POST, return 500."""
        with patch(
            "vetinari.rules_manager.get_rules_manager",
            side_effect=RuntimeError("db gone"),
        ):
            response = admin_client.post(
                "/api/v1/rules/global",
                json={"rules": ["rule 1"]},
                headers=_CSRF,
            )
        _assert_500_error(response)


class TestRulesProjectExceptionBounding:
    """GET /api/v1/rules/project/{project_id}  -  failure must be bounded to 500."""

    def test_rules_manager_raise_returns_500(self, client: object) -> None:
        """When get_rules_manager raises, return 500."""
        with patch(
            "vetinari.rules_manager.get_rules_manager",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.get("/api/v1/rules/project/proj-123")
        _assert_500_error(response)


class TestRulesModelIdLeadingSlash:
    """GET /api/v1/rules/model/{model_id}  -  model_id must never carry a leading slash."""

    def test_plain_model_id_no_leading_slash(self, client: object) -> None:
        """Plain model ID must not gain a leading slash in the persisted key."""
        mock_rm = MagicMock()
        mock_rm.get_model_rules.return_value = []
        with patch("vetinari.rules_manager.get_rules_manager", return_value=mock_rm):
            response = client.get("/api/v1/rules/model/plain-model")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:200]}"
        body = response.json()
        model_id = body.get("model_id", "")
        assert not model_id.startswith("/"), f"model_id must not start with '/': got {model_id!r}"
        assert model_id == "plain-model", f"Expected model_id='plain-model', got {model_id!r}"

    def test_slash_containing_model_id_no_leading_slash(self, client: object) -> None:
        """Model ID with internal slashes (e.g. org/model) must not gain a leading slash."""
        mock_rm = MagicMock()
        mock_rm.get_model_rules.return_value = []
        with patch("vetinari.rules_manager.get_rules_manager", return_value=mock_rm):
            response = client.get("/api/v1/rules/model/org/model-name")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:200]}"
        body = response.json()
        model_id = body.get("model_id", "")
        assert not model_id.startswith("/"), f"model_id must not start with '/': got {model_id!r}"

    def test_rules_manager_raise_returns_500(self, client: object) -> None:
        """When get_rules_manager raises, return 500."""
        with patch(
            "vetinari.rules_manager.get_rules_manager",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.get("/api/v1/rules/model/plain-model")
        _assert_500_error(response)
