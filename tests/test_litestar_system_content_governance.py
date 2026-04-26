"""Mounted request-level governance tests for native system-content routes.

Exercises path-leak prevention, graceful-degradation, and input-rejection
contracts for 8 routes across two source files.  Every test goes through the
full Litestar HTTP stack via TestClient so routing, serialisation, guards, and
middleware are exercised alongside handler logic.

Routes under test:
  GET /api/v1/workflow           — path leak prevention, filesystem-error degradation
  GET /api/v1/system-prompts     — corrupted-file graceful degradation
  GET /api/v1/preferences        — 503 on manager failure
  GET /api/v1/settings           — 503 on settings/hardware failure
  GET /api/v1/variant            — 503 on manager failure
  GET /api/v1/download           — unsafe identifier rejection, no JSON content-type
  GET /api/v1/artifacts          — path leak prevention
  GET /api/artifacts             — path leak prevention (admin route in projects_api)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Minimal Litestar app with all handlers registered, shutdown suppressed.

    Returns:
        A Litestar application instance safe to use in tests.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        app = create_app(debug=True)
    return app


# ---------------------------------------------------------------------------
# GET /api/v1/workflow
# ---------------------------------------------------------------------------


class TestWorkflowRoute:
    """GET /api/v1/workflow — path leak prevention and filesystem-error degradation."""

    def test_workflow_does_not_leak_absolute_paths(self, litestar_app: object, tmp_path: Path) -> None:
        """Project data must not contain absolute filesystem paths under any key.

        Exposing the host filesystem layout to API callers enables directory
        traversal inference even when traversal itself is blocked.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        # Create a minimal project directory
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        proj_dir = projects_dir / "myproject"
        proj_dir.mkdir()
        (proj_dir / "project.yaml").write_text(
            "project_name: My Project\ndescription: test\nstatus: active\n",
            encoding="utf-8",
        )

        with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/workflow")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        projects = body.get("projects", [])
        assert len(projects) >= 1, "Expected at least one project in response"

        for project in projects:
            # The 'path' key must not appear — only 'id' carries the identifier
            assert "path" not in project, (
                f"Project data must not include 'path' key — leaks filesystem layout: {project}"
            )
            # Verify no string value contains an absolute path
            for key, value in project.items():
                if isinstance(value, str):
                    assert not os.path.isabs(value), f"Project field '{key}' contains an absolute path: {value!r}"

    def test_workflow_filesystem_error_does_not_500(self, litestar_app: object, tmp_path: Path) -> None:
        """When iterdir() on projects_dir raises PermissionError, route must not return 500.

        A hard 500 exposes implementation details and is unhelpful to callers.
        The route should degrade gracefully with an empty project list.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()

        with (
            patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            patch.object(Path, "iterdir", side_effect=PermissionError("access denied")),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/workflow")

        assert response.status_code == 200, (
            f"Expected 200 (degraded) when filesystem error occurs, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        assert body["projects"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/system-prompts
# ---------------------------------------------------------------------------


class TestSystemPromptsRoute:
    """GET /api/v1/system-prompts — corrupted-file graceful degradation."""

    def test_corrupted_prompt_file_graceful_degradation(self, litestar_app: object, tmp_path: Path) -> None:
        """When one prompt file is unreadable the route must still return 200 with readable ones.

        An unreadable file must appear in the 'warnings' list but must not
        cause the handler to abort and return 500.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        prompts_dir = tmp_path / "system_prompts"
        prompts_dir.mkdir()
        good_file = prompts_dir / "good.txt"
        good_file.write_text("This is a good prompt.", encoding="utf-8")
        bad_file = prompts_dir / "bad.txt"
        bad_file.write_text("This will fail to read.", encoding="utf-8")

        original_read_text = Path.read_text

        def _selective_read_text(self: Path, **kwargs: object) -> str:
            if self.name == "bad.txt":
                raise OSError("permission denied")
            return original_read_text(self, **kwargs)

        with (
            patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            patch.object(Path, "read_text", _selective_read_text),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/system-prompts")

        assert response.status_code == 200, (
            f"Expected 200 when one prompt file fails, got {response.status_code}: {response.text[:300]}"
        )
        body = response.json()
        prompts = body.get("prompts", [])
        # The good file must still appear
        names = [p["name"] for p in prompts]
        assert "good" in names, f"Good prompt should be present in response, got: {names}"
        # The bad file must trigger a warning
        warnings = body.get("warnings", [])
        assert len(warnings) > 0, f"Expected at least one warning when a prompt file is unreadable, got: {warnings}"


# ---------------------------------------------------------------------------
# GET /api/v1/preferences
# ---------------------------------------------------------------------------


class TestPreferencesGetRoute:
    """GET /api/v1/preferences — 503 on preferences manager failure."""

    def test_manager_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_preferences_manager raises, the route must return 503 not 500.

        A 503 signals transient service unavailability so clients can retry,
        whereas 500 implies a permanent coding defect.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.preferences.get_preferences_manager",
            side_effect=RuntimeError("prefs manager down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/preferences")

        assert response.status_code == 503, (
            f"Expected 503 when preferences manager unavailable, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/settings
# ---------------------------------------------------------------------------


class TestSettingsGetRoute:
    """GET /api/v1/settings — 503 on settings or hardware detection failure."""

    def test_settings_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_settings raises, the route must return 503 not 500.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.config.settings.get_settings",
            side_effect=RuntimeError("settings subsystem down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/settings")

        assert response.status_code == 503, (
            f"Expected 503 when settings unavailable, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/variant
# ---------------------------------------------------------------------------


class TestVariantGetRoute:
    """GET /api/v1/variant — 503 on variant manager failure."""

    def test_variant_manager_unavailable_returns_503(self, litestar_app: object) -> None:
        """When get_variant_manager raises, the route must return 503 not 500.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.variant_system.get_variant_manager",
            side_effect=RuntimeError("variant manager down"),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/variant")

        assert response.status_code == 503, (
            f"Expected 503 when variant manager unavailable, got {response.status_code}: {response.text[:300]}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/download
# ---------------------------------------------------------------------------


class TestDownloadRoute:
    """GET /api/v1/download — unsafe identifier rejection and media-type correctness."""

    def test_invalid_identifiers_rejected_not_normalized(self, litestar_app: object) -> None:
        """Path-traversal identifiers must be rejected with 400, not silently normalised.

        If ../etc is normalised to ____etc, the handler may serve an unrelated
        file while returning 200 success — this is a path confusion vulnerability.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = client.get(
                "/api/v1/download",
                params={"project_id": "../../etc", "task_id": "valid-task", "filename": "output.txt"},
            )

        assert response.status_code == 400, (
            f"Expected 400 for unsafe project_id, got {response.status_code}: {response.text[:300]}"
        )

    def test_unsafe_task_id_rejected(self, litestar_app: object) -> None:
        """Unsafe task_id containing path separators must be rejected with 400.

        Args:
            litestar_app: Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            response = client.get(
                "/api/v1/download",
                params={"project_id": "valid-project", "task_id": "../../../windows", "filename": "output.txt"},
            )

        assert response.status_code == 400, (
            f"Expected 400 for unsafe task_id, got {response.status_code}: {response.text[:300]}"
        )

    def test_valid_download_does_not_advertise_json_content_type(self, litestar_app: object, tmp_path: Path) -> None:
        """A successful file download must NOT return Content-Type: application/json.

        The route previously declared media_type=MediaType.JSON while serving a
        binary file response — the Content-Type header would lie to callers.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        # Build the expected directory structure projects/<pid>/outputs/<tid>/generated/<fn>
        project_id = "testproject"
        task_id = "testtask"
        filename = "output.txt"
        generated_dir = tmp_path / "projects" / project_id / "outputs" / task_id / "generated"
        generated_dir.mkdir(parents=True)
        (generated_dir / filename).write_text("file content", encoding="utf-8")

        with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
            with TestClient(app=litestar_app) as client:
                response = client.get(
                    "/api/v1/download",
                    params={"project_id": project_id, "task_id": task_id, "filename": filename},
                )

        assert response.status_code == 200, (
            f"Expected 200 for valid download, got {response.status_code}: {response.text[:300]}"
        )
        content_type = response.headers.get("content-type", "")
        assert "application/json" not in content_type, (
            f"File download must not advertise JSON content-type, got: {content_type!r}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/artifacts
# ---------------------------------------------------------------------------


class TestArtifactsRoute:
    """GET /api/v1/artifacts — path leak prevention."""

    def test_artifacts_does_not_leak_absolute_paths(self, litestar_app: object, tmp_path: Path) -> None:
        """Artifact 'path' field must be the filename only — no absolute paths.

        Exposing the server's build directory layout enables filesystem
        reconnaissance.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        artifacts_dir = tmp_path / "build" / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "output.whl").write_bytes(b"fake wheel content")

        with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/v1/artifacts")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        artifacts = body.get("artifacts", [])
        assert len(artifacts) >= 1, "Expected at least one artifact in response"

        for artifact in artifacts:
            path_value = artifact.get("path", "")
            assert not os.path.isabs(path_value), f"Artifact 'path' must not be absolute — got: {path_value!r}"
            # Should be just the filename
            assert path_value == artifact["name"], (
                f"Artifact 'path' should equal 'name' (filename only), got path={path_value!r} name={artifact['name']!r}"
            )


# ---------------------------------------------------------------------------
# GET /api/artifacts (admin route in litestar_projects_api)
# ---------------------------------------------------------------------------


class TestProjectArtifactsRoute:
    """GET /api/artifacts — path leak prevention for the admin artifacts endpoint."""

    def test_project_artifacts_does_not_leak_absolute_paths(self, litestar_app: object, tmp_path: Path) -> None:
        """Admin artifact 'path' field must be filename only — no absolute paths.

        This is the admin variant of the artifacts endpoint registered in
        litestar_projects_api.py.  It must not expose the server's directory
        structure even to admin callers.

        The admin guard is bypassed by patching ``is_admin_connection`` so the
        test focuses exclusively on the path-leak defect, not authentication.

        Args:
            litestar_app: Litestar application fixture.
            tmp_path: Pytest temporary directory.
        """
        from litestar.testing import TestClient

        artifacts_dir = tmp_path / "build" / "artifacts"
        artifacts_dir.mkdir(parents=True)
        (artifacts_dir / "vetinari.whl").write_bytes(b"fake wheel content")

        # Patch is_admin_connection at the guards module so admin_guard passes.
        # The guard function is called at request time, not app-creation time,
        # so this patch is effective without rebuilding the app.
        with (
            patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
            patch("vetinari.web.litestar_guards.is_admin_connection", return_value=True),
        ):
            with TestClient(app=litestar_app) as client:
                response = client.get("/api/artifacts")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text[:300]}"
        body = response.json()
        artifacts = body.get("artifacts", [])
        assert len(artifacts) >= 1, "Expected at least one artifact in admin artifacts response"

        for artifact in artifacts:
            path_value = artifact.get("path", "")
            assert not os.path.isabs(path_value), f"Admin artifact 'path' must not be absolute — got: {path_value!r}"
            assert path_value == artifact["name"], (
                f"Admin artifact 'path' should equal 'name' (filename only), got path={path_value!r} name={artifact['name']!r}"
            )
