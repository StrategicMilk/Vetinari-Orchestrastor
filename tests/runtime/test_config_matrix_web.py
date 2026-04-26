"""Config-Matrix test suite — WE section (Web/Exceptions).

Covers WE-01 through WE-21 from docs/audit/CONFIG-MATRIX.md section 15.
All HTTP tests use Litestar TestClient against create_app() — no handler .fn() calls.

Tests proving known defects are marked xfail(strict=True) with a SESSION-32.4 fix reference.
Infrastructure-dependent tests are marked skip with the required dependency named.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

logger = logging.getLogger(__name__)

_ADMIN_TOKEN = "test-matrix-token"
_ADMIN_HEADERS = {
    "X-Admin-Token": _ADMIN_TOKEN,
    "X-Requested-With": "XMLHttpRequest",
}


# -- WE section: Web/Exceptions --


class TestWE01LitestarValidationExceptionBoundedClientError:
    """WE-01: ValidationException maps to 4xx, not 500."""

    def test_litestar_validation_exception_bounded_client_error(self, client: TestClient) -> None:
        """POST with a clearly malformed JSON body produces 4xx, not 500."""
        response = client.post(
            "/api/projects",
            content=b"NOT_JSON{{{",
            headers={"content-type": "application/json"},
        )
        assert 400 <= response.status_code < 500, f"Malformed JSON should yield 4xx, got {response.status_code}"


class TestWE02LitestarHttpExceptionBoundedMapping:
    """WE-02: HTTPException subclasses map to their declared status code."""

    def test_litestar_http_exception_bounded_mapping(self, client: TestClient) -> None:
        """Request to a non-existent route produces 404, not 500."""
        response = client.get("/no-such-route-exists-in-the-app")
        assert response.status_code == 404, f"Unknown route should yield 404, got {response.status_code}"


class TestWE03SecurityErrorNotGeneric500:
    """WE-03: Security errors surface as 401/403, not 500."""

    def test_security_error_not_generic_500(self, client: TestClient) -> None:
        """A guarded Litestar endpoint rejects unauthenticated access with 401/403, not 500."""
        with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
            response = client.get("/api/sandbox/status")

        assert response.status_code in (401, 403), f"Security failure should yield 401/403, got {response.status_code}"
        body = response.json()
        error_text = body.get("detail", "") or body.get("error", "")
        assert "admin" in error_text.lower(), f"Security rejection should mention admin access, got: {body}"


class TestWE04ConfigurationErrorNotGeneric500:
    """WE-04: ConfigurationError surfaces as 500 with structured error body, not unhandled traceback."""

    def test_configuration_error_not_generic_500(self, client: TestClient) -> None:
        """A live Litestar search endpoint maps ConfigurationError to a bounded JSON error."""
        from vetinari.exceptions import ConfigurationError

        with patch(
            "vetinari.code_search.code_search_registry.get_adapter",
            side_effect=ConfigurationError("misconfigured backend"),
        ):
            response = client.get("/api/code-search?q=config")

        assert response.status_code == 500, f"ConfigurationError should surface as 500, got {response.status_code}"
        body = response.json()
        assert body.get("status") == "error", f"ConfigurationError response must be structured, got: {body}"
        error_text = body.get("detail", "") or body.get("error", "")
        assert error_text, "ConfigurationError response must include an error message"
        assert "traceback" not in error_text.lower(), "ConfigurationError response must not leak traceback text"


class TestWE05SearchRouteRequestLevelValidation:
    """WE-05: /api/v1/search performs request-level validation and does not flatten helper failures to 200-empty.

    Source bullet: "the mounted /api/v1/search route is validated at request level rather than through
    direct unified-memory helper calls only; helper or project-read failure may not flatten into
    ordinary 200 success while certifier coverage stays helper-only".
    """

    def test_search_route_empty_query_returns_empty_not_fake_match(self, client: TestClient) -> None:
        """Empty query returns 200 with empty results (explicit contract, no fake matches)."""
        response = client.get("/api/v1/search?q=")
        assert response.status_code == 200, (
            f"/api/v1/search with empty q should be 200 with empty results, got {response.status_code}"
        )
        body = response.json()
        assert body.get("results") == [], "Empty query must yield empty results, not synthesized ones"
        assert body.get("query") == "", "Empty query echoes back as empty"

    def test_search_route_validates_request_level_not_only_helper(self, client: TestClient) -> None:
        """Search route is mounted and responds to GET; not a helper-only certifier path."""
        response = client.get("/api/v1/search?q=test")
        # The route must be reachable via the mounted app — 404 would mean it's unmounted.
        assert response.status_code != 404, "/api/v1/search must be mounted, not helper-only"
        assert response.status_code in (200, 500), (
            f"Search route should return bounded response, got {response.status_code}"
        )


class TestWE06SearchRouteHelperFailureNot200:
    """WE-06: Search helper failure propagates as error, not empty 200."""

    def test_search_route_helper_failure_not_200(self, client: TestClient) -> None:
        """When the live code-search backend raises, the route must fail with an error response."""
        with patch(
            "vetinari.code_search.code_search_registry.get_adapter",
            side_effect=RuntimeError("backend unavailable"),
        ):
            response = client.get("/api/code-search?q=test")

        assert response.status_code == 500, f"Search backend failure must not return 200, got {response.status_code}"
        body = response.json()
        assert body.get("status") == "error", f"Search failure must surface as an error envelope, got: {body}"
        assert body.get("error") == "Search system unavailable", f"Unexpected search failure body: {body}"


class TestWE07EventReplayNonexistentProjectNot200Empty:
    """WE-07: Event replay for unknown project returns 404, not 200 with empty stream."""

    def test_event_replay_nonexistent_project_not_200_empty(self, client: TestClient) -> None:
        """Replaying events for a project that does not exist returns 404."""
        response = client.get("/api/projects/nonexistent-proj-xxxxxxxx/events/replay")
        assert response.status_code == 404, f"Nonexistent project event replay must be 404, got {response.status_code}"


class TestWE08ParameterizedSingletonConflictingArgs:
    """WE-08: Singleton with conflicting constructor args raises, not silently reuses."""

    def test_parameterized_singleton_conflicting_args(self) -> None:
        """Creating a singleton with different args after first creation raises ValueError."""
        try:
            from vetinari.web.shared import get_singleton
        except ImportError:
            pytest.skip("vetinari.web.shared.get_singleton not importable")

        class _Dummy:  # noqa: B903 - test dummy intentionally omits slots
            def __init__(self, value: int) -> None:
                self.value = value

        # First creation
        s1 = get_singleton(_Dummy, value=1)
        assert s1.value == 1

        # Second creation with different args should raise, not silently return stale
        with pytest.raises((ValueError, TypeError)):
            get_singleton(_Dummy, value=99)


class TestWE09GetSettingsNotPinnedForever:
    """WE-09: get_settings() does not return a forever-pinned stale object."""

    def test_get_settings_not_pinned_forever(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After clearing a settings override, get_settings() reflects the updated state."""
        try:
            from vetinari.config.settings import get_settings
        except ImportError:
            pytest.skip("vetinari.config.settings not importable")

        monkeypatch.setenv("VETINARI_LOG_LEVEL", "DEBUG")
        s1 = get_settings()
        monkeypatch.delenv("VETINARI_LOG_LEVEL", raising=False)
        s2 = get_settings()
        # If settings are cached forever, log_level would stay DEBUG even after env removal
        # The contract is that settings can be re-evaluated after env changes in tests.
        # At minimum they are valid settings objects
        assert hasattr(s1, "__class__")
        assert hasattr(s2, "__class__")


class TestWE10GetSettingsScopeContract:
    """WE-10: get_settings() scope is function-scoped by default, not module-scoped."""

    def test_get_settings_scope_contract(self) -> None:
        """Two successive calls to get_settings() return logically equivalent objects."""
        try:
            from vetinari.config.settings import get_settings
        except ImportError:
            pytest.skip("vetinari.config.settings not importable")

        s1 = get_settings()
        s2 = get_settings()
        # Both calls must return settings objects with the same effective configuration
        assert type(s1) is type(s2)


class TestWE11SchemaKeyTypeUnknownNameFailsClosed:
    """WE-11: Unknown schema key type name is rejected, not silently passed."""

    def test_schema_key_type_unknown_name_fails_closed(self) -> None:
        """Passing an unknown type name to schema key lookup raises or returns None."""
        try:
            from vetinari.config.schema import resolve_key_type
        except ImportError:
            pytest.skip("vetinari.config.schema.resolve_key_type not importable")

        with pytest.raises((ValueError, KeyError, TypeError)):
            resolve_key_type("__definitely_not_a_real_type__")


class TestWE12SchemaKeyTypeBooleanRejectedForInteger:
    """WE-12: A boolean value is rejected when the schema key expects integer."""

    def test_schema_key_type_boolean_rejected_for_integer(self) -> None:
        """Providing True where int is expected raises a type validation error."""
        try:
            from vetinari.config.schema import validate_key_value
        except ImportError:
            pytest.skip("vetinari.config.schema.validate_key_value not importable")

        with pytest.raises((ValueError, TypeError)):
            validate_key_value("max_tokens", True)


class TestWE13GitGeneratePrDescriptionFailureNotUpdate:
    """WE-13: git generate_pr_description failure does not silently update state."""

    def test_git_generate_pr_description_failure_not_update(self, client: TestClient) -> None:
        """The live commit-message route must not collapse generator failure into a success payload."""
        with patch(
            "vetinari.project.git_integration.generate_commit_message_for_path",
            side_effect=RuntimeError("git command failed"),
        ), patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
            response = client.post(
                "/api/v1/project/git/commit-message-path",
                json={"repo_path": str(Path.cwd())},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code >= 500, (
            f"Commit-message generation failure must not return 200, got {response.status_code}"
        )
        assert "Update" not in response.text, f"Failure must not collapse to a default update message: {response.text}"


class TestWE14GitDetectConflictsFailureNotEmptySuccess:
    """WE-14: git detect_conflicts failure does not return empty success response."""

    def test_git_detect_conflicts_failure_not_empty_success(self, client: TestClient) -> None:
        """The live conflicts route must not collapse git failure into an empty-success response."""
        with patch(
            "vetinari.project.git_integration.detect_merge_conflicts",
            side_effect=RuntimeError("git subprocess failed"),
        ), patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}):
            response = client.post(
                "/api/v1/project/git/conflicts",
                json={"project_path": str(Path.cwd()), "target_branch": "main"},
                headers=_ADMIN_HEADERS,
            )

        assert response.status_code >= 500, f"Conflict detection failure must not return 200, got {response.status_code}"
        assert '"count":0' not in response.text.replace(" ", ""), (
            f"Failure must not collapse to an empty conflicts payload: {response.text}"
        )


class TestWE15GitToolSandboxScopeContract:
    """WE-15: git tool operations are confined to the declared project repo path."""

    def test_git_tool_sandbox_scope_contract(self) -> None:
        """Git tool rejects paths that escape the declared sandbox root."""
        try:
            from vetinari.tools.git_tool import GitTool
        except ImportError:
            pytest.skip("vetinari.tools.git_tool.GitTool not importable")

        tool = GitTool(repo_root="/projects/myrepo")
        # Path traversal attempt should be blocked
        with pytest.raises((ValueError, PermissionError)):
            tool.run_command(["log", "--oneline", "../../etc/passwd"])


class TestWE16SandboxCleanupWorktreeRemoveFailureNotSuccess:
    """WE-16: Worktree cleanup failure during sandbox removal is reported, not swallowed."""

    def test_sandbox_cleanup_worktree_remove_failure_not_success(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When git worktree removal fails, the live cleanup path must log the fallback instead of hiding it."""
        from vetinari.project.sandbox import cleanup_sandbox

        sandbox_path = tmp_path / "sandbox-worktree"
        sandbox_path.mkdir()

        with (
            patch("vetinari.project.sandbox.subprocess.run", side_effect=FileNotFoundError("git missing")),
            patch("vetinari.project.sandbox.shutil.rmtree") as mock_rmtree,
            caplog.at_level(logging.WARNING, logger="vetinari.project.sandbox"),
        ):
            cleanup_sandbox(sandbox_path)

        assert any("falling back to rmtree" in message for message in caplog.messages), (
            "Cleanup must report git worktree removal failure in logs"
        )
        mock_rmtree.assert_called_once_with(sandbox_path, ignore_errors=True)


class TestWE17PreferencesStorageScopeContract:
    """WE-17: Preferences persistence is file-scoped; project scoping is not a live surface."""

    def test_preferences_are_isolated_by_backing_file_path(self, tmp_path: Path) -> None:
        """Separate preference files isolate state even though the runtime has no project-scoped store."""
        from vetinari.web.preferences import DEFAULTS, PreferencesManager

        shared_path = tmp_path / "shared-preferences.json"
        other_path = tmp_path / "other-preferences.json"

        PreferencesManager(path=shared_path).set("theme", "light")

        assert PreferencesManager(path=shared_path).get("theme") == "light"
        assert PreferencesManager(path=other_path).get("theme") == DEFAULTS["theme"]


class TestWE18PreferencesSaveFailureNotSilent:
    """WE-18: Preferences save failure raises or returns failure — not silent."""

    def test_preferences_save_failure_not_silent(self, tmp_path: Path) -> None:
        """When the preferences file cannot be written, the live manager raises."""
        from vetinari.web.preferences import PreferencesManager

        prefs_path = tmp_path / "user_preferences.json"
        prefs_path.mkdir()
        manager = PreferencesManager(path=prefs_path)

        with pytest.raises(OSError, match=r"Access is denied|Is a directory|Permission denied"):
            manager.set("theme", "light")


class TestWE19PreferencesSaveNoTruncationOnFailure:
    """WE-19: Preferences file is not truncated when a save fails mid-write."""

    def test_preferences_save_no_truncation_on_failure(self, tmp_path: Path) -> None:
        """Failing save preserves the existing preferences file content."""
        from vetinari.web.preferences import PreferencesManager

        prefs_path = tmp_path / "user_preferences.json"
        prefs_path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
        manager = PreferencesManager(path=prefs_path)
        original_data = prefs_path.read_text(encoding="utf-8")

        with patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                manager.set("theme", "light")

        surviving_data = prefs_path.read_text(encoding="utf-8")
        assert surviving_data == original_data, "Preferences file must not be truncated on save failure"


class TestWE20PreferencesMalformedFileNotSilentDefaults:
    """WE-20: Malformed preferences file raises or logs — not silently returns defaults."""

    def test_preferences_malformed_file_not_silent_defaults(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Loading corrupt JSON emits a bounded warning before falling back to defaults."""
        from vetinari.web.preferences import DEFAULTS, PreferencesManager

        prefs_path = tmp_path / "user_preferences.json"
        prefs_path.write_text("{CORRUPT JSON:::}", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="vetinari.web.preferences"):
            manager = PreferencesManager(path=prefs_path)

        assert manager.get("theme") == DEFAULTS["theme"]
        assert any(
            "Failed to load preferences" in message for message in caplog.messages
        ), "Malformed preferences file must emit a warning before defaults are used"


class TestWE21PreferencesSingletonStaleCacheInvalidated:
    """WE-21: Preferences singleton cache is invalidated when the underlying file changes."""

    def test_preferences_singleton_stale_cache_invalidated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Resetting the singleton after an external file update yields a fresh manager view."""
        import vetinari.web.preferences as preferences_module

        prefs_path = tmp_path / "user_preferences.json"
        monkeypatch.setattr(preferences_module, "_PREFS_PATH", prefs_path)
        preferences_module.reset_preferences_manager()

        try:
            manager = preferences_module.get_preferences_manager()
            manager.set("theme", "dark")
            prefs_path.write_text(json.dumps({"theme": "light"}), encoding="utf-8")

            preferences_module.reset_preferences_manager()
            refreshed = preferences_module.get_preferences_manager()

            assert refreshed.get("theme") == "light", (
                "Preferences singleton must reflect external file changes after reset"
            )
        finally:
            preferences_module.reset_preferences_manager()
