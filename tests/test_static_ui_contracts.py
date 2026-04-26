"""Static UI contract probes — SESSION-32C.

Verifies that the Svelte source api.js no longer exports dead route wrappers
and that retained exports target live backend routes. These are source-text
probes, not runtime execution — they catch regressions before a build step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SVELTE_SRC = Path(__file__).parent.parent / "ui" / "svelte" / "src"
_API_JS = _SVELTE_SRC / "lib" / "api.js"
_STORES_JS = _SVELTE_SRC / "lib" / "stores" / "app.svelte.js"
_SETTINGS_SVELTE = _SVELTE_SRC / "views" / "SettingsView.svelte"


@pytest.fixture(scope="module")
def api_source() -> str:
    """Content of ui/svelte/src/lib/api.js."""
    return _API_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stores_source() -> str:
    """Content of ui/svelte/src/lib/stores/app.svelte.js."""
    return _STORES_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def settings_source() -> str:
    """Content of ui/svelte/src/views/SettingsView.svelte."""
    return _SETTINGS_SVELTE.read_text(encoding="utf-8")


class TestDeadApiWrappers:
    """Removed wrappers must not exist in api.js."""

    def test_no_create_rule(self, api_source: str) -> None:
        assert "createRule" not in api_source, (
            "createRule() targets POST /api/v1/rules which is unmounted — must not exist"
        )

    def test_no_update_rule(self, api_source: str) -> None:
        assert "updateRule" not in api_source, (
            "updateRule() targets PUT /api/v1/rules/:id which is unmounted — must not exist"
        )

    def test_no_delete_rule(self, api_source: str) -> None:
        assert "deleteRule" not in api_source, (
            "deleteRule() targets DELETE /api/v1/rules/:id which is unmounted — must not exist"
        )

    def test_no_list_skills(self, api_source: str) -> None:
        assert "listSkills" not in api_source, (
            "listSkills() targets GET /api/v1/skills which is unmounted — must not exist"
        )

    def test_no_get_skill(self, api_source: str) -> None:
        assert "getSkill" not in api_source, (
            "getSkill() targets GET /api/v1/skills/:id which is unmounted — must not exist"
        )


class TestLiveApiWrappers:
    """Read-only rules wrapper must be retained (live GET route)."""

    def test_get_rules_retained(self, api_source: str) -> None:
        assert "export function getRules" in api_source, "getRules() targets live GET /api/v1/rules — must be retained"

    def test_list_projects_uses_get(self, api_source: str) -> None:
        # listProjects must call get(), not post()
        match = re.search(
            r"export function listProjects\([^)]*\)\s*\{([^}]+)\}",
            api_source,
        )
        assert match is not None, "listProjects() must be defined in api.js"
        body = match.group(1)
        assert "get(" in body, "listProjects() must use GET, not POST"
        assert "return post(" not in body, "listProjects() must not call post() — /api/projects is GET-only"


class TestDeadAccessibilityState:
    """reducedMotion/compactMode state removed from stores — never applied to DOM."""

    def test_no_reduced_motion_state(self, stores_source: str) -> None:
        assert "_reducedMotion" not in stores_source, (
            "_reducedMotion $state was never applied to the DOM — must be removed"
        )

    def test_no_compact_mode_state(self, stores_source: str) -> None:
        assert "_compactMode" not in stores_source, "_compactMode $state was never applied to the DOM — must be removed"

    def test_no_reduced_motion_storage_key(self, stores_source: str) -> None:
        assert "reducedMotion: 'reducedMotion'" not in stores_source, (
            "reducedMotion STORAGE_KEY must be removed alongside its state"
        )

    def test_no_compact_mode_storage_key(self, stores_source: str) -> None:
        assert "compactMode: 'compactMode'" not in stores_source, (
            "compactMode STORAGE_KEY must be removed alongside its state"
        )


class TestDeadAccessibilityUi:
    """SettingsView UI toggles for unapplied flags must be removed."""

    def test_no_toggle_reduced_motion(self, settings_source: str) -> None:
        assert "toggleReducedMotion" not in settings_source, (
            "toggleReducedMotion() referenced unapplied state — must be removed"
        )

    def test_no_toggle_compact_mode(self, settings_source: str) -> None:
        assert "toggleCompactMode" not in settings_source, (
            "toggleCompactMode() referenced unapplied state — must be removed"
        )
