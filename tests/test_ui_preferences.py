"""Tests for user preferences API and variant system."""

import json
import pytest
from dataclasses import asdict
from pathlib import Path

from vetinari.web.preferences import (
    PreferencesManager,
    UserPreferences,
    get_preferences_manager,
    preferences_bp,
    reset_preferences_manager,
)
from vetinari.web.variant_system import (
    VARIANT_CONFIGS,
    VariantConfig,
    VariantLevel,
    VariantManager,
)


# ---------------------------------------------------------------------------
# UserPreferences defaults
# ---------------------------------------------------------------------------


class TestUserPreferencesDefaults:
    """Verify UserPreferences dataclass defaults."""

    def test_default_nicknames_are_discworld(self):
        prefs = UserPreferences()
        assert prefs.agent_nicknames["PLANNER"] == "Vetinari"
        assert prefs.agent_nicknames["RESEARCHER"] == "Ponder"
        assert prefs.agent_nicknames["ARCHITECT"] == "Ridcully"
        assert prefs.agent_nicknames["BUILDER"] == "Igor"
        assert prefs.agent_nicknames["TESTER"] == "Vimes"
        assert prefs.agent_nicknames["DOCUMENTER"] == "Carrot"
        assert prefs.agent_nicknames["RESILIENCE"] == "Rincewind"
        assert prefs.agent_nicknames["META"] == "Death"

    def test_default_nicknames_count(self):
        prefs = UserPreferences()
        assert len(prefs.agent_nicknames) == 8

    def test_default_theme(self):
        prefs = UserPreferences()
        assert prefs.theme == "dark"

    def test_default_compact_mode(self):
        prefs = UserPreferences()
        assert prefs.compact_mode is False

    def test_default_suggestion_frequency(self):
        prefs = UserPreferences()
        assert prefs.suggestion_frequency == "normal"

    def test_default_variant_level(self):
        prefs = UserPreferences()
        assert prefs.variant_level == "medium"

    def test_default_show_learning_dashboard(self):
        prefs = UserPreferences()
        assert prefs.show_learning_dashboard is True

    def test_default_show_agent_status(self):
        prefs = UserPreferences()
        assert prefs.show_agent_status is True

    def test_default_auto_approve_milestones(self):
        prefs = UserPreferences()
        assert prefs.auto_approve_milestones is False

    def test_default_agent_icons_empty(self):
        prefs = UserPreferences()
        assert prefs.agent_icons == {}

    def test_asdict_roundtrip(self):
        prefs = UserPreferences()
        data = asdict(prefs)
        restored = UserPreferences(**data)
        assert asdict(restored) == data


# ---------------------------------------------------------------------------
# PreferencesManager core logic
# ---------------------------------------------------------------------------


class TestPreferencesManager:
    """Test load / save / update / reset cycle."""

    def test_load_returns_defaults_when_no_file(self, tmp_path):
        mgr = PreferencesManager(path=str(tmp_path / "prefs.json"))
        prefs = mgr.load()
        assert prefs.theme == "dark"
        assert prefs.agent_nicknames["PLANNER"] == "Vetinari"

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "sub" / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        prefs = UserPreferences(theme="light")
        mgr.save(prefs)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["theme"] == "light"

    def test_load_reads_saved_data(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        mgr.save(UserPreferences(theme="light", compact_mode=True))
        # Force re-read
        mgr._preferences = None
        prefs = mgr.load()
        assert prefs.theme == "light"
        assert prefs.compact_mode is True

    def test_load_caches_result(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        p1 = mgr.load()
        p2 = mgr.load()
        assert p1 is p2

    def test_update_merges_partial(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        prefs = mgr.update({"theme": "light"})
        assert prefs.theme == "light"
        # Other fields unchanged
        assert prefs.compact_mode is False
        assert prefs.agent_nicknames["PLANNER"] == "Vetinari"

    def test_update_ignores_unknown_keys(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        prefs = mgr.update({"nonexistent_key": 42, "theme": "light"})
        assert prefs.theme == "light"
        assert not hasattr(prefs, "nonexistent_key")

    def test_update_persists_to_disk(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        mgr.update({"compact_mode": True})
        data = json.loads(path.read_text())
        assert data["compact_mode"] is True

    def test_reset_returns_defaults(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        mgr.update({"theme": "light", "compact_mode": True})
        prefs = mgr.reset()
        assert prefs.theme == "dark"
        assert prefs.compact_mode is False

    def test_reset_persists_defaults(self, tmp_path):
        path = tmp_path / "prefs.json"
        mgr = PreferencesManager(path=str(path))
        mgr.update({"theme": "light"})
        mgr.reset()
        data = json.loads(path.read_text())
        assert data["theme"] == "dark"

    def test_load_handles_corrupt_json(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text("{invalid json!!!")
        mgr = PreferencesManager(path=str(path))
        prefs = mgr.load()
        # Falls back to defaults
        assert prefs.theme == "dark"

    def test_load_ignores_extra_keys_in_file(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"theme": "light", "extra_field": "ignored"}))
        mgr = PreferencesManager(path=str(path))
        prefs = mgr.load()
        assert prefs.theme == "light"
        assert not hasattr(prefs, "extra_field")


# ---------------------------------------------------------------------------
# Nickname lookup
# ---------------------------------------------------------------------------


class TestPreferencesManagerNickname:
    """Nickname lookup tests."""

    def test_known_agent_returns_nickname(self, tmp_path):
        mgr = PreferencesManager(path=str(tmp_path / "p.json"))
        assert mgr.get_nickname("PLANNER") == "Vetinari"
        assert mgr.get_nickname("TESTER") == "Vimes"

    def test_unknown_agent_returns_raw_type(self, tmp_path):
        mgr = PreferencesManager(path=str(tmp_path / "p.json"))
        assert mgr.get_nickname("UNKNOWN_AGENT") == "UNKNOWN_AGENT"

    def test_custom_nickname_after_update(self, tmp_path):
        mgr = PreferencesManager(path=str(tmp_path / "p.json"))
        mgr.update(
            {
                "agent_nicknames": {
                    **UserPreferences().agent_nicknames,
                    "PLANNER": "Lord Havelock",
                }
            }
        )
        assert mgr.get_nickname("PLANNER") == "Lord Havelock"


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


class TestSingleton:
    """Test get_preferences_manager / reset_preferences_manager."""

    def test_singleton_returns_same_instance(self):
        reset_preferences_manager()
        m1 = get_preferences_manager()
        m2 = get_preferences_manager()
        assert m1 is m2
        reset_preferences_manager()

    def test_reset_clears_singleton(self):
        reset_preferences_manager()
        m1 = get_preferences_manager()
        reset_preferences_manager()
        m2 = get_preferences_manager()
        assert m1 is not m2
        reset_preferences_manager()


# ---------------------------------------------------------------------------
# VariantLevel enum
# ---------------------------------------------------------------------------


class TestVariantLevel:
    """Test VariantLevel enum."""

    def test_values(self):
        assert VariantLevel.LOW.value == "low"
        assert VariantLevel.MEDIUM.value == "medium"
        assert VariantLevel.HIGH.value == "high"

    def test_from_string(self):
        assert VariantLevel("low") is VariantLevel.LOW
        assert VariantLevel("medium") is VariantLevel.MEDIUM
        assert VariantLevel("high") is VariantLevel.HIGH

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            VariantLevel("turbo")

    def test_all_levels_in_configs(self):
        for level in VariantLevel:
            assert level in VARIANT_CONFIGS


# ---------------------------------------------------------------------------
# VariantConfig for each level
# ---------------------------------------------------------------------------


class TestVariantConfigs:
    """Verify the built-in configs."""

    def test_low_config(self):
        cfg = VARIANT_CONFIGS[VariantLevel.LOW]
        assert cfg.max_context_tokens == 4096
        assert cfg.max_planning_depth == 2
        assert cfg.enable_verification is False
        assert cfg.enable_self_improvement is False

    def test_medium_config(self):
        cfg = VARIANT_CONFIGS[VariantLevel.MEDIUM]
        assert cfg.max_context_tokens == 16384
        assert cfg.max_planning_depth == 5
        assert cfg.enable_verification is True
        assert cfg.enable_self_improvement is True

    def test_high_config(self):
        cfg = VARIANT_CONFIGS[VariantLevel.HIGH]
        assert cfg.max_context_tokens == 32768
        assert cfg.max_planning_depth == 10
        assert cfg.enable_verification is True
        assert cfg.enable_self_improvement is True

    def test_configs_have_descriptions(self):
        for cfg in VARIANT_CONFIGS.values():
            assert cfg.description
            assert len(cfg.description) > 10

    def test_context_tokens_increase(self):
        tokens = [
            VARIANT_CONFIGS[VariantLevel.LOW].max_context_tokens,
            VARIANT_CONFIGS[VariantLevel.MEDIUM].max_context_tokens,
            VARIANT_CONFIGS[VariantLevel.HIGH].max_context_tokens,
        ]
        assert tokens == sorted(tokens)
        assert tokens[0] < tokens[1] < tokens[2]

    def test_planning_depth_increases(self):
        depths = [
            VARIANT_CONFIGS[VariantLevel.LOW].max_planning_depth,
            VARIANT_CONFIGS[VariantLevel.MEDIUM].max_planning_depth,
            VARIANT_CONFIGS[VariantLevel.HIGH].max_planning_depth,
        ]
        assert depths == sorted(depths)
        assert depths[0] < depths[1] < depths[2]


# ---------------------------------------------------------------------------
# VariantManager
# ---------------------------------------------------------------------------


class TestVariantManager:
    """Test VariantManager level switching."""

    def test_default_level(self):
        mgr = VariantManager()
        assert mgr.current_level == "medium"

    def test_custom_default(self):
        mgr = VariantManager(default_level="high")
        assert mgr.current_level == "high"

    def test_get_config_returns_current(self):
        mgr = VariantManager()
        cfg = mgr.get_config()
        assert cfg.level is VariantLevel.MEDIUM

    def test_set_level_switches(self):
        mgr = VariantManager()
        cfg = mgr.set_level("high")
        assert mgr.current_level == "high"
        assert cfg.level is VariantLevel.HIGH

    def test_set_level_low(self):
        mgr = VariantManager()
        cfg = mgr.set_level("low")
        assert mgr.current_level == "low"
        assert cfg.enable_verification is False

    def test_set_invalid_level_raises(self):
        mgr = VariantManager()
        with pytest.raises(ValueError):
            mgr.set_level("invalid")

    def test_get_all_levels(self):
        mgr = VariantManager()
        levels = mgr.get_all_levels()
        assert len(levels) == 3
        level_values = [lv["level"] for lv in levels]
        assert "low" in level_values
        assert "medium" in level_values
        assert "high" in level_values
        for lv in levels:
            assert "description" in lv

    def test_set_level_returns_config(self):
        mgr = VariantManager()
        cfg = mgr.set_level("low")
        assert isinstance(cfg, VariantConfig)
        assert cfg.max_context_tokens == 4096


# ---------------------------------------------------------------------------
# Flask route tests (using Flask test client)
# ---------------------------------------------------------------------------


class TestFlaskRoutes:
    """Test the preferences Flask blueprint routes."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        """Create a minimal Flask app with the preferences blueprint."""
        from flask import Flask

        self.app = Flask(__name__)
        self.app.register_blueprint(preferences_bp)
        self.client = self.app.test_client()

        # Point the singleton at a temp file
        reset_preferences_manager()
        self.prefs_path = str(tmp_path / "test_prefs.json")
        get_preferences_manager(self.prefs_path)
        yield
        reset_preferences_manager()

    def test_get_preferences(self):
        resp = self.client.get("/api/preferences")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["theme"] == "dark"
        assert data["agent_nicknames"]["PLANNER"] == "Vetinari"

    def test_update_preferences(self):
        resp = self.client.post(
            "/api/preferences",
            json={"theme": "light", "compact_mode": True},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["theme"] == "light"
        assert data["compact_mode"] is True
        # Unchanged fields persist
        assert data["agent_nicknames"]["PLANNER"] == "Vetinari"

    def test_update_persists_across_get(self):
        self.client.post("/api/preferences", json={"theme": "light"})
        resp = self.client.get("/api/preferences")
        data = resp.get_json()
        assert data["theme"] == "light"

    def test_reset_preferences(self):
        self.client.post("/api/preferences", json={"theme": "light"})
        resp = self.client.post("/api/preferences/reset")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["theme"] == "dark"

    def test_get_nickname_known(self):
        resp = self.client.get("/api/preferences/nickname/PLANNER")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["agent_type"] == "PLANNER"
        assert data["nickname"] == "Vetinari"

    def test_get_nickname_unknown(self):
        resp = self.client.get("/api/preferences/nickname/UNKNOWN")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["nickname"] == "UNKNOWN"

    def test_update_with_empty_body(self):
        resp = self.client.post(
            "/api/preferences",
            content_type="application/json",
            data="{}",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        # Nothing changed
        assert data["theme"] == "dark"

    def test_update_with_no_json(self):
        resp = self.client.post("/api/preferences")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["theme"] == "dark"
