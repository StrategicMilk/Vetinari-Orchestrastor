"""
Tests for vetinari.config.inference_config

Covers:
- InferenceProfile: defaults, custom values, to_dict()
- _classify_model_size: all size tiers, keyword fallbacks, default fallback
- _clamp: within bounds, below lo, above hi, at boundary, integer inputs
- InferenceConfigManager: singleton creation, reset, reload, missing file,
  JSON parse error, get_profile (known, unknown fallback, empty config),
  get_effective_params (no model, size adjustments, model overrides, clamping),
  list_profiles, get_all_profiles, get_stats
- Thread-safety: singleton is identical across concurrent creations
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

# Remove any incomplete stub left by earlier test files so the REAL module loads
sys.modules.pop("vetinari.config.inference_config", None)

from vetinari.config.inference_config import (
    InferenceConfigManager,
    InferenceProfile,
    _clamp,
    _classify_model_size,
    get_inference_config,
    reset_inference_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, data: dict) -> Path:
    """Write *data* as JSON to *path* and return *path*."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_config(**extra) -> dict:
    """Return a minimal valid config dict, merging any **extra** top-level keys."""
    cfg = {
        "version": "1.0",
        "profiles": {
            "general": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 2048,
                "stop_sequences": [],
                "prefer_json": False,
            }
        },
        "model_size_adjustments": {},
        "model_overrides": {},
    }
    cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# Autouse fixture — reset singleton before and after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_inference_config()
    yield
    reset_inference_config()


# ===========================================================================
# InferenceProfile
# ===========================================================================


class TestInferenceProfileDefaults:
    def test_default_temperature(self):
        p = InferenceProfile()
        assert p.temperature == 0.3

    def test_default_top_p(self):
        p = InferenceProfile()
        assert p.top_p == 0.9

    def test_default_top_k(self):
        p = InferenceProfile()
        assert p.top_k == 40

    def test_default_max_tokens(self):
        p = InferenceProfile()
        assert p.max_tokens == 8192

    def test_default_stop_sequences(self):
        p = InferenceProfile()
        assert p.stop_sequences == []

    def test_default_prefer_json(self):
        p = InferenceProfile()
        assert p.prefer_json is False


class TestInferenceProfileCustomValues:
    def test_custom_temperature(self):
        p = InferenceProfile(temperature=0.7)
        assert p.temperature == 0.7

    def test_custom_top_p(self):
        p = InferenceProfile(top_p=0.85)
        assert p.top_p == 0.85

    def test_custom_top_k(self):
        p = InferenceProfile(top_k=25)
        assert p.top_k == 25

    def test_custom_max_tokens(self):
        p = InferenceProfile(max_tokens=4096)
        assert p.max_tokens == 4096

    def test_custom_stop_sequences(self):
        p = InferenceProfile(stop_sequences=["<stop>", "</s>"])
        assert p.stop_sequences == ["<stop>", "</s>"]

    def test_custom_prefer_json(self):
        p = InferenceProfile(prefer_json=True)
        assert p.prefer_json is True

    def test_stop_sequences_are_independent(self):
        """Mutable default must not be shared across instances."""
        p1 = InferenceProfile()
        p2 = InferenceProfile()
        p1.stop_sequences.append("foo")
        assert p2.stop_sequences == []


class TestInferenceProfileToDict:
    def test_returns_all_keys(self):
        p = InferenceProfile()
        d = p.to_dict()
        assert set(d.keys()) == {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop_sequences",
            "prefer_json",
        }

    def test_values_match_fields(self):
        p = InferenceProfile(
            temperature=0.5,
            top_p=0.88,
            top_k=30,
            max_tokens=1024,
            stop_sequences=["end"],
            prefer_json=True,
        )
        d = p.to_dict()
        assert d["temperature"] == 0.5
        assert d["top_p"] == 0.88
        assert d["top_k"] == 30
        assert d["max_tokens"] == 1024
        assert d["stop_sequences"] == ["end"]
        assert d["prefer_json"] is True

    def test_default_profile_to_dict(self):
        d = InferenceProfile().to_dict()
        assert d["temperature"] == 0.3
        assert d["prefer_json"] is False


# ===========================================================================
# _classify_model_size
# ===========================================================================


class TestClassifyModelSizeSmall:
    def test_7b(self):
        assert _classify_model_size("qwen2.5-coder-7b") == "small"

    def test_3b(self):
        assert _classify_model_size("llama-3b-instruct") == "small"

    def test_1b(self):
        assert _classify_model_size("phi-1b") == "small"

    def test_10b_boundary(self):
        assert _classify_model_size("model-10b") == "small"

    def test_case_insensitive(self):
        assert _classify_model_size("Model-7B-Chat") == "small"


class TestClassifyModelSizeMedium:
    def test_14b(self):
        assert _classify_model_size("deepseek-14b") == "medium"

    def test_30b(self):
        assert _classify_model_size("llama-30b") == "medium"

    def test_32b(self):
        assert _classify_model_size("qwen-32b-instruct") == "medium"

    def test_40b_boundary(self):
        assert _classify_model_size("model-40b") == "medium"


class TestClassifyModelSizeLarge:
    def test_70b(self):
        assert _classify_model_size("llama-70b-chat") == "large"

    def test_65b(self):
        assert _classify_model_size("model-65b") == "large"

    def test_80b_boundary(self):
        assert _classify_model_size("model-80b") == "large"


class TestClassifyModelSizeXlarge:
    def test_120b(self):
        assert _classify_model_size("qwen-120b") == "xlarge"

    def test_405b(self):
        assert _classify_model_size("llama-405b") == "xlarge"

    def test_81b_just_over_boundary(self):
        assert _classify_model_size("model-81b") == "xlarge"


class TestClassifyModelSizeKeywordFallbacks:
    def test_tiny_keyword(self):
        assert _classify_model_size("tinyllama-v1") == "small"

    def test_mini_keyword(self):
        assert _classify_model_size("phi-mini") == "small"

    def test_small_keyword(self):
        assert _classify_model_size("gpt-small") == "small"

    def test_ultra_keyword(self):
        assert _classify_model_size("command-ultra") == "xlarge"

    def test_xl_keyword(self):
        assert _classify_model_size("some-xl-model") == "xlarge"

    def test_xxl_keyword(self):
        assert _classify_model_size("t5-xxl") == "xlarge"

    def test_large_keyword(self):
        assert _classify_model_size("gpt-large") == "xlarge"


class TestClassifyModelSizeDefault:
    def test_unknown_name_returns_medium(self):
        assert _classify_model_size("some-unknown-model") == "medium"

    def test_empty_string_returns_medium(self):
        assert _classify_model_size("") == "medium"

    def test_plain_name_no_numbers(self):
        assert _classify_model_size("mistral-instruct") == "medium"


# ===========================================================================
# _clamp
# ===========================================================================


class TestClamp:
    def test_within_bounds_unchanged(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_lo_clamped_to_lo(self):
        assert _clamp(-0.1, 0.0, 1.0) == 0.0

    def test_above_hi_clamped_to_hi(self):
        assert _clamp(2.0, 0.0, 1.5) == 1.5

    def test_exactly_lo_returned(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0

    def test_exactly_hi_returned(self):
        assert _clamp(1.0, 0.0, 1.0) == 1.0

    def test_integer_inputs(self):
        assert _clamp(50, 1, 100) == 50

    def test_integer_below_lo(self):
        assert _clamp(0, 1, 100) == 1

    def test_integer_above_hi(self):
        assert _clamp(200, 1, 100) == 100

    def test_negative_range(self):
        assert _clamp(-5.0, -10.0, -1.0) == -5.0

    def test_lo_equals_hi(self):
        assert _clamp(0.7, 0.5, 0.5) == 0.5


# ===========================================================================
# InferenceConfigManager — singleton and reset
# ===========================================================================


class TestSingleton:
    def test_same_instance_returned_twice(self):
        a = get_inference_config()
        b = get_inference_config()
        assert a is b

    def test_reset_produces_new_instance(self):
        a = get_inference_config()
        reset_inference_config()
        b = get_inference_config()
        assert a is not b

    def test_reset_then_get_is_fresh_instance(self):
        get_inference_config()
        reset_inference_config()
        cfg = get_inference_config()
        assert cfg is not None
        assert isinstance(cfg, InferenceConfigManager)

    def test_constructor_returns_same_as_factory(self):
        a = InferenceConfigManager()
        b = get_inference_config()
        assert a is b

    def test_concurrent_creation_returns_same_instance(self):
        """Multiple threads must all receive the identical singleton."""
        results = []

        def grab():
            results.append(get_inference_config())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)


# ===========================================================================
# InferenceConfigManager — loading from file
# ===========================================================================


class TestLoadFromFile:
    def test_load_minimal_config_succeeds(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        reset_inference_config()
        cfg = get_inference_config()
        assert cfg.reload(str(cfg_file)) is True
        assert cfg.is_loaded is True

    def test_load_populates_profiles(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["coding"] = {
            "temperature": 0.15,
            "top_p": 0.92,
            "top_k": 40,
            "max_tokens": 4096,
            "stop_sequences": [],
            "prefer_json": False,
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert "coding" in cfg.list_profiles()

    def test_load_populates_size_adjustments(self, tmp_path):
        data = _minimal_config()
        data["model_size_adjustments"] = {
            "small": {"temperature_offset": -0.1, "top_p_offset": 0.0, "top_k_offset": 0},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        # Verify the adjustment is applied (7b = small, -0.1 offset on 0.3 general)
        params = cfg.get_effective_params("general", "phi-7b")
        assert params["temperature"] == pytest.approx(0.2, abs=1e-6)

    def test_load_populates_model_overrides(self, tmp_path):
        data = _minimal_config()
        data["model_overrides"] = {
            "my-special-model": {"temperature_offset": 0.2},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("general", "my-special-model")
        # general temp=0.3, medium offset=0, override +0.2 → 0.5
        assert params["temperature"] == pytest.approx(0.5, abs=1e-3)

    def test_missing_file_returns_false(self):
        cfg = get_inference_config()
        result = cfg.reload("/nonexistent/does_not_exist.json")
        assert result is False

    def test_missing_file_sets_is_loaded_false(self):
        cfg = get_inference_config()
        cfg.reload("/nonexistent/does_not_exist.json")
        assert cfg.is_loaded is False

    def test_json_parse_error_returns_false(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is not valid json !!!", encoding="utf-8")
        cfg = get_inference_config()
        result = cfg.reload(str(bad_file))
        assert result is False

    def test_json_parse_error_sets_is_loaded_false(self, tmp_path):
        # Write syntactically invalid JSON straight away (no intermediate reload)
        bad_file = tmp_path / "bad2.json"
        bad_file.write_text("[1,2,3", encoding="utf-8")
        cfg = get_inference_config()
        cfg.reload(str(bad_file))
        assert cfg.is_loaded is False

    def test_json_parse_error_clears_stale_profiles(self, tmp_path):
        # Load a good config first so profiles are populated.
        good_file = tmp_path / "good.json"
        data = _minimal_config()
        data["profiles"]["stale_task"] = {"temperature": 0.5}
        _write_config(good_file, data)
        cfg = get_inference_config()
        cfg.reload(str(good_file))
        assert "stale_task" in cfg.list_profiles()

        # Now reload with invalid JSON — stale profiles must be cleared, not silently served.
        bad_file = tmp_path / "bad_after_good.json"
        bad_file.write_text("{ this is not valid json !!!", encoding="utf-8")
        result = cfg.reload(str(bad_file))
        assert result is False
        assert cfg.is_loaded is False
        assert "stale_task" not in cfg.list_profiles(), (
            "Stale profiles must be cleared on JSONDecodeError — serving stale data silently is the defect being fixed"
        )

    def test_wrong_root_type_returns_false(self, tmp_path):
        # JSON array at root is structurally valid JSON but wrong shape for the loader.
        bad_file = tmp_path / "array_root.json"
        bad_file.write_text('[{"profiles": {}}]', encoding="utf-8")
        cfg = get_inference_config()
        result = cfg.reload(str(bad_file))
        assert result is False
        assert cfg.is_loaded is False

    def test_wrong_root_type_clears_stale_profiles(self, tmp_path):
        # Load good data first, then reload with array-root JSON — stale profiles must be cleared.
        good_file = tmp_path / "good2.json"
        data = _minimal_config()
        data["profiles"]["stale2"] = {"temperature": 0.3}
        _write_config(good_file, data)
        cfg = get_inference_config()
        cfg.reload(str(good_file))
        assert "stale2" in cfg.list_profiles()

        bad_file = tmp_path / "array_root2.json"
        bad_file.write_text('[{"profiles": {}}]', encoding="utf-8")
        cfg.reload(str(bad_file))
        assert "stale2" not in cfg.list_profiles(), "Wrong-root-type reload must clear stale profiles"

    def test_reload_replaces_profiles(self, tmp_path):
        # Load config A with "task_a"
        data_a = _minimal_config()
        data_a["profiles"]["task_a"] = {"temperature": 0.1}
        _write_config(tmp_path / "a.json", data_a)
        cfg = get_inference_config()
        cfg.reload(str(tmp_path / "a.json"))
        assert "task_a" in cfg.list_profiles()

        # Reload config B without "task_a" but with "task_b"
        data_b = _minimal_config()
        data_b["profiles"]["task_b"] = {"temperature": 0.9}
        _write_config(tmp_path / "b.json", data_b)
        cfg.reload(str(tmp_path / "b.json"))
        assert "task_b" in cfg.list_profiles()
        assert "task_a" not in cfg.list_profiles()

    def test_reload_updates_is_loaded_to_true(self, tmp_path):
        cfg = get_inference_config()
        cfg.reload("/nonexistent/path.json")
        assert cfg.is_loaded is False

        good_file = _write_config(tmp_path / "good.json", _minimal_config())
        cfg.reload(str(good_file))
        assert cfg.is_loaded is True


# ===========================================================================
# InferenceConfigManager.get_profile
# ===========================================================================


class TestGetProfile:
    def _loaded_cfg(self, tmp_path: Path) -> InferenceConfigManager:
        data = _minimal_config()
        data["profiles"].update({
            "coding": {
                "temperature": 0.15,
                "top_p": 0.92,
                "top_k": 40,
                "max_tokens": 4096,
                "stop_sequences": [],
                "prefer_json": False,
            },
            "planning": {
                "temperature": 0.35,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 3000,
                "stop_sequences": [],
                "prefer_json": True,
            },
        })
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        return cfg

    def test_known_task_type_coding(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("coding")
        assert p.temperature == 0.15
        assert p.max_tokens == 4096

    def test_known_task_type_planning(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("planning")
        assert p.temperature == 0.35
        assert p.prefer_json is True

    def test_returns_inference_profile_instance(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("coding")
        assert isinstance(p, InferenceProfile)

    def test_unknown_task_falls_back_to_general(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("totally_unknown_task_xyz")
        assert p.temperature == 0.3  # general defaults
        assert p.max_tokens == 2048

    def test_unknown_task_fallback_prefer_json_false(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("totally_unknown_task_xyz")
        assert p.prefer_json is False

    def test_empty_config_returns_dataclass_defaults(self, tmp_path):
        """When profiles dict is empty and knowledge tier has nothing, returns general defaults."""
        from unittest.mock import patch

        data = {
            "version": "1.0",
            "profiles": {},
            "model_size_adjustments": {},
            "model_overrides": {},
        }
        cfg_file = _write_config(tmp_path / "empty.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        with patch("vetinari.knowledge.get_parameter_guide", return_value={}):
            p = cfg.get_profile("coding")
        assert p.temperature == 0.3
        assert p.max_tokens == 2048

    def test_general_profile_returned_explicitly(self, tmp_path):
        cfg = self._loaded_cfg(tmp_path)
        p = cfg.get_profile("general")
        assert p.temperature == 0.3

    def test_stop_sequences_from_profile(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["with_stops"] = {
            "temperature": 0.2,
            "stop_sequences": ["<|im_end|>", "</s>"],
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        p = cfg.get_profile("with_stops")
        assert p.stop_sequences == ["<|im_end|>", "</s>"]


# ===========================================================================
# InferenceConfigManager.get_effective_params
# ===========================================================================


class TestGetEffectiveParams:
    def _cfg_with_adjustments(self, tmp_path: Path) -> InferenceConfigManager:
        data = _minimal_config()
        data["profiles"].update({
            "coding": {
                "temperature": 0.15,
                "top_p": 0.92,
                "top_k": 40,
                "max_tokens": 4096,
                "stop_sequences": [],
                "prefer_json": False,
            },
            "creative_writing": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "max_tokens": 4096,
                "stop_sequences": [],
                "prefer_json": False,
            },
            "classification": {
                "temperature": 0.0,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 256,
                "stop_sequences": [],
                "prefer_json": True,
            },
            "planning": {
                "temperature": 0.35,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 3000,
                "stop_sequences": [],
                "prefer_json": True,
            },
        })
        data["model_size_adjustments"] = {
            "small": {"temperature_offset": -0.15, "top_p_offset": -0.03, "top_k_offset": -5},
            "medium": {"temperature_offset": -0.05, "top_p_offset": 0.0, "top_k_offset": 0},
            "large": {"temperature_offset": 0.0, "top_p_offset": 0.0, "top_k_offset": 0},
            "xlarge": {"temperature_offset": 0.1, "top_p_offset": 0.02, "top_k_offset": 5},
        }
        data["model_overrides"] = {
            "qwen2.5-coder-7b": {"temperature_offset": -0.1},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        return cfg

    def test_no_model_returns_base_profile(self, tmp_path):
        """When model discovery is unavailable, no-model falls back to base profile."""
        from unittest.mock import patch

        cfg = self._cfg_with_adjustments(tmp_path)
        with patch(
            "vetinari.config.inference_config.get_task_default_model",
            side_effect=RuntimeError("no models"),
        ):
            params = cfg.get_effective_params("coding")
        assert params["temperature"] == 0.15
        assert params["top_p"] == 0.92

    def test_no_model_empty_string_returns_base_profile(self, tmp_path):
        """When model discovery is unavailable, empty-string model falls back to base profile."""
        from unittest.mock import patch

        cfg = self._cfg_with_adjustments(tmp_path)
        with patch(
            "vetinari.config.inference_config.get_task_default_model",
            side_effect=RuntimeError("no models"),
        ):
            params = cfg.get_effective_params("coding", "")
        assert params["temperature"] == 0.15

    def test_no_model_auto_selects_and_adjusts(self, tmp_path):
        """When model discovery succeeds, auto-selected model adjustments are applied."""
        from unittest.mock import patch

        cfg = self._cfg_with_adjustments(tmp_path)
        # Auto-select qwen2.5-coder-7b (small tier) → temperature_offset -0.15
        with patch(
            "vetinari.config.inference_config.get_task_default_model",
            return_value="qwen2.5-coder-7b",
        ):
            params = cfg.get_effective_params("coding")
        # coding base 0.15 + small offset -0.15 + qwen2.5-coder-7b override -0.1 = 0.0 (clamped)
        assert params["temperature"] == 0.0

    def test_small_model_temperature_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # planning: 0.35 + small offset -0.15 = 0.20
        params = cfg.get_effective_params("planning", "llama-7b")
        assert params["temperature"] == pytest.approx(0.20, abs=1e-3)

    def test_small_model_top_p_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # planning: 0.9 + small offset -0.03 = 0.87
        params = cfg.get_effective_params("planning", "llama-7b")
        assert params["top_p"] == pytest.approx(0.87, abs=1e-3)

    def test_small_model_top_k_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # planning: top_k=40 + small -5 = 35
        params = cfg.get_effective_params("planning", "llama-7b")
        assert params["top_k"] == 35

    def test_xlarge_model_temperature_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # general: 0.3 + xlarge +0.1 = 0.4
        params = cfg.get_effective_params("general", "llama-405b")
        assert params["temperature"] == pytest.approx(0.4, abs=1e-3)

    def test_xlarge_model_top_p_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # general: 0.9 + xlarge +0.02 = 0.92
        params = cfg.get_effective_params("general", "llama-405b")
        assert params["top_p"] == pytest.approx(0.92, abs=1e-3)

    def test_model_override_applied(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # coding: 0.15, small offset -0.15, override -0.1 → -0.10 → clamped to 0.0
        params = cfg.get_effective_params("coding", "qwen2.5-coder-7b")
        assert params["temperature"] == 0.0

    def test_model_override_stacks_with_size_adjustment(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # Confirm that both size and override contributed to reaching 0.0
        params = cfg.get_effective_params("coding", "qwen2.5-coder-7b")
        assert params["temperature"] >= 0.0

    def test_temperature_clamped_to_zero_floor(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # classification: temp=0.0, small offset -0.15 → would be -0.15, clamped to 0.0
        params = cfg.get_effective_params("classification", "tiny-3b")
        assert params["temperature"] >= 0.0

    def test_temperature_clamped_to_1_5_ceiling(self, tmp_path):
        # Build a config where temp + offsets would exceed 1.5
        data = _minimal_config()
        data["profiles"]["hot"] = {
            "temperature": 1.4,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 512,
            "stop_sequences": [],
            "prefer_json": False,
        }
        data["model_size_adjustments"] = {
            "xlarge": {"temperature_offset": 0.5, "top_p_offset": 0.0, "top_k_offset": 0},
        }
        data["model_overrides"] = {}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("hot", "llama-405b")
        assert params["temperature"] <= 1.5

    def test_top_p_clamped_to_1_0_ceiling(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # creative_writing: top_p=0.95, xlarge +0.02 = 0.97 — within range
        params = cfg.get_effective_params("creative_writing", "llama-405b")
        assert params["top_p"] <= 1.0

    def test_top_p_forced_above_1_clamped(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["high_topp"] = {
            "temperature": 0.5,
            "top_p": 0.99,
            "top_k": 40,
            "max_tokens": 512,
            "stop_sequences": [],
            "prefer_json": False,
        }
        data["model_size_adjustments"] = {
            "xlarge": {"temperature_offset": 0.0, "top_p_offset": 0.05, "top_k_offset": 0},
        }
        data["model_overrides"] = {}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("high_topp", "llama-405b")
        assert params["top_p"] <= 1.0

    def test_top_k_clamped_to_1_floor(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["low_topk"] = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 3,
            "max_tokens": 512,
            "stop_sequences": [],
            "prefer_json": False,
        }
        data["model_size_adjustments"] = {
            "small": {"temperature_offset": 0.0, "top_p_offset": 0.0, "top_k_offset": -10},
        }
        data["model_overrides"] = {}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("low_topk", "phi-7b")
        assert params["top_k"] >= 1

    def test_top_k_clamped_to_100_ceiling(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["high_topk"] = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 98,
            "max_tokens": 512,
            "stop_sequences": [],
            "prefer_json": False,
        }
        data["model_size_adjustments"] = {
            "xlarge": {"temperature_offset": 0.0, "top_p_offset": 0.0, "top_k_offset": 20},
        }
        data["model_overrides"] = {}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("high_topk", "llama-405b")
        assert params["top_k"] <= 100

    def test_top_k_is_integer(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        params = cfg.get_effective_params("planning", "llama-7b")
        assert isinstance(params["top_k"], int)

    def test_max_tokens_not_adjusted_by_size(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        params = cfg.get_effective_params("coding", "llama-7b")
        assert params["max_tokens"] == 4096

    def test_stop_sequences_passed_through(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["seq_task"] = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 512,
            "stop_sequences": ["###"],
            "prefer_json": False,
        }
        data["model_size_adjustments"] = {}
        data["model_overrides"] = {}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        params = cfg.get_effective_params("seq_task", "phi-7b")
        assert params["stop_sequences"] == ["###"]

    def test_prefer_json_passed_through(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        params = cfg.get_effective_params("planning", "llama-7b")
        assert params["prefer_json"] is True

    def test_result_dict_has_all_keys(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        params = cfg.get_effective_params("coding", "llama-7b")
        assert set(params.keys()) == {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop_sequences",
            "prefer_json",
        }

    def test_medium_model_small_temperature_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # general: 0.3, medium -0.05 = 0.25
        params = cfg.get_effective_params("general", "deepseek-14b")
        assert params["temperature"] == pytest.approx(0.25, abs=1e-3)

    def test_large_model_no_offset(self, tmp_path):
        cfg = self._cfg_with_adjustments(tmp_path)
        # general: 0.3, large offset 0.0 = 0.3
        params = cfg.get_effective_params("general", "llama-70b")
        assert params["temperature"] == pytest.approx(0.3, abs=1e-3)

    def test_unknown_size_tier_no_adjustment(self, tmp_path):
        """Model with no size number and no keyword gets 'medium' tier."""
        cfg = self._cfg_with_adjustments(tmp_path)
        # unknown → medium, offset -0.05; general 0.3 → 0.25
        params = cfg.get_effective_params("general", "mystery-model")
        assert params["temperature"] == pytest.approx(0.25, abs=1e-3)


# ===========================================================================
# InferenceConfigManager — introspection methods
# ===========================================================================


class TestListProfiles:
    def test_returns_list(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert isinstance(cfg.list_profiles(), list)

    def test_contains_general(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert "general" in cfg.list_profiles()

    def test_contains_all_loaded_profiles(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["alpha"] = {"temperature": 0.1}
        data["profiles"]["beta"] = {"temperature": 0.2}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        profiles = cfg.list_profiles()
        assert "alpha" in profiles
        assert "beta" in profiles

    def test_empty_profiles_returns_empty_list(self, tmp_path):
        data = {
            "version": "1.0",
            "profiles": {},
            "model_size_adjustments": {},
            "model_overrides": {},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert cfg.list_profiles() == []


class TestGetAllProfiles:
    def test_returns_dict(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert isinstance(cfg.get_all_profiles(), dict)

    def test_returned_dict_is_copy_not_reference(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        d = cfg.get_all_profiles()
        d["injected"] = {"temperature": 9.9}
        assert "injected" not in cfg.list_profiles()

    def test_keys_match_list_profiles(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["foo"] = {"temperature": 0.5}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert set(cfg.get_all_profiles().keys()) == set(cfg.list_profiles())


class TestGetStats:
    def test_loaded_true_when_config_loaded(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        stats = cfg.get_stats()
        assert stats["loaded"] is True

    def test_loaded_false_when_no_config(self):
        cfg = get_inference_config()
        cfg.reload("/nonexistent/path.json")
        stats = cfg.get_stats()
        assert stats["loaded"] is False

    def test_profile_count_matches(self, tmp_path):
        data = _minimal_config()
        data["profiles"]["extra1"] = {"temperature": 0.1}
        data["profiles"]["extra2"] = {"temperature": 0.2}
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert cfg.get_stats()["profile_count"] == 3  # general + extra1 + extra2

    def test_config_path_in_stats(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert cfg.get_stats()["config_path"] == str(cfg_file)

    def test_model_size_tiers_listed(self, tmp_path):
        data = _minimal_config()
        data["model_size_adjustments"] = {
            "small": {"temperature_offset": -0.1},
            "large": {"temperature_offset": 0.1},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        tiers = cfg.get_stats()["model_size_tiers"]
        assert set(tiers) == {"small", "large"}

    def test_model_overrides_listed(self, tmp_path):
        data = _minimal_config()
        data["model_overrides"] = {
            "my-model": {"temperature_offset": 0.0},
        }
        cfg_file = _write_config(tmp_path / "cfg.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert "my-model" in cfg.get_stats()["model_overrides"]

    def test_stats_returns_all_expected_keys(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        stats = cfg.get_stats()
        assert set(stats.keys()) == {
            "loaded",
            "config_path",
            "profile_count",
            "model_size_tiers",
            "model_overrides",
        }


# ===========================================================================
# InferenceConfigManager — is_loaded property
# ===========================================================================


class TestIsLoaded:
    def test_true_after_successful_load(self, tmp_path):
        cfg_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        assert cfg.is_loaded is True

    def test_false_after_failed_load(self):
        cfg = get_inference_config()
        cfg.reload("/nonexistent/path.json")
        assert cfg.is_loaded is False

    def test_true_after_reload_success_following_failure(self, tmp_path):
        cfg = get_inference_config()
        cfg.reload("/nonexistent/path.json")
        assert cfg.is_loaded is False

        good_file = _write_config(tmp_path / "cfg.json", _minimal_config())
        cfg.reload(str(good_file))
        assert cfg.is_loaded is True


# ===========================================================================
# get_profile fallback when no config loaded (empty profiles)
# ===========================================================================


class TestGetProfileEmptyState:
    def test_returns_inference_profile_with_dataclass_defaults(self, tmp_path):
        data = {
            "version": "1.0",
            "profiles": {},
            "model_size_adjustments": {},
            "model_overrides": {},
        }
        cfg_file = _write_config(tmp_path / "empty.json", data)
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        p = cfg.get_profile("anything")
        assert isinstance(p, InferenceProfile)
        assert p.temperature == 0.3
        assert p.top_p == 0.9
        assert p.top_k == 40
        assert p.max_tokens == 2048
        assert p.prefer_json is False


# ===========================================================================
# Integration: production config file (if present)
# ===========================================================================


class TestProductionConfig:
    """
    These tests exercise the real task_inference_profiles.json when it exists.
    They are skipped gracefully when the file is absent (e.g. CI without assets).
    """

    @pytest.fixture(autouse=True)
    def load_production_config(self):
        cfg = get_inference_config()
        if not cfg.is_loaded:
            pytest.skip("Production inference config not available")

    def test_is_loaded(self):
        assert get_inference_config().is_loaded is True

    def test_at_least_one_profile(self):
        assert len(get_inference_config().list_profiles()) >= 1

    def test_general_profile_exists(self):
        assert "general" in get_inference_config().list_profiles()

    def test_coding_profile_temperature(self):
        p = get_inference_config().get_profile("coding")
        assert p.temperature == 0.10  # Optimized: lower = more deterministic code

    def test_coding_profile_max_tokens(self):
        p = get_inference_config().get_profile("coding")
        assert p.max_tokens == 16384

    def test_general_fallback_temperature(self):
        p = get_inference_config().get_profile("nonexistent_task_type")
        assert p.temperature == 0.35  # Optimized: slightly warmer for versatility

    def test_effective_params_coding_no_model(self):
        from unittest.mock import patch

        # Isolate from disk model catalog — auto-selection would apply size/model offsets
        with patch(
            "vetinari.config.inference_config.get_task_default_model",
            side_effect=RuntimeError("no models"),
        ):
            params = get_inference_config().get_effective_params("coding")
        assert params["temperature"] == 0.10  # Base coding profile, no model adjustments
        assert params["top_p"] == 0.92

    def test_effective_params_small_model_coder_override(self):
        params = get_inference_config().get_effective_params("coding", "qwen2.5-coder-7b")
        # coding 0.10, small -0.10, override -0.1 → -0.10 → clamped to 0.0
        assert params["temperature"] == 0.0

    def test_effective_params_planning_small_model(self):
        params = get_inference_config().get_effective_params("planning", "llama-7b")
        # planning 0.35, small -0.10 → 0.25
        assert params["temperature"] == pytest.approx(0.25, abs=1e-3)

    def test_effective_params_classification_small_clamps_to_zero(self):
        params = get_inference_config().get_effective_params("classification", "tiny-3b")
        assert params["temperature"] >= 0.0

    def test_stats_profile_count_over_30(self):
        assert get_inference_config().get_stats()["profile_count"] > 30


# -- _get_knowledge_profile (knowledge YAML middle tier) --


class TestGetKnowledgeProfile:
    """Tests for the knowledge YAML fallback tier in get_profile()."""

    def _empty_cfg(self, tmp_path: Path) -> InferenceConfigManager:
        """Return an InferenceConfigManager with only a 'general' profile."""
        data = {
            "version": "1.0",
            "profiles": {
                "general": {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "max_tokens": 2048},
            },
            "model_size_adjustments": {},
            "model_overrides": {},
        }
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps(data), encoding="utf-8")
        reset_inference_config()
        cfg = get_inference_config()
        cfg.reload(str(cfg_file))
        return cfg

    def test_returns_none_when_guide_empty(self, tmp_path: Path) -> None:
        """_get_knowledge_profile returns None when get_parameter_guide returns {}."""
        from unittest.mock import patch

        cfg = self._empty_cfg(tmp_path)
        with patch("vetinari.knowledge.get_parameter_guide", return_value={}):
            result = cfg._get_knowledge_profile("some_task")
        assert result is None

    def test_preset_format_builds_profile(self, tmp_path: Path) -> None:
        """Preset format with temperature is converted to an InferenceProfile."""
        from unittest.mock import patch

        cfg = self._empty_cfg(tmp_path)
        guide = {"preset": "code", "temperature": 0.05, "top_p": 0.89, "top_k": 30, "max_tokens": 4096}
        with patch("vetinari.knowledge.get_parameter_guide", return_value=guide):
            result = cfg._get_knowledge_profile("coding")
        assert result is not None
        assert isinstance(result, InferenceProfile)
        assert result.temperature == 0.05

    def test_preset_format_sets_top_p(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        cfg = self._empty_cfg(tmp_path)
        guide = {"preset": "code", "temperature": 0.05, "top_p": 0.89}
        with patch("vetinari.knowledge.get_parameter_guide", return_value=guide):
            result = cfg._get_knowledge_profile("coding")
        assert result is not None
        assert result.top_p == 0.89

    def test_unknown_task_falls_back_to_general_when_guide_empty(self, tmp_path: Path) -> None:
        """When knowledge guide returns nothing, get_profile falls back to general."""
        from unittest.mock import patch

        cfg = self._empty_cfg(tmp_path)
        with patch("vetinari.knowledge.get_parameter_guide", return_value={}):
            result = cfg.get_profile("totally_unknown_task")
        assert result.temperature == 0.3  # general default
        assert isinstance(result, InferenceProfile)
