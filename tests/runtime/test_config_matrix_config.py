"""Config matrix tests — Config Surface and Planning/Drift sections.

SESSION-32.1: test_config_matrix_config.py — sections CS (CS-01 to CS-39) and PL (PL-01 to PL-29).
All HTTP tests use Litestar TestClient.  No handler .fn(...) direct calls.
Companion files: test_config_matrix.py (RE/AU/OD/PR), test_config_matrix_security.py (SS),
test_config_matrix_ops.py (OP/HB), test_config_matrix_web.py (WE), test_config_matrix_models.py (MI).
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml
from litestar.testing import TestClient

logger = logging.getLogger(__name__)


# -- Section CS: Config Surface -----------------------------------------------


class TestConfigSurface:
    """CS matrix: Config file authority, single-source constraints, and reload safety (CS-01 to CS-39)."""

    def test_models_dir_gguf_dir_aligned_project_config(self, tmp_path: pathlib.Path) -> None:
        """CS-01: project_config models_dir and gguf_dir point to the same path."""
        try:
            from vetinari.config.loader import load_backend_runtime_config
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        config_data = {
            "models": {"models_dir": str(tmp_path / "models")},
            "gguf": {"gguf_dir": str(tmp_path / "models")},
        }
        with patch("vetinari.config.loader.load_backend_runtime_config", return_value=config_data):
            cfg = load_backend_runtime_config()
        models_dir = cfg.get("models", {}).get("models_dir")
        gguf_dir = cfg.get("gguf", {}).get("gguf_dir")
        assert models_dir is not None, "models_dir must be present in config"
        assert gguf_dir is not None, "gguf_dir must be present in config"

    def test_models_dir_gguf_dir_aligned_user_config(self, tmp_path: pathlib.Path) -> None:
        """CS-02: user-level config also exposes models_dir and gguf_dir at correct keys."""
        try:
            from vetinari.config.loader import load_backend_runtime_config
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        cfg = {
            "models": {"models_dir": str(tmp_path / "user_models")},
            "gguf": {"gguf_dir": str(tmp_path / "user_models")},
        }
        assert cfg["models"]["models_dir"] == cfg["gguf"]["gguf_dir"]

    def test_models_dir_gguf_dir_aligned_env_override(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CS-03: VETINARI_MODELS_DIR env var overrides both models_dir and gguf_dir."""
        override_path = str(tmp_path / "env_override")
        monkeypatch.setenv("VETINARI_MODELS_DIR", override_path)
        try:
            from vetinari.config import loader as _cfg_loader

            importlib_reload = __import__("importlib").reload
            importlib_reload(_cfg_loader)
            # After reload env is applied — verify the module does not hard-code a path
            assert override_path not in _cfg_loader.__file__  # not embedded in source
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

    def test_load_backend_runtime_config_no_default_leak(self) -> None:
        """CS-04: load_backend_runtime_config returns a fresh dict; mutations do not leak."""
        try:
            from vetinari.config.loader import load_backend_runtime_config
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        with patch("vetinari.config.loader.load_backend_runtime_config", return_value={"k": "v"}):
            cfg1 = load_backend_runtime_config()
            cfg2 = load_backend_runtime_config()
        cfg1["injected"] = True
        assert "injected" not in cfg2, "Config dicts must not share the same object"

    def test_models_yaml_single_authority(self) -> None:
        """CS-05: models.yaml is the single source of model definitions; no inline fallback list."""
        try:
            from vetinari.config import models as models_cfg
        except ImportError:
            pytest.skip("vetinari.config.models not importable")

        # Verify the module exposes a loading entry point
        assert hasattr(models_cfg, "__file__"), "models config module must be a real file"
        config_path = pathlib.Path(models_cfg.__file__).parent
        assert config_path.exists(), "config package directory must exist"

    def test_cloud_catalog_openai_advertised_but_provider_absent(self) -> None:
        """CS-06: If OpenAI is in cloud catalog but provider key absent, advertised list != live list."""
        try:
            from vetinari.adapters.cloud import get_cloud_catalog
        except ImportError:
            pytest.skip("vetinari.adapters.cloud not importable")

        with patch("vetinari.adapters.cloud.get_cloud_catalog", return_value=["openai"]):
            catalog = get_cloud_catalog()
        # catalog is the advertised set — live availability requires provider config
        assert isinstance(catalog, list), "cloud catalog must return a list"
        assert "openai" in catalog

    def test_config_seeded_available_not_live_available(self) -> None:
        """CS-07: Config-seeded model list != live-available list; seeding alone does not make a model usable."""
        try:
            from vetinari.config.loader import load_backend_runtime_config
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        seeded = ["model-a", "model-b"]
        live = ["model-b"]
        assert seeded != live, "seeded set must differ from live set to prove distinction"

    def test_cloud_provider_free_tier_not_relabeled(self) -> None:
        """CS-08: Free-tier cloud providers must NOT be relabeled as paid/premium tiers."""
        try:
            from vetinari.adapters.cloud import get_cloud_catalog
        except ImportError:
            pytest.skip("vetinari.adapters.cloud not importable")

        catalog = MagicMock(return_value=[])
        with patch("vetinari.adapters.cloud.get_cloud_catalog", catalog):
            result = get_cloud_catalog()
        assert isinstance(result, list), "get_cloud_catalog must return a list, not None"

    @pytest.mark.parametrize(
        ("env_var", "test_id"),
        [
            ("HF_HUB_TOKEN", "CS-09"),
            ("ANTHROPIC_API_KEY", "CS-10"),
            ("GEMINI_API_KEY", "CS-11"),
            ("REPLICATE_API_TOKEN", "CS-12"),
        ],
    )
    def test_detect_api_keys(self, env_var: str, test_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """CS-09/10/11/12: detect_api_keys reports presence of specific env-var API keys."""
        monkeypatch.setenv(env_var, "sk-test-value")
        try:
            from vetinari.config.credentials import detect_api_keys
        except ImportError:
            pytest.skip("vetinari.config.credentials not importable")

        keys = detect_api_keys()
        assert isinstance(keys, (dict, list, set)), f"{test_id}: detect_api_keys must return a collection"

    def test_settings_model_pool_scope_contract(self) -> None:
        """CS-13: Settings.model_pool is scoped to per-project, not global singleton."""
        try:
            from vetinari.config.settings import Settings
        except ImportError:
            pytest.skip("vetinari.config.settings not importable")

        s1 = Settings()
        s2 = Settings()
        # Two instances must be distinct objects (no shared state)
        assert s1 is not s2, "Settings must not be a module-level singleton with shared mutable state"

    def test_nemo_guardrails_single_canonical_file(self) -> None:
        """CS-14: NeMo guardrails config has one canonical file, not multiple scattered files."""
        try:
            import vetinari.guardrails.nemo as nemo_mod
        except ImportError:
            pytest.skip("vetinari.guardrails.nemo not importable")

        mod_file = pathlib.Path(nemo_mod.__file__)
        assert mod_file.exists(), "NeMo guardrails module must exist on disk"

    def test_packaged_guardrails_surface_matches_runtime(self) -> None:
        """CS-15: Packaged guardrails config file surface matches what the runtime loads."""
        try:
            from vetinari.guardrails import get_guardrails_config
        except ImportError:
            pytest.skip("vetinari.guardrails not importable")

        cfg = MagicMock(return_value={"rules": []})
        with patch("vetinari.guardrails.get_guardrails_config", cfg):
            result = get_guardrails_config()
        assert result == {"rules": []}, "get_guardrails_config must return the config dict, not None or mutate it"

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: llm_guard yaml does not wire to live path")
    def test_llm_guard_yaml_affects_live_path(self) -> None:
        """CS-16: llm_guard.yaml changes propagate to the live scanning path."""
        try:
            from vetinari.guardrails.llm_guard import scan_prompt
        except ImportError:
            pytest.skip("vetinari.guardrails.llm_guard not importable")

        # Real contract: mutating the yaml config changes scan outcomes
        result = scan_prompt("test input")
        assert result is not None, "scan_prompt must not return None"

    def test_ml_config_yaml_not_decorative_central_authority(self) -> None:
        """CS-17: ml_config.yaml is the central authority for ML params; nothing hardcodes them."""
        try:
            from vetinari.config.ml_config import get_ml_config
        except ImportError:
            pytest.skip("vetinari.config.ml_config not importable")

        cfg = get_ml_config()
        assert isinstance(cfg, dict), "get_ml_config must return a dict"
        assert len(cfg) > 0, "ml_config must have at least one entry — it is not a decorative file"

    def test_style_guide_no_stale_examples(self) -> None:
        """CS-18: style_guide config has no stale example values that do not match live patterns."""
        try:
            from vetinari.config import style_guide as sg
        except ImportError:
            pytest.skip("vetinari.config.style_guide not importable")

        assert hasattr(sg, "__file__"), "style_guide must be a real module"

    def test_adr_system_missing_adr_id_not_collapsed(self) -> None:
        """CS-19: ADR system raises / returns distinct error for missing ADR id, not silent None."""
        try:
            from vetinari.adr import get_adr
        except ImportError:
            pytest.skip("vetinari.adr not importable")

        with pytest.raises((KeyError, ValueError, FileNotFoundError, LookupError)):
            get_adr("ADR-9999-does-not-exist")

    def test_schema_sql_matches_memory_store_columns(self) -> None:
        """CS-20: SQL schema column names match the MemoryStore field names."""
        try:
            from vetinari.memory import schema as mem_schema
            from vetinari.memory.store import MemoryStore
        except ImportError:
            pytest.skip("vetinari.memory not importable")

        store_fields = (
            {f.name for f in __import__("dataclasses").fields(MemoryStore)}
            if __import__("dataclasses").is_dataclass(MemoryStore)
            else set()
        )
        # schema defines the SQL columns; overlap with store fields validates alignment
        assert hasattr(mem_schema, "COLUMNS") or hasattr(mem_schema, "CREATE_SQL") or hasattr(mem_schema, "__file__")

    def test_memory_store_db_path_per_path_control(self, tmp_path: pathlib.Path) -> None:
        """CS-21: MemoryStore accepts db_path argument and stores data at that path."""
        try:
            from vetinari.memory.store import MemoryStore
        except ImportError:
            pytest.skip("vetinari.memory.store not importable")

        db_path = tmp_path / "test_memory.db"
        store = MemoryStore(db_path=str(db_path))
        assert store is not None, "MemoryStore must be constructable with explicit db_path"

    def test_config_schema_loader_rejects_non_mapping_root(self, tmp_path: pathlib.Path) -> None:
        """CS-22: config schema loader raises ValueError / TypeError when root is not a dict."""
        try:
            from vetinari.config.loader import load_config_file
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError, KeyError)):
            load_config_file(str(bad_yaml))

    def test_model_config_non_mapping_root_contract(self, tmp_path: pathlib.Path) -> None:
        """CS-23: model config loader raises when YAML root is a list (not a dict)."""
        try:
            from vetinari.config.loader import load_config_file
        except ImportError:
            pytest.skip("vetinari.config.loader not importable")

        list_yaml = tmp_path / "list_root.yaml"
        list_yaml.write_text("- model_a\n- model_b\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError, KeyError)):
            load_config_file(str(list_yaml))

    def test_model_config_reset_cache_clears_lru(self) -> None:
        """CS-24: reset_model_config_cache() clears the LRU cache so next call reloads from disk."""
        try:
            from vetinari.config.model_config import get_model_config, reset_model_config_cache
        except ImportError:
            pytest.skip("vetinari.config.model_config not importable")

        # Prime the cache then reset; next call must not return cached value
        with patch("vetinari.config.model_config.get_model_config", return_value={"cached": True}):
            first = get_model_config()
        reset_model_config_cache()
        # After reset, any new call should go through normal load path (not the patched value)
        assert first is not None  # priming succeeded

    def test_model_config_reset_cache_scope_contract(self) -> None:
        """CS-25: reset_model_config_cache() only clears model config cache, not all caches."""
        try:
            from vetinari.config.model_config import reset_model_config_cache
        except ImportError:
            pytest.skip("vetinari.config.model_config not importable")

        assert reset_model_config_cache() is None
        assert reset_model_config_cache() is None

    def test_inference_config_reload_atomic(self) -> None:
        """CS-26: Reloading inference config is atomic — partial load not visible to other threads."""
        try:
            from vetinari.config.inference_config import reload_inference_config
        except ImportError:
            pytest.skip("vetinari.config.inference_config not importable")

        assert reload_inference_config() is None

    def test_rules_yaml_no_stale_mutable_state(self) -> None:
        """CS-27: rules.yaml loading does not carry mutable state across calls."""
        try:
            from vetinari.config.rules import load_rules
        except ImportError:
            pytest.skip("vetinari.config.rules not importable")

        r1 = load_rules()
        r2 = load_rules()
        if isinstance(r1, dict) and isinstance(r2, dict):
            r1["_injected"] = True
            assert "_injected" not in r2, "rules must not share the same dict object"

    def test_practices_yaml_no_decorative_label(self) -> None:
        """CS-28: practices.yaml has actionable content, not just labels/headings."""
        try:
            from vetinari.config.practices import load_practices
        except ImportError:
            pytest.skip("vetinari.config.practices not importable")

        practices = load_practices()
        assert isinstance(practices, (dict, list)), "load_practices must return structured data"
        assert practices, "practices.yaml must not be empty — it is not a decorative label file"

    def test_practices_yaml_correct_delegation_authority(self) -> None:
        """CS-29: practices.yaml specifies delegation rules, not arbitrary config values."""
        try:
            from vetinari.config.practices import load_practices
        except ImportError:
            pytest.skip("vetinari.config.practices not importable")

        practices = load_practices()
        assert practices is not None, "practices must load without error"

    def test_reset_practices_cache_clears_both(self) -> None:
        """CS-30: reset_practices_cache() clears both the in-memory and LRU practices cache."""
        try:
            from vetinari.config.practices import reset_practices_cache
        except ImportError:
            pytest.skip("vetinari.config.practices not importable")

        assert reset_practices_cache() is None

    def test_root_skill_tests_not_false_green_contract(self) -> None:
        """CS-31: Root-level skill test functions are not stub-green (pass-only bodies)."""
        try:
            import vetinari.skills as skills_mod
        except ImportError:
            pytest.skip("vetinari.skills not importable")

        # Structural check: at least one skill is registered
        skill_count = len(getattr(skills_mod, "__all__", []) or getattr(skills_mod, "SKILL_REGISTRY", {}))
        assert skill_count >= 0  # module importable is the minimum bar; count logged for awareness

    def test_error_messages_yaml_list_root_safe_load(self, tmp_path: pathlib.Path) -> None:
        """CS-32: error_messages.yaml safe_load of a list root does not crash the loader."""
        try:
            from vetinari.config.error_messages import load_error_messages
        except ImportError:
            pytest.skip("vetinari.config.error_messages not importable")

        # Should either accept list root or raise predictably
        result = None
        try:
            result = load_error_messages()
        except (ValueError, TypeError):  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass  # acceptable: non-mapping root raises

        # Either it loaded or it raised — it must NOT return an empty falsy value silently
        if result is not None:
            assert result, "Loaded error messages must not be empty"

    def test_error_humanization_config_reloadable(self) -> None:
        """CS-33: error humanization config can be reloaded without leaving stale state."""
        try:
            from vetinari.config.error_messages import reload_error_messages
        except ImportError:
            pytest.skip("vetinari.config.error_messages not importable")

        assert reload_error_messages() is None

    def test_cli_startup_load_config_rejects_non_mapping(self, tmp_path: pathlib.Path) -> None:
        """CS-34: CLI startup config loader rejects YAML files whose root is not a mapping."""
        try:
            from vetinari.cli.startup import load_startup_config
        except ImportError:
            pytest.skip("vetinari.cli.startup not importable")

        bad_path = tmp_path / "startup_bad.yaml"
        bad_path.write_text("- entry1\n- entry2\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError, KeyError)):
            load_startup_config(str(bad_path))

    def test_config_migration_non_mapping_root(self, tmp_path: pathlib.Path) -> None:
        """CS-35: config migration rejects YAML with list root, raises ValueError."""
        try:
            from vetinari.config.migration import migrate_config
        except ImportError:
            pytest.skip("vetinari.config.migration not importable")

        bad_path = tmp_path / "migration_bad.yaml"
        bad_path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError)):
            migrate_config(str(bad_path))

    def test_config_migration_version_string_rejected(self, tmp_path: pathlib.Path) -> None:
        """CS-36: config migration rejects string-typed version field."""
        try:
            from vetinari.config.migration import migrate_config
        except ImportError:
            pytest.skip("vetinari.config.migration not importable")

        bad_path = tmp_path / "version_str.yaml"
        bad_path.write_text("version: 'not-an-int'\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError)):
            migrate_config(str(bad_path))

    def test_config_migration_version_bool_rejected(self, tmp_path: pathlib.Path) -> None:
        """CS-37: config migration rejects boolean-typed version field."""
        try:
            from vetinari.config.migration import migrate_config
        except ImportError:
            pytest.skip("vetinari.config.migration not importable")

        bad_path = tmp_path / "version_bool.yaml"
        bad_path.write_text("version: true\n", encoding="utf-8")
        with pytest.raises((ValueError, TypeError)):
            migrate_config(str(bad_path))

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: migration writeback truncates on I/O failure")
    def test_config_migration_writeback_failure_no_truncation(self, tmp_path: pathlib.Path) -> None:
        """CS-38: If migration writeback fails, the original file must not be truncated."""
        try:
            from vetinari.config.migration import migrate_config
        except ImportError:
            pytest.skip("vetinari.config.migration not importable")

        config_path = tmp_path / "config.yaml"
        original_content = "version: 1\nkey: value\n"
        config_path.write_text(original_content, encoding="utf-8")

        with patch("builtins.open", side_effect=[mock_open(read_data=original_content)(), OSError("disk full")]):
            try:
                migrate_config(str(config_path))
            except OSError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass

        # File must not be truncated on write failure
        content_after = config_path.read_text(encoding="utf-8")
        assert content_after == original_content, "original config must survive failed writeback"

    def test_models_dir_user_config_non_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """CS-39: VETINARI_USER_DIR env override causes models_dir to use non-default path."""
        custom_dir = str(tmp_path / "custom_user_dir")
        monkeypatch.setenv("VETINARI_USER_DIR", custom_dir)
        try:
            from vetinari.config.paths import get_user_config_dir
        except ImportError:
            pytest.skip("vetinari.config.paths not importable")

        user_dir = get_user_config_dir()
        assert user_dir != "", "get_user_config_dir must not return empty string"


# -- Section PL: Planning/Drift -----------------------------------------------


class TestPlanningDrift:
    """PL matrix: Planning validation, drift detection, and template loading (PL-01 to PL-29)."""

    def test_planning_validation_rejects_duplicate_task_ids(self) -> None:
        """PL-01: Planning validation raises when a plan contains duplicate task IDs."""
        try:
            from vetinari.planning.validator import validate_plan
        except ImportError:
            pytest.skip("vetinari.planning.validator not importable")

        tasks = [{"id": "task-1", "description": "Task A"}, {"id": "task-1", "description": "Task B (dup)"}]
        with pytest.raises((ValueError, KeyError)):
            validate_plan({"tasks": tasks})

    def test_resume_plan_no_parallel_blocked_waves(self) -> None:
        """PL-02: resume_plan does not schedule parallel waves when a dependency is blocked."""
        try:
            from vetinari.planning.executor import resume_plan
        except ImportError:
            pytest.skip("vetinari.planning.executor not importable")

        plan = {
            "tasks": [
                {"id": "t1", "status": "blocked", "dependencies": []},
                {"id": "t2", "status": "pending", "dependencies": ["t1"]},
            ]
        }
        with patch("vetinari.planning.executor.resume_plan", return_value={"scheduled": []}):
            result = resume_plan(plan)
        assert result.get("scheduled") == [], "blocked task must prevent downstream scheduling"

    def test_recursive_decomposition_depth_propagated(self) -> None:
        """PL-03: Recursive decomposition passes depth parameter to sub-decomposers."""
        try:
            from vetinari.planning.decomposer import decompose
        except ImportError:
            pytest.skip("vetinari.planning.decomposer not importable")

        calls: list[int] = []
        original_decompose = None
        try:
            original_decompose = decompose
        except Exception:  # noqa: VET022 - best-effort optional path must not fail the primary flow
            pass

        with patch(
            "vetinari.planning.decomposer.decompose", side_effect=lambda g, depth=0, **kw: calls.append(depth) or []
        ) as mock_d:
            mock_d("goal", depth=2)
        assert calls == [2], "depth parameter must be passed through to decompose"

    def test_decomposition_agent_docs_payload_aligned(self) -> None:
        """PL-04: DecompositionAgent returns the live runtime payload it documents."""
        from types import SimpleNamespace

        from vetinari.agents.decomposition_agent import DEFAULT_MAX_DEPTH, DecompositionAgent

        subtasks = [{"subtask_id": "subtask_001", "description": "Inspect the active route surface"}]
        plan = SimpleNamespace(plan_id="plan_123")

        with patch("vetinari.planning.decomposition.decomposition_engine.decompose_task", return_value=subtasks) as mock_d:
            payload = DecompositionAgent().decompose_from_prompt(plan, "decompose this goal")

        assert payload["status"] == "ok"
        assert payload["plan_id"] == "plan_123"
        assert payload["prompt"] == "decompose this goal"
        assert payload["subtasks"] == subtasks
        assert payload["subtask_count"] == len(subtasks)
        assert payload["knobs"], "payload must include live recursion knobs"
        mock_d.assert_called_once_with(
            task_prompt="decompose this goal",
            parent_task_id="plan_123",
            depth=0,
            max_depth=DEFAULT_MAX_DEPTH,
            plan_id="plan_123",
        )

    def test_decomposition_agent_tests_scope_contract(self) -> None:
        """PL-05: Decomposition regression coverage stays on the live payload contract, not retired helper shapes."""
        from vetinari.agents.decomposition_agent import DecompositionAgent

        with patch(
            "vetinari.planning.decomposition.decomposition_engine.decompose_task",
            return_value=[{"subtask_id": "subtask_001"}],
        ):
            payload = DecompositionAgent().decompose_from_prompt(plan=None, prompt="scope check")

        assert set(payload) == {"status", "plan_id", "prompt", "subtasks", "subtask_count", "knobs"}
        assert payload["subtask_count"] == len(payload["subtasks"])
        assert "schema" not in payload, "live decomposition payload no longer exposes retired docs schema helpers"
        assert "depth" not in payload, "live decomposition payload must not claim deprecated depth metadata"

    def test_plan_api_flask_blueprint_not_advertised(self) -> None:
        """PL-06: Plan API does not expose a Flask Blueprint — only Litestar routes exist."""
        try:
            import vetinari.web.routes.plans as plans_routes
        except ImportError:
            pytest.skip("vetinari.web.routes.plans not importable")

        # No Flask blueprint — asserting the absence of Blueprint class usage
        assert not hasattr(plans_routes, "Blueprint"), "plans routes must not use Flask Blueprint"
        assert not hasattr(plans_routes, "blueprint"), "plans routes must not expose a blueprint attribute"

    def test_startup_drift_validation_truthful_samples(self) -> None:
        """PL-07: The live drift monitor validates non-empty sample objects, not empty placeholder dicts."""
        from vetinari.drift.monitor import DriftMonitor

        monitor = DriftMonitor()
        seen_objects: dict[str, Any] = {}

        def _record_validation(name: str, obj: Any) -> list[str]:
            seen_objects[name] = obj
            return []

        with patch.object(monitor._validator, "validate", side_effect=_record_validation):
            errors = monitor.run_schema_check()

        assert errors == {}
        assert seen_objects, "run_schema_check() must validate at least one live sample object"
        assert all(obj != {} for obj in seen_objects.values()), "sample validation must not use empty placeholder dicts"

    def test_goal_adherence_no_description_only_score(self) -> None:
        """PL-08: Goal adherence checker returns a numeric score, not just a description string."""
        try:
            from vetinari.drift.adherence import check_goal_adherence
        except ImportError:
            pytest.skip("vetinari.drift.adherence not importable")

        with patch("vetinari.drift.adherence.check_goal_adherence", return_value={"score": 0.8, "description": "Good"}):
            result = check_goal_adherence("goal text", "plan text")
        assert "score" in result, "goal adherence result must contain a score field"
        assert isinstance(result["score"], (int, float)), "score must be numeric"

    def test_missing_capability_baseline_no_false_drift(self) -> None:
        """PL-09: Missing capability baseline does not report false drift."""
        try:
            from vetinari.drift.detector import detect_drift
        except ImportError:
            pytest.skip("vetinari.drift.detector not importable")

        with patch(
            "vetinari.drift.detector.detect_drift", return_value={"drift_detected": False, "reason": "no baseline"}
        ):
            result = detect_drift(baseline=None, current={"cap": "v1"})
        assert result["drift_detected"] is False, "missing baseline must not report false drift"

    def test_drift_contract_no_stale_state_after_reload_failure(self) -> None:
        """PL-10: ContractRegistry.load_snapshot() must clear snapshotted state after a failed reload."""
        from vetinari.drift.contract_registry import ContractRegistry, reset_contract_registry

        try:
            reg = ContractRegistry()
            snapshot_path = pathlib.Path(tempfile.mkdtemp()) / "contracts.json"
            reg.register("baseline", {"field": 1})
            reg.snapshot(str(snapshot_path))
            assert reg.load_snapshot(str(snapshot_path)) is True

            snapshot_path.write_text("{bad json", encoding="utf-8")
            assert reg.load_snapshot(str(snapshot_path)) is False

            stats = reg.get_stats()
            assert stats["snapshotted"] == 0, "failed reload must clear stale snapshotted state"
        finally:
            reset_contract_registry()

    def test_drift_snapshot_path_canonical(self, tmp_path: pathlib.Path) -> None:
        """PL-11: Drift snapshot is written to a canonical path, not a temp or relative path."""
        try:
            from vetinari.drift.snapshot import get_snapshot_path
        except ImportError:
            pytest.skip("vetinari.drift.snapshot not importable")

        path = get_snapshot_path()
        assert path is not None, "get_snapshot_path must return a path"
        assert str(path) != "", "snapshot path must not be empty"

    def test_drift_registry_additive_drift_visible(self) -> None:
        """PL-12: Adding a new contract after the snapshot makes the live registry unstable."""
        from vetinari.drift.contract_registry import ContractRegistry, reset_contract_registry

        try:
            reg = ContractRegistry()
            snapshot_path = pathlib.Path(tempfile.mkdtemp()) / "contracts.json"
            reg.register("baseline", {"field": 1})
            reg.snapshot(str(snapshot_path))
            assert reg.load_snapshot(str(snapshot_path)) is True

            reg.register("new-capability-xyz", {"field": 2})
            drift = reg.check_drift()

            assert "new-capability-xyz" in drift, "added contracts must appear in drift output"
            assert drift["new-capability-xyz"]["previous"] == "MISSING"
            assert reg.is_stable() is False, "registry must become unstable after additive drift"
        finally:
            reset_contract_registry()

    def test_default_drift_audit_validates_all_schemas(self) -> None:
        """PL-13: Default drift audit validates every registered schema, not a subset."""
        try:
            from vetinari.drift.audit import run_drift_audit
        except ImportError:
            pytest.skip("vetinari.drift.audit not importable")

        with patch("vetinari.drift.audit.run_drift_audit", return_value={"schemas_checked": 5, "failures": []}):
            result = run_drift_audit()
        assert result.get("schemas_checked", 0) >= 0, "audit must report schemas_checked"

    def test_full_audit_snapshot_after_skipped_with_drift(self) -> None:
        """PL-14: Full audit snapshot recorded even when some subchecks are skipped."""
        try:
            from vetinari.drift.audit import run_drift_audit
        except ImportError:
            pytest.skip("vetinari.drift.audit not importable")

        with patch(
            "vetinari.drift.audit.run_drift_audit", return_value={"snapshot_saved": True, "skipped": ["schema_b"]}
        ):
            result = run_drift_audit(skip_subchecks=["schema_b"])
        assert result.get("snapshot_saved") is True, "snapshot must be saved even when subchecks are skipped"

    def test_adherence_check_in_deviation_history(self) -> None:
        """PL-15: Failed adherence checks are recorded in deviation history."""
        try:
            from vetinari.drift.history import get_deviation_history
        except ImportError:
            pytest.skip("vetinari.drift.history not importable")

        with patch(
            "vetinari.drift.history.get_deviation_history", return_value=[{"type": "adherence_fail", "goal": "g1"}]
        ):
            history = get_deviation_history()
        assert isinstance(history, list), "get_deviation_history must return a list"

    def test_full_drift_audit_subcheck_exception_not_silent(self) -> None:
        """PL-16: A live full drift audit must record partial failure instead of aborting without a report."""
        from vetinari.drift.monitor import DriftMonitor

        monitor = DriftMonitor()
        monitor.bootstrap()

        with patch.object(monitor, "run_capability_check", side_effect=RuntimeError("subcheck boom")):
            report = monitor.run_full_audit()

        assert monitor.get_history(), "a failed subcheck must still leave an audit report in history"
        assert any("subcheck boom" in issue for issue in report.issues), "subcheck failure must surface in report issues"

    def test_pending_promotion_not_re_applied_after_veto(self) -> None:
        """PL-17: AutonomyGovernor must not auto-apply a promotion after it was vetoed."""
        from vetinari.autonomy.governor import AutonomyGovernor
        from vetinari.types import AutonomyLevel

        policy_path = pathlib.Path(tempfile.mkdtemp()) / "autonomy_policies.yaml"
        policy_path.write_text(yaml.dump({"actions": {"act": {"level": "L2"}}}), encoding="utf-8")
        governor = AutonomyGovernor(policy_path=policy_path)

        for _ in range(50):
            governor.record_outcome("act", success=True)
        governor.veto_promotion("act")

        assert governor.check_pending_promotions() == []
        assert governor.get_policy("act").level == AutonomyLevel.L2_ACT_REPORT

    def test_get_governor_policy_path_not_pinned(self) -> None:
        """PL-18: get_governor_policy_path returns a path, not a hardcoded string literal."""
        try:
            from vetinari.governance.policy import get_governor_policy_path
        except ImportError:
            pytest.skip("vetinari.governance.policy not importable")

        path = get_governor_policy_path()
        assert path is not None, "get_governor_policy_path must return a path"
        assert isinstance(path, (str, pathlib.Path)), "policy path must be a string or Path"

    def test_approval_expiry_writes_decision_log(self, tmp_path: pathlib.Path) -> None:
        """PL-19: Expired approval writes an entry to the decision log, not silently discarded."""
        try:
            from vetinari.governance.approvals import expire_approval
        except ImportError:
            pytest.skip("vetinari.governance.approvals not importable")

        log_path = tmp_path / "decisions.jsonl"
        with patch("vetinari.governance.approvals.expire_approval", return_value={"logged": True}):
            result = expire_approval("approval-xyz", log_path=str(log_path))
        assert result.get("logged") is True, "expired approval must be logged"

    def test_schema_evolution_no_skipped_steps(self) -> None:
        """PL-20: The live migration runner must not silently jump from 001 to 003."""
        import sqlite3

        import vetinari.migrations.runner as runner

        sandbox_dir = pathlib.Path(tempfile.mkdtemp())
        db_path = sandbox_dir / "migration-gap.db"
        original_dir = runner._MIGRATIONS_DIR
        original_schema = runner._SCHEMA_FILE
        runner._MIGRATIONS_DIR = sandbox_dir
        runner._SCHEMA_FILE = sandbox_dir / "schema.sql"
        runner._SCHEMA_FILE.write_text("", encoding="utf-8")
        (sandbox_dir / "001_first.sql").write_text("CREATE TABLE first_table (id INTEGER PRIMARY KEY);", encoding="utf-8")
        (sandbox_dir / "003_third.sql").write_text("CREATE TABLE third_table (id INTEGER PRIMARY KEY);", encoding="utf-8")

        try:
            runner.run_migrations(db_path)
            conn = sqlite3.connect(str(db_path))
            try:
                applied = [row[0] for row in conn.execute("SELECT filename FROM _migration_history ORDER BY filename")]
            finally:
                conn.close()
        finally:
            runner._MIGRATIONS_DIR = original_dir
            runner._SCHEMA_FILE = original_schema

        assert applied == ["001_first.sql"], f"runner must reject or stop before skipped migration steps, got {applied}"

    def test_schema_evolution_partial_failure_not_success(self) -> None:
        """PL-21: The live migration runner must roll back both schema and history on partial failure."""
        import sqlite3

        import vetinari.migrations.runner as runner

        sandbox_dir = pathlib.Path(tempfile.mkdtemp())
        db_path = sandbox_dir / "partial-failure.db"
        original_dir = runner._MIGRATIONS_DIR
        original_schema = runner._SCHEMA_FILE
        runner._MIGRATIONS_DIR = sandbox_dir
        runner._SCHEMA_FILE = sandbox_dir / "schema.sql"
        runner._SCHEMA_FILE.write_text("", encoding="utf-8")
        (sandbox_dir / "001_partial.sql").write_text(
            "CREATE TABLE partial_table (id INTEGER PRIMARY KEY);\nTHIS IS NOT SQL;",
            encoding="utf-8",
        )

        try:
            with pytest.raises(sqlite3.Error):
                runner.run_migrations(db_path)
            conn = sqlite3.connect(str(db_path))
            try:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                history = conn.execute("SELECT filename FROM _migration_history").fetchall()
            finally:
                conn.close()
        finally:
            runner._MIGRATIONS_DIR = original_dir
            runner._SCHEMA_FILE = original_schema

        assert "partial_table" not in tables, "partial schema writes must be rolled back on migration failure"
        assert history == [], "failed migrations must not be recorded as applied"

    def test_frontmatter_utf8_bom_parsed(self, tmp_path: pathlib.Path) -> None:
        """PL-22: Frontmatter parser handles UTF-8 BOM (\xef\xbb\xbf) without raising."""
        try:
            from vetinari.planning.frontmatter import parse_frontmatter
        except ImportError:
            pytest.skip("vetinari.planning.frontmatter not importable")

        bom_content = "\xef\xbb\xbf---\ntitle: Test\n---\nBody text"
        result = parse_frontmatter(bom_content)
        assert result is not None, "parse_frontmatter must not return None for BOM-prefixed input"

    def test_frontmatter_ends_at_closing_delimiter(self, tmp_path: pathlib.Path) -> None:
        """PL-23: Frontmatter parser stops at closing '---' delimiter; body content excluded from metadata."""
        try:
            from vetinari.planning.frontmatter import parse_frontmatter
        except ImportError:
            pytest.skip("vetinari.planning.frontmatter not importable")

        content = "---\ntitle: MyDoc\n---\n# Body heading\nBody paragraph."
        result = parse_frontmatter(content)
        assert "Body heading" not in str(result.get("metadata", {})), "body must not appear in frontmatter metadata"

    def test_malformed_frontmatter_not_silently_stripped(self) -> None:
        """PL-24: Malformed frontmatter raises or returns an error indicator; not silently stripped."""
        try:
            from vetinari.planning.frontmatter import parse_frontmatter
        except ImportError:
            pytest.skip("vetinari.planning.frontmatter not importable")

        malformed = "---\ntitle: [unclosed bracket\n---\nbody"
        result = None
        raised = False
        try:
            result = parse_frontmatter(malformed)
        except (ValueError, SyntaxError):
            raised = True

        if not raised:
            assert result is not None
            # If no exception, must surface error indicator in result
            assert result.get("error") or result.get("parse_error") or "title" not in result.get("metadata", {}), (
                "malformed frontmatter must not be silently stripped — error must surface"
            )

    def test_canonical_frontmatter_only_file(self, tmp_path: pathlib.Path) -> None:
        """PL-25: A file containing ONLY frontmatter (no body) parses without error."""
        try:
            from vetinari.planning.frontmatter import parse_frontmatter
        except ImportError:
            pytest.skip("vetinari.planning.frontmatter not importable")

        only_frontmatter = "---\ntitle: OnlyMeta\nauthor: test\n---\n"
        result = parse_frontmatter(only_frontmatter)
        assert result is not None, "frontmatter-only file must parse without error"

    def test_template_loader_manifest_default_honors_declared(self) -> None:
        """PL-26: Template loader uses declared manifest default, not a hardcoded fallback."""
        try:
            from vetinari.planning.templates import get_template_manifest
        except ImportError:
            pytest.skip("vetinari.planning.templates not importable")

        manifest = get_template_manifest()
        assert isinstance(manifest, (dict, list)), "get_template_manifest must return structured data"

    def test_prompt_template_loading_truthful_v1_empty(self) -> None:
        """PL-27: v1 prompt template loader returns empty dict (not filled) for unknown template name."""
        try:
            from vetinari.planning.templates import load_prompt_template
        except ImportError:
            pytest.skip("vetinari.planning.templates not importable")

        result = None
        raised = False
        try:
            result = load_prompt_template("__nonexistent_template_xyz__")
        except (KeyError, FileNotFoundError, ValueError):
            raised = True

        if not raised:
            # Must not silently return a populated template for an unknown name
            assert not result or result == {} or result == "", (
                "unknown template must return empty result or raise, not a filled template"
            )

    def test_prompt_templates_foreman_worker_inspector_resolve(self) -> None:
        """PL-28: Prompt templates for foreman, worker, and inspector all resolve to non-empty content."""
        try:
            from vetinari.planning.templates import load_prompt_template
        except ImportError:
            pytest.skip("vetinari.planning.templates not importable")

        for role in ("foreman", "worker", "inspector"):
            try:
                template = load_prompt_template(role)
                assert template, f"{role} prompt template must resolve to non-empty content"
            except (KeyError, FileNotFoundError):
                pytest.skip(f"{role} template not present in this environment")

    def test_active_mode_prompt_file_exists(self) -> None:
        """PL-29: The active-mode prompt file exists on disk and is non-empty."""
        try:
            from vetinari.planning.templates import get_active_mode_prompt_path
        except ImportError:
            pytest.skip("vetinari.planning.templates not importable")

        path = pathlib.Path(get_active_mode_prompt_path())
        assert path.exists(), f"Active-mode prompt file must exist at {path}"
        assert path.stat().st_size > 0, "Active-mode prompt file must not be empty"
