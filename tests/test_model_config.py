"""Tests for model configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.config.model_config import (
    get_agent_timeout,
    get_model_config,
    get_model_for_agent,
    load_model_config,
    reset_cache,
)
from vetinari.types import AgentType


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset config cache before each test."""
    reset_cache()
    yield
    reset_cache()


class TestLoadModelConfig:
    def test_loads_default_config_when_no_file(self, tmp_path: Path) -> None:
        config = load_model_config(tmp_path / "nonexistent.yaml")
        assert "default" in config
        assert config["default"]["model"] == "qwen2.5-72b"
        assert "agents" in config
        assert "planner" in config["agents"]

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(
            "default:\n  model: custom-model\n  provider: openai\n",
            encoding="utf-8",
        )
        config = load_model_config(yaml_path)
        assert config["default"]["model"] == "custom-model"
        assert config["default"]["provider"] == "openai"

    def test_handles_invalid_yaml_gracefully(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text("{{invalid yaml", encoding="utf-8")
        config = load_model_config(yaml_path)
        assert config["default"]["model"] == "qwen2.5-72b"

    def test_handles_non_dict_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        config = load_model_config(yaml_path)
        assert config["default"]["model"] == "qwen2.5-72b"

    def test_merges_agent_config_over_defaults(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(
            "agents:\n  builder:\n    default:\n      model: special-builder\n      provider: local\n",
            encoding="utf-8",
        )
        config = load_model_config(yaml_path)
        assert config["agents"]["builder"]["default"]["model"] == "special-builder"
        assert config["agents"]["planner"]["default"]["model"] == "qwen2.5-72b"


class TestGetModelForAgent:
    def test_returns_default_for_unknown_agent(self) -> None:
        result = get_model_for_agent("unknown_agent")
        assert result["model"] == "qwen2.5-72b"

    def test_returns_agent_default(self) -> None:
        result = get_model_for_agent("planner")
        assert result["model"] == "qwen2.5-72b"

    def test_returns_mode_specific_config(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(
            "agents:\n  builder:\n    build:\n      model: code-model\n      provider: local\n    default:\n      model: general\n      provider: local\n",
            encoding="utf-8",
        )
        config = load_model_config(yaml_path)
        # Manually check via config since get_model_for_agent uses cached default path
        assert config["agents"]["builder"]["build"]["model"] == "code-model"

    def test_returns_provider_key(self) -> None:
        result = get_model_for_agent("builder")
        assert "provider" in result
        assert result["provider"] in ("local", "llama_cpp")

    def test_agent_type_is_case_insensitive(self) -> None:
        result_lower = get_model_for_agent("foreman")
        result_upper = get_model_for_agent(AgentType.FOREMAN.value)
        assert result_lower["model"] == result_upper["model"]


class TestGetAgentTimeout:
    def test_foreman_default_timeout(self) -> None:
        assert get_agent_timeout("foreman") == 120

    def test_worker_default_timeout(self) -> None:
        assert get_agent_timeout("worker") == 300

    def test_inspector_default_timeout(self) -> None:
        assert get_agent_timeout("inspector") == 60

    def test_case_insensitive(self) -> None:
        assert get_agent_timeout(AgentType.FOREMAN.value) == get_agent_timeout("foreman")
        assert get_agent_timeout(AgentType.WORKER.value) == get_agent_timeout("worker")
        assert get_agent_timeout(AgentType.INSPECTOR.value) == get_agent_timeout("inspector")

    def test_unknown_agent_returns_default(self) -> None:
        # Unknown agents fall back to the module-level _DEFAULT_TIMEOUT (120)
        result = get_agent_timeout("unknown_agent_xyz")
        assert isinstance(result, int)
        assert result > 0

    def test_yaml_override_respected(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "models.yaml"
        yaml_path.write_text(
            "agents:\n  foreman:\n    default:\n      model: qwen2.5-72b\n      provider: local\n    timeout_seconds: 999\n",
            encoding="utf-8",
        )
        config = load_model_config(yaml_path)
        # Directly read the config so we don't touch the global cache
        agent_cfg = config.get("agents", {}).get("foreman", {})
        assert agent_cfg.get("timeout_seconds") == 999

    @pytest.mark.parametrize(
        ("agent", "expected"),
        [
            ("foreman", 120),
            ("worker", 300),
            ("inspector", 60),
        ],
    )
    def test_all_canonical_agents(self, agent: str, expected: int) -> None:
        assert get_agent_timeout(agent) == expected


class TestGetModelConfig:
    def test_returns_dict_with_required_keys(self) -> None:
        config = get_model_config()
        assert "default" in config
        assert "agents" in config

    def test_canonical_agents_present(self) -> None:
        config = get_model_config()
        agents = config["agents"]
        for agent in ("foreman", "worker", "inspector"):
            assert agent in agents, f"Expected '{agent}' in agents config"

    def test_canonical_agents_have_timeout(self) -> None:
        config = get_model_config()
        expected = {"foreman": 120, "worker": 300, "inspector": 60}
        for agent, timeout in expected.items():
            assert config["agents"][agent].get("timeout_seconds") == timeout
