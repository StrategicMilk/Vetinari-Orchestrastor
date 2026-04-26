"""
Comprehensive tests for three Vetinari modules:
  1. vetinari.model_registry   - Model discovery, inference, singleton registry
  2. vetinari.repo_map          - AST-based repository structure mapping
  3. vetinari.benchmarks.suite  - Agent benchmark evaluation and persistence

Organized into test classes per logical unit:
  TestInferCapabilities, TestInferContextWindow, TestModelInfo, TestModelRegistry
  TestModuleInfo, TestRepoMap
  TestBenchmarkHelpers, TestBenchmarkSuite
"""

import json
import os
import sys
import time

# ── Clean stubs that may have been created by other test files ──────────
for _stubname in (
    "vetinari.model_registry",
    "vetinari.repo_map",
    "vetinari.benchmarks.suite",
    "vetinari.benchmarks",
):
    sys.modules.pop(_stubname, None)

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from vetinari.benchmarks.suite import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    _score_by_keys,
    run_benchmark,
)

# ── Now import the modules under test ──────────────────────────────────
from vetinari.models.model_registry import (
    ModelInfo,
    ModelRegistry,
    _infer_capabilities,
    _infer_context_window,
    get_model_registry,
)
from vetinari.repo_map import ModuleInfo, RepoMap, get_repo_map
from vetinari.types import AgentType

# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons before each test."""
    import vetinari.models.model_registry as mr_mod
    import vetinari.repo_map as rm_mod

    # Reset ModelRegistry singleton
    ModelRegistry._instance = None
    mr_mod._registry = None

    # Reset RepoMap singleton
    rm_mod._repo_map = None

    yield

    # Cleanup after test as well
    ModelRegistry._instance = None
    mr_mod._registry = None
    rm_mod._repo_map = None


@pytest.fixture
def mock_no_config():
    """Patch _config_path.exists() to return False so no YAML is loaded."""
    with patch.object(Path, "exists", return_value=False):
        yield


@pytest.fixture
def registry_no_io():
    """Create a ModelRegistry that skips static config and network."""
    with patch.object(ModelRegistry, "_load_static_config"):
        reg = ModelRegistry()
    # Prevent auto-refresh in query methods
    reg._last_refresh = time.time() + 9999
    return reg


@pytest.fixture
def sample_model_info():
    """A ready-made ModelInfo for reuse."""
    return ModelInfo(
        model_id="test-qwen3-8b-coder-q4_k_m",
        display_name="Test Qwen3 8B Coder",
        provider="local",
        capabilities=["coding", "fast", "general"],
        context_window=32768,
        memory_requirements_gb=6,
        quantization="q4_k_m",
        latency_hint="fast",
        is_loaded=True,
        last_seen=time.time(),
        endpoint="local",
        source="discovered",
    )


# ========================================================================
# 1. vetinari.model_registry
# ========================================================================


class TestInferCapabilities:
    """Tests for _infer_capabilities() helper."""

    def test_coder_model(self):
        caps = _infer_capabilities("deepseek-coder-33b-instruct")
        assert "coding" in caps
        assert "general" in caps

    def test_vision_language_model(self):
        caps = _infer_capabilities("qwen3-vl-32b-gemini")
        assert "vision" in caps
        assert "coding" in caps
        assert "reasoning" in caps

    def test_uncensored_model(self):
        caps = _infer_capabilities("llama3-heretic-uncensored-8b")
        assert "uncensored" in caps
        assert "reasoning" in caps

    def test_thinking_model(self):
        caps = _infer_capabilities("qwen3-thinking-32b")
        assert "reasoning" in caps
        assert "analysis" in caps

    def test_math_model(self):
        caps = _infer_capabilities("mathstral-7b")
        assert "reasoning" in caps
        assert "analysis" in caps

    def test_instruct_model(self):
        caps = _infer_capabilities("mistral-7b-instruct")
        assert "coding" in caps
        assert "reasoning" in caps

    def test_qwen_family(self):
        caps = _infer_capabilities("qwen3-14b")
        assert "reasoning" in caps
        assert "coding" in caps

    def test_llama3_family(self):
        caps = _infer_capabilities("llama-3.1-70b")
        assert "reasoning" in caps
        assert "coding" in caps

    def test_phi_family(self):
        caps = _infer_capabilities("phi-3-mini-4k")
        assert "reasoning" in caps
        assert "fast" in caps

    def test_gemma_family(self):
        caps = _infer_capabilities("gemma-2-9b")
        assert "reasoning" in caps
        assert "general" in caps

    def test_yi_family(self):
        caps = _infer_capabilities("yi-34b")
        assert "reasoning" in caps

    def test_unknown_model_gets_general(self):
        caps = _infer_capabilities("totally-unknown-model")
        assert caps == ["general"]

    def test_always_includes_general(self):
        caps = _infer_capabilities("deepseek-coder-v2")
        assert "general" in caps

    def test_result_is_sorted(self):
        caps = _infer_capabilities("qwen3-vl-coder-uncensored-thinking")
        assert caps == sorted(caps)

    def test_multimodal_keyword(self):
        caps = _infer_capabilities("multimodal-llm")
        assert "vision" in caps

    def test_starcoder(self):
        caps = _infer_capabilities("starcoder-15b")
        assert "coding" in caps
        assert "fast" in caps

    def test_mixtral(self):
        caps = _infer_capabilities("mixtral-8x7b")
        assert "reasoning" in caps
        assert "coding" in caps


class TestInferContextWindow:
    """Tests for _infer_context_window() helper."""

    def test_llama3(self):
        assert _infer_context_window("llama-3.1-8b") == 131072

    def test_qwen3(self):
        assert _infer_context_window("qwen3-32b") == 32768

    def test_qwen25(self):
        assert _infer_context_window("qwen2.5-coder-7b") == 32768

    def test_gemma2(self):
        assert _infer_context_window("gemma-2-9b") == 8192

    def test_phi3(self):
        assert _infer_context_window("phi-3-mini") == 4096

    def test_mistral(self):
        assert _infer_context_window("mistral-7b-instruct") == 32768

    def test_mixtral(self):
        assert _infer_context_window("mixtral-8x7b") == 32768

    def test_yi(self):
        assert _infer_context_window("yi-34b") == 4096

    def test_unknown_defaults_8192(self):
        assert _infer_context_window("totally-unknown-model") == 8192


class TestModelInfo:
    """Tests for ModelInfo dataclass and to_dict()."""

    def test_to_dict_has_all_keys(self, sample_model_info):
        d = sample_model_info.to_dict()
        required_keys = {
            "id",
            "model_id",
            "name",
            "provider",
            "capabilities",
            "context_window",
            "context_len",
            "context_length",
            "memory_requirements_gb",
            "quantization",
            "latency_hint",
            "privacy_level",
            "cost_per_1k_tokens",
            "requires_cpu_offload",
            "preferred_for",
            "is_loaded",
            "last_seen",
            "endpoint",
            "source",
            "tags",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_compat_aliases(self, sample_model_info):
        d = sample_model_info.to_dict()
        assert d["id"] == d["model_id"]
        assert d["context_len"] == d["context_window"]
        assert d["context_length"] == d["context_window"]
        assert d["tags"] == d["capabilities"]

    def test_name_alias(self, sample_model_info):
        d = sample_model_info.to_dict()
        assert d["name"] == "Test Qwen3 8B Coder"

    def test_default_values(self):
        m = ModelInfo(model_id="m", display_name="M", provider="local")
        assert m.capabilities == []
        assert m.context_window == 8192
        assert m.privacy_level == "local"
        assert m.cost_per_1k_tokens == 0.0
        assert m.is_loaded is False
        assert m.source == "discovered"

    def test_to_dict_preserves_loaded_state(self):
        m = ModelInfo(
            model_id="loaded",
            display_name="L",
            provider="local",
            is_loaded=True,
        )
        assert m.to_dict()["is_loaded"] is True


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_singleton_get_instance(self, registry_no_io):
        """get_instance returns the same object."""
        with patch.object(ModelRegistry, "_load_static_config"):
            a = ModelRegistry.get_instance()
            b = ModelRegistry.get_instance()
        assert a is b

    def test_register_model(self, registry_no_io, sample_model_info):
        registry_no_io.register_model(sample_model_info)
        assert registry_no_io.get_model_info(sample_model_info.model_id) is sample_model_info

    def test_get_model_info_missing(self, registry_no_io):
        result = registry_no_io.get_model_info("nonexistent-model")
        assert result is None

    def test_get_available_models_no_filter(self, registry_no_io, sample_model_info):
        registry_no_io.register_model(sample_model_info)
        models = registry_no_io.get_available_models()
        assert len(models) == 1
        assert models[0].model_id == sample_model_info.model_id

    def test_get_available_models_by_provider(self, registry_no_io):
        m1 = ModelInfo(model_id="local-1", display_name="L1", provider="local", is_loaded=True)
        m2 = ModelInfo(model_id="cloud-1", display_name="C1", provider="openai")
        registry_no_io.register_model(m1)
        registry_no_io.register_model(m2)
        result = registry_no_io.get_available_models(provider="openai")
        assert len(result) == 1
        assert result[0].provider == "openai"

    def test_get_available_models_loaded_only(self, registry_no_io):
        m1 = ModelInfo(model_id="loaded", display_name="L", provider="local", is_loaded=True)
        m2 = ModelInfo(model_id="unloaded", display_name="U", provider="local", is_loaded=False)
        registry_no_io.register_model(m1)
        registry_no_io.register_model(m2)
        result = registry_no_io.get_available_models(loaded_only=True)
        assert len(result) == 1
        assert result[0].model_id == "loaded"

    def test_get_available_models_by_capability(self, registry_no_io):
        m = ModelInfo(
            model_id="coder",
            display_name="C",
            provider="local",
            capabilities=["coding", "general"],
        )
        registry_no_io.register_model(m)
        assert len(registry_no_io.get_available_models(capability="coding")) == 1
        assert len(registry_no_io.get_available_models(capability="vision")) == 0

    def test_get_loaded_local_models(self, registry_no_io):
        m = ModelInfo(
            model_id="local-loaded",
            display_name="LL",
            provider="local",
            is_loaded=True,
        )
        registry_no_io.register_model(m)
        loaded = registry_no_io.get_loaded_local_models()
        assert len(loaded) == 1

    def test_get_all_as_dicts(self, registry_no_io, sample_model_info):
        registry_no_io.register_model(sample_model_info)
        dicts = registry_no_io.get_all_as_dicts()
        assert len(dicts) == 1
        assert isinstance(dicts[0], dict)
        assert dicts[0]["model_id"] == sample_model_info.model_id

    def test_get_loaded_as_dicts(self, registry_no_io):
        m = ModelInfo(
            model_id="loaded-d",
            display_name="LD",
            provider="local",
            is_loaded=True,
        )
        registry_no_io.register_model(m)
        dicts = registry_no_io.get_loaded_as_dicts()
        assert len(dicts) == 1

    def test_list_loaded_models_compat(self, registry_no_io):
        m = ModelInfo(
            model_id="compat",
            display_name="C",
            provider="local",
            is_loaded=True,
        )
        registry_no_io.register_model(m)
        result = registry_no_io.list_loaded_models()
        assert result == registry_no_io.get_loaded_as_dicts()

    def test_get_registry_stats(self, registry_no_io):
        m1 = ModelInfo(model_id="a", display_name="A", provider="local", is_loaded=True)
        m2 = ModelInfo(model_id="b", display_name="B", provider="openai", is_loaded=False)
        registry_no_io.register_model(m1)
        registry_no_io.register_model(m2)
        stats = registry_no_io.get_registry_stats()
        assert stats["total"] == 2
        assert stats["loaded"] == 1
        assert stats["local"] == 1
        assert stats["cloud"] == 1
        assert "a" in stats["loaded_ids"]

    def test_infer_quantization(self):
        assert ModelRegistry._infer_quantization("model-q8-gguf") == "q8"
        assert ModelRegistry._infer_quantization("model-q4_k_m") == "q4_k_m"
        assert ModelRegistry._infer_quantization("model-q4_k_s") == "q4_k_s"
        assert ModelRegistry._infer_quantization("model-f16") == "f16"
        assert ModelRegistry._infer_quantization("model-f32") == "f32"
        assert ModelRegistry._infer_quantization("unknown-model") == "q4_k_m"

    def test_infer_latency_slow(self):
        assert ModelRegistry._infer_latency("llama-3-70b") == "slow"
        assert ModelRegistry._infer_latency("qwen3-72b") == "slow"

    def test_infer_latency_medium(self):
        assert ModelRegistry._infer_latency("qwen3-32b") == "medium"
        assert ModelRegistry._infer_latency("llama-34b") == "medium"

    def test_infer_latency_fast(self):
        assert ModelRegistry._infer_latency("phi-3-7b") == "fast"
        assert ModelRegistry._infer_latency("gemma-2b") == "fast"

    def test_infer_latency_default_medium(self):
        assert ModelRegistry._infer_latency("unknown-model") == "medium"

    def test_build_auth_headers_returns_empty_dict(self):
        """_build_auth_headers returns {} — local inference needs no auth."""
        assert ModelRegistry._build_auth_headers() == {}

    def test_build_auth_headers_always_empty(self):
        """_build_auth_headers returns {} regardless of environment variables."""
        with patch.dict(os.environ, {"VETINARI_API_TOKEN": "tok123"}):
            assert ModelRegistry._build_auth_headers() == {}

    def test_refresh_skips_within_interval(self, registry_no_io):
        """refresh() is a no-op when called within REFRESH_INTERVAL."""
        registry_no_io._last_refresh = time.time()
        with patch.object(registry_no_io, "_do_refresh") as mock_do:
            registry_no_io.refresh()
        mock_do.assert_not_called()

    def test_refresh_force(self, registry_no_io):
        """refresh(force=True) always calls _do_refresh."""
        registry_no_io._last_refresh = time.time()
        with patch.object(registry_no_io, "_do_refresh") as mock_do:
            registry_no_io.refresh(force=True)
        mock_do.assert_called_once_with()

    def test_do_refresh_success(self, registry_no_io, tmp_path):
        """_do_refresh() creates ModelInfo entries by scanning models directory."""
        (tmp_path / "qwen3-8b-q4_k_m.gguf").write_bytes(b"\x00" * 128)
        (tmp_path / "llama-3.1-70b-instruct.gguf").write_bytes(b"\x00" * 128)

        registry_no_io._models_dir = str(tmp_path)
        with patch("vetinari.utils.estimate_model_memory_gb", return_value=8):
            registry_no_io._do_refresh()

        assert "qwen3-8b-q4_k_m" in registry_no_io._models
        assert "llama-3.1-70b-instruct" in registry_no_io._models
        assert registry_no_io._models["qwen3-8b-q4_k_m"].is_loaded is True

    def test_do_refresh_no_dir(self, registry_no_io):
        """_do_refresh() does nothing when models directory is not set."""
        registry_no_io._models_dir = None
        registry_no_io._do_refresh()
        assert len(registry_no_io._models) == 0

    def test_do_refresh_missing_dir(self, registry_no_io, tmp_path):
        """_do_refresh() does nothing when models directory does not exist."""
        registry_no_io._models_dir = str(tmp_path / "nonexistent")
        registry_no_io._do_refresh()
        assert len(registry_no_io._models) == 0

    def test_do_refresh_marks_old_models_unloaded(self, registry_no_io, tmp_path):
        """Previously loaded models are marked unloaded when not found in directory."""
        old = ModelInfo(
            model_id="old-model",
            display_name="Old",
            provider="local",
            is_loaded=True,
            source="discovered",
        )
        registry_no_io.register_model(old)

        # Only new-model.gguf exists — old-model should be marked unloaded
        (tmp_path / "new-model.gguf").write_bytes(b"\x00" * 128)
        registry_no_io._models_dir = str(tmp_path)
        with patch("vetinari.utils.estimate_model_memory_gb", return_value=4):
            registry_no_io._do_refresh()

        assert registry_no_io._models["old-model"].is_loaded is False
        assert registry_no_io._models["new-model"].is_loaded is True

    def test_do_refresh_updates_existing_discovered(self, registry_no_io, tmp_path):
        """Existing 'discovered' entries get capabilities re-inferred on refresh."""
        existing = ModelInfo(
            model_id="qwen3-coder-8b",
            display_name="QC",
            provider="local",
            capabilities=["old"],
            source="discovered",
            is_loaded=False,
        )
        registry_no_io.register_model(existing)

        (tmp_path / "qwen3-coder-8b.gguf").write_bytes(b"\x00" * 128)
        registry_no_io._models_dir = str(tmp_path)
        registry_no_io._do_refresh()

        updated = registry_no_io._models["qwen3-coder-8b"]
        assert updated.is_loaded is True
        assert "coding" in updated.capabilities  # re-inferred
        assert updated.capabilities != ["old"]

    def test_do_refresh_preserves_config_source(self, registry_no_io, tmp_path):
        """Existing 'config' entries keep original capabilities on refresh."""
        cfg_model = ModelInfo(
            model_id="cfg-model",
            display_name="CFG",
            provider="local",
            capabilities=["custom-cap"],
            source="config",
            is_loaded=False,
        )
        registry_no_io.register_model(cfg_model)

        (tmp_path / "cfg-model.gguf").write_bytes(b"\x00" * 128)
        registry_no_io._models_dir = str(tmp_path)
        registry_no_io._do_refresh()

        updated = registry_no_io._models["cfg-model"]
        assert updated.is_loaded is True
        assert updated.capabilities == ["custom-cap"]  # preserved

    def test_load_static_config_with_yaml(self):
        """_load_static_config loads models from YAML."""
        yaml_data = {
            "models": [
                {
                    "model_id": "yaml-model",
                    "display_name": "YAML Model",
                    "provider": "local",
                    "capabilities": ["reasoning"],
                    "context_window": 16384,
                }
            ],
            "cloud_models": [
                {
                    "model_id": "cloud-model",
                    "provider": "openai",
                    "cost_per_1k_tokens": 0.01,
                }
            ],
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("vetinari.models.model_registry.load_yaml", create=True):
                # We need to patch the import inside _load_static_config
                with patch.dict(sys.modules, {}):
                    mock_utils = MagicMock()
                    mock_utils.load_yaml.return_value = yaml_data
                    with patch.dict(sys.modules, {"vetinari.utils": mock_utils}):
                        reg = ModelRegistry()

        assert "yaml-model" in reg._models
        assert "cloud-model" in reg._models
        assert reg._models["yaml-model"].capabilities == ["reasoning"]
        assert reg._models["cloud-model"].cost_per_1k_tokens == 0.01

    def test_load_static_config_missing_file(self):
        """_load_static_config does nothing when models.yaml is absent."""
        with patch.object(Path, "exists", return_value=False):
            reg = ModelRegistry()
        assert len(reg._models) == 0

    def test_get_model_registry_module_singleton(self):
        """get_model_registry() returns the same instance on repeat calls."""
        with patch.object(ModelRegistry, "_load_static_config"):
            r1 = get_model_registry()
            r2 = get_model_registry()
        assert r1 is r2


# ========================================================================
# 2. vetinari.repo_map
# ========================================================================


class TestModuleInfo:
    """Tests for ModuleInfo dataclass."""

    def test_defaults(self):
        m = ModuleInfo(path="foo.py", name="foo")
        assert m.classes == []
        assert m.functions == []
        assert m.imports == []
        assert m.docstring == ""
        assert m.line_count == 0

    def test_fields(self):
        m = ModuleInfo(
            path="pkg/bar.py",
            name="pkg.bar",
            classes=["MyClass"],
            functions=["do_stuff(x, y)"],
            imports=["os", "sys"],
            docstring="Bar module",
            line_count=42,
        )
        assert m.name == "pkg.bar"
        assert len(m.classes) == 1


class TestRepoMap:
    """Tests for RepoMap class."""

    def _write_py(self, tmp_path, rel_path, content):
        p = tmp_path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_generate_nonexistent_path(self):
        mapper = RepoMap()
        result = mapper.generate("/nonexistent/path/xyzzy")
        assert "not found" in result.lower()

    def test_generate_empty_directory(self, tmp_path):
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "no python files" in result.lower()

    def test_generate_single_file(self, tmp_path):
        self._write_py(tmp_path, "hello.py", "def greet(name):\n    return f'Hello {name}'\n")
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "hello.py" in result
        assert "greet" in result

    def test_generate_with_class(self, tmp_path):
        code = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
            "    def subtract(self, a, b):\n"
            "        return a - b\n"
        )
        self._write_py(tmp_path, "calc.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "Calculator" in result

    def test_generate_with_docstring(self, tmp_path):
        code = '"""This is a great module."""\nimport os\ndef foo(): pass\n'
        self._write_py(tmp_path, "doc.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "great module" in result

    def test_generate_with_docstring_file_still_parseable(self, tmp_path):
        """Files without module-level docstrings parse fine on all Python versions."""
        code = "import os\ndef foo(): pass\n"
        self._write_py(tmp_path, "nodoc.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "nodoc.py" in result
        assert "foo" in result

    def test_generate_imports_extracted(self, tmp_path):
        code = "import os\nimport sys\nfrom pathlib import Path\n\ndef work(): pass\n"
        self._write_py(tmp_path, "imp.py", code)
        mapper = RepoMap()
        modules = mapper._scan_directory(tmp_path, None, False)
        assert len(modules) == 1
        assert "os" in modules[0].imports
        assert "pathlib" in modules[0].imports

    def test_generate_skips_pycache(self, tmp_path):
        self._write_py(tmp_path, "__pycache__/cached.py", "x = 1\n")
        self._write_py(tmp_path, "real.py", "y = 2\n")
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "cached" not in result
        assert "real.py" in result

    def test_generate_skips_venv(self, tmp_path):
        self._write_py(tmp_path, "venv/lib/site.py", "x = 1\n")
        self._write_py(tmp_path, "app.py", "def main(): pass\n")
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path))
        assert "site" not in result
        assert "app.py" in result

    def test_generate_token_truncation(self, tmp_path):
        """Output is truncated when it exceeds max_tokens * 4 chars."""
        for i in range(50):
            code = f"def function_{i}(a, b, c, d, e):\n    return {i}\n"
            self._write_py(tmp_path, f"mod_{i:02d}.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path), max_tokens=50)
        # With max_tokens=50 => max_chars=200, should truncate
        assert "truncated" in result.lower() or "not shown" in result.lower()

    def test_generate_include_private_false(self, tmp_path):
        code = "def _hidden_fn(): pass\ndef public(): pass\n"
        self._write_py(tmp_path, "priv.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path), include_private=False)
        assert "public" in result
        assert "_hidden_fn" not in result

    def test_generate_include_private_true(self, tmp_path):
        code = "def _hidden_fn(): pass\ndef public(): pass\n"
        self._write_py(tmp_path, "priv.py", code)
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path), include_private=True)
        assert "_hidden_fn" in result
        assert "public" in result

    def test_generate_focus_paths(self, tmp_path):
        self._write_py(tmp_path, "a.py", "def aa(): pass\n")
        self._write_py(tmp_path, "b.py", "def bb(): pass\n")
        mapper = RepoMap()
        result = mapper.generate(str(tmp_path), focus_paths=["a.py"])
        assert "aa" in result
        assert "bb" not in result

    def test_parse_file_syntax_error(self, tmp_path):
        self._write_py(tmp_path, "bad.py", "def broken(:\n")
        mapper = RepoMap()
        mod = mapper._parse_file(tmp_path / "bad.py", tmp_path)
        assert mod is None

    def test_parse_file_huge_file_skipped(self, tmp_path):
        # >100k chars should be skipped
        self._write_py(tmp_path, "huge.py", "x = 1\n" * 20000)
        mapper = RepoMap()
        mod = mapper._parse_file(tmp_path / "huge.py", tmp_path)
        assert mod is None

    def test_format_class_with_bases(self, tmp_path):
        code = "class Child(Parent):\n    def method(self): pass\n"
        self._write_py(tmp_path, "cls.py", code)
        mapper = RepoMap()
        modules = mapper._scan_directory(tmp_path, None, False)
        assert len(modules) == 1
        cls = modules[0].classes[0]
        assert "Child" in cls
        assert "Parent" in cls
        assert "method" in cls

    def test_format_function_async(self, tmp_path):
        code = "async def fetch_data(url, timeout): pass\n"
        self._write_py(tmp_path, "async_mod.py", code)
        mapper = RepoMap()
        modules = mapper._scan_directory(tmp_path, None, False)
        assert len(modules) == 1
        assert any("async" in f for f in modules[0].functions)

    def test_format_function_truncates_args(self, tmp_path):
        code = "def many_args(a, b, c, d, e, f, g): pass\n"
        self._write_py(tmp_path, "args.py", code)
        mapper = RepoMap()
        modules = mapper._scan_directory(tmp_path, None, False)
        func_str = modules[0].functions[0]
        assert "..." in func_str  # >4 args truncated

    def test_generate_for_task_relevance(self, tmp_path):
        self._write_py(tmp_path, "auth.py", "class AuthManager:\n    def login(self): pass\n")
        self._write_py(tmp_path, "math_utils.py", "def add(a, b): return a + b\n")
        mapper = RepoMap()
        result = mapper.generate_for_task(str(tmp_path), "authentication login")
        # auth.py should appear (relevant keywords) — it may or may not include math_utils
        assert "auth" in result.lower()

    def test_generate_for_task_nonexistent_path(self, tmp_path):
        mapper = RepoMap()
        result = mapper.generate_for_task("/nonexistent/path/xyz", "anything")
        assert result == ""

    def test_generate_for_task_empty_dir(self, tmp_path):
        mapper = RepoMap()
        result = mapper.generate_for_task(str(tmp_path), "anything")
        assert result == ""

    def test_get_repo_map_singleton(self):
        a = get_repo_map()
        b = get_repo_map()
        assert a is b

    def test_line_count_tracked(self, tmp_path):
        code = "line1 = 1\nline2 = 2\nline3 = 3\n"
        self._write_py(tmp_path, "counted.py", code)
        mapper = RepoMap()
        modules = mapper._scan_directory(tmp_path, None, False)
        assert modules[0].line_count == 3

    def test_skip_extensions(self, tmp_path):
        # .pyc should be skipped
        p = tmp_path / "compiled.pyc"
        p.write_bytes(b"\x00")
        self._write_py(tmp_path, "real.py", "x = 1\n")
        mapper = RepoMap()
        files = list(mapper._iter_python_files(tmp_path))
        names = [f.name for f in files]
        assert "compiled.pyc" not in names
        assert "real.py" in names


# ========================================================================
# 3. vetinari.benchmarks.suite
# ========================================================================


class TestBenchmarkHelpers:
    """Tests for _score_by_keys and dataclasses."""

    def test_score_by_keys_dict_all_present(self):
        output = {"tasks": [1, 2], "dependencies": ["a"]}
        score = _score_by_keys(output, ["tasks", "dependencies"])
        assert score == 1.0

    def test_score_by_keys_dict_partial(self):
        output = {"tasks": [1], "extra": "x"}
        score = _score_by_keys(output, ["tasks", "dependencies"])
        assert score == 0.5

    def test_score_by_keys_dict_none_present(self):
        output = {"other": "val"}
        score = _score_by_keys(output, ["tasks", "dependencies"])
        assert score == 0.0

    def test_score_by_keys_dict_empty_value_not_counted(self):
        """output.get(k) returns falsy for empty list/empty string."""
        output = {"tasks": [], "dependencies": ""}
        score = _score_by_keys(output, ["tasks", "dependencies"])
        assert score == 0.0

    def test_score_by_keys_json_string(self):
        output = json.dumps({"tasks": [1], "dependencies": [2]})
        score = _score_by_keys(output, ["tasks", "dependencies"])
        assert score == 1.0

    def test_score_by_keys_invalid_json_string(self):
        score = _score_by_keys("not json at all", ["tasks"])
        assert score == 0.3

    def test_score_by_keys_non_dict_non_string(self):
        score = _score_by_keys(42, ["tasks"])
        assert score == 0.1

    def test_score_by_keys_empty_required(self):
        score = _score_by_keys({"a": 1}, [])
        # found=0, max(0,1) = 1, so 0/1 = 0.0
        assert score == 0.0

    def test_score_by_keys_none_input(self):
        score = _score_by_keys(None, ["key"])
        assert score == 0.1

    def test_benchmark_case_dataclass(self):
        bc = BenchmarkCase(
            case_id="test_001",
            agent_type=AgentType.FOREMAN.value,
            task_type="planning",
            description="test case",
            input="test input",
            evaluator=lambda o: 1.0,
        )
        assert bc.case_id == "test_001"
        assert bc.expected_keys == []

    def test_benchmark_result_dataclass(self):
        br = BenchmarkResult(
            agent_type=AgentType.WORKER.value,
            timestamp="2025-01-01",
            cases_run=3,
            cases_passed=2,
            avg_score=0.8,
        )
        assert br.scores == []
        assert br.details == []
        assert br.duration_ms == 0.0
        assert br.error == ""

    def test_benchmark_result_to_dict(self):
        br = BenchmarkResult(
            agent_type="X",
            timestamp="t",
            cases_run=1,
            cases_passed=1,
            avg_score=0.9,
        )
        d = asdict(br)
        assert d["agent_type"] == "X"
        assert d["avg_score"] == 0.9


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_build_cases_creates_cases(self):
        suite = BenchmarkSuite()
        assert len(suite._cases) >= 10

    def test_all_cases_have_evaluators(self):
        suite = BenchmarkSuite()
        for case in suite._cases:
            assert callable(case.evaluator)

    def test_all_cases_have_unique_ids(self):
        suite = BenchmarkSuite()
        ids = [c.case_id for c in suite._cases]
        assert len(ids) == len(set(ids))

    def test_run_agent_unknown_type(self):
        suite = BenchmarkSuite()
        result = suite.run_agent("NONEXISTENT_AGENT_TYPE")
        assert result.cases_run == 0
        assert result.avg_score == 0.0
        assert "No benchmark cases" in result.error

    def test_run_case_import_failure(self):
        """_run_case handles import errors gracefully."""
        suite = BenchmarkSuite()
        case = BenchmarkCase(
            case_id="fail_001",
            agent_type="FAKE",
            task_type="test",
            description="will fail",
            input="test",
            evaluator=lambda o: 1.0,
        )
        score, detail = suite._run_case(case)
        assert score == 0.0
        assert "error" in detail

    def test_run_agent_with_mock_agent(self):
        """Run a known agent type with fully mocked agent graph."""
        suite = BenchmarkSuite()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"tasks": ["a", "b"], "dependencies": ["a->b"]}
        mock_result.errors = []

        mock_agent = MagicMock()
        mock_agent.execute.return_value = mock_result

        mock_graph = MagicMock()
        mock_graph.get_agent.return_value = mock_agent

        with patch("vetinari.orchestration.agent_graph.get_agent_graph", return_value=mock_graph):
            with patch("vetinari.types.AgentType", side_effect=lambda x: x):
                with patch("vetinari.agents.contracts.AgentTask"):
                    result = suite.run_agent(AgentType.FOREMAN.value)

        assert result.cases_run >= 1
        assert result.avg_score > 0

    def test_run_agent_agent_not_in_graph(self):
        """Agent returns None from get_agent => score 0."""
        suite = BenchmarkSuite()

        mock_graph = MagicMock()
        mock_graph.get_agent.return_value = None

        with patch("vetinari.orchestration.agent_graph.get_agent_graph", return_value=mock_graph):
            with patch("vetinari.types.AgentType", side_effect=lambda x: x):
                result = suite.run_agent(AgentType.FOREMAN.value)

        assert result.avg_score == 0.0

    def test_run_agent_execution_failure(self):
        """Agent execute returns failure => score 0.2."""
        suite = BenchmarkSuite()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["boom"]

        mock_agent = MagicMock()
        mock_agent.execute.return_value = mock_result

        mock_graph = MagicMock()
        mock_graph.get_agent.return_value = mock_agent

        with patch("vetinari.orchestration.agent_graph.get_agent_graph", return_value=mock_graph):
            with patch("vetinari.types.AgentType", side_effect=lambda x: x):
                with patch("vetinari.agents.contracts.AgentTask"):
                    result = suite.run_agent(AgentType.FOREMAN.value)

        assert result.avg_score == pytest.approx(0.2, abs=0.01)

    def test_persist_writes_jsonl(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        bsuite._RESULTS_PATH = tmp_path / "results.jsonl"
        try:
            suite = BenchmarkSuite()
            br = BenchmarkResult(
                agent_type="TEST",
                timestamp="now",
                cases_run=1,
                cases_passed=1,
                avg_score=0.95,
            )
            suite._persist(br)

            content = bsuite._RESULTS_PATH.read_text()
            data = json.loads(content.strip())
            assert data["agent_type"] == "TEST"
            assert data["avg_score"] == 0.95
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_load_historical_empty(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        bsuite._RESULTS_PATH = tmp_path / "nonexistent.jsonl"
        try:
            suite = BenchmarkSuite()
            assert suite._load_historical() == {}
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_load_historical_with_data(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(
            json.dumps({"agent_type": "A", "avg_score": 0.8})
            + "\n"
            + json.dumps({"agent_type": "A", "avg_score": 0.9})
            + "\n"
            + json.dumps({"agent_type": "B", "avg_score": 0.7})
            + "\n"
        )
        bsuite._RESULTS_PATH = results_file
        try:
            suite = BenchmarkSuite()
            hist = suite._load_historical()
            assert hist["A"] == pytest.approx(0.85, abs=0.01)
            assert hist["B"] == pytest.approx(0.7, abs=0.01)
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_check_regression_no_historical(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        bsuite._RESULTS_PATH = tmp_path / "nonexistent.jsonl"
        try:
            suite = BenchmarkSuite()
            results = [
                BenchmarkResult(
                    agent_type="A",
                    timestamp="t",
                    cases_run=1,
                    cases_passed=1,
                    avg_score=0.9,
                )
            ]
            regressions = suite.check_regression(results)
            assert regressions == []
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_check_regression_detects_drop(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(json.dumps({"agent_type": "A", "avg_score": 0.9}) + "\n")
        bsuite._RESULTS_PATH = results_file
        try:
            suite = BenchmarkSuite()
            new_results = [
                BenchmarkResult(
                    agent_type="A",
                    timestamp="t",
                    cases_run=1,
                    cases_passed=0,
                    avg_score=0.5,
                )
            ]
            regressions = suite.check_regression(new_results, threshold=0.05)
            assert len(regressions) == 1
            assert "A" in regressions[0]
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_check_regression_within_threshold(self, tmp_path):
        import vetinari.benchmarks.suite as bsuite

        original_path = bsuite._RESULTS_PATH
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(json.dumps({"agent_type": "A", "avg_score": 0.9}) + "\n")
        bsuite._RESULTS_PATH = results_file
        try:
            suite = BenchmarkSuite()
            new_results = [
                BenchmarkResult(
                    agent_type="A",
                    timestamp="t",
                    cases_run=1,
                    cases_passed=1,
                    avg_score=0.88,
                )
            ]
            regressions = suite.check_regression(new_results, threshold=0.05)
            assert regressions == []
        finally:
            bsuite._RESULTS_PATH = original_path

    def test_print_report_does_not_crash(self):
        suite = BenchmarkSuite()
        results = [
            BenchmarkResult(
                agent_type="TEST",
                timestamp="now",
                cases_run=2,
                cases_passed=1,
                avg_score=0.75,
                duration_ms=42.1,
            )
        ]
        # Verify print_report completes without raising and logs output
        import logging

        with patch.object(logging.getLogger("vetinari.benchmarks.suite"), "info") as mock_log:
            suite.print_report(results)
        formatted = []
        for log_call in mock_log.call_args_list:
            template, *args = log_call.args
            formatted.append(template % tuple(args) if args else template)
        assert any("VETINARI BENCHMARK REPORT" in message for message in formatted)
        assert any("[PASS] TEST" in message for message in formatted)
        assert any("OVERALL AVG: 0.750" in message for message in formatted)

    def test_run_all_filters_by_agent_types(self):
        """run_all(agent_types) only runs specified agents."""
        suite = BenchmarkSuite()
        with patch.object(suite, "run_agent") as mock_run:
            mock_run.return_value = BenchmarkResult(
                agent_type=AgentType.FOREMAN.value,
                timestamp="t",
                cases_run=1,
                cases_passed=1,
                avg_score=0.9,
            )
            with patch.object(suite, "_persist"):
                results = suite.run_all(agent_types=[AgentType.FOREMAN.value])

        mock_run.assert_called_once_with(AgentType.FOREMAN.value)
        assert len(results) == 1

    def test_run_all_persists_each_result(self):
        suite = BenchmarkSuite()
        with patch.object(suite, "run_agent") as mock_run:
            mock_run.return_value = BenchmarkResult(
                agent_type="A",
                timestamp="t",
                cases_run=1,
                cases_passed=1,
                avg_score=0.9,
            )
            with patch.object(suite, "_persist") as mock_persist:
                suite.run_all(agent_types=["A", "B"])
        assert mock_persist.call_count == 2

    def test_run_benchmark_convenience(self):
        """run_benchmark() creates suite, runs, prints, checks regression."""
        with patch.object(BenchmarkSuite, "run_all") as mock_all:
            mock_all.return_value = [
                BenchmarkResult(
                    agent_type="X",
                    timestamp="t",
                    cases_run=1,
                    cases_passed=1,
                    avg_score=0.9,
                )
            ]
            with patch.object(BenchmarkSuite, "print_report"):
                with patch.object(BenchmarkSuite, "check_regression", return_value=[]):
                    results = run_benchmark(["X"])
        assert len(results) == 1

    def test_pass_threshold(self):
        assert BenchmarkSuite.PASS_THRESHOLD == 0.6

    def test_cases_cover_multiple_agent_types(self):
        suite = BenchmarkSuite()
        agent_types = {c.agent_type for c in suite._cases}
        # All 3 canonical agent types should be covered
        assert len(agent_types) >= 3
        assert agent_types == {AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value}
