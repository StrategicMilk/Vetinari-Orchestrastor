"""Tests for vetinari/cli_packaging.py.

Verifies that cmd_init, cmd_doctor, and cmd_models return the expected exit
codes and that _register_packaging_commands wires up all three argparse
subparsers correctly.  All external dependencies (psutil, pynvml,
huggingface_hub, llama_cpp, vetinari sub-modules) are mocked so that tests
run without real hardware, models, or network access.
"""

from __future__ import annotations

import argparse
import builtins
import json
import os as _os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_stub_module as _make_stub

# ---------------------------------------------------------------------------
# Minimal sys.modules stubs — installed BEFORE importing cli_packaging so
# that optional third-party imports at module level do not fail.
# ---------------------------------------------------------------------------

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# vetinari package root
_vet_pkg = sys.modules.get("vetinari") or _make_stub("vetinari")
if not hasattr(_vet_pkg, "__path__"):
    _vet_pkg.__path__ = [_os.path.join(_ROOT, "vetinari")]
    _vet_pkg.__package__ = "vetinari"
sys.modules.setdefault("vetinari", _vet_pkg)

# vetinari.constants — load the real module (stdlib-only, no side-effects)
import importlib.util as _ilu

_const_spec = _ilu.spec_from_file_location("vetinari.constants", _os.path.join(_ROOT, "vetinari", "constants.py"))
_const_mod = _ilu.module_from_spec(_const_spec)
sys.modules.setdefault("vetinari.constants", _const_mod)
_const_spec.loader.exec_module(_const_mod)

# vetinari.database stub
_mock_db_conn = MagicMock()
_mock_db_conn.__enter__ = lambda s: _mock_db_conn
_mock_db_conn.__exit__ = MagicMock(return_value=False)
_mock_db_conn.execute = MagicMock()
sys.modules.setdefault(
    "vetinari.database",
    _make_stub("vetinari.database", get_connection=MagicMock(return_value=_mock_db_conn)),
)

# vetinari.security stub
sys.modules.setdefault("vetinari.security", _make_stub("vetinari.security"))

# vetinari.orchestration package + two_layer stub
if "vetinari.orchestration" not in sys.modules:
    _orch_pkg = _make_stub("vetinari.orchestration")
    _orch_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "orchestration")]
    _orch_pkg.__package__ = "vetinari.orchestration"
    sys.modules["vetinari.orchestration"] = _orch_pkg
sys.modules.setdefault(
    "vetinari.orchestration.two_layer",
    _make_stub("vetinari.orchestration.two_layer", get_two_layer_orchestrator=MagicMock()),
)

# vetinari.memory package + unified stub
if "vetinari.memory" not in sys.modules:
    _mem_pkg = _make_stub("vetinari.memory")
    _mem_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "memory")]
    _mem_pkg.__package__ = "vetinari.memory"
    sys.modules["vetinari.memory"] = _mem_pkg
sys.modules.setdefault("vetinari.memory.unified", _make_stub("vetinari.memory.unified"))

# vetinari.learning package stubs
if "vetinari.learning" not in sys.modules:
    _learn_pkg = _make_stub("vetinari.learning")
    _learn_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "learning")]
    _learn_pkg.__package__ = "vetinari.learning"
    sys.modules["vetinari.learning"] = _learn_pkg

_mock_collector = MagicMock()
_mock_collector.get_stats.return_value = {"total_records": 42, "avg_score": 0.75}
sys.modules.setdefault(
    "vetinari.learning.training_data",
    _make_stub("vetinari.learning.training_data", get_training_collector=MagicMock(return_value=_mock_collector)),
)

_mock_episode_mem = MagicMock()
_mock_episode_mem.get_stats.return_value = {"total_episodes": 1}
_MockEpisodeMemory = MagicMock(return_value=_mock_episode_mem)
sys.modules.setdefault(
    "vetinari.learning.episode_memory",
    _make_stub("vetinari.learning.episode_memory", EpisodeMemory=_MockEpisodeMemory),
)

# rich stub — test the non-rich path by default (rich presence tested separately)
sys.modules.setdefault("rich", _make_stub("rich"))
sys.modules.setdefault("rich.console", _make_stub("rich.console", Console=MagicMock()))
sys.modules.setdefault("rich.table", _make_stub("rich.table", Table=MagicMock()))
sys.modules.setdefault(
    "rich.progress",
    _make_stub(
        "rich.progress",
        Progress=MagicMock(),
        BarColumn=MagicMock(),
        SpinnerColumn=MagicMock(),
        TaskProgressColumn=MagicMock(),
        TextColumn=MagicMock(),
        TimeElapsedColumn=MagicMock(),
    ),
)

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------

from vetinari.cli_packaging import (
    _CHECK_FAIL,
    _CHECK_PASS,
    _CHECK_WARN,
    _detect_hardware,
    _get_recommended_models,
    _guess_family,
    _guess_quantization,
    _models_download,
    _models_files,
    _models_info,
    _models_list,
    _models_recommend,
    _models_remove,
    _register_packaging_commands,
    cmd_doctor,
    cmd_init,
    cmd_models,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Return a temporary directory with a few fake .gguf files."""
    models = tmp_path / "models"
    models.mkdir()
    for name in [
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "llama-3-8b.Q6_K.gguf",
    ]:
        fp = models / name
        fp.write_bytes(b"GGUF" + b"\x00" * 100)  # valid GGUF magic
    return models


@pytest.fixture
def tmp_nested_models_dir(tmp_path: Path) -> Path:
    """Return a nested models directory with GGUF files below the root."""
    models = tmp_path / "models"
    nested = models / "publisher" / "repo"
    nested.mkdir(parents=True)
    (nested / "nested-model.Q4_K_M.gguf").write_bytes(b"GGUF" + b"\x00" * 100)
    return models


@pytest.fixture
def base_args() -> SimpleNamespace:
    """Minimal args namespace used as the base for cmd_* calls."""
    return SimpleNamespace(
        skip_download=True,
        json=False,
        models_action="list",
        repo=None,
        filename=None,
        name=None,
    )


# ---------------------------------------------------------------------------
# _detect_hardware
# ---------------------------------------------------------------------------


class TestDetectHardware:
    def test_returns_dict_with_required_keys(self) -> None:
        """Hardware dict must always contain the expected keys even when psutil
        and pynvml are unavailable."""
        with (
            patch.dict(sys.modules, {"psutil": None, "pynvml": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            # Direct call — ImportError from psutil/pynvml is handled gracefully
            hw = _detect_hardware()
        # When mocked import raises, cpu_count falls back to os.cpu_count
        assert isinstance(hw, dict)
        required = {"cpu_count", "ram_gb", "gpu_name", "vram_gb", "cuda_available"}
        assert required.issubset(hw.keys())

    def test_returns_dict_with_psutil(self) -> None:
        """When psutil is available, ram_gb should be positive."""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * 1024**3
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            hw = _detect_hardware()
        assert hw["ram_gb"] == 16.0

    def test_gpu_detected_via_pynvml(self) -> None:
        """When pynvml succeeds, gpu_name and vram_gb are populated."""
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value.total = 24 * 1024**3
        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            hw = _detect_hardware()
        assert hw["gpu_name"] == "NVIDIA RTX 4090"
        assert hw["vram_gb"] == 24.0
        assert hw["cuda_available"] is True


# ---------------------------------------------------------------------------
# _get_recommended_models
# ---------------------------------------------------------------------------


class TestGetRecommendedModels:
    @pytest.mark.parametrize(
        "vram_gb",
        [
            0.0,
            2.0,
            4.0,
            6.0,
            8.0,
            12.0,
            16.0,
            32.0,
        ],
    )
    def test_tier_selection_returns_candidates(self, vram_gb: float) -> None:
        """Each VRAM tier returns at least one backend-aware candidate."""
        models = _get_recommended_models(vram_gb)
        assert len(models) > 0
        assert all("backend" in model for model in models)
        assert all("format" in model for model in models)

    def test_gpu_tier_prefers_native_formats(self) -> None:
        models = _get_recommended_models(16.0)
        assert models[0]["backend"] in {"nim", "vllm"}
        assert models[0]["format"] in {"awq", "gptq", "safetensors"}

    def test_each_model_has_required_keys(self) -> None:
        """Every recommended model dict must have name, repo, filename, url."""
        for vram in (0.0, 4.0, 8.0, 16.0):
            for model in _get_recommended_models(vram):
                assert "name" in model
                assert "repo" in model
                assert "filename" in model
                assert "url" in model


# ---------------------------------------------------------------------------
# _guess_quantization and _guess_family
# ---------------------------------------------------------------------------


class TestGuessQuantization:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("mistral-7b-instruct-v0.2.Q4_K_M.gguf", "Q4_K_M"),
            ("llama-3-8b.Q6_K.gguf", "Q6_K"),
            ("model.F16.gguf", "F16"),
            ("model.Q8_0.gguf", "Q8_0"),
            ("unknown-model.gguf", "unknown"),
        ],
    )
    def test_known_quantizations(self, filename: str, expected: str) -> None:
        assert _guess_quantization(filename) == expected


class TestGuessFamily:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("mistral-7b-instruct.Q4_K_M.gguf", "mistral"),
            ("Meta-Llama-3.1-8B.gguf", "llama"),
            ("phi-2.Q4_K_M.gguf", "phi"),
            ("tinyllama-1.1b.gguf", "tinyllama"),
            ("codestral-22b.Q4_K_M.gguf", "codestral"),
            ("random-unnamed-model.gguf", "unknown"),
        ],
    )
    def test_family_detection(self, filename: str, expected: str) -> None:
        assert _guess_family(filename) == expected


# ---------------------------------------------------------------------------
# _models_list
# ---------------------------------------------------------------------------


class TestModelsList:
    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        """Listing an empty models directory is informational and returns 0."""
        result = _models_list(tmp_path)
        assert result == 0

    def test_lists_gguf_files(self, tmp_models_dir: Path, capsys: pytest.CaptureFixture) -> None:
        """Present .gguf files are shown in plain-text output."""
        import vetinari.cli_packaging_models as _models_pkg

        with (
            patch.object(_models_pkg, "_RICH_AVAILABLE", False),
            patch.object(_models_pkg, "_console", None),
        ):
            result = _models_list(tmp_models_dir)
        assert result == 0
        captured = capsys.readouterr()
        assert "mistral" in captured.out.lower() or "llama" in captured.out.lower()

    def test_lists_nested_gguf_files(self, tmp_nested_models_dir: Path) -> None:
        """Nested GGUF files should be discovered under the models root."""
        import vetinari.cli_packaging_models as _models_pkg

        discovered = _models_pkg._iter_model_files(tmp_nested_models_dir)
        assert [path.name for path in discovered] == ["nested-model.Q4_K_M.gguf"]

    def test_filters_local_models_by_family_quant_and_type(self, tmp_models_dir: Path) -> None:
        """Local listing filters should narrow the file scan deterministically."""
        import vetinari.cli_packaging_models as _models_pkg

        discovered = _models_pkg._iter_model_files(
            tmp_models_dir,
            family="llama",
            quantization="Q6_K",
            file_type="gguf",
        )
        assert [path.name for path in discovered] == ["llama-3-8b.Q6_K.gguf"]


# ---------------------------------------------------------------------------
# _models_download
# ---------------------------------------------------------------------------


class TestModelsDownload:
    def test_missing_repo_returns_one(self, tmp_path: Path) -> None:
        result = _models_download(None, "model.gguf", tmp_path)
        assert result == 1

    def test_missing_filename_returns_one(self, tmp_path: Path) -> None:
        result = _models_download("owner/repo", None, tmp_path, backend="llama_cpp")
        assert result == 1

    def test_default_download_without_filename_uses_native_snapshot(self, tmp_path: Path) -> None:
        mock_hf = MagicMock()
        snapshot_dir = tmp_path / "native" / "vllm" / "safetensors" / "owner--repo" / ("d" * 40)
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch(
                "vetinari.model_discovery.ModelDiscovery.download_model",
                return_value={
                    "path": str(snapshot_dir),
                    "revision": "d" * 40,
                    "backend": "vllm",
                    "format": "safetensors",
                    "file_count": 2,
                },
            ) as mock_download,
        ):
            result = _models_download("owner/repo", None, tmp_path)
        assert result == 0
        mock_download.assert_called_once_with(
            "owner/repo",
            None,
            models_dir=tmp_path,
            revision=None,
            backend="vllm",
            model_format=None,
        )

    def test_invalid_existing_file_returns_one(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        dest.write_bytes(b"not a gguf")
        mock_hf = MagicMock()
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch(
                "vetinari.model_discovery.ModelDiscovery.download_model",
                side_effect=ValueError("not a valid GGUF file"),
            ),
        ):
            result = _models_download("owner/repo", "model.gguf", tmp_path)
        assert result == 1

    def test_valid_existing_file_returns_zero_after_validation(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        dest.write_bytes(b"GGUFdata")
        mock_hf = MagicMock()
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch(
                "vetinari.model_discovery.ModelDiscovery.download_model",
                return_value={"path": str(dest), "revision": "a" * 40, "sha256": "b" * 64},
            ),
        ):
            result = _models_download("owner/repo", "model.gguf", tmp_path)
        assert result == 0

    def test_huggingface_hub_unavailable(self, tmp_path: Path) -> None:
        """When huggingface_hub is not installed, returns 1 with helpful message."""
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            result = _models_download("owner/repo", "model.gguf", tmp_path)
        assert result == 1

    def test_successful_download(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.gguf"
        mock_hf = MagicMock()
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch(
                "vetinari.model_discovery.ModelDiscovery.download_model",
                return_value={"path": str(dest), "revision": "a" * 40, "sha256": "b" * 64},
            ),
        ):
            result = _models_download("owner/repo", "model.gguf", tmp_path)
        assert result == 0

    def test_download_exception_returns_one(self, tmp_path: Path) -> None:
        mock_hf = MagicMock()
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch("vetinari.model_discovery.ModelDiscovery.download_model", side_effect=OSError("network error")),
        ):
            result = _models_download("owner/repo", "model.gguf", tmp_path)
        assert result == 1

    def test_native_download_does_not_require_filename(self, tmp_path: Path) -> None:
        mock_hf = MagicMock()
        snapshot_dir = tmp_path / "native" / "vllm" / "safetensors" / "owner--repo" / ("c" * 40)
        with (
            patch.dict(sys.modules, {"huggingface_hub": mock_hf}),
            patch(
                "vetinari.model_discovery.ModelDiscovery.download_model",
                return_value={
                    "path": str(snapshot_dir),
                    "revision": "c" * 40,
                    "backend": "vllm",
                    "format": "safetensors",
                    "manifest_path": str(snapshot_dir / ".vetinari-download.json"),
                    "file_count": 3,
                },
            ) as mock_download,
        ):
            result = _models_download(
                "owner/repo",
                None,
                tmp_path / "native",
                backend="vllm",
                model_format="safetensors",
            )
        assert result == 0
        mock_download.assert_called_once_with(
            "owner/repo",
            None,
            models_dir=tmp_path / "native",
            revision=None,
            backend="vllm",
            model_format="safetensors",
        )


class TestModelsFiles:
    def test_lists_repo_files_with_filters(self, capsys: pytest.CaptureFixture) -> None:
        with patch(
            "vetinari.model_discovery.ModelDiscovery.get_repo_files",
            return_value=[
                {
                    "filename": "model.safetensors",
                    "file_type": "safetensors",
                    "size": 1024**3,
                    "quantization": "AWQ",
                    "revision": "c" * 40,
                }
            ],
        ) as mock_files:
            result = _models_files(
                "owner/repo",
                backend="vllm",
                model_format="awq",
                objective="coding",
                family="qwen",
                file_type="safetensors",
            )
        assert result == 0
        captured = capsys.readouterr()
        assert "model.safetensors" in captured.out
        mock_files.assert_called_once()


# ---------------------------------------------------------------------------
# _models_remove
# ---------------------------------------------------------------------------


class TestModelsRemove:
    def test_missing_name_returns_one(self, tmp_path: Path) -> None:
        result = _models_remove(None, tmp_path)
        assert result == 1

    def test_no_match_returns_one(self, tmp_path: Path) -> None:
        result = _models_remove("nonexistent", tmp_path)
        assert result == 1

    def test_multiple_matches_returns_one(self, tmp_models_dir: Path) -> None:
        # "llama" and "mistral" are in the dir; searching for ".gguf" hits all
        result = _models_remove(".gguf", tmp_models_dir)
        assert result == 1

    def test_confirm_no_aborts(self, tmp_models_dir: Path) -> None:
        with patch("builtins.input", return_value="n"):
            result = _models_remove("mistral", tmp_models_dir)
        assert result == 1

    def test_confirm_yes_removes_file(self, tmp_models_dir: Path) -> None:
        target = tmp_models_dir / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        assert target.exists()
        with (
            patch("builtins.input", return_value="y"),
            patch.object(Path, "unlink", autospec=True) as mock_unlink,
        ):
            result = _models_remove("mistral", tmp_models_dir)
        assert result == 0
        mock_unlink.assert_called_once_with(target)


# ---------------------------------------------------------------------------
# _models_info
# ---------------------------------------------------------------------------


class TestModelsInfo:
    def test_missing_name_returns_one(self, tmp_path: Path) -> None:
        result = _models_info(None, tmp_path)
        assert result == 1

    def test_no_match_returns_one(self, tmp_path: Path) -> None:
        result = _models_info("ghost", tmp_path)
        assert result == 1

    def test_valid_model_info(self, tmp_models_dir: Path, capsys: pytest.CaptureFixture) -> None:
        result = _models_info("mistral", tmp_models_dir)
        assert result == 0
        captured = capsys.readouterr()
        assert "Q4_K_M" in captured.out
        assert "mistral" in captured.out.lower()
        assert "valid" in captured.out.lower()

    def test_nested_model_info(self, tmp_nested_models_dir: Path, capsys: pytest.CaptureFixture) -> None:
        result = _models_info("nested-model", tmp_nested_models_dir)
        assert result == 0
        captured = capsys.readouterr()
        assert "nested-model.Q4_K_M.gguf" in captured.out
        assert "valid" in captured.out.lower()


# ---------------------------------------------------------------------------
# _models_recommend
# ---------------------------------------------------------------------------


class TestModelsRecommend:
    def test_prints_models_and_returns_zero(self, capsys: pytest.CaptureFixture) -> None:
        result = _models_recommend(0.0)
        assert result == 0
        captured = capsys.readouterr()
        assert "Repo" in captured.out or "repo" in captured.out.lower()

    def test_16gb_tier(self, capsys: pytest.CaptureFixture) -> None:
        result = _models_recommend(16.0)
        assert result == 0
        captured = capsys.readouterr()
        assert "Backend" in captured.out
        assert "vllm" in captured.out.lower() or "nim" in captured.out.lower()


# ---------------------------------------------------------------------------
# cmd_models
# ---------------------------------------------------------------------------


class TestCmdModels:
    def test_list_action(self, base_args: SimpleNamespace, tmp_models_dir: Path) -> None:
        base_args.models_action = "list"
        with patch("vetinari.cli_packaging._find_models_dir", return_value=tmp_models_dir):
            result = cmd_models(base_args)
        assert result == 0

    def test_recommend_action(self, base_args: SimpleNamespace) -> None:
        base_args.models_action = "recommend"
        with patch(
            "vetinari.cli_packaging._detect_hardware",
            return_value={
                "cpu_count": 8,
                "ram_gb": 32.0,
                "gpu_name": None,
                "vram_gb": 0.0,
                "cuda_available": False,
            },
        ):
            result = cmd_models(base_args)
        assert result == 0

    def test_unknown_action(self, base_args: SimpleNamespace) -> None:
        base_args.models_action = "explode"
        result = cmd_models(base_args)
        assert result == 1

    def test_download_missing_args(self, base_args: SimpleNamespace, tmp_path: Path) -> None:
        base_args.models_action = "download"
        with patch("vetinari.cli_packaging._find_models_dir", return_value=tmp_path):
            result = cmd_models(base_args)
        assert result == 1

    def test_remove_missing_name(self, base_args: SimpleNamespace, tmp_path: Path) -> None:
        base_args.models_action = "remove"
        with patch("vetinari.cli_packaging._find_models_dir", return_value=tmp_path):
            result = cmd_models(base_args)
        assert result == 1

    def test_info_missing_name(self, base_args: SimpleNamespace, tmp_path: Path) -> None:
        base_args.models_action = "info"
        with patch("vetinari.cli_packaging._find_models_dir", return_value=tmp_path):
            result = cmd_models(base_args)
        assert result == 1


# ---------------------------------------------------------------------------
# cmd_doctor
# ---------------------------------------------------------------------------


class TestCmdDoctor:
    def test_returns_0_when_no_check_fails(self, base_args: SimpleNamespace) -> None:
        """cmd_doctor should return 0 when the diagnostic suite has no failures.

        Patches environment-dependent checks that fail in a clean dev environment
        (no .gguf files, no registered local backend) so the test exercises the
        cmd_doctor exit-code logic, not environment-specific state.
        """
        with patch.dict(
            cmd_doctor.__globals__,
            {
                "_check_database": lambda: ("SQLite database", _CHECK_PASS, "connection OK"),
                "_check_backend_registration": lambda: (
                    "Backend registration",
                    _CHECK_PASS,
                    "registered=['local'] fallback=['local']",
                ),
            },
        ):
            result = cmd_doctor(base_args)
        assert result == 0

    def test_json_flag_emits_valid_json(self, base_args: SimpleNamespace, capsys: pytest.CaptureFixture) -> None:
        base_args.json = True
        cmd_doctor(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 26
        for item in data:
            assert "label" in item
            assert "status" in item
            assert "detail" in item

    def test_all_expected_checks_present_in_json(
        self, base_args: SimpleNamespace, capsys: pytest.CaptureFixture
    ) -> None:
        base_args.json = True
        cmd_doctor(base_args)
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 26
        labels = {item["label"] for item in data}
        assert "vLLM package" in labels
        assert "vLLM endpoint" in labels
        assert "NIM endpoint" in labels
        assert "Dependency readiness matrix" in labels
        assert "Backend registration" in labels

        matrix_item = next(item for item in data if item["label"] == "Dependency readiness matrix")
        assert "matrix" in matrix_item
        assert "summary" in matrix_item
        packages = {entry["package"] for entry in matrix_item["matrix"]}
        assert {
            "pydantic",
            "litestar",
            "llama_cpp",
            "torch",
            "pynvml",
            "vllm",
            "duckduckgo_search",
            "pytest_cov",
            "pytest_asyncio",
            "pytest_xdist",
            "schemathesis",
        }.issubset(packages)

    def test_episode_memory_check_reports_stats(
        self, base_args: SimpleNamespace, capsys: pytest.CaptureFixture
    ) -> None:
        """Episode memory doctor output should reflect the runtime stats API."""
        base_args.json = True
        _mock_episode_mem.get_stats.reset_mock()

        cmd_doctor(base_args)

        data = json.loads(capsys.readouterr().out)
        episode_memory = next(item for item in data if item["label"] == "Episode memory")

        assert episode_memory["status"] == _CHECK_PASS
        assert "stored episode(s)" in episode_memory["detail"]
        _mock_episode_mem.get_stats.assert_called_once_with()

    def test_plain_text_output(self, base_args: SimpleNamespace, capsys: pytest.CaptureFixture) -> None:
        """Plain-text output includes the summary line with check counts."""
        import vetinari.cli_packaging_doctor as _doc_pkg

        base_args.json = False
        with patch.object(_doc_pkg, "_RICH_AVAILABLE", False):
            cmd_doctor(base_args)
        captured = capsys.readouterr()
        # Summary line is always printed to stdout regardless of rich availability
        assert "checks:" in captured.out
        assert "passed" in captured.out

    def test_fails_on_broken_database(self, base_args: SimpleNamespace) -> None:
        """If the database check raises, cmd_doctor should return 1."""
        broken_db = MagicMock()
        broken_db.get_connection.side_effect = RuntimeError("db gone")
        with patch.dict(sys.modules, {"vetinari.database": broken_db}):
            result = cmd_doctor(base_args)
        assert result == 1


# ---------------------------------------------------------------------------
# cmd_init
# ---------------------------------------------------------------------------


class TestCmdInit:
    def setup_method(self) -> None:
        """Force the non-rich output path for all TestCmdInit tests.

        When the full test suite runs, some earlier test may load the real
        ``rich`` package, replacing the module-level stubs installed at the
        top of this file.  ``cli_packaging_data`` would then hold a real
        ``rich.console.Console`` instance as ``_console``.  Calling
        ``_console.rule(...)`` triggers rich's lazy internal import of
        ``rich.rule`` — which goes through ``builtins.__import__`` and hits
        the ``_raise_import`` mock in ``test_import_failure_returns_one``,
        causing an unexpected ``ImportError``.

        Resetting ``_RICH_AVAILABLE`` to ``False`` forces the plain-text
        fallback path, so no rich submodule imports occur while
        ``builtins.__import__`` is patched.
        """
        import vetinari.cli_packaging_data as _data

        self._orig_rich_available = _data._RICH_AVAILABLE
        self._orig_console = _data._console
        _data._RICH_AVAILABLE = False
        _data._console = None

    def teardown_method(self) -> None:
        """Restore the rich availability state changed in setup_method."""
        import vetinari.cli_packaging_data as _data

        _data._RICH_AVAILABLE = self._orig_rich_available
        _data._console = self._orig_console

    @pytest.fixture
    def init_args(self) -> SimpleNamespace:
        """Minimal args namespace for cmd_init with skip_download=True."""
        return SimpleNamespace(skip_download=True)

    def test_wizard_failure_returns_one(self, init_args: SimpleNamespace) -> None:
        """When the maintained setup wizard reports failure, cmd_init returns 1."""
        with patch(
            "vetinari.setup.init_wizard.run_wizard",
            return_value=SimpleNamespace(success=False),
        ) as mock_wizard:
            result = cmd_init(init_args)
        assert result == 1
        mock_wizard.assert_called_once_with(skip_download=True)

    def test_successful_init_uses_maintained_wizard(self, init_args: SimpleNamespace) -> None:
        """A successful init delegates to setup.init_wizard.run_wizard."""
        with patch(
            "vetinari.setup.init_wizard.run_wizard",
            return_value=SimpleNamespace(success=True),
        ) as mock_wizard:
            result = cmd_init(init_args)

        assert result == 0
        mock_wizard.assert_called_once_with(skip_download=True)

    def test_skip_download_flag_respected(
        self, init_args: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """When --skip-download is set, the download step is skipped."""
        with patch(
            "vetinari.setup.init_wizard.run_wizard",
            return_value=SimpleNamespace(success=True),
        ) as mock_wizard:
            result = cmd_init(init_args)

        assert result == 0
        mock_wizard.assert_called_once_with(skip_download=True)


# ---------------------------------------------------------------------------
# _register_packaging_commands
# ---------------------------------------------------------------------------


class TestRegisterPackagingCommands:
    @pytest.fixture
    def parser(self) -> argparse.ArgumentParser:
        """ArgumentParser with all packaging subcommands registered."""
        p = argparse.ArgumentParser(prog="vetinari")
        subparsers = p.add_subparsers(dest="command")
        _register_packaging_commands(subparsers)
        return p

    def test_init_subparser_registered(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.skip_download is False

    def test_init_skip_download_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["init", "--skip-download"])
        assert args.skip_download is True

    def test_doctor_subparser_registered(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"
        assert args.json is False

    def test_doctor_json_flag(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["doctor", "--json"])
        assert args.json is True

    def test_models_subparser_registered_list(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["models", "list"])
        assert args.command == "models"
        assert args.models_action == "list"
        assert args.backend == "auto"

    @pytest.mark.parametrize("action", ["list", "files", "download", "status", "remove", "info", "recommend"])
    def test_models_all_actions_accepted(self, parser: argparse.ArgumentParser, action: str) -> None:
        args = parser.parse_args(["models", action])
        assert args.models_action == action

    def test_models_download_with_repo_and_filename(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args([
            "models",
            "download",
            "--repo",
            "TheBloke/Mistral-7B-GGUF",
            "--filename",
            "mistral.Q4_K_M.gguf",
        ])
        assert args.repo == "TheBloke/Mistral-7B-GGUF"
        assert args.filename == "mistral.Q4_K_M.gguf"

    def test_models_download_with_native_backend_and_filters(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args([
            "models",
            "download",
            "--repo",
            "Qwen/Qwen2.5-Coder-7B",
            "--backend",
            "vllm",
            "--format",
            "safetensors",
            "--objective",
            "coding",
            "--family",
            "qwen",
            "--min-size-gb",
            "1",
            "--max-size-gb",
            "20",
        ])
        assert args.backend == "vllm"
        assert args.model_format == "safetensors"
        assert args.objective == "coding"
        assert args.family == "qwen"
        assert args.min_size_gb == 1.0
        assert args.max_size_gb == 20.0

    def test_models_remove_with_name(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["models", "remove", "--name", "mistral"])
        assert args.name == "mistral"

    def test_models_info_with_name(self, parser: argparse.ArgumentParser) -> None:
        args = parser.parse_args(["models", "info", "--name", "llama"])
        assert args.name == "llama"

    def test_models_invalid_action_raises(self, parser: argparse.ArgumentParser) -> None:
        with pytest.raises(SystemExit):
            parser.parse_args(["models", "frobnicate"])
