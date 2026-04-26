"""Tests for vetinari.preflight — dependency checker, CUDA readiness, and installer."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, patch

import pytest

from vetinari.preflight import (
    DependencyGroupStatus,
    DependencyReadiness,
    HardwareInfo,
    PreflightReport,
    _build_pip_environment,
    build_dependency_readiness_matrix,
    check_cuda_readiness,
    check_dependency_groups,
    detect_hardware,
    print_preflight_report,
    prompt_and_install,
    run_preflight,
    summarize_dependency_readiness,
)

# ── HardwareInfo ─────────────────────────────────────────────────────────────


class TestHardwareInfo:
    def test_defaults(self):
        hw = HardwareInfo()
        assert hw.has_nvidia_gpu is False
        assert hw.gpu_name == ""
        assert hw.vram_gb == 0.0
        assert hw.has_cuda_toolkit is False
        assert hw.torch_has_cuda is False
        assert hw.llama_cpp_has_cuda is False
        assert hw.has_docker is False

    def test_repr(self):
        hw = HardwareInfo(has_nvidia_gpu=True, gpu_name="RTX 5090")
        assert repr(hw) == "HardwareInfo(...)"


# ── DependencyGroupStatus ────────────────────────────────────────────────────


class TestDependencyGroupStatus:
    def test_complete_group(self):
        g = DependencyGroupStatus(
            name="Test",
            extra="test",
            description="Test group",
            import_names=("foo", "bar"),
            installed=("foo", "bar"),
            missing=(),
        )
        assert g.is_complete is True
        assert g.is_partial is False

    def test_partial_group(self):
        g = DependencyGroupStatus(
            name="Test",
            extra="test",
            description="Test group",
            import_names=("foo", "bar"),
            installed=("foo",),
            missing=("bar",),
        )
        assert g.is_complete is False
        assert g.is_partial is True

    def test_empty_group(self):
        g = DependencyGroupStatus(
            name="Test",
            extra="test",
            description="Test group",
            import_names=("foo", "bar"),
            installed=(),
            missing=("foo", "bar"),
        )
        assert g.is_complete is False
        assert g.is_partial is False

    def test_repr(self):
        g = DependencyGroupStatus(name="Cloud", extra="cloud", description="d", import_names=("x",))
        assert "Cloud" in repr(g)
        assert "cloud" in repr(g)


# ── detect_hardware ──────────────────────────────────────────────────────────


class TestDetectHardware:
    @patch("vetinari.preflight.Path")
    @patch("vetinari.preflight.shutil.which", return_value=None)
    @patch("vetinari.preflight.importlib.util.find_spec", return_value=None)
    def test_no_gpu_no_tools(self, mock_find_spec, mock_which, mock_path):
        """When no GPU, tools, or packages are available, returns empty HardwareInfo."""
        # Mock Path so CUDA toolkit directory check returns False
        mock_path.return_value.exists.return_value = False
        # Patch pynvml import to raise ImportError
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("pynvml")):
                hw = detect_hardware()
        assert hw.has_nvidia_gpu is False
        assert hw.has_cuda_toolkit is False
        assert hw.has_docker is False

    @patch("vetinari.preflight.shutil.which")
    @patch("vetinari.preflight.importlib.util.find_spec", return_value=None)
    def test_nvidia_smi_fallback(self, mock_find_spec, mock_which):
        """When pynvml is unavailable but nvidia-smi exists, detect GPU via fallback."""

        def which_side_effect(name):
            if name == "nvidia-smi":
                return "/usr/bin/nvidia-smi"
            return None

        mock_which.side_effect = which_side_effect

        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("builtins.__import__", side_effect=_import_blocker("pynvml")):
                hw = detect_hardware()
        assert hw.has_nvidia_gpu is True
        assert "nvidia-smi" in hw.gpu_name.lower()


# ── check_dependency_groups ──────────────────────────────────────────────────


class TestCheckDependencyGroups:
    def test_returns_all_groups(self):
        hw = HardwareInfo()
        groups = check_dependency_groups(hw)
        assert len(groups) > 0
        names = {g.name for g in groups}
        assert "Local Inference (GGUF)" in names
        assert "Cloud LLM Providers" in names

    def test_gpu_sets_recommended_flag(self):
        hw_gpu = HardwareInfo(has_nvidia_gpu=True, gpu_name="RTX 5090", vram_gb=32.0)
        groups = check_dependency_groups(hw_gpu)
        training = next(g for g in groups if g.name == "Training & Fine-tuning")
        assert training.recommended is True

    def test_no_gpu_base_recommended(self):
        hw_cpu = HardwareInfo()
        groups = check_dependency_groups(hw_cpu)
        training = next(g for g in groups if g.name == "Training & Fine-tuning")
        # Training is not in _BASE_RECOMMENDED_GROUPS
        assert training.recommended is False
        # Encryption is in base recommended
        encryption = next(g for g in groups if g.name == "Encryption")
        assert encryption.recommended is True

    @patch("vetinari.preflight.importlib.util.find_spec")
    def test_all_missing(self, mock_find_spec):
        mock_find_spec.return_value = None
        hw = HardwareInfo()
        groups = check_dependency_groups(hw)
        for g in groups:
            assert g.is_complete is False
            assert len(g.missing) == len(g.import_names)

    @patch("vetinari.preflight.importlib.util.find_spec")
    def test_all_installed(self, mock_find_spec):
        mock_find_spec.return_value = MagicMock()  # non-None = found
        hw = HardwareInfo()
        groups = check_dependency_groups(hw)
        for g in groups:
            assert g.is_complete is True
            assert len(g.missing) == 0


class TestDependencyReadinessMatrix:
    @patch("vetinari.preflight.importlib_metadata.version")
    @patch("vetinari.preflight.importlib.util.find_spec")
    def test_matrix_includes_required_optional_and_dev_packages(self, mock_find_spec, mock_version):
        installed = {
            "pydantic",
            "litestar",
            "httpx",
            "defusedxml",
            "llama_cpp",
            "torch",
            "bitsandbytes",
            "pynvml",
            "ddgs",
            "pytest_cov",
            "pytest_asyncio",
            "xdist",
            "schemathesis",
        }

        def find_spec_side_effect(name):
            return MagicMock() if name in installed else None

        def version_side_effect(name):
            known = {
                "pydantic": "2.10.0",
                "litestar": "2.14.0",
                "httpx": "0.28.1",
                "defusedxml": "0.7.1",
                "llama-cpp-python": "0.3.20",
                "torch": "2.8.0",
                "bitsandbytes": "0.49.2",
                "nvidia-ml-py": "12.570.0",
                "ddgs": "9.1.0",
                "pytest-cov": "6.0.0",
                "pytest-asyncio": "0.24.0",
                "pytest-xdist": "3.6.1",
                "schemathesis": "3.39.0",
            }
            if name not in known:
                raise PackageNotFoundError(name)
            return known[name]

        mock_find_spec.side_effect = find_spec_side_effect
        mock_version.side_effect = version_side_effect

        hw = HardwareInfo(
            has_nvidia_gpu=True,
            gpu_name="RTX 5090",
            vram_gb=32.0,
            torch_has_cuda=True,
            llama_cpp_has_cuda=True,
        )
        matrix = build_dependency_readiness_matrix(hw)

        packages = {item.package for item in matrix}
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

        ddgs_item = next(item for item in matrix if item.package == "duckduckgo_search")
        assert ddgs_item.import_name == "ddgs"
        assert ddgs_item.status == "installed-optional"

        xdist_item = next(item for item in matrix if item.package == "pytest_xdist")
        assert xdist_item.import_name == "xdist"
        assert xdist_item.status == "installed-dev"

    @patch("vetinari.preflight.importlib_metadata.version")
    @patch("vetinari.preflight.importlib.util.find_spec")
    def test_gpu_packages_can_be_installed_but_unverified(self, mock_find_spec, mock_version):
        installed = {"pydantic", "litestar", "httpx", "defusedxml", "llama_cpp", "torch", "pynvml"}

        def find_spec_side_effect(name):
            return MagicMock() if name in installed else None

        def version_side_effect(name):
            known = {
                "pydantic": "2.10.0",
                "litestar": "2.14.0",
                "httpx": "0.28.1",
                "defusedxml": "0.7.1",
                "llama-cpp-python": "0.3.20",
                "torch": "2.8.0",
                "nvidia-ml-py": "12.570.0",
            }
            if name not in known:
                raise PackageNotFoundError(name)
            return known[name]

        mock_find_spec.side_effect = find_spec_side_effect
        mock_version.side_effect = version_side_effect

        hw = HardwareInfo(
            has_nvidia_gpu=True,
            gpu_name="RTX 5090",
            vram_gb=32.0,
            torch_has_cuda=False,
            llama_cpp_has_cuda=False,
        )
        matrix = build_dependency_readiness_matrix(hw)
        torch_item = next(item for item in matrix if item.package == "torch")
        llama_item = next(item for item in matrix if item.package == "llama_cpp")
        vllm_item = next(item for item in matrix if item.package == "vllm")

        assert torch_item.status == "installed-unverified"
        assert "CUDA is not available" in torch_item.detail
        assert llama_item.status == "installed-unverified"
        assert "GPU offload is not available" in llama_item.detail
        assert vllm_item.status == "missing-optional"

    def test_summary_tracks_missing_and_unverified_packages(self):
        matrix = [
            DependencyReadiness(
                package="pydantic",
                import_name="pydantic",
                channel="core",
                expected="required",
                description="Core runtime",
                install_command="pip install vetinari",  # noqa: VET301 — user guidance string
                distribution="pydantic",
                installed=True,
                version="2.10.0",
                runtime_verified=True,
                status="ready",
                detail="importable",
            ),
            DependencyReadiness(
                package="torch",
                import_name="torch",
                channel="extra",
                expected="recommended",
                description="CUDA runtime",
                install_command='pip install "vetinari[image,training]"',  # noqa: VET301 — user guidance string
                distribution="torch",
                installed=True,
                version="2.8.0",
                runtime_verified=False,
                status="installed-unverified",
                detail="importable but CUDA is not available",
            ),
            DependencyReadiness(
                package="litestar",
                import_name=None,
                channel="core",
                expected="required",
                description="ASGI framework",
                install_command="pip install vetinari",  # noqa: VET301 — user guidance string
                distribution="litestar",
                installed=False,
                version=None,
                runtime_verified=None,
                status="missing-required",
                detail="missing",
            ),
            DependencyReadiness(
                package="pytest_xdist",
                import_name=None,
                channel="dev",
                expected="dev",
                description="Parallel test execution",
                install_command='pip install "vetinari[dev]"',  # noqa: VET301 — user guidance string
                distribution="pytest-xdist",
                installed=False,
                version=None,
                runtime_verified=None,
                status="missing-dev",
                detail="missing",
            ),
        ]

        summary = summarize_dependency_readiness(matrix)
        assert summary["total"] == 4
        assert summary["installed"] == 2
        assert summary["missing_required"] == ["litestar"]
        assert summary["installed_unverified"] == ["torch"]
        assert summary["missing_dev"] == ["pytest_xdist"]


# ── check_cuda_readiness ────────────────────────────────────────────────────


class TestCheckCudaReadiness:
    def test_no_gpu_returns_empty(self):
        hw = HardwareInfo()
        actions = check_cuda_readiness(hw)
        assert actions == []

    def test_fully_ready_returns_empty(self):
        hw = HardwareInfo(
            has_nvidia_gpu=True,
            has_cuda_toolkit=True,
            torch_has_cuda=True,
            llama_cpp_has_cuda=True,
            has_docker=True,
        )
        with patch("vetinari.preflight.importlib.util.find_spec", return_value=MagicMock()):
            actions = check_cuda_readiness(hw)
        assert actions == []

    def test_missing_cuda_toolkit(self):
        hw = HardwareInfo(
            has_nvidia_gpu=True,
            has_cuda_toolkit=False,
            torch_has_cuda=True,
            llama_cpp_has_cuda=True,
            has_docker=True,
        )
        with patch("vetinari.preflight.importlib.util.find_spec", return_value=MagicMock()):
            actions = check_cuda_readiness(hw)
        assert any("CUDA Toolkit" in a for a in actions)

    def test_torch_cpu_only(self):
        hw = HardwareInfo(
            has_nvidia_gpu=True,
            has_cuda_toolkit=True,
            torch_has_cuda=False,
            llama_cpp_has_cuda=True,
            has_docker=True,
        )
        with patch("vetinari.preflight.importlib.util.find_spec", return_value=MagicMock()):
            actions = check_cuda_readiness(hw)
        assert any("PyTorch" in a for a in actions)

    def test_no_docker(self):
        hw = HardwareInfo(
            has_nvidia_gpu=True,
            has_cuda_toolkit=True,
            torch_has_cuda=True,
            llama_cpp_has_cuda=True,
            has_docker=False,
        )
        with patch("vetinari.preflight.importlib.util.find_spec", return_value=MagicMock()):
            actions = check_cuda_readiness(hw)
        assert any("Docker" in a for a in actions)


# ── print_preflight_report ───────────────────────────────────────────────────


class TestPrintPreflightReport:
    def test_prints_without_error(self, capsys):
        report = PreflightReport(
            hardware=HardwareInfo(has_nvidia_gpu=True, gpu_name="RTX 5090", vram_gb=32.0),
            groups=[
                DependencyGroupStatus(
                    name="Test Group",
                    extra="test",
                    description="A test",
                    import_names=("foo",),
                    installed=("foo",),
                    missing=(),
                    recommended=True,
                ),
                DependencyGroupStatus(
                    name="Missing Group",
                    extra="missing",
                    description="Not installed",
                    import_names=("bar",),
                    installed=(),
                    missing=("bar",),
                    recommended=True,
                ),
            ],
            system_actions=["Install CUDA Toolkit"],
        )
        print_preflight_report(report)
        captured = capsys.readouterr()
        assert "RTX 5090" in captured.out
        assert "CUDA Toolkit" in captured.out
        assert "Test Group" in captured.out
        assert "Missing Group" in captured.out

    def test_no_gpu_output(self, capsys):
        report = PreflightReport(hardware=HardwareInfo(), groups=[], system_actions=[])
        print_preflight_report(report)
        captured = capsys.readouterr()
        assert "not detected" in captured.out


# ── prompt_and_install ───────────────────────────────────────────────────────


class TestPromptAndInstall:
    def test_nothing_to_install(self):
        report = PreflightReport(
            hardware=HardwareInfo(),
            groups=[
                DependencyGroupStatus(
                    name="G",
                    extra="g",
                    description="d",
                    import_names=("x",),
                    installed=("x",),
                    missing=(),
                    recommended=True,
                ),
            ],
        )
        result = prompt_and_install(report)
        assert result is False

    @patch("builtins.input", return_value="n")
    def test_user_declines(self, mock_input):
        report = PreflightReport(
            hardware=HardwareInfo(),
            groups=[
                DependencyGroupStatus(
                    name="G",
                    extra="g",
                    description="d",
                    import_names=("x",),
                    installed=(),
                    missing=("x",),
                    recommended=True,
                ),
            ],
        )
        result = prompt_and_install(report)
        assert result is False

    @patch("vetinari.preflight.subprocess.check_call")
    @patch(
        "vetinari.preflight._build_pip_environment",
        return_value={"TEMP": "C:/tmp", "TMP": "C:/tmp", "TMPDIR": "C:/tmp", "PIP_CACHE_DIR": "C:/cache"},
    )
    @patch("builtins.input", return_value="y")
    def test_user_approves(self, mock_input, mock_build_env, mock_check_call):
        report = PreflightReport(
            hardware=HardwareInfo(),
            groups=[
                DependencyGroupStatus(
                    name="G",
                    extra="g",
                    description="d",
                    import_names=("x",),
                    installed=(),
                    missing=("x",),
                    recommended=True,
                ),
            ],
        )
        result = prompt_and_install(report)
        assert result is True
        mock_check_call.assert_called_once()
        cmd = mock_check_call.call_args[0][0]
        env = mock_check_call.call_args.kwargs["env"]
        assert cmd[1] == "-c"
        assert "vetinari[g]" in cmd[-1]
        assert env["TEMP"] == "C:/tmp"
        assert env["PIP_CACHE_DIR"] == "C:/cache"


class TestBuildPipEnvironment:
    @patch("vetinari.preflight.Path.home")
    @patch("vetinari.preflight.Path.cwd")
    def test_prefers_writable_repo_local_root(self, mock_cwd, mock_home, tmp_path):
        mock_cwd.return_value = tmp_path
        mock_home.return_value = tmp_path / "home"

        env = _build_pip_environment()

        assert env["TEMP"].startswith(str(tmp_path / ".pip-work"))
        assert env["PIP_CACHE_DIR"].startswith(str(tmp_path / ".pip-work"))


# ── run_preflight ────────────────────────────────────────────────────────────


class TestRunPreflight:
    @patch("vetinari.preflight.prompt_and_install")  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.print_preflight_report")  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.check_cuda_readiness", return_value=[])  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.check_dependency_groups", return_value=[])  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.detect_hardware", return_value=HardwareInfo())  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    def test_non_interactive_skips_prompt(self, mock_hw, mock_groups, mock_cuda, mock_print, mock_install):
        report = run_preflight(interactive=False)
        assert isinstance(report, PreflightReport)
        mock_print.assert_called_once()
        mock_install.assert_not_called()

    @patch("vetinari.preflight.prompt_and_install")  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.print_preflight_report")  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.check_cuda_readiness", return_value=[])  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.check_dependency_groups", return_value=[])  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    @patch("vetinari.preflight.detect_hardware", return_value=HardwareInfo())  # noqa: VET242 - run_preflight unit intentionally isolates helper orchestration
    def test_interactive_calls_prompt(self, mock_hw, mock_groups, mock_cuda, mock_print, mock_install):
        report = run_preflight(interactive=True)
        mock_install.assert_called_once()
        assert isinstance(report, PreflightReport)
        assert mock_print.call_args.args[0] is report
        assert mock_install.call_args.args[0] is report


# ── Helpers ──────────────────────────────────────────────────────────────────


def _import_blocker(*blocked_names):
    """Return an __import__ replacement that blocks specific module names."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocked_import(name, *args, **kwargs):
        if name in blocked_names:
            raise ImportError(f"Blocked for test: {name}")
        return original_import(name, *args, **kwargs)

    return _blocked_import
