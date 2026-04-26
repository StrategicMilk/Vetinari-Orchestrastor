"""Release packaging metadata and manifest boundary checks."""

from __future__ import annotations

import ast
import configparser
import re
from email.parser import Parser
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"
PKG_INFO = ROOT / "vetinari.egg-info" / "PKG-INFO"
SOURCES = ROOT / "vetinari.egg-info" / "SOURCES.txt"
ENTRY_POINTS = ROOT / "vetinari.egg-info" / "entry_points.txt"
REQUIRES = ROOT / "vetinari.egg-info" / "requires.txt"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_vetinari.py"
ALLOWED_TOP_LEVEL_RELEASE_FILES = {
    "CHANGELOG.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "README.md",
    "THIRD-PARTY-LICENSES.md",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
}
ALLOWED_EGG_INFO_FILES = {
    "vetinari.egg-info/PKG-INFO",
    "vetinari.egg-info/SOURCES.txt",
    "vetinari.egg-info/dependency_links.txt",
    "vetinari.egg-info/entry_points.txt",
    "vetinari.egg-info/requires.txt",
    "vetinari.egg-info/top_level.txt",
}


def _load_pyproject() -> dict:
    if tomllib is None:
        pytest.skip("tomllib is required to parse pyproject.toml")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _normalized_requirement(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).replace("_", "-").lower()


def _metadata() -> object:
    return Parser().parsestr(PKG_INFO.read_text(encoding="utf-8"))


def _requires_sections() -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for raw_line in REQUIRES.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def test_package_python_classifiers_match_verified_horizon() -> None:
    pyproject = _load_pyproject()
    classifiers = pyproject["project"]["classifiers"]
    pkg_classifiers = _metadata().get_all("Classifier") or []

    assert "Programming Language :: Python :: 3.14" not in classifiers
    assert "Programming Language :: Python :: 3.14" not in pkg_classifiers
    assert set(classifiers) == set(pkg_classifiers)


def test_package_setup_py_delegates_to_pyproject_metadata() -> None:
    source = SETUP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    setup_calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    ]

    assert len(setup_calls) == 1
    assert setup_calls[0].keywords == []
    assert "install_requires" not in source
    assert "extras_require" not in source
    assert "Programming Language :: Python :: 3.14" not in source
    assert "flask" not in source.lower()
    assert "duckduckgo-search" not in source.lower()


def test_package_requires_metadata_matches_pyproject_dependencies() -> None:
    pyproject = _load_pyproject()
    requires = _requires_sections()
    pyproject_base = {_normalized_requirement(req) for req in pyproject["project"]["dependencies"]}
    generated_base = {_normalized_requirement(req) for req in requires[""]}
    pkg_requires = {_normalized_requirement(req) for req in (_metadata().get_all("Requires-Dist") or [])}

    assert generated_base == pyproject_base
    assert pyproject_base.issubset(pkg_requires)
    assert not any(req.startswith("flask") for req in generated_base)
    assert not any(req.startswith("duckduckgo-search") for req in generated_base)


def test_package_optional_dependency_metadata_matches_release_constraints() -> None:
    pyproject = _load_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    observability = {_normalized_requirement(req) for req in optional["observability"]}
    vllm = {_normalized_requirement(req) for req in optional["vllm"]}
    all_extra = {_normalized_requirement(req) for req in optional["all"]}

    assert "openllmetry-sdk>=0.30" not in observability
    assert "opentelemetry-api>=1.20" in observability
    assert "opentelemetry-sdk>=1.20" in observability
    assert "opentelemetry-exporter-otlp-proto-http>=1.20" in observability
    assert "vllm>=0.13,!=0.18.1;platform-system!='windows'" in vllm
    guardrails = {_normalized_requirement(req) for req in optional["guardrails"]}
    assert "llm-guard==0.3.16" in guardrails
    assert "fastembed>=0.8.0" in guardrails
    assert "semgrep==1.161.0;platform-system!='windows'" in {_normalized_requirement(req) for req in optional["dev"]}
    assert (
        "vetinari[local,cloud,vllm,crypto,image,search,ml,guardrails,observability,training,watcher,notifications,llmlingua,dev]"
        in all_extra
    )


def test_package_entrypoints_do_not_publish_asgi_factory_as_console_script() -> None:
    parser = configparser.ConfigParser()
    parser.read(ENTRY_POINTS, encoding="utf-8")
    console_scripts = dict(parser.items("console_scripts"))

    assert console_scripts == {"vetinari": "vetinari.__main__:main"}
    assert "vetinari-asgi" not in console_scripts


def test_package_entrypoint_bootstraps_environment_before_cli_import() -> None:
    source = (ROOT / "vetinari" / "__main__.py").read_text(encoding="utf-8")

    assert source.index("vetinari.bootstrap_environment()") < source.index("from vetinari.cli import main")


def test_package_data_declares_runtime_configs_without_parent_globs() -> None:
    pyproject = _load_pyproject()
    patterns = pyproject["tool"]["setuptools"]["package-data"]["vetinari"]

    assert all(".." not in Path(pattern).parts for pattern in patterns)
    assert "../ui/**/*" not in patterns
    assert "../config/**/*" not in patterns
    for expected in {
        "py.typed",
        "context_registry.json",
        "skills_registry.json",
        "config/**/*.yaml",
        "config/**/*.json",
        "config/**/*.md",
        "config/runtime/**/*",
        "migrations/*.sql",
        "skills/catalog/**/*.md",
    }:
        assert expected in patterns


def test_packaged_runtime_config_fallbacks_cover_root_config_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    import vetinari.config_paths as config_paths

    monkeypatch.setattr(config_paths, "_PROJECT_CONFIG_DIR", ROOT / ".missing-package-config")

    for parts in [
        ("models.yaml",),
        ("mcp_servers.yaml",),
        ("cloud_providers.yaml",),
        ("error_messages.yaml",),
        ("notifications.yaml",),
        ("document_profiles.yaml",),
        ("writing_style.yaml",),
        ("ml_config.yaml",),
        ("knowledge", "benchmarks.yaml"),
        ("knowledge", "model_families.yaml"),
        ("knowledge", "parameters.yaml"),
        ("knowledge", "architecture.yaml"),
        ("knowledge", "quantization.yaml"),
    ]:
        resolved = config_paths.resolve_config_path(*parts)
        assert resolved.is_file()
        assert ROOT / "vetinari" / "config" / "runtime" in resolved.parents


def test_package_sources_manifest_is_rebuilt_and_release_bounded() -> None:
    lines = [line.strip() for line in SOURCES.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [line for line in lines if not (ROOT / line).exists()]
    top_level_entries = [line.replace("\\", "/") for line in lines if "/" not in line.replace("\\", "/")]
    forbidden_fragments = (
        "../",
        ".agents/",
        ".ai-codex/",
        ".audit_probe",
        ".claude/",
        ".claire/",
        ".codex/",
        ".omc/",
        ".pytest-tmp-root",
        "AGENTS.md",
        "CLAUDE.md",
        "CLEANUP_PLAN.md",
        "node_modules/",
        "ui/",
    )
    forbidden_suffixes = (".map", ".safetensors", ".gguf", ".bin", ".onnx", ".pt", ".pth", ".ckpt")
    forbidden = [
        line
        for line in lines
        if any(fragment in line.replace("\\", "/") for fragment in forbidden_fragments)
        or line.endswith(forbidden_suffixes)
    ]
    egg_info_entries = [
        line.replace("\\", "/") for line in lines if line.replace("\\", "/").startswith("vetinari.egg-info/")
    ]
    unexpected_egg_info = sorted(set(egg_info_entries) - ALLOWED_EGG_INFO_FILES)
    unexpected_top_level = sorted(set(top_level_entries) - ALLOWED_TOP_LEVEL_RELEASE_FILES)

    assert missing == []
    assert forbidden == []
    assert unexpected_egg_info == []
    assert unexpected_top_level == []


def test_package_release_certifier_uses_explicit_failures_not_bare_asserts() -> None:
    tree = ast.parse(VALIDATE_SCRIPT.read_text(encoding="utf-8"))
    bare_asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]

    assert bare_asserts == []
