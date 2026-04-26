"""Runtime configuration path resolution."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PACKAGE_CONFIG_DIR = _PACKAGE_ROOT / "config"
_PACKAGE_RUNTIME_CONFIG_DIR = _PACKAGE_CONFIG_DIR / "runtime"
_PROJECT_CONFIG_DIR = _PACKAGE_ROOT.parent / "config"


def resolve_config_path(*parts: str) -> Path:
    """Return a source-tree config path or the packaged runtime fallback.

    Returns:
        The first existing config path for ``parts``, or the source-tree path
        where the config should be created.
    """
    project_path = _PROJECT_CONFIG_DIR.joinpath(*parts)
    if project_path.exists():
        return project_path

    packaged_runtime_path = _PACKAGE_RUNTIME_CONFIG_DIR.joinpath(*parts)
    if packaged_runtime_path.exists():
        return packaged_runtime_path

    packaged_path = _PACKAGE_CONFIG_DIR.joinpath(*parts)
    if packaged_path.exists():
        return packaged_path

    return project_path
