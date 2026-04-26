"""Run mypy through a pure-Python package copy on Windows.

The compiled mypy wheels can crash on this host during real type-check runs.
This wrapper copies the installed package into a repo-local cache without the
compiled ``*.pyd`` modules and then runs mypy from that pure-Python copy.
"""

from __future__ import annotations

import runpy
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PURE_ROOT = ROOT / ".vetinari" / "tool-cache" / "mypy-pure"
PURE_PACKAGE = PURE_ROOT / "mypy"
VERSION_FILE = PURE_ROOT / ".version"


def _installed_mypy_root() -> Path:
    distribution = metadata.distribution("mypy")
    for file in distribution.files or []:
        if str(file) == "mypy/__init__.py":
            return Path(distribution.locate_file(file)).parent
    msg = "Could not locate installed mypy package"
    raise RuntimeError(msg)


def _copy_package_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "*.pyd",
            "*.so",
            "__pycache__",
            "_compiled_backup",
        ),
    )


def ensure_pure_mypy() -> Path:
    version = metadata.version("mypy")
    source = _installed_mypy_root()
    PURE_ROOT.mkdir(parents=True, exist_ok=True)
    current_version = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else ""
    if not PURE_PACKAGE.exists() or current_version != version:
        _copy_package_tree(source, PURE_PACKAGE)
        VERSION_FILE.write_text(version, encoding="utf-8")
    return PURE_ROOT


def _default_cache_dir() -> Path:
    """Return a repo-relative path for the mypy cache directory.

    Returns:
        Path under the repository root, always consistent across platforms.
    """
    return ROOT / ".mypy_cache"


def _apply_safe_defaults(argv: list[str]) -> list[str]:
    args = list(argv)
    if not any(arg.startswith("--follow-imports") for arg in args):
        args[:0] = ["--follow-imports=silent"]
    if "--sqlite-cache" not in args and "--no-sqlite-cache" not in args:
        args[:0] = ["--no-sqlite-cache"]
    if "--cache-dir" not in args:
        cache_dir = _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        args[:0] = ["--cache-dir", str(cache_dir)]
    return args


def main() -> int:
    pure_root = ensure_pure_mypy()
    sys.path.insert(0, str(pure_root))
    sys.argv = ["mypy", *_apply_safe_defaults(sys.argv[1:])]
    runpy.run_module("mypy", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
