"""Lazy import utility — consolidates try/import/except patterns.

Replaces dozens of identical ``try: import X; HAS_X = True except: HAS_X = False``
blocks scattered across adapter and integration modules.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)


class _BrokenImportProxy(ModuleType):
    """Falsey module placeholder for imports that found code but failed to load."""

    def __init__(self, module_name: str, error: BaseException) -> None:
        super().__init__(module_name)
        self.import_error = error

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> ModuleType:
        raise ImportError(f"Module {self.__name__!r} failed during import") from self.import_error


def _resolve_module_name(module_name: str, package: str | None) -> str:
    """Resolve relative module names so missing-target checks compare the real path."""
    if module_name.startswith("."):
        return importlib.util.resolve_name(module_name, package)
    return module_name


def _is_missing_target(exc: ImportError, resolved_name: str) -> bool:
    """Return True only when the requested module itself is missing."""
    missing_name = getattr(exc, "name", "") or ""
    return bool(missing_name) and (resolved_name == missing_name or resolved_name.startswith(f"{missing_name}."))


def lazy_import(module_name: str, *, package: str | None = None) -> tuple[ModuleType | None, bool]:
    """Attempt to import a module, returning (module, available) without raising.

    Consolidates the common pattern::

        try:
            import llama_cpp
            HAS_LLAMA_CPP = True
        except ImportError:
            HAS_LLAMA_CPP = False

    Into::

        llama_cpp, HAS_LLAMA_CPP = lazy_import("llama_cpp")

    Args:
        module_name: Fully qualified module name to import.
        package: Package context for relative imports (passed to
            ``importlib.import_module``).

    Returns:
        A ``(module, is_available)`` tuple.  When the import fails,
        ``module`` is ``None`` and ``is_available`` is ``False``.
    """
    resolved_name = _resolve_module_name(module_name, package)
    try:
        mod = importlib.import_module(module_name, package=package)
        return mod, True
    except ImportError as exc:
        if _is_missing_target(exc, resolved_name):
            logger.debug("Optional dependency %r not available", resolved_name)
            return None, False
        logger.warning("Optional dependency %r failed during import", resolved_name, exc_info=True)
        return _BrokenImportProxy(resolved_name, exc), False
    except Exception as exc:
        logger.warning("Optional dependency %r failed during import", resolved_name, exc_info=True)
        return _BrokenImportProxy(resolved_name, exc), False


def require_import(module_name: str, *, feature: str = "") -> ModuleType:
    """Import a module or raise a clear error explaining which feature needs it.

    Args:
        module_name: Fully qualified module name to import.
        feature: Human-readable feature name for the error message
            (e.g. ``"LLM Guard scanning"``).

    Returns:
        The imported module.

    Raises:
        ImportError: With a descriptive message when the module is missing.
    """
    mod, available = lazy_import(module_name)
    if not available or mod is None:
        feature_clause = f" for {feature}" if feature else ""
        raise ImportError(
            f"Package {module_name!r} is required{feature_clause}. Install it with: pip install {module_name}"
        )
    return mod
