"""Subprocess utility functions for the code sandbox.

Provides module-level helpers for wrapping user code before subprocess
execution, parsing the structured output block emitted by the wrapper,
and applying Windows Job Object memory limits.

These are extracted from ``CodeSandbox`` methods so that
``sandbox_manager.py`` and tests can import them without depending on
the full ``CodeSandbox`` class.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def wrap_python_code(
    code: str,
    input_data: dict[str, Any] | None = None,
    blocked_modules: list[str] | None = None,
    allow_network: bool = False,
    filesystem_allowlist: list[str] | None = None,
) -> str:
    """Wrap user Python code with the Vetinari sandbox harness.

    The wrapper script:
    - Redirects stdout/stderr to capture output
    - Installs a restricted ``__import__`` that blocks forbidden modules
    - Installs a filesystem guard when a path allowlist is provided
    - Executes the user code via ``compile``/``eval`` in an isolated namespace
    - Emits a JSON result block delimited by ``===VETINARI_OUTPUT_START===``
      and ``===VETINARI_OUTPUT_END===`` markers on real stdout

    Args:
        code: User Python code to execute inside the harness.
        input_data: Optional dict made available as ``INPUT_DATA`` in the
            executed namespace.
        blocked_modules: List of top-level module names the sandbox should
            refuse to import.
        allow_network: When ``False``, adds common network modules to the
            block list.
        filesystem_allowlist: List of path prefixes allowed for file access.
            An empty list (default) permits all paths.

    Returns:
        Complete Python source string ready to be written to a temp file and
        executed via ``sys.executable``.
    """
    import base64 as _b64

    input_json = json.dumps(input_data or {})  # noqa: VET112 - empty fallback preserves optional request metadata contract
    code_b64 = _b64.b64encode(code.encode("utf-8")).decode("ascii")
    blocked_json = json.dumps(blocked_modules or [])  # noqa: VET112 - empty fallback preserves optional request metadata contract
    allow_network_str = "True" if allow_network else "False"
    fs_allowlist_json = json.dumps(filesystem_allowlist or [])  # noqa: VET112 - empty fallback preserves optional request metadata contract

    # NOTE: The code execution below is the intentional purpose of this
    # sandbox module — it runs user code in an isolated subprocess.
    return f"""
import sys as _sys
import json as _json
import traceback as _tb
import base64 as _b64
import builtins as _builtins
import pathlib as _pathlib

# Save real stdout/stderr BEFORE capturing
_real_stdout = _sys.stdout
_real_stderr = _sys.stderr

INPUT_DATA = {input_json}

# --- Module restriction enforcement ---
_BLOCKED_MODULES = {blocked_json}
_ALLOW_NETWORK = {allow_network_str}
_WRAPPER_NEEDS = {{"sys", "json", "traceback", "base64", "builtins"}}
_original_import = _builtins.__import__

def _restricted_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(
            "Module %r is blocked in the Vetinari sandbox" % name
        )
    if not _ALLOW_NETWORK and top_level in ("socket", "requests", "urllib", "httpx", "aiohttp"):
        raise ImportError(
            "Network module %r is blocked (allow_network=False)" % name
        )
    return _original_import(name, *args, **kwargs)

_builtins.__import__ = _restricted_import
for _mod in list(_sys.modules):
    _top = _mod.split(".")[0]
    if _top in _BLOCKED_MODULES and _top not in _WRAPPER_NEEDS:
        del _sys.modules[_mod]

# --- Filesystem allowlist enforcement ---
# The write guard is always installed. When the allowlist is empty all writes
# are blocked (fail-closed) and reads are unrestricted (for imports). When a
# non-empty allowlist is provided, BOTH reads and writes are confined to the
# listed paths plus the Python installation directory (so imports still work).
_FS_ALLOWLIST = {fs_allowlist_json}
_original_open = _builtins.open
_original_path_open = _pathlib.Path.open
_FS_ALLOWLIST_PATHS = [_pathlib.Path(_prefix).resolve() for _prefix in _FS_ALLOWLIST]
# Python installation prefix — always readable so stdlib imports work.
_PYTHON_PREFIX = str(_sys.prefix).replace("\\\\", "/")
_PYTHON_BASE_PREFIX = str(_sys.base_prefix).replace("\\\\", "/")

def _is_allowlisted(_resolved_path):
    if not _FS_ALLOWLIST_PATHS:
        return False
    return any(_resolved_path.is_relative_to(_prefix) for _prefix in _FS_ALLOWLIST_PATHS)

def _restricted_open(file, mode="r", *args, **kwargs):
    _is_write = any(m in str(mode) for m in ("w", "a", "x")) or "+" in str(mode)
    from pathlib import Path as _Path
    _resolved_path = _Path(file).resolve()
    _resolved = str(_resolved_path)
    _resolved_fwd = _resolved.replace("\\\\", "/")
    if _is_write:
        if _FS_ALLOWLIST:
            if not _is_allowlisted(_resolved_path):
                raise PermissionError(
                    "Write access to path %r is blocked by the Vetinari sandbox filesystem allowlist" % _resolved
                )
        else:
            raise PermissionError(
                "Write access to path %r is blocked — no filesystem allowlist configured for this sandbox" % _resolved
            )
    elif _FS_ALLOWLIST:
        # Non-empty allowlist: reads are also confined (except Python stdlib).
        _is_python = _resolved_fwd.startswith(_PYTHON_PREFIX) or _resolved_fwd.startswith(_PYTHON_BASE_PREFIX)
        if not _is_python and not _is_allowlisted(_resolved_path):
            raise PermissionError(
                "Read access to path %r is blocked by the Vetinari sandbox filesystem allowlist" % _resolved
            )
    return _original_open(file, mode, *args, **kwargs)

_builtins.open = _restricted_open

def _restricted_path_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
    return _restricted_open(
        self,
        mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
    )

def _restricted_path_read_text(self, encoding=None, errors=None, newline=None):
    with _restricted_open(self, "r", encoding=encoding, errors=errors, newline=newline) as _fh:
        return _fh.read()

def _restricted_path_write_text(self, data, encoding=None, errors=None, newline=None):
    with _restricted_open(self, "w", encoding=encoding, errors=errors, newline=newline) as _fh:
        return _fh.write(data)

_pathlib.Path.open = _restricted_path_open
_pathlib.Path.read_text = _restricted_path_read_text
_pathlib.Path.write_text = _restricted_path_write_text

_output = []
_errors = []

class _OutputCapture:
    def write(self, text):
        if text.strip():
            _output.append(text)
    def flush(self):
        pass

_sys.stdout = _OutputCapture()
_sys.stderr = _OutputCapture()

_user_code = _b64.b64decode("{code_b64}").decode("utf-8")
try:
    _sandbox_globals = {{}}
    _code_obj = compile(_user_code, "<vetinari_sandbox>", "exec")
    _builtins.eval(_code_obj, _sandbox_globals)
except Exception as _e:
    _errors.append(_tb.format_exc())

# Restore real stdout for final JSON output
_sys.stdout = _real_stdout
_sys.stderr = _real_stderr

_result = {{
    "success": len(_errors) == 0,
    "output": "".join(_output),
    "errors": "".join(_errors),
    "input_received": INPUT_DATA
}}

print("===VETINARI_OUTPUT_START===")  # noqa: T201, VET035
print(_json.dumps(_result))  # noqa: T201, VET035
print("===VETINARI_OUTPUT_END===")  # noqa: T201, VET035
"""


def parse_sandbox_output(raw_stdout: str) -> tuple[bool | None, str, str]:
    """Extract success status from the structured wrapper output block.

    The wrapper script emits a JSON blob enclosed between
    ``===VETINARI_OUTPUT_START===`` and ``===VETINARI_OUTPUT_END===``
    markers.  This function parses that block and returns the user code's
    real success flag, captured output, and error text.

    Args:
        raw_stdout: The full stdout string from the sandbox subprocess.

    Returns:
        A 3-tuple ``(success, output, error)`` where *success* is a bool
        when the JSON block was found and parseable, or ``None`` when no
        structured block was present (caller should fall back to the
        subprocess return code).
    """
    start_marker = "===VETINARI_OUTPUT_START==="
    end_marker = "===VETINARI_OUTPUT_END==="
    start_idx = raw_stdout.find(start_marker)
    end_idx = raw_stdout.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return None, raw_stdout, ""
    json_text = raw_stdout[start_idx + len(start_marker) : end_idx].strip()
    try:
        data = json.loads(json_text)
        return (
            bool(data.get("success", False)),
            data.get("output", ""),
            data.get("errors", ""),
        )
    except (json.JSONDecodeError, ValueError):
        logger.warning("Sandbox output is not JSON — returning raw stdout as plain text output")
        return None, raw_stdout, ""
