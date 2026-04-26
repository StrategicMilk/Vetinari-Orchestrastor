"""Session 34I1 operator truth regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_python(code: str, env_updates: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    env.update(env_updates or {})
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_package_import_does_not_load_cwd_dotenv(tmp_path: Path) -> None:
    env_key = "VETINARI_IMPORT_MUTATION_PROBE"
    (tmp_path / ".env").write_text(f"{env_key}=from-dotenv\n", encoding="utf-8")

    result = _run_isolated_python(
        f"""
import json
import os
os.environ.pop({env_key!r}, None)
old_cwd = os.getcwd()
os.chdir({str(tmp_path)!r})
try:
    import vetinari  # noqa: F401
finally:
    os.chdir(old_cwd)
print(json.dumps({{"value": os.environ.get({env_key!r})}}))
""",
    )

    assert result["value"] is None


def test_explicit_dotenv_bootstrap_strips_documented_quotes(tmp_path: Path) -> None:
    result = _run_isolated_python(
        f"""
import json
import os
from pathlib import Path
import vetinari

fake_repo = Path({str(tmp_path)!r}) / "repo"
fake_pkg = fake_repo / "vetinari"
fake_pkg.mkdir(parents=True)
(fake_pkg / "__init__.py").write_text("", encoding="utf-8")
(fake_repo / ".env").write_text('GEMINI_API_KEY="gemini-token"\\nREPLICATE_API_TOKEN=\\'rep-token\\'\\n', encoding="utf-8")
vetinari.__file__ = str(fake_pkg / "__init__.py")
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("REPLICATE_API_TOKEN", None)
vetinari.bootstrap_environment()
print(json.dumps({{"gemini": os.environ.get("GEMINI_API_KEY"), "replicate": os.environ.get("REPLICATE_API_TOKEN")}}))
""",
    )

    assert result == {"gemini": "gemini-token", "replicate": "rep-token"}


def test_backend_config_reads_live_user_config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vetinari.backend_config import load_backend_runtime_config

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.yaml").write_text("inference:\n  context_length: 4096\n", encoding="utf-8")
    (second / "config.yaml").write_text("inference:\n  context_length: 16384\n", encoding="utf-8")

    monkeypatch.setenv("VETINARI_USER_DIR", str(first))
    assert load_backend_runtime_config()["local_inference"]["context_length"] == 4096

    monkeypatch.setenv("VETINARI_USER_DIR", str(second))
    assert load_backend_runtime_config()["local_inference"]["context_length"] == 16384


def test_init_wizard_writes_to_live_user_config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vetinari.setup.init_wizard import _write_config
    from vetinari.system.hardware_detect import HardwareProfile

    user_dir = tmp_path / "operator"
    monkeypatch.setenv("VETINARI_USER_DIR", str(user_dir))

    config_path = _write_config(HardwareProfile(cpu_count=4, ram_gb=16.0))

    assert config_path == user_dir / "config.yaml"
    assert config_path.exists()


def test_training_pause_resume_are_unsupported_not_success() -> None:
    from vetinari.cli_training import cmd_train

    assert cmd_train(SimpleNamespace(train_action="pause")) == 1
    assert cmd_train(SimpleNamespace(train_action="resume")) == 1


def test_serve_malformed_env_port_returns_bounded_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from vetinari.cli_commands import cmd_serve

    monkeypatch.setenv("VETINARI_WEB_PORT", "not-an-int")
    with patch("builtins.print") as mock_print:
        rc = cmd_serve(SimpleNamespace(port=None, web_host=None, debug=False, verbose=False))

    assert rc == 1
    assert "Invalid web port" in " ".join(str(call) for call in mock_print.call_args_list)


def test_health_returns_nonzero_when_check_reports_failure() -> None:
    from vetinari.cli_commands import cmd_health

    with patch("vetinari.cli_commands._health_check_quiet", return_value=False):
        assert cmd_health(SimpleNamespace(verbose=False)) == 1
