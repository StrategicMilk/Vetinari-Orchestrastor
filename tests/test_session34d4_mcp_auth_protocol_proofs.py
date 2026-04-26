"""SESSION-34D4 mounted/auth/protocol proof fixtures.

These tests intentionally exercise request-level and execution-boundary
contracts from convergence pass 2373.  Runtime repairs belong to the 34F/34G/
34I owner shards; this file owns the proof that the unsafe behavior is visible
through executable checks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

from vetinari.code_sandbox import ExecutionResult
from vetinari.sandbox_manager import CodeExecutor
from vetinari.tools.tool_registry_integration import CodeExecutionToolWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_python(code: str, env_updates: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a small Python probe in a fresh interpreter with controlled env."""
    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(  # noqa: S603 - fixed interpreter probe with repo-controlled code.
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


class TestSession34D4AuthEnvProtocol:
    """Auth/provider env parsing and mounted settings-readiness proof."""

    def test_auth_dotenv_loader_strips_quoted_provider_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Documented quoted .env credentials must reach providers unquoted."""
        import vetinari

        fake_repo = tmp_path / "repo"
        fake_pkg = fake_repo / "vetinari"
        fake_pkg.mkdir(parents=True)
        fake_init = fake_pkg / "__init__.py"
        fake_init.write_text("", encoding="utf-8")
        (fake_repo / ".env").write_text(
            'GEMINI_API_KEY="gemini-token"\nREPLICATE_API_TOKEN=\'rep-token\'\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(vetinari, "__file__", str(fake_init))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)

        vetinari._load_env_file()

        assert os.environ["GEMINI_API_KEY"] == "gemini-token"
        assert os.environ["REPLICATE_API_TOKEN"] == "rep-token"

    def test_auth_settings_route_reports_gemini_and_replicate_readiness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/v1/settings must expose every documented provider key."""
        from vetinari.config.settings import reset_settings
        from vetinari.web.litestar_app import create_app

        monkeypatch.setenv("GEMINI_API_KEY", "gemini-token")
        monkeypatch.setenv("REPLICATE_API_TOKEN", "rep-token")
        reset_settings()

        with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
            app = create_app(debug=True)

        try:
            with TestClient(app=app) as client:
                response = client.get("/api/v1/settings")
        finally:
            reset_settings()

        assert response.status_code == 200, response.text[:300]
        api_keys = response.json()["api_keys"]
        assert api_keys["gemini"] is True
        assert api_keys["replicate"] is True


class TestSession34D4McpExecutionBoundaries:
    """MCP/tool execution proof for sandbox and validation boundaries."""

    def test_mcp_tool_execute_code_rejects_policy_blocked_eval(self) -> None:
        """Tool-registry code execution must cross the canonical sandbox policy."""
        wrapper = CodeExecutionToolWrapper()
        wrapper._executor = SimpleNamespace(
            run=lambda code, **kwargs: {"success": True, "output": "2", "error": ""}
        )

        result = wrapper.execute(code="eval('1 + 1')", language="python")

        assert result.success is False
        assert "sandbox" in (result.error or json.dumps(result.output)).lower()

    def test_mcp_run_and_validate_does_not_pass_with_zero_assertions_after_failure(
        self,
    ) -> None:
        """A failed execution with no assertions is not successful validation."""
        sandbox = MagicMock()
        sandbox.execute_python.return_value = ExecutionResult(success=False, output="", error="boom")
        executor = CodeExecutor(sandbox=sandbox)

        result = executor.run_and_validate("raise RuntimeError('boom')")

        assert result["success"] is False
        assert result["validations"], "validation must record that no proof assertions ran"
        assert result["all_validations_passed"] is False


class TestSession34D4AuthConfigImportBounds:
    """Operator env values must be bounded before import-time/runtime use."""

    @pytest.mark.parametrize(
        ("env_name", "bad_value"),
        [
            ("VETINARI_GPU_LAYERS", "not-an-int"),
            ("VETINARI_CONTEXT_LENGTH", "8192.5"),
        ],
    )
    def test_auth_local_model_env_import_errors_are_bounded(self, env_name: str, bad_value: str) -> None:
        """Malformed local-model env must not surface as raw int() crashes."""
        code = """
import json
try:
    import vetinari.constants as constants
    print(json.dumps({"ok": True, "gpu": constants.DEFAULT_GPU_LAYERS, "ctx": constants.DEFAULT_CONTEXT_LENGTH}))
except Exception as exc:
    print(json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}))
"""
        proc = _run_isolated_python(code, {env_name: bad_value})
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)

        assert result["ok"] or (
            result["type"] != "ValueError" and env_name in result["message"]
        ), result

    @pytest.mark.parametrize("bad_value", ["nan", "inf", "-0.1", "1.1", "not-a-float"])
    def test_auth_ponder_cloud_weight_is_rejected_or_bounded(self, bad_value: str) -> None:
        """PONDER_CLOUD_WEIGHT must be finite and within the routing range."""
        code = """
import json
import math
try:
    import vetinari.models.ponder as ponder
    weight = ponder.PONDER_CLOUD_WEIGHT
    print(json.dumps({"ok": True, "finite": math.isfinite(weight), "weight": weight}))
except Exception as exc:
    print(json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}))
"""
        proc = _run_isolated_python(code, {"PONDER_CLOUD_WEIGHT": bad_value})
        assert proc.returncode == 0, proc.stderr
        result = json.loads(proc.stdout)

        assert not result["ok"] or (
            result["finite"] and 0.0 <= float(result["weight"]) <= 1.0
        ), result
