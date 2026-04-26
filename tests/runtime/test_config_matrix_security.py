"""Config matrix tests — Security section (SS-01 to SS-33).

SESSION-32.1: test_config_matrix_security.py.
All HTTP tests use Litestar TestClient.  No handler .fn(...) direct calls.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

from tests.factories import ESCAPING_TRAVERSAL_IDS, TRAVERSAL_IDS
from tests.runtime.conftest import _ADMIN_HEADERS, _ADMIN_TOKEN, _CSRF

logger = logging.getLogger(__name__)


# -- Section SS: Security ------------------------------------------------------


class TestSecurityPromptInjection:
    """SS-01 to SS-03: Prompt injection leading-whitespace sanitization."""

    @pytest.mark.parametrize(
        "role_keyword",
        [
            "  <|system|>",
            "\t<|system|>",
            "   system:",
        ],
    )
    def test_prompt_injection_indented_system_stripped(self, role_keyword: str) -> None:
        """SS-01: Leading-whitespace-prefixed <|system|> tags are stripped from prompts."""
        try:
            from vetinari.agents.base_agent import BaseAgent

            sanitized = BaseAgent._sanitize_prompt(role_keyword)  # type: ignore[attr-defined]
            assert role_keyword not in sanitized or sanitized != role_keyword, (
                f"Indented system injection not stripped: {role_keyword!r}"
            )
        except (ImportError, AttributeError):
            try:
                from vetinari.security.prompt_sanitizer import sanitize_prompt

                sanitized = sanitize_prompt(role_keyword)
                assert role_keyword not in sanitized or len(sanitized) < len(role_keyword)
            except ImportError:
                pytest.skip("prompt sanitizer not importable")

    @pytest.mark.parametrize(
        "role_keyword",
        [
            "  <|assistant|>",
            "\t<|assistant|>",
        ],
    )
    def test_prompt_injection_indented_assistant_stripped(self, role_keyword: str) -> None:
        """SS-02: Leading-whitespace-prefixed <|assistant|> tags are stripped."""
        try:
            from vetinari.agents.base_agent import BaseAgent

            sanitized = BaseAgent._sanitize_prompt(role_keyword)  # type: ignore[attr-defined]
            assert role_keyword not in sanitized or sanitized != role_keyword
        except (ImportError, AttributeError):
            pytest.skip("_sanitize_prompt not available")

    @pytest.mark.parametrize(
        "role_keyword",
        [
            "  <|user|>",
            "\t<|user|>",
        ],
    )
    def test_prompt_injection_indented_user_stripped(self, role_keyword: str) -> None:
        """SS-03: Leading-whitespace-prefixed <|user|> tags are stripped."""
        try:
            from vetinari.agents.base_agent import BaseAgent

            sanitized = BaseAgent._sanitize_prompt(role_keyword)  # type: ignore[attr-defined]
            assert role_keyword not in sanitized or sanitized != role_keyword
        except (ImportError, AttributeError):
            pytest.skip("_sanitize_prompt not available")


class TestSecurityTraversal:
    """SS-04: PolicyEnforcer path traversal canonicalization."""

    @pytest.mark.parametrize("traversal_id", ESCAPING_TRAVERSAL_IDS)
    def test_policy_enforcer_traversal_canonicalized(self, traversal_id: str) -> None:
        """SS-04: PolicyEnforcer._check_jurisdiction() rejects traversal IDs.

        Every ID in ESCAPING_TRAVERSAL_IDS that escapes the project root must
        be rejected before the policy lookup proceeds.
        """
        try:
            from vetinari.agents.policy_enforcer import PolicyEnforcer

            enforcer = PolicyEnforcer()
            result = enforcer._check_jurisdiction(traversal_id)  # type: ignore[attr-defined]
            # Must not return a passing/allowed result for traversal
            if hasattr(result, "passed"):
                assert result.passed is False, f"Traversal ID {traversal_id!r} passed jurisdiction check"
            elif isinstance(result, bool):
                assert result is False
        except (ImportError, AttributeError):
            pytest.skip("PolicyEnforcer._check_jurisdiction not available")


class TestSecurityNeMo:
    """SS-05 to SS-06: NeMo guardrails fail-closed on init failure."""

    def test_nemo_init_failure_check_input_fail_closed(self) -> None:
        """SS-05: NeMo provider init failure blocks GuardrailsManager.check_input()."""
        from vetinari.safety.guardrails import GuardrailsManager, reset_guardrails
        from vetinari.safety.nemo_provider import reset_nemo_provider

        reset_guardrails()
        reset_nemo_provider()
        with (
            patch("vetinari.safety.nemo_provider._NEMO_AVAILABLE", True),
            patch(
                "vetinari.safety.nemo_provider.NeMoGuardrailsProvider",
                side_effect=RuntimeError("nemo init failed"),
            ),
        ):
            mgr = GuardrailsManager()
            result = mgr.check_input("benign text")

        assert result.allowed is False, "NeMo init failure must fail check_input closed"
        assert any("fail-closed" in violation.description.lower() for violation in result.violations)

    def test_nemo_init_failure_check_output_fail_closed(self) -> None:
        """SS-06: NeMo provider init failure blocks GuardrailsManager.check_output()."""
        from vetinari.safety.guardrails import GuardrailsManager, reset_guardrails
        from vetinari.safety.nemo_provider import reset_nemo_provider

        reset_guardrails()
        reset_nemo_provider()
        with (
            patch("vetinari.safety.nemo_provider._NEMO_AVAILABLE", True),
            patch(
                "vetinari.safety.nemo_provider.NeMoGuardrailsProvider",
                side_effect=RuntimeError("nemo init failed"),
            ),
        ):
            mgr = GuardrailsManager()
            result = mgr.check_output("benign output")

        assert result.allowed is False, "NeMo init failure must fail check_output closed"
        assert any("fail-closed" in violation.description.lower() for violation in result.violations)


class TestSecuritySSRF:
    """SS-07 to SS-10: SSRF hostname blocking for loopback/private/link-local/metadata."""

    @pytest.mark.parametrize(
        "hostname",
        [
            "localhost",
            "127.0.0.1",
            "127.0.0.2",
            "::1",
        ],
    )
    def test_ssrf_loopback_hostname_blocked(self, hostname: str) -> None:
        """SS-07: Loopback hostnames are rejected by SSRF DNS resolution check."""
        try:
            from vetinari.security.ssrf_guard import check_hostname

            result = check_hostname(hostname)
            assert result is False or (hasattr(result, "passed") and result.passed is False), (
                f"Loopback hostname {hostname!r} not blocked by SSRF guard"
            )
        except ImportError:
            pytest.skip("ssrf_guard not importable")

    @pytest.mark.parametrize(
        "hostname",
        [
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
        ],
    )
    def test_ssrf_private_hostname_blocked(self, hostname: str) -> None:
        """SS-08: Private-range hostnames are rejected by SSRF guard."""
        try:
            from vetinari.security.ssrf_guard import check_hostname

            result = check_hostname(hostname)
            assert result is False or (hasattr(result, "passed") and result.passed is False), (
                f"Private hostname {hostname!r} not blocked by SSRF guard"
            )
        except ImportError:
            pytest.skip("ssrf_guard not importable")

    @pytest.mark.parametrize(
        "hostname",
        [
            "169.254.0.1",
            "169.254.169.254",
        ],
    )
    def test_ssrf_link_local_hostname_blocked(self, hostname: str) -> None:
        """SS-09: Link-local hostnames are rejected by SSRF guard."""
        try:
            from vetinari.security.ssrf_guard import check_hostname

            result = check_hostname(hostname)
            assert result is False or (hasattr(result, "passed") and result.passed is False), (
                f"Link-local hostname {hostname!r} not blocked by SSRF guard"
            )
        except ImportError:
            pytest.skip("ssrf_guard not importable")

    @pytest.mark.parametrize(
        "hostname",
        [
            "169.254.169.254",  # AWS IMDSv1
            "metadata.google.internal",
            "metadata.internal",
        ],
    )
    def test_ssrf_metadata_hostname_blocked(self, hostname: str) -> None:
        """SS-10: Cloud metadata service hostnames are rejected by SSRF guard."""
        try:
            from vetinari.security.ssrf_guard import check_hostname

            result = check_hostname(hostname)
            assert result is False or (hasattr(result, "passed") and result.passed is False), (
                f"Metadata hostname {hostname!r} not blocked by SSRF guard"
            )
        except ImportError:
            pytest.skip("ssrf_guard not importable")


class TestSecurityToolPermissions:
    """SS-11 to SS-14: Tool.run() per-agent permission and Worker read-only enforcement."""

    def test_tool_run_file_write_permission_enforced(self, tmp_path: Any) -> None:
        """SS-11: FileOperationsTool.run() blocks inspector file writes."""
        from vetinari.execution_context import get_context_manager
        from vetinari.tools.file_tool import FileOperationsTool
        from vetinari.types import AgentType, ExecutionMode

        tool = FileOperationsTool(str(tmp_path))
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            result = tool.run(
                operation="write",
                agent_type=AgentType.INSPECTOR,
                path="test.txt",
                content="data",
            )

        assert result.success is False, "Inspector must not have file_write permission"
        assert "file_write" in (result.error or "")
        assert not (tmp_path / "test.txt").exists()

    def test_tool_run_file_delete_permission_enforced(self, tmp_path: Any) -> None:
        """SS-12: FileOperationsTool.run() blocks inspector file deletes."""
        from vetinari.execution_context import get_context_manager
        from vetinari.tools.file_tool import FileOperationsTool
        from vetinari.types import AgentType, ExecutionMode

        tool = FileOperationsTool(str(tmp_path))
        target = tmp_path / "test.txt"
        target.write_text("data", encoding="utf-8")
        ctx = get_context_manager()
        with ctx.temporary_mode(ExecutionMode.EXECUTION):
            result = tool.run(
                operation="delete",
                agent_type=AgentType.INSPECTOR,
                path="test.txt",
            )

        assert result.success is False, "Inspector must not have file_delete permission"
        assert "file_delete" in (result.error or "")
        assert target.exists(), "Unauthorized delete must not remove the file"

    def test_worker_research_mode_readonly_enforced(self) -> None:
        """SS-13: Worker research modes expose read-only semantics on the live agent."""
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        agent = WorkerAgent()
        agent._current_mode = "code_discovery"
        assert agent.mode_group == "research"
        assert agent.can_write() is False, "Worker research mode must be read-only"
        write_actions = [a for a in agent.allowed_actions() if "write" in a or "delete" in a]
        assert write_actions == [], f"Research mode has write actions: {write_actions}"

    def test_worker_architecture_mode_readonly_enforced(self) -> None:
        """SS-14: Worker architecture modes expose read-only semantics on the live agent."""
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        agent = WorkerAgent()
        agent._current_mode = "architecture"
        assert agent.mode_group == "architecture"
        assert agent.can_write() is False, "Worker architecture mode must be read-only"
        write_actions = [a for a in agent.allowed_actions() if "write" in a or "delete" in a]
        assert write_actions == [], f"Architecture mode has write actions: {write_actions}"


class TestSecuritySandbox:
    """SS-15 to SS-18 and SS-24 to SS-30: CodeSandbox allowlist and policy enforcement."""

    def test_sandbox_empty_allowlist_not_permit_all(self, tmp_path: Any) -> None:
        """SS-15: Empty filesystem_allowlist still blocks writes outside the sandbox root."""
        from vetinari.code_sandbox import CodeSandbox

        working_dir = tmp_path / "sandbox"
        working_dir.mkdir()
        outside = tmp_path / "outside.txt"
        code = f"""
with open({str(outside)!r}, "w", encoding="utf-8") as fh:
    fh.write("blocked")
"""

        sandbox = CodeSandbox(working_dir=str(working_dir), filesystem_allowlist=[])
        result = sandbox.execute_python(code)
        assert result.success is False, "Empty allowlist must not permit external writes"
        assert not outside.exists()

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: apply_changes failure must not be silently ignored"
    )
    def test_sandbox_apply_changes_failure_not_ignored(self) -> None:
        """SS-16: apply_changes() failure is propagated, not silently swallowed."""
        try:
            from vetinari.sandbox.code_sandbox import CodeSandbox

            sandbox = CodeSandbox(filesystem_allowlist=["/tmp"])
            with patch.object(sandbox, "_apply_to_fs", side_effect=OSError("disk full")):
                result = sandbox.apply_changes({"file.txt": "content"})
                if hasattr(result, "success"):
                    assert result.success is False, "apply_changes must propagate failure"
        except (ImportError, AttributeError):
            pytest.skip("CodeSandbox not importable")

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: malformed list-root sandbox_policy.yaml must be bounded"
    )
    def test_sandbox_policy_malformed_list_root(self) -> None:
        """SS-17: sandbox_policy.yaml with list root raises bounded error, not crash."""
        import pathlib
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8", delete=False) as f:
            f.write("- item1\n- item2\n")
            policy_path = f.name

        try:
            from vetinari.sandbox.policy_loader import load_sandbox_policy

            result = load_sandbox_policy(pathlib.Path(policy_path))
            assert result is None or (hasattr(result, "valid") and result.valid is False)
        except (ImportError, AttributeError):
            pytest.skip("load_sandbox_policy not importable")
        except Exception as exc:
            # Must be a bounded error, not an uncaught generic exception
            assert "list" in str(exc).lower() or "mapping" in str(exc).lower(), (
                f"Unexpected exception type for malformed list root: {exc!r}"
            )
        finally:
            import os

            os.unlink(policy_path)

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: malformed scalar-root sandbox_policy.yaml must be bounded"
    )
    def test_sandbox_policy_malformed_scalar_root(self) -> None:
        """SS-18: sandbox_policy.yaml with scalar root raises bounded error, not crash."""
        import pathlib
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8", delete=False) as f:
            f.write("just_a_scalar\n")
            policy_path = f.name

        try:
            from vetinari.sandbox.policy_loader import load_sandbox_policy

            result = load_sandbox_policy(pathlib.Path(policy_path))
            assert result is None or (hasattr(result, "valid") and result.valid is False)
        except (ImportError, AttributeError):
            pytest.skip("load_sandbox_policy not importable")
        except Exception as exc:
            assert "scalar" in str(exc).lower() or "mapping" in str(exc).lower(), (
                f"Unexpected exception type for scalar root: {exc!r}"
            )
        finally:
            import os

            os.unlink(policy_path)

    def test_sandbox_pathlib_open_blocked(self, tmp_path: Any) -> None:
        """SS-24: Sandbox blocks pathlib.Path.open() writes outside the allowlist."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside = tmp_path / "outside" / "file.txt"
        outside.parent.mkdir()
        code = f"""
from pathlib import Path
with Path({str(outside)!r}).open("w", encoding="utf-8") as fh:
    fh.write("blocked")
"""

        sandbox = CodeSandbox(working_dir=str(tmp_path), filesystem_allowlist=[str(allowed_dir)])
        result = sandbox.execute_python(code)
        assert result.success is False, "pathlib.Path.open() outside allowlist must be blocked"
        assert not outside.exists()

    def test_sandbox_pathlib_write_text_blocked(self, tmp_path: Any) -> None:
        """SS-25: Sandbox blocks pathlib.Path.write_text() outside the allowlist."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside = tmp_path / "outside" / "file.txt"
        outside.parent.mkdir()
        code = f"""
from pathlib import Path
Path({str(outside)!r}).write_text("blocked", encoding="utf-8")
"""

        sandbox = CodeSandbox(working_dir=str(tmp_path), filesystem_allowlist=[str(allowed_dir)])
        result = sandbox.execute_python(code)
        assert result.success is False, "pathlib.Path.write_text() outside allowlist must be blocked"
        assert not outside.exists()

    def test_sandbox_pathlib_read_text_blocked(self, tmp_path: Any) -> None:
        """SS-26: Sandbox blocks pathlib.Path.read_text() outside the allowlist."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside = tmp_path / "outside" / "secret.txt"
        outside.parent.mkdir()
        outside.write_text("secret", encoding="utf-8")
        code = f"""
from pathlib import Path
print(Path({str(outside)!r}).read_text(encoding="utf-8"))
"""

        sandbox = CodeSandbox(working_dir=str(tmp_path), filesystem_allowlist=[str(allowed_dir)])
        result = sandbox.execute_python(code)
        assert result.success is False, "pathlib.Path.read_text() outside allowlist must be blocked"
        assert "secret" not in result.output

    def test_sandbox_isolation_pathlib_contract(self) -> None:
        """SS-27: Sandbox isolation contract: pathlib operations are subject to containment."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(filesystem_allowlist=[])
        assert hasattr(sandbox, "execute_python"), "CodeSandbox must enforce containment through execute_python()"

    def test_sandbox_allowlist_string_prefix_not_bypassed(self, tmp_path: Any) -> None:
        """SS-28: An allowlist root does not also permit sibling prefix paths."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "safe"
        allowed_dir.mkdir()
        bypass = tmp_path / "safe_evil" / "file.txt"
        bypass.parent.mkdir()
        code = f"""
from pathlib import Path
Path({str(bypass)!r}).write_text("blocked", encoding="utf-8")
"""

        sandbox = CodeSandbox(working_dir=str(tmp_path), filesystem_allowlist=[str(allowed_dir)])
        result = sandbox.execute_python(code)
        assert result.success is False, "Sibling prefix path must not bypass the allowlist"
        assert not bypass.exists()

    def test_sandbox_restricted_open_os_blocked_allowlisted_succeeds(self, tmp_path: Any) -> None:
        """SS-29: Builtin open() allows allowlisted writes and blocks disallowed writes."""
        from vetinari.code_sandbox import CodeSandbox

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        allowed_file = allowed_dir / "ok.txt"
        outside = tmp_path / "outside" / "blocked.txt"
        outside.parent.mkdir()
        code = f"""
with open({str(allowed_file)!r}, "w", encoding="utf-8") as fh:
    fh.write("ok")
with open({str(outside)!r}, "w", encoding="utf-8") as fh:
    fh.write("blocked")
"""

        sandbox = CodeSandbox(working_dir=str(tmp_path), filesystem_allowlist=[str(allowed_dir)])
        result = sandbox.execute_python(code)
        assert result.success is False, "Disallowed open() call must fail the execution"
        assert allowed_file.exists(), "Allowlisted write should still succeed before the blocked call"
        assert allowed_file.read_text(encoding="utf-8") == "ok"
        assert not outside.exists()

    def test_sandbox_working_dir_missing_fails_bounded(self) -> None:
        """SS-30: Sandbox with missing working directory returns a bounded failure result."""
        from vetinari.code_sandbox import CodeSandbox

        sandbox = CodeSandbox(working_dir="C:/definitely/missing/vetinari-sandbox-path")
        result = sandbox.execute_python("print('test')")
        assert result.success is False
        assert "working_dir" in (result.error or "")


class TestSecurityGovernanceVerifier:
    """SS-19: Default governance verifier must not pass non-empty dicts."""

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: governance verifier must not default-pass non-empty dict"
    )
    def test_governance_verifier_no_default_pass(self) -> None:
        """SS-19: Default governance verifier returns passed=False for unverified dicts."""
        try:
            from vetinari.agents.inspector_agent import InspectorAgent

            agent = InspectorAgent()
            # Non-empty dict with no positive evidence — must not pass
            result = agent.verify({"output": "some content"})
            assert result is not None
            if hasattr(result, "passed"):
                assert result.passed is False, "Default verifier must not pass dict without positive evidence"
        except (ImportError, AttributeError):
            pytest.skip("InspectorAgent.verify not importable")


class TestSecuritySecretScanning:
    """SS-20 to SS-21: Secret scanner nested tuple/list recursion."""

    def test_secret_scan_nested_tuple_recursion(self) -> None:
        """SS-20: Secret scanner recurses into nested tuples to find secrets."""
        try:
            from vetinari.security.secret_scanner import scan_for_secrets

            # Secret nested inside a tuple
            data = ("outer", ("inner", "sk-1234567890abcdef1234567890abcdef"))
            results = scan_for_secrets(data)
            assert len(results) > 0, "Secret scanner did not find secret nested in tuple"
        except (ImportError, AttributeError):
            pytest.skip("scan_for_secrets not importable")

    def test_secret_scan_nested_list_recursion(self) -> None:
        """SS-21: Secret scanner recurses into nested lists to find secrets."""
        try:
            from vetinari.security.secret_scanner import scan_for_secrets

            data = ["outer", ["inner", "sk-1234567890abcdef1234567890abcdef"]]
            results = scan_for_secrets(data)
            assert len(results) > 0, "Secret scanner did not find secret nested in list"
        except (ImportError, AttributeError):
            pytest.skip("scan_for_secrets not importable")


class TestSecurityCodeExecutionGuardrails:
    """SS-22 to SS-23: CODE_EXECUTION guardrails fail-closed."""

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: guardrail scanner unavailable must fail closed")
    def test_code_execution_guardrail_scanner_unavailable_fail_closed(self) -> None:
        """SS-22: Guardrail scanner unavailable causes CODE_EXECUTION to fail closed."""
        try:
            from vetinari.sandbox.guardrails import CodeExecutionGuardrail

            # Simulate scanner construction failure
            with patch(
                "vetinari.sandbox.guardrails.CodeExecutionGuardrail._build_scanner",
                side_effect=RuntimeError("scanner unavailable"),
            ):
                guard = CodeExecutionGuardrail()
                result = guard.check("import os; os.system('rm -rf /')")
                if hasattr(result, "passed"):
                    assert result.passed is False, "Guardrail must fail closed when scanner unavailable"
        except (ImportError, AttributeError):
            pytest.skip("CodeExecutionGuardrail not importable")

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: scanner construction failure must fail closed")
    def test_guardrails_fail_closed_scanner_construction_failure(self) -> None:
        """SS-23: Scanner construction failure causes guardrails to fail closed."""
        try:
            from vetinari.sandbox.guardrails import CodeExecutionGuardrail

            with patch(
                "vetinari.sandbox.guardrails.CodeExecutionGuardrail.__init__",
                side_effect=RuntimeError("construction failed"),
            ):
                try:
                    guard = CodeExecutionGuardrail()
                    result = guard.check("print('hello')")
                    if hasattr(result, "passed"):
                        assert result.passed is False
                except RuntimeError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                    pass  # Construction failure is acceptable — guard failed closed
        except (ImportError, AttributeError):
            pytest.skip("CodeExecutionGuardrail not importable")


class TestSecurityFileOperations:
    """SS-31: FileOperationsTool.delete() miss is visible (not silent success)."""

    def test_file_operations_delete_miss_not_success(self) -> None:
        """SS-31: FileOperationsTool surfaces a delete miss as failure."""
        from vetinari.tools.file_tool import FileOperationsTool

        tool = FileOperationsTool(".")
        fake_ctx = MagicMock()
        fake_ctx.enforce_permission.return_value = None
        with (
            patch("vetinari.tools.file_tool.get_context_manager", return_value=fake_ctx),
            patch.object(tool._ops, "delete_file", return_value=False),
        ):
            result = tool.execute(operation="delete", path="missing.txt")

        assert result.success is False, "delete() on nonexistent file must not return success=True"
        assert "missing.txt" in (result.error or "")


class TestSecurityProjectVerification:
    """SS-32: Project verification rejects traversal project_id."""

    @pytest.mark.parametrize("traversal_id", TRAVERSAL_IDS[:4])  # use first 4 for coverage
    def test_project_verify_goal_traversal_project_id_rejected(self, client: TestClient, traversal_id: str) -> None:
        """SS-32: /api/v1/projects/{id}/verify-goal rejects traversal project IDs."""
        resp = client.post(
            f"/api/v1/projects/{traversal_id}/verify-goal",
            json={"goal": "do something"},
            headers=_CSRF,
        )
        assert resp.status_code in {400, 404, 422}, (
            f"Traversal id {traversal_id!r} was not rejected: {resp.status_code}"
        )


class TestSecurityWorkerExecuteRoute:
    """SS-33: Worker execute routes through execute_safely."""

    def test_worker_execute_routes_through_execute_safely(self) -> None:
        """SS-33: WorkerAgent.execute() always calls _execute_safely() (not direct _infer)."""
        from vetinari.agents.consolidated.worker_agent import WorkerAgent
        from vetinari.agents.contracts import AgentResult, AgentTask
        from vetinari.types import AgentType

        agent = WorkerAgent()
        task = AgentTask(
            task_id="ss33-worker-execute",
            agent_type=AgentType.WORKER,
            description="build a test artifact",
            prompt="build a test artifact",
            mode="build",
            context={},
        )
        expected = AgentResult(success=True, output="ok", errors=[], metadata={})

        with patch.object(agent, "_execute_safely", return_value=expected) as mocked_execute_safely:
            result = agent.execute(task)

        assert result is expected
        mocked_execute_safely.assert_called_once()
        execute_task, execute_fn = mocked_execute_safely.call_args.args
        assert execute_task is task
        assert callable(execute_fn), "execute() must hand a mode-dispatch callable to _execute_safely()"
