"""Tests for per-agent permission model (US-040)."""

from __future__ import annotations

import pytest

from vetinari.exceptions import SecurityError
from vetinari.execution_context import (
    AGENT_PERMISSION_MAP,
    ToolPermission,
    enforce_agent_permissions,
)
from vetinari.types import AgentType


class TestAgentPermissionMap:
    """Tests for AGENT_PERMISSION_MAP configuration."""

    def test_all_active_agent_types_have_permission_entries(self) -> None:
        """Every active agent type must have an entry in the permission map."""
        active_types = {
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        }
        for agent_type in active_types:
            assert agent_type in AGENT_PERMISSION_MAP, f"{agent_type.value} missing from AGENT_PERMISSION_MAP"

    def test_all_permission_sets_are_frozensets(self) -> None:
        """All permission sets must be frozensets to prevent accidental mutation."""
        for agent_type, perms in AGENT_PERMISSION_MAP.items():
            assert isinstance(perms, frozenset), f"{agent_type.value} permissions should be frozenset"

    def test_all_agents_have_model_inference(self) -> None:
        """All active agents must be allowed MODEL_INFERENCE as a minimum."""
        for agent_type, perms in AGENT_PERMISSION_MAP.items():
            assert ToolPermission.MODEL_INFERENCE in perms, f"{agent_type.value} should have MODEL_INFERENCE"


class TestForemanPermissions:
    """Tests for Foreman agent permissions."""

    @pytest.mark.parametrize(
        "permission",
        [
            ToolPermission.FILE_READ,
            ToolPermission.MODEL_INFERENCE,
            ToolPermission.MODEL_DISCOVERY,
        ],
    )
    def test_foreman_allowed(self, permission: ToolPermission) -> None:
        """Foreman must be allowed FILE_READ, MODEL_INFERENCE, and MODEL_DISCOVERY."""
        result = enforce_agent_permissions(AgentType.FOREMAN, permission)
        assert result is None  # returns None when permission is granted

    @pytest.mark.parametrize(
        ("permission", "pattern"),
        [
            (ToolPermission.FILE_WRITE, r"FOREMAN.*not allowed.*file_write"),
            (ToolPermission.BASH_EXECUTE, r"FOREMAN.*not allowed.*bash_execute"),
            (ToolPermission.GIT_PUSH, r"FOREMAN.*not allowed.*git_push"),
        ],
    )
    def test_foreman_denied(self, permission: ToolPermission, pattern: str) -> None:
        """Foreman must not be allowed FILE_WRITE, BASH_EXECUTE, or GIT_PUSH."""
        with pytest.raises(SecurityError, match=pattern):
            enforce_agent_permissions(AgentType.FOREMAN, permission)


class TestWorkerPermissions:
    """Tests for Worker agent permissions — the execution agent with broad access."""

    @pytest.mark.parametrize(
        "permission",
        [
            ToolPermission.FILE_WRITE,
            ToolPermission.BASH_EXECUTE,
            ToolPermission.PYTHON_EXECUTE,
            ToolPermission.NETWORK_REQUEST,
        ],
    )
    def test_worker_allowed(self, permission: ToolPermission) -> None:
        """Worker must be allowed broad execution permissions."""
        result = enforce_agent_permissions(AgentType.WORKER, permission)
        assert result is None  # returns None when permission is granted

    def test_worker_denied_git_push(self) -> None:
        """Worker must not be allowed to push to git without explicit authorization."""
        with pytest.raises(SecurityError):
            enforce_agent_permissions(AgentType.WORKER, ToolPermission.GIT_PUSH)


class TestInspectorPermissions:
    """Tests for Inspector agent permissions — read + run tests, no file write."""

    @pytest.mark.parametrize(
        "permission",
        [
            ToolPermission.BASH_EXECUTE,
            ToolPermission.FILE_READ,
        ],
    )
    def test_inspector_allowed(self, permission: ToolPermission) -> None:
        """Inspector must be allowed to run tests and read files."""
        result = enforce_agent_permissions(AgentType.INSPECTOR, permission)
        assert result is None  # returns None when permission is granted

    def test_inspector_denied_file_write(self) -> None:
        """Inspector must not be allowed to write files."""
        with pytest.raises(SecurityError):
            enforce_agent_permissions(AgentType.INSPECTOR, ToolPermission.FILE_WRITE)


class TestEnforceAgentPermissions:
    """Tests for the enforce_agent_permissions function."""

    def test_raises_for_unknown_agent_type(self) -> None:
        """An agent type not in AGENT_PERMISSION_MAP must be denied with a clear message."""
        for at in AgentType:
            if at not in AGENT_PERMISSION_MAP:
                with pytest.raises(SecurityError, match="no permission mapping"):
                    enforce_agent_permissions(at, ToolPermission.FILE_READ)
                break

    def test_permission_error_includes_allowed_list(self) -> None:
        """PermissionError messages must include the list of allowed permissions."""
        with pytest.raises(SecurityError, match="Allowed permissions"):
            enforce_agent_permissions(AgentType.FOREMAN, ToolPermission.FILE_WRITE)
