"""Tests for vetinari.orchestration.task_manifest.

Covers TaskContextManifest, ManifestBuilder, manifest hashing,
prompt formatting, and singleton management.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.task_manifest import (
    ManifestBuilder,
    TaskManifestContext,
    get_manifest_builder,
)
from vetinari.types import AgentType

# ── TaskContextManifest Tests ──────────────────────────────────────────


class TestTaskContextManifest:
    """Test the TaskContextManifest dataclass."""

    def test_default_values(self) -> None:
        manifest = TaskManifestContext(task_spec="Build feature X")
        assert manifest.task_spec == "Build feature X"
        assert manifest.acceptance_criteria == []
        assert manifest.relevant_rules == []
        assert manifest.constraints == {}
        assert manifest.verification_checklist == []
        assert manifest.relevant_episodes == []
        assert manifest.defect_warnings == []
        assert manifest.escalation_triggers == []
        assert manifest.manifest_hash == ""

    def test_compute_hash_deterministic(self) -> None:
        manifest = TaskManifestContext(
            task_spec="test",
            acceptance_criteria=["criterion 1"],
        )
        hash1 = manifest.compute_hash()
        hash2 = manifest.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_hash_varies_with_content(self) -> None:
        m1 = TaskManifestContext(task_spec="task A")
        m2 = TaskManifestContext(task_spec="task B")
        assert m1.compute_hash() != m2.compute_hash()

    def test_to_dict(self) -> None:
        manifest = TaskManifestContext(
            task_spec="test",
            acceptance_criteria=["crit 1"],
            relevant_rules=["rule 1"],
            constraints={"max_tokens": 4096},
        )
        manifest.compute_hash()
        d = manifest.to_dict()
        assert d["task_spec"] == "test"
        assert d["acceptance_criteria"] == ["crit 1"]
        assert d["relevant_rules"] == ["rule 1"]
        assert d["constraints"]["max_tokens"] == 4096
        assert d["manifest_hash"] != ""

    def test_from_dict_roundtrip(self) -> None:
        original = TaskManifestContext(
            task_spec="test",
            acceptance_criteria=["criterion"],
            relevant_rules=["rule"],
            constraints={"max_tokens": 8192},
            verification_checklist=["check 1"],
            defect_warnings=["warning 1"],
            escalation_triggers=["trigger 1"],
        )
        original.compute_hash()

        restored = TaskManifestContext.from_dict(original.to_dict())
        assert restored.task_spec == original.task_spec
        assert restored.acceptance_criteria == original.acceptance_criteria
        assert restored.manifest_hash == original.manifest_hash

    def test_format_for_prompt_empty(self) -> None:
        manifest = TaskManifestContext(task_spec="test")
        text = manifest.format_for_prompt()
        assert "## Your Work Instructions" in text

    def test_format_for_prompt_with_content(self) -> None:
        manifest = TaskManifestContext(
            task_spec="Build X",
            acceptance_criteria=["Tests pass", "No lint errors"],
            verification_checklist=["Type hints present"],
            defect_warnings=["Unwired features"],
            escalation_triggers=["Ambiguous requirements"],
            constraints={"max_tokens": 32768, "timeout_seconds": 120, "max_retries": 2},
        )
        text = manifest.format_for_prompt()
        assert "Acceptance Criteria" in text
        assert "Tests pass" in text
        assert "Quality Will Check" in text
        assert "Common Mistakes" in text
        assert "Stop and Ask for Help" in text
        assert "32768 tokens" in text

    def test_format_for_prompt_skips_empty_sections(self) -> None:
        manifest = TaskManifestContext(
            task_spec="test",
            acceptance_criteria=["crit 1"],
            # No verification, defect warnings, or escalation triggers
        )
        text = manifest.format_for_prompt()
        assert "Acceptance Criteria" in text
        assert "Quality Will Check" not in text


# ── ManifestBuilder Tests ──────────────────────────────────────────────


class TestManifestBuilder:
    """Test ManifestBuilder with mocked dependencies."""

    def test_build_returns_manifest(self) -> None:
        builder = ManifestBuilder()
        with (
            patch.object(builder, "_get_rules", return_value=["rule 1"]),
            patch.object(builder, "_get_constraints", return_value={"max_tokens": 16384}),
            patch.object(builder, "_get_verification", return_value=["check 1"]),
            patch.object(builder, "_get_defect_warnings", return_value=["warning 1"]),
            patch.object(builder, "_get_episodes", return_value=[]),
        ):
            manifest = builder.build("Build feature", AgentType.WORKER.value, "build")

        assert manifest.task_spec == "Build feature"
        assert manifest.relevant_rules == ["rule 1"]
        assert manifest.constraints == {"max_tokens": 16384}
        assert manifest.verification_checklist == ["check 1"]
        assert manifest.defect_warnings == ["warning 1"]
        assert manifest.manifest_hash != ""  # Hash computed

    def test_build_with_acceptance_criteria(self) -> None:
        builder = ManifestBuilder()
        with (
            patch.object(builder, "_get_rules", return_value=[]),
            patch.object(builder, "_get_constraints", return_value={}),
            patch.object(builder, "_get_verification", return_value=[]),
            patch.object(builder, "_get_defect_warnings", return_value=[]),
            patch.object(builder, "_get_episodes", return_value=[]),
        ):
            manifest = builder.build(
                "Build feature",
                AgentType.WORKER.value,
                "build",
                acceptance_criteria=["Tests pass", "No lint errors"],
            )

        assert manifest.acceptance_criteria == ["Tests pass", "No lint errors"]

    def test_build_includes_escalation_triggers(self) -> None:
        builder = ManifestBuilder()
        with (
            patch.object(builder, "_get_rules", return_value=[]),
            patch.object(builder, "_get_constraints", return_value={}),
            patch.object(builder, "_get_verification", return_value=[]),
            patch.object(builder, "_get_defect_warnings", return_value=[]),
            patch.object(builder, "_get_episodes", return_value=[]),
        ):
            manifest = builder.build("Task", AgentType.WORKER.value, "build")

        assert len(manifest.escalation_triggers) >= 5
        assert any("Ambiguous" in t for t in manifest.escalation_triggers)

    def test_get_rules_handles_import_error(self) -> None:
        builder = ManifestBuilder()
        with patch(
            "vetinari.orchestration.task_manifest.get_manifest_builder",
            side_effect=ImportError,
        ):
            # Direct call should handle gracefully
            rules = builder._get_rules(AgentType.WORKER.value, None)
            # May return rules from real manager or empty list
            assert isinstance(rules, list)

    def test_get_episodes_handles_error(self) -> None:
        builder = ManifestBuilder()
        with patch(
            "vetinari.memory.unified.get_unified_memory_store",
            side_effect=Exception("no store"),
        ):
            episodes = builder._get_episodes("some task")
            assert episodes == []


# ── Singleton Tests ────────────────────────────────────────────────────


class TestManifestBuilderSingleton:
    """Test singleton management."""

    def test_get_manifest_builder_returns_same_instance(self) -> None:
        b1 = get_manifest_builder()
        b2 = get_manifest_builder()
        assert b1 is b2

    def test_instance_is_manifest_builder(self) -> None:
        b = get_manifest_builder()
        assert isinstance(b, ManifestBuilder)


# ── Integration with Task.metadata ─────────────────────────────────────


class TestTaskMetadataIntegration:
    """Test that Task.metadata field works with manifest."""

    def test_task_has_metadata_field(self) -> None:
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        task = Task(
            id="test-1",
            description="Test task",
            assigned_agent=AgentType.WORKER,
        )
        assert task.metadata == {}

    def test_task_metadata_stores_manifest(self) -> None:
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        task = Task(
            id="test-1",
            description="Test task",
            assigned_agent=AgentType.WORKER,
        )
        manifest = TaskManifestContext(task_spec="test")
        manifest.compute_hash()
        task.metadata["manifest"] = manifest.to_dict()

        assert "manifest" in task.metadata
        assert task.metadata["manifest"]["task_spec"] == "test"
        assert task.metadata["manifest"]["manifest_hash"] != ""

    def test_task_to_dict_includes_metadata(self) -> None:
        from vetinari.agents.contracts import Task
        from vetinari.types import AgentType

        task = Task(
            id="test-1",
            description="Test task",
            assigned_agent=AgentType.WORKER,
            metadata={"manifest": {"task_spec": "test"}},
        )
        d = task.to_dict()
        assert "metadata" in d
        assert d["metadata"]["manifest"]["task_spec"] == "test"

    def test_task_from_dict_preserves_metadata(self) -> None:
        from vetinari.agents.contracts import Task

        data = {
            "id": "test-1",
            "description": "Test task",
            "assigned_agent": AgentType.WORKER.value,
            "metadata": {"manifest": {"task_spec": "test"}},
        }
        task = Task.from_dict(data)
        assert task.metadata["manifest"]["task_spec"] == "test"
