"""Tests proving persistence-layer identifier confinement against path traversal.

Each persistence layer that constructs filesystem paths from caller-supplied IDs
must reject IDs containing path traversal sequences (`..`, absolute paths, null
bytes, etc.) with a ``ValueError`` before any I/O occurs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.factories import ESCAPING_TRAVERSAL_IDS

# -- ADRSystem -----------------------------------------------------------------


class TestADRSystemConfinement:
    """ADRSystem rejects path-like adr_id values."""

    def test_create_adr_rejects_traversal_id(self, tmp_path: Path) -> None:
        """IDs that resolve outside the storage directory must be rejected with ValueError."""
        from vetinari.adr import ADRSystem

        storage = tmp_path / "adr"
        storage.mkdir()
        system = ADRSystem(storage_path=storage)

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                system.create_adr(
                    title="Test",
                    context="Test context",
                    decision="Test decision",
                    consequences="none",
                    category="architecture",
                    adr_id=bad_id,
                )

    def test_create_adr_accepts_safe_id(self, tmp_path: Path) -> None:
        """A well-formed ADR ID must succeed and produce a file inside storage."""
        from vetinari.adr import ADRSystem

        storage = tmp_path / "adr"
        storage.mkdir()
        system = ADRSystem(storage_path=storage)

        adr = system.create_adr(
            title="Test",
            context="Test context",
            decision="Test decision",
            consequences=["none"],
            category="architecture",
            adr_id="ADR-0001",
        )

        assert adr is not None
        assert adr.adr_id == "ADR-0001"
        assert (storage / "ADR-0001.json").exists()


# -- SubtaskTree ---------------------------------------------------------------


class TestSubtaskTreeConfinement:
    """SubtaskTree rejects path-like plan_id values."""

    def test_create_subtask_rejects_traversal_plan_id(self, tmp_path: Path) -> None:
        """Plan IDs that resolve outside the storage directory must be rejected."""
        from vetinari.planning.subtask_tree import SubtaskTree

        storage = tmp_path / "subtasks"
        storage.mkdir()
        tree = SubtaskTree(storage_path=storage)

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                tree.create_subtask(
                    plan_id=bad_id,
                    parent_id="root",
                    depth=0,
                    description="test subtask",
                    prompt="do the thing",
                    agent_type="worker",
                )

    def test_create_subtask_accepts_safe_plan_id(self, tmp_path: Path) -> None:
        """A well-formed plan ID must succeed and write inside storage."""
        from vetinari.planning.subtask_tree import SubtaskTree

        storage = tmp_path / "subtasks"
        storage.mkdir()
        tree = SubtaskTree(storage_path=storage)

        subtask = tree.create_subtask(
            plan_id="plan_safe0001",
            parent_id="root",
            depth=0,
            description="test subtask",
            prompt="do the thing",
            agent_type="worker",
        )

        assert subtask is not None
        assert (storage / "plan_safe0001.json").exists()


# -- PlanManager ---------------------------------------------------------------


class TestPlanManagerConfinement:
    """PlanManager._save_plan and delete_plan reject path-like plan_id values."""

    def test_save_plan_rejects_traversal_via_update(self, tmp_path: Path) -> None:
        """update_plan must propagate the ValueError from _save_plan."""
        from vetinari.planning.planning import PlanManager, PlanningExecutionPlan

        storage = tmp_path / "plans"
        storage.mkdir()
        mgr = PlanManager(storage_path=storage)

        # Inject a plan with a traversal ID directly (bypassing create_plan
        # which always generates a safe UUID-based ID).
        bad_id = "../escape"
        plan = PlanningExecutionPlan(
            plan_id=bad_id,
            title="Escape plan",
            prompt="",
            created_by="test",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        mgr.plans[bad_id] = plan

        with pytest.raises(ValueError, match="path traversal"):
            mgr._save_plan(plan)

    def test_delete_plan_rejects_traversal_id(self, tmp_path: Path) -> None:
        """delete_plan must reject IDs that resolve outside the storage directory."""
        from vetinari.planning.planning import PlanManager

        storage = tmp_path / "plans"
        storage.mkdir()
        mgr = PlanManager(storage_path=storage)

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                mgr.delete_plan(bad_id)

    def test_save_plan_accepts_safe_id(self, tmp_path: Path) -> None:
        """A UUID-style plan ID must be saved inside storage without error."""
        from vetinari.planning.planning import PlanManager

        storage = tmp_path / "plans"
        storage.mkdir()
        mgr = PlanManager(storage_path=storage)

        plan = mgr.create_plan(title="Safe plan", prompt="build something")

        # File must be inside storage (not escaped)
        plan_file = storage / f"{plan.plan_id}.json"
        assert plan_file.exists()
        assert plan_file.resolve().is_relative_to(storage.resolve())


# -- PipelineStateStore --------------------------------------------------------


class TestPipelineStateStoreConfinement:
    """PipelineStateStore rejects path-like task_id values via _state_file."""

    def test_mark_stage_rejects_traversal_task_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """mark_stage_complete must raise ValueError for traversal task IDs."""
        import vetinari.orchestration.pipeline_state as ps_mod

        monkeypatch.setattr(ps_mod, "_STATE_DIR", tmp_path)
        store = ps_mod.PipelineStateStore()

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                store.mark_stage_complete(bad_id, "stage1", {})

    def test_get_resume_point_rejects_traversal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_resume_point must raise ValueError for traversal task IDs."""
        import vetinari.orchestration.pipeline_state as ps_mod

        monkeypatch.setattr(ps_mod, "_STATE_DIR", tmp_path)
        store = ps_mod.PipelineStateStore()

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                store.get_resume_point(bad_id)

    def test_clear_state_rejects_traversal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """clear_state must raise ValueError for traversal task IDs."""
        import vetinari.orchestration.pipeline_state as ps_mod

        monkeypatch.setattr(ps_mod, "_STATE_DIR", tmp_path)
        store = ps_mod.PipelineStateStore()

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                store.clear_state(bad_id)

    def test_safe_task_id_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A normal task ID must complete without error."""
        import vetinari.orchestration.pipeline_state as ps_mod

        monkeypatch.setattr(ps_mod, "_STATE_DIR", tmp_path)
        store = ps_mod.PipelineStateStore()

        store.mark_stage_complete("task_abc123", "intake", {"ok": True})
        result = store.get_resume_point("task_abc123")
        assert result is not None
        assert result[0] == "intake"


# -- PromptVersionManager ------------------------------------------------------


class TestPromptVersionManagerConfinement:
    """PromptVersionManager rejects path-like agent_type values."""

    def test_save_version_rejects_traversal(self, tmp_path: Path) -> None:
        """Agent type values that resolve outside the versions dir must be rejected."""
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)

        for bad_id in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                mgr.save_version(agent_type=bad_id, mode="default", prompt_text="test")

    def test_save_version_accepts_safe_agent_type(self, tmp_path: Path) -> None:
        """A well-formed agent_type must be saved inside the versions directory."""
        from vetinari.prompts.version_manager import PromptVersionManager

        mgr = PromptVersionManager(versions_dir=tmp_path)
        version = mgr.save_version(agent_type="WORKER", mode="build", prompt_text="You are a worker agent.")

        assert version is not None
        version_file = tmp_path / "worker_build.json"
        assert version_file.exists()
        assert version_file.resolve().is_relative_to(tmp_path.resolve())


# -- prompt_loader -------------------------------------------------------------


class TestPromptLoaderConfinement:
    """load_agent_prompt rejects path-like agent names."""

    def test_rejects_traversal_agent_name(self) -> None:
        """Agent names that resolve outside the agents directory must be rejected."""
        from vetinari.agents.prompt_loader import load_agent_prompt

        for bad_name in ESCAPING_TRAVERSAL_IDS:
            with pytest.raises(ValueError, match="path traversal"):
                load_agent_prompt(bad_name)

    def test_safe_agent_name_does_not_raise(self) -> None:
        """A legitimate agent name must not raise — it may return empty string
        if the agent file does not exist, but must not raise ValueError."""
        from vetinari.agents.prompt_loader import load_agent_prompt

        # "worker" is a known agent; if the file is absent it returns "".
        # Either way no ValueError should be raised.
        result = load_agent_prompt("worker")
        assert isinstance(result, str)
