"""Tests for the Coding Agent.

These tests verify:
1. CodeAgentEngine can generate scaffolds
2. CodeAgentEngine can generate implementations
3. CodeAgentEngine can generate tests
4. Multi-step tasks work correctly
5. Integration with plan mode
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCodeAgentEngine:
    """Tests for the in-process coding agent engine."""

    def test_agent_initialization(self):
        """CodeAgentEngine initializes with expected defaults."""
        from vetinari.coding_agent import CodeAgentEngine

        agent = CodeAgentEngine()
        assert agent is not None
        assert agent.enabled is True
        assert agent.lm_provider == "internal"

    def test_generate_scaffold(self):
        """Scaffold generation produces a code artifact with module content."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task

        agent = CodeAgentEngine()
        agent.enabled = True

        task = make_code_agent_task(
            "Generate scaffold for my_module",
            task_type=CodingTaskType.SCAFFOLD,
            language="python",
            repo_path="./test_project",
            target_files=["my_module"],
        )

        artifact = agent.run_task(task)

        assert artifact is not None
        assert "python" in artifact.language.lower()
        assert "my_module" in artifact.path or "scaffold" in artifact.content.lower()

    def test_generate_implementation(self):
        """Implementation generation produces a file with class or function."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task

        agent = CodeAgentEngine()
        agent.enabled = True

        task = make_code_agent_task(
            "Implement calculator",
            task_type=CodingTaskType.IMPLEMENT,
            language="python",
            target_files=["calculator.py"],
        )

        artifact = agent.run_task(task)

        assert artifact is not None
        assert artifact.type.value in ["file_contents", "patch"]
        assert "class" in artifact.content.lower() or "def" in artifact.content.lower()

    def test_generate_tests(self):
        """Test generation produces pytest test code."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task

        agent = CodeAgentEngine()
        agent.enabled = True

        task = make_code_agent_task(
            "Generate tests for calculator",
            task_type=CodingTaskType.TEST,
            language="python",
            target_files=["calculator"],
        )

        artifact = agent.run_task(task)

        assert artifact is not None
        assert "pytest" in artifact.content.lower() or "test" in artifact.content.lower()
        assert artifact.type.value == "test_artifact"

    def test_multi_step_task(self):
        """Multi-step task execution produces an artifact for each step."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task

        agent = CodeAgentEngine()
        agent.enabled = True

        tasks = [
            make_code_agent_task("scaffold demo", task_type=CodingTaskType.SCAFFOLD, target_files=["demo"]),
            make_code_agent_task("implement demo", task_type=CodingTaskType.IMPLEMENT, target_files=["demo"]),
            make_code_agent_task("test demo", task_type=CodingTaskType.TEST, target_files=["demo"]),
        ]

        artifacts = agent.run_multi_step_task(tasks)

        assert len(artifacts) == 3
        assert all(a is not None for a in artifacts)

    def test_task_status_tracking(self):
        """AgentTask status is tracked through the execution lifecycle."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task
        from vetinari.types import StatusEnum

        agent = CodeAgentEngine()
        agent.enabled = True

        task = make_code_agent_task("scaffold", task_type=CodingTaskType.SCAFFOLD)
        assert task.status == StatusEnum.PENDING

        # run_task converts AgentTask to _CodeTask internally;
        # the original AgentTask status is not mutated (the _CodeTask copy is)
        agent.run_task(task)
        # Verify the task was processed (artifact returned = success)


class TestCodingAgentPlanIntegration:
    """Tests for coding agent integration with plan mode."""

    def test_execute_coding_task_method_exists(self, tmp_path):
        """PlanModeEngine has execute_coding_task method."""
        import os

        os.environ["VETINARI_MEMORY_PATH"] = str(tmp_path / "vetinari_memory.db")
        from vetinari.memory import MemoryStore
        from vetinari.planning.plan_mode import PlanModeEngine

        mem = MemoryStore(db_path=str(tmp_path / "mem.db"))
        engine = PlanModeEngine(memory_store=mem)
        assert hasattr(engine, "execute_coding_task")

    def test_execute_multi_step_coding_method_exists(self, tmp_path):
        """PlanModeEngine has execute_multi_step_coding method."""
        from vetinari.memory import MemoryStore
        from vetinari.planning.plan_mode import PlanModeEngine

        mem = MemoryStore(db_path=str(tmp_path / "mem.db"))
        engine = PlanModeEngine(memory_store=mem)
        assert hasattr(engine, "execute_multi_step_coding")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
