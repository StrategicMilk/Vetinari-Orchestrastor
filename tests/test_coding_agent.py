"""
Tests for the Coding Agent.

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
        """Test that the coding agent initializes correctly."""
        from vetinari.coding_agent import CodeAgentEngine
        
        agent = CodeAgentEngine()
        assert agent is not None
        assert agent.enabled is True
        assert agent.lm_provider == "internal"
    
    def test_generate_scaffold(self):
        """Test scaffold generation."""
        from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType
        
        agent = CodeAgentEngine()
        agent.enabled = True
        
        task = CodeTask(
            type=CodingTaskType.SCAFFOLD,
            language="python",
            repo_path="./test_project",
            target_files=["my_module"]
        )
        
        artifact = agent.run_task(task)
        
        assert artifact is not None
        assert artifact.task_id == task.task_id
        assert "python" in artifact.language.lower()
        assert "my_module" in artifact.path or "scaffold" in artifact.content.lower()
    
    def test_generate_implementation(self):
        """Test implementation generation."""
        from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType
        
        agent = CodeAgentEngine()
        agent.enabled = True
        
        task = CodeTask(
            type=CodingTaskType.IMPLEMENT,
            language="python",
            target_files=["calculator.py"]
        )
        
        artifact = agent.run_task(task)
        
        assert artifact is not None
        assert artifact.type.value in ["file_contents", "patch"]
        assert "class" in artifact.content.lower() or "def" in artifact.content.lower()
    
    def test_generate_tests(self):
        """Test test generation."""
        from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType
        
        agent = CodeAgentEngine()
        agent.enabled = True
        
        task = CodeTask(
            type=CodingTaskType.TEST,
            language="python",
            target_files=["calculator"]
        )
        
        artifact = agent.run_task(task)
        
        assert artifact is not None
        assert "pytest" in artifact.content.lower() or "test" in artifact.content.lower()
        assert artifact.type.value == "test_artifact"
    
    def test_multi_step_task(self):
        """Test multi-step task execution."""
        from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType
        
        agent = CodeAgentEngine()
        agent.enabled = True
        
        tasks = [
            CodeTask(type=CodingTaskType.SCAFFOLD, target_files=["demo"]),
            CodeTask(type=CodingTaskType.IMPLEMENT, target_files=["demo"]),
            CodeTask(type=CodingTaskType.TEST, target_files=["demo"])
        ]
        
        artifacts = agent.run_multi_step_task(tasks)
        
        assert len(artifacts) == 3
        assert all(a is not None for a in artifacts)
    
    def test_task_status_tracking(self):
        """Test that task status is tracked correctly."""
        from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType, CodingTaskStatus
        
        agent = CodeAgentEngine()
        agent.enabled = True
        
        task = CodeTask(type=CodingTaskType.SCAFFOLD)
        
        assert task.status == CodingTaskStatus.PENDING
        
        agent.run_task(task)
        
        assert task.status == CodingTaskStatus.COMPLETED


class TestCodeBridge:
    """Tests for the external code bridge."""
    
    def test_bridge_initialization(self):
        """Test that the bridge initializes correctly."""
        from vetinari.agents.coding_bridge import CodingBridge

        bridge = CodingBridge()
        assert bridge is not None
        assert bridge.enabled is False  # Default disabled

    def test_bridge_disabled_returns_error(self):
        """Test that disabled bridge returns error."""
        from vetinari.agents.coding_bridge import CodingBridge, BridgeTaskSpec

        bridge = CodingBridge()
        bridge.enabled = False

        spec = BridgeTaskSpec()
        result = bridge.submit_task(spec)

        assert result.success is False
        assert "not enabled" in result.error.lower()


class TestCodingAgentPlanIntegration:
    """Tests for coding agent integration with plan mode."""
    
    def test_execute_coding_task_method_exists(self, tmp_path):
        """Test that PlanModeEngine has execute_coding_task method."""
        import os
        os.environ["VETINARI_MEMORY_PATH"] = str(tmp_path / "vetinari_memory.db")
        from vetinari.plan_mode import PlanModeEngine
        from vetinari.memory import MemoryStore
        mem = MemoryStore(db_path=str(tmp_path / "mem.db"))
        engine = PlanModeEngine(memory_store=mem)
        assert hasattr(engine, 'execute_coding_task')
    
    def test_execute_multi_step_coding_method_exists(self, tmp_path):
        """Test that PlanModeEngine has execute_multi_step_coding method."""
        from vetinari.plan_mode import PlanModeEngine
        from vetinari.memory import MemoryStore
        mem = MemoryStore(db_path=str(tmp_path / "mem.db"))
        engine = PlanModeEngine(memory_store=mem)
        assert hasattr(engine, 'execute_multi_step_coding')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
