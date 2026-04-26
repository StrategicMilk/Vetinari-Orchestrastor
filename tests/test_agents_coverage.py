"""Coverage tests for vetinari/agents/* agent classes — Phase 7C"""

from __future__ import annotations

import pytest

from vetinari.types import AgentType


@pytest.fixture
def make_agent():
    """Return a factory that creates a minimal concrete BaseAgent subclass.

    Cannot use tests.factories because MinimalAgent subclasses the abstract
    BaseAgent defined in this module's test context.

    Returns:
        A callable that accepts an agent_type_value string and returns a
        MinimalAgent instance.
    """
    from vetinari.agents.multi_mode_agent import MultiModeAgent

    class MinimalAgent(MultiModeAgent):
        """Minimal concrete agent for testing BaseAgent and MultiModeAgent behaviour.

        Inherits the real verify() from MultiModeAgent so tests exercise production
        verification logic, not a stub.  execute() is overridden with a stub that
        avoids real inference while still satisfying the AgentResult contract.
        """

        MODES: dict = {}
        DEFAULT_MODE: str = ""

        def execute(self, task, context=None):
            from vetinari.agents.contracts import AgentResult

            return AgentResult(success=True, output="done")

        def get_system_prompt(self) -> str:
            return "You are a test agent."

    def _factory(agent_type_value: str = AgentType.WORKER.value) -> MinimalAgent:
        return MinimalAgent(AgentType[agent_type_value])

    return _factory


# ─── BaseAgent ───────────────────────────────────────────────────────────────


class TestBaseAgentCoverage:
    def test_properties(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        assert agent.agent_type is not None
        assert isinstance(agent.name, str)
        assert isinstance(agent.description, str)
        assert not agent.is_initialized

    def test_initialize(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        agent.initialize({"project": "test"})
        assert agent.is_initialized

    def test_execute(self, make_agent) -> None:
        from vetinari.agents.contracts import AgentTask

        agent = make_agent(AgentType.INSPECTOR.value)
        task = AgentTask(task_id="t1", agent_type=AgentType.INSPECTOR, description="evaluate this", prompt="check it")
        result = agent.execute(task)
        assert result.success

    def test_verify(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        # Verify requires structured dict output — not raw strings or AgentResult objects
        structured_output = {"content": "analysis results", "status": "complete"}
        vr = agent.verify(structured_output)
        assert vr.passed

    def test_verify_rejects_garbage(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        # Raw strings, empty dicts, and None must fail verification
        assert not agent.verify("random garbage").passed
        assert not agent.verify({}).passed
        assert not agent.verify(None).passed

    def test_get_system_prompt(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)

    def test_get_capabilities(self, make_agent) -> None:
        agent = make_agent(AgentType.WORKER.value)
        caps = agent.get_capabilities()
        assert isinstance(caps, list)


# ─── ForemanAgent ─────────────────────────────────────────────────────────────


class TestPlannerAgent:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from vetinari.agents.planner_agent import ForemanAgent

        self.agent = ForemanAgent()

    def test_properties(self) -> None:
        assert self.agent.agent_type == AgentType.FOREMAN

    def test_get_system_prompt(self) -> None:
        prompt = self.agent.get_system_prompt()
        assert "Foreman" in prompt or "Plan" in prompt

    def test_get_capabilities(self) -> None:
        caps = self.agent.get_capabilities()
        assert "plan_generation" in caps

    def test_initialize(self) -> None:
        self.agent.initialize({"project_root": "/tmp"})
        assert self.agent.is_initialized

    def test_execute_returns_result(self) -> None:
        from vetinari.agents.contracts import AgentResult, AgentTask

        task = AgentTask(
            task_id="t1", agent_type=AgentType.FOREMAN, description="Plan task", prompt="Plan: build a REST API"
        )
        result = self.agent.execute(task)
        assert isinstance(result, AgentResult)
        assert isinstance(result.success, bool)
        assert result.output is not None
        assert result.output != ""


# ─── BuilderAgent ─────────────────────────────────────────────────────────────


class TestBuilderAgent:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from vetinari.agents.builder_agent import BuilderAgent

        self.agent = BuilderAgent()

    def test_agent_type(self) -> None:
        assert self.agent.agent_type == AgentType.WORKER

    def test_get_system_prompt(self) -> None:
        prompt = self.agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 10

    def test_get_capabilities(self) -> None:
        caps = self.agent.get_capabilities()
        assert isinstance(caps, list)
        assert len(caps) > 0
