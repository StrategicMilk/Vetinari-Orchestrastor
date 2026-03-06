"""Tests for PhaseCoordinator and AGENT_CAPABILITIES helpers."""
import pytest
from vetinari.phase_coordinator import PhaseCoordinator, get_phase_coordinator
from vetinari.agents.contracts import get_agents_for_expertise, get_agents_for_phase


class TestClassifyTask:
    def test_classify_implementation_keywords(self):
        pc = PhaseCoordinator()
        assert pc.classify_task("build a REST API endpoint") == "implementation"

    def test_classify_analysis_keywords(self):
        pc = PhaseCoordinator()
        assert pc.classify_task("research available libraries and explore options") == "analysis"

    def test_classify_quality_keywords(self):
        pc = PhaseCoordinator()
        assert pc.classify_task("test and verify the security audit") == "quality"

    def test_classify_defaults_to_implementation(self):
        pc = PhaseCoordinator()
        # No keyword matches -> default
        assert pc.classify_task("do something") == "implementation"


class TestShouldUsePhases:
    def test_below_threshold(self):
        pc = PhaseCoordinator()
        assert pc.should_use_phases(5) is False

    def test_above_threshold(self):
        pc = PhaseCoordinator()
        assert pc.should_use_phases(6) is True


class TestGroupTasksByPhase:
    def test_groups_correctly(self):
        pc = PhaseCoordinator()

        class FakeTask:
            def __init__(self, description):
                self.description = description

        tasks = [
            FakeTask("build the login page"),
            FakeTask("research authentication libraries"),
            FakeTask("test the login flow"),
        ]
        groups = pc.group_tasks_by_phase(tasks)
        assert "implementation" in groups
        assert "analysis" in groups
        assert "quality" in groups
        assert len(groups["implementation"]) == 1
        assert len(groups["analysis"]) == 1
        assert len(groups["quality"]) == 1


class TestGetExecutionOrder:
    def test_returns_all_phases(self):
        pc = PhaseCoordinator()
        order = pc.get_execution_order()
        expected = {"analysis", "planning", "implementation", "quality", "documentation", "meta"}
        assert set(order) == expected
        assert len(order) == 6


class TestAgentCapabilityHelpers:
    def test_get_agents_for_expertise_finds_correct_agents(self):
        agents = get_agents_for_expertise("code")
        assert "BUILDER" in agents

    def test_get_agents_for_expertise_testing(self):
        agents = get_agents_for_expertise("testing")
        assert "TESTER" in agents

    def test_get_agents_for_phase_finds_correct_agents(self):
        agents = get_agents_for_phase("quality")
        assert "TESTER" in agents

    def test_get_agents_for_phase_analysis(self):
        agents = get_agents_for_phase("analysis")
        assert "PLANNER" in agents
        assert "RESEARCHER" in agents
        assert "ARCHITECT" in agents
