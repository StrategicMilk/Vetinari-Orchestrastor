"""Tests for planner intelligence improvements."""

import pytest
from vetinari.agents.planner_agent import PlannerAgent
from vetinari.agents.contracts import AgentType, Task


class TestDynamicAgentDiscovery:
    def test_dynamic_agent_list(self):
        planner = PlannerAgent()
        agent_list = planner._get_dynamic_agent_list()
        assert "Available agents" in agent_list
        # Should exclude self — check no line starts with "- PLANNER:"
        assert "- PLANNER:" not in agent_list

    def test_system_prompt_contains_agents(self):
        planner = PlannerAgent()
        prompt = planner.get_system_prompt()
        assert "Available agents" in prompt
        assert "Planning Master" in prompt


class TestGoalClarity:
    def test_clear_goal(self):
        planner = PlannerAgent()
        score = planner._assess_goal_clarity(
            "Build a REST API in Python using Flask with user authentication and PostgreSQL database"
        )
        assert score >= 0.6

    def test_vague_goal(self):
        planner = PlannerAgent()
        score = planner._assess_goal_clarity("do something")
        assert score < 0.4

    def test_medium_goal(self):
        planner = PlannerAgent()
        score = planner._assess_goal_clarity("create a web scraper")
        assert 0.3 <= score <= 0.8

    def test_empty_goal(self):
        planner = PlannerAgent()
        score = planner._assess_goal_clarity("")
        assert score <= 0.5


class TestDAGValidation:
    def test_valid_dag(self):
        tasks = [
            Task(id="t1", description="Analyze requirements", dependencies=[], assigned_agent=AgentType.EXPLORER),
            Task(id="t2", description="Implement feature", dependencies=["t1"], assigned_agent=AgentType.BUILDER),
            Task(id="t3", description="Write tests", dependencies=["t2"], assigned_agent=AgentType.TEST_AUTOMATION),
        ]
        issues = PlannerAgent.validate_dag(tasks)
        assert len(issues) == 0

    def test_circular_dependency(self):
        tasks = [
            Task(id="t1", description="Task A", dependencies=["t2"], assigned_agent=AgentType.BUILDER),
            Task(id="t2", description="Task B", dependencies=["t1"], assigned_agent=AgentType.BUILDER),
        ]
        issues = PlannerAgent.validate_dag(tasks)
        assert any("Circular" in i for i in issues)

    def test_missing_dependency(self):
        tasks = [
            Task(id="t1", description="Task A", dependencies=["t99"], assigned_agent=AgentType.BUILDER),
        ]
        issues = PlannerAgent.validate_dag(tasks)
        assert any("non-existent" in i for i in issues)

    def test_empty_description(self):
        tasks = [
            Task(id="t1", description="", dependencies=[], assigned_agent=AgentType.BUILDER),
        ]
        issues = PlannerAgent.validate_dag(tasks)
        assert any("empty" in i.lower() or "short" in i.lower() for i in issues)
