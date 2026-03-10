"""Tests for creative improvement suggestions."""

import pytest
from vetinari.agents.architect_agent import ArchitectAgent
from vetinari.agents.contracts import AgentTask, AgentType


class TestArchitectSuggestions:
    def test_suggest_routing(self):
        agent = ArchitectAgent()
        agent.initialize({})
        task = AgentTask(
            task_id="test-suggest",
            agent_type=AgentType.ARCHITECT,
            description="suggest improvements for the project",
            prompt="Suggest improvements for a Flask REST API",
        )
        result = agent.execute(task)
        assert result.success
        assert result.metadata.get("mode") == "suggest"
        assert "suggestions" in result.output

    def test_fallback_suggestions(self):
        agent = ArchitectAgent()
        suggestions = agent._fallback_suggestions("Build a REST API with Flask")
        assert len(suggestions) >= 1
        assert all(isinstance(s, dict) for s in suggestions)
        assert all("title" in s for s in suggestions)

    def test_fallback_includes_security(self):
        agent = ArchitectAgent()
        suggestions = agent._fallback_suggestions("Build a web app")
        categories = [s.get("category") for s in suggestions]
        assert "security" in categories

    def test_fallback_includes_testing(self):
        agent = ArchitectAgent()
        suggestions = agent._fallback_suggestions("Create a data pipeline")
        categories = [s.get("category") for s in suggestions]
        assert "testing" in categories

    def test_fallback_skips_test_when_in_goal(self):
        agent = ArchitectAgent()
        suggestions = agent._fallback_suggestions("Build and test a CLI tool")
        titles = [s.get("title", "").lower() for s in suggestions]
        assert not any("test" in t for t in titles)

    def test_suggest_with_completed_outputs(self):
        agent = ArchitectAgent()
        agent.initialize({})
        task = AgentTask(
            task_id="test-suggest-2",
            agent_type=AgentType.ARCHITECT,
            description="suggest improvements",
            prompt="Suggest improvements",
            context={
                "insertion_point": "post_execution",
                "completed_outputs": ["Created Flask app", "Added user model"],
            },
        )
        result = agent.execute(task)
        assert result.success
        assert "suggestions" in result.output

    def test_empty_goal_fallback(self):
        agent = ArchitectAgent()
        suggestions = agent._fallback_suggestions("")
        assert len(suggestions) >= 1
