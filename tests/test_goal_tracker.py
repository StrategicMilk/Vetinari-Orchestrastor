"""Tests for goal tracker and anti-drift system."""

import pytest
from vetinari.drift.goal_tracker import GoalTracker, AdherenceResult, ScopeCreepItem, create_goal_tracker


class TestGoalTracker:
    def test_init(self):
        tracker = GoalTracker("Build a REST API with Flask")
        assert tracker.original_goal == "Build a REST API with Flask"
        assert len(tracker._goal_keywords) > 0

    def test_high_adherence(self):
        tracker = GoalTracker("Build a REST API with Flask and PostgreSQL")
        result = tracker.check_adherence(
            "Created Flask REST API endpoints with PostgreSQL database connection",
            "Implement REST API endpoints",
        )
        assert result.score >= 0.5

    def test_low_adherence(self):
        tracker = GoalTracker("Build a REST API with Flask and PostgreSQL")
        result = tracker.check_adherence(
            "Configured Kubernetes deployment manifests for production cluster",
            "Set up deployment infrastructure",
        )
        assert result.score < 0.5

    def test_empty_output(self):
        tracker = GoalTracker("Build a web scraper")
        result = tracker.check_adherence("", "Execute task")
        assert isinstance(result.score, float)

    def test_empty_goal(self):
        tracker = GoalTracker("")
        result = tracker.check_adherence("some output", "some task")
        assert result.score == 1.0  # No goal to drift from


class TestScopeCreep:
    def test_relevant_tasks(self):
        tracker = GoalTracker("Build a Python web scraper for news articles")

        class MockTask:
            def __init__(self, id, description):
                self.id = id
                self.description = description

        tasks = [
            MockTask("t1", "Research web scraping libraries in Python"),
            MockTask("t2", "Implement article parser for news sites"),
            MockTask("t3", "Write Python scraper with BeautifulSoup"),
        ]
        flagged = tracker.detect_scope_creep(tasks)
        assert len(flagged) == 0

    def test_irrelevant_task(self):
        tracker = GoalTracker("Build a Python web scraper for news articles")

        class MockTask:
            def __init__(self, id, description):
                self.id = id
                self.description = description

        tasks = [
            MockTask("t1", "Implement web scraper"),
            MockTask("t2", "Configure Kubernetes cluster networking and service mesh"),
        ]
        flagged = tracker.detect_scope_creep(tasks)
        assert len(flagged) >= 1
        assert any(f.task_id == "t2" for f in flagged)


class TestDriftTrend:
    def test_trend_with_insufficient_data(self):
        tracker = GoalTracker("test goal")
        trend = tracker.get_drift_trend()
        assert trend["trend"] == "unknown"

    def test_stable_trend(self):
        tracker = GoalTracker("Build a REST API with authentication")
        # Simulate multiple adherence checks
        for _ in range(5):
            tracker.check_adherence(
                "Implemented REST API authentication endpoint",
                "Build authentication for REST API",
            )
        trend = tracker.get_drift_trend()
        assert trend["samples"] == 5
        assert trend["trend"] in ("stable", "insufficient_data", "improving")


class TestFactory:
    def test_create_goal_tracker(self):
        tracker = create_goal_tracker("Build something")
        assert isinstance(tracker, GoalTracker)
