"""Tests for vetinari.phase_coordinator — rule-based phase routing."""

from __future__ import annotations

from vetinari.phase_coordinator import (
    PhaseCoordinator,
    PhaseRoute,
    get_phase_coordinator,
)


class TestPhaseRoute:
    """Tests for the PhaseRoute dataclass."""

    def test_default_parallel_is_true(self):
        route = PhaseRoute("test", ["A"], ["B"])
        assert route.parallel is True

    def test_explicit_parallel_false(self):
        route = PhaseRoute("test", ["A"], ["B"], parallel=False)
        assert route.parallel is False

    def test_fields(self):
        route = PhaseRoute("analysis", ["RESEARCHER"], ["PLANNER"], parallel=True)
        assert route.phase == "analysis"
        assert route.primary_agents == ["RESEARCHER"]
        assert route.fallback_agents == ["PLANNER"]


class TestPhaseCoordinator:
    """Tests for PhaseCoordinator classification and routing."""

    def setup_method(self):
        self.coord = PhaseCoordinator()

    def test_should_use_phases_below_threshold(self):
        assert self.coord.should_use_phases(3) is False
        assert self.coord.should_use_phases(5) is False

    def test_should_use_phases_above_threshold(self):
        assert self.coord.should_use_phases(6) is True
        assert self.coord.should_use_phases(100) is True

    def test_classify_task_analysis(self):
        assert self.coord.classify_task("research the API documentation") == "analysis"
        assert self.coord.classify_task("investigate the bug") == "analysis"

    def test_classify_task_planning(self):
        assert self.coord.classify_task("plan the architecture") == "planning"
        assert self.coord.classify_task("design the decomposition") == "planning"

    def test_classify_task_implementation(self):
        assert self.coord.classify_task("build the login page") == "implementation"
        assert self.coord.classify_task("implement user auth") == "implementation"

    def test_classify_task_quality(self):
        assert self.coord.classify_task("test the endpoints") == "quality"
        assert self.coord.classify_task("verify and validate the output") == "quality"

    def test_classify_task_documentation(self):
        assert self.coord.classify_task("document the API") == "documentation"

    def test_classify_task_meta(self):
        assert self.coord.classify_task("optimize performance") == "meta"

    def test_classify_task_default_implementation(self):
        """Unknown descriptions default to implementation."""
        assert self.coord.classify_task("do something random") == "implementation"

    def test_get_route_known_phase(self):
        route = self.coord.get_route("analysis")
        assert route.phase == "analysis"
        assert "RESEARCHER" in route.primary_agents

    def test_get_route_unknown_phase_defaults_to_implementation(self):
        route = self.coord.get_route("nonexistent")
        assert route.phase == "implementation"

    def test_group_tasks_by_phase(self):
        tasks = ["research the API", "build the login", "test the output"]
        groups = self.coord.group_tasks_by_phase(tasks)
        assert "analysis" in groups
        assert "implementation" in groups
        assert "quality" in groups

    def test_get_execution_order(self):
        order = self.coord.get_execution_order()
        assert order[0] == "analysis"
        assert order[-1] == "meta"
        assert len(order) == 6

    def test_phases_dict_has_six_entries(self):
        assert len(PhaseCoordinator.PHASES) == 6


class TestGetPhaseCoordinator:
    """Tests for the singleton accessor."""

    def test_returns_phase_coordinator(self):
        coord = get_phase_coordinator()
        assert isinstance(coord, PhaseCoordinator)

    def test_returns_same_instance(self):
        a = get_phase_coordinator()
        b = get_phase_coordinator()
        assert a is b
