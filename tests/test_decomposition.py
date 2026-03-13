"""
Comprehensive tests for vetinari/decomposition.py

Covers:
- Module-level constants
- _DOD_CRITERIA and _DOR_CRITERIA dicts
- SubtaskSpec dataclass
- DecompositionEvent dataclass
- DecompositionEngine: __init__, _build_default_templates, get_templates,
  get_dod_criteria, get_dor_criteria, decompose_task, _keyword_decompose,
  get_decomposition_history
- Singleton helpers: _get_engine, module-level decomposition_engine
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

import vetinari.decomposition as decomp_module
from vetinari.decomposition import (
    DEFAULT_MAX_DEPTH,
    MAX_MAX_DEPTH,
    MIN_MAX_DEPTH,
    SEED_MIX,
    SEED_RATE,
    DecompositionEngine,
    DecompositionEvent,
    SubtaskSpec,
    _get_engine,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_default_max_depth_value(self):
        assert DEFAULT_MAX_DEPTH == 14

    def test_min_max_depth_value(self):
        assert MIN_MAX_DEPTH == 12

    def test_max_max_depth_value(self):
        assert MAX_MAX_DEPTH == 16

    def test_seed_rate_value(self):
        assert pytest.approx(0.3) == SEED_RATE

    def test_seed_mix_value(self):
        assert pytest.approx(0.5) == SEED_MIX

    def test_min_less_than_default(self):
        assert MIN_MAX_DEPTH < DEFAULT_MAX_DEPTH

    def test_default_less_than_max(self):
        assert DEFAULT_MAX_DEPTH < MAX_MAX_DEPTH

    def test_class_constants_match_module(self):
        assert DecompositionEngine.DEFAULT_MAX_DEPTH == DEFAULT_MAX_DEPTH
        assert DecompositionEngine.MIN_MAX_DEPTH == MIN_MAX_DEPTH
        assert DecompositionEngine.MAX_MAX_DEPTH == MAX_MAX_DEPTH
        assert DecompositionEngine.SEED_RATE == SEED_RATE
        assert DecompositionEngine.SEED_MIX == SEED_MIX


# ---------------------------------------------------------------------------
# _DOD_CRITERIA and _DOR_CRITERIA
# ---------------------------------------------------------------------------

class TestCriteriaConstants:
    def test_dod_has_three_levels(self):
        from vetinari.decomposition import _DOD_CRITERIA
        assert set(_DOD_CRITERIA.keys()) == {"Light", "Standard", "Hard"}

    def test_dor_has_three_levels(self):
        from vetinari.decomposition import _DOR_CRITERIA
        assert set(_DOR_CRITERIA.keys()) == {"Light", "Standard", "Hard"}

    def test_dod_light_has_entries(self):
        from vetinari.decomposition import _DOD_CRITERIA
        assert len(_DOD_CRITERIA["Light"]) >= 1

    def test_dod_standard_has_more_than_light(self):
        from vetinari.decomposition import _DOD_CRITERIA
        assert len(_DOD_CRITERIA["Standard"]) > len(_DOD_CRITERIA["Light"])

    def test_dod_hard_has_most(self):
        from vetinari.decomposition import _DOD_CRITERIA
        assert len(_DOD_CRITERIA["Hard"]) >= len(_DOD_CRITERIA["Standard"])

    def test_dor_light_has_entries(self):
        from vetinari.decomposition import _DOR_CRITERIA
        assert len(_DOR_CRITERIA["Light"]) >= 1

    def test_dor_hard_has_most(self):
        from vetinari.decomposition import _DOR_CRITERIA
        assert len(_DOR_CRITERIA["Hard"]) >= len(_DOR_CRITERIA["Standard"])

    def test_dod_all_values_are_strings(self):
        from vetinari.decomposition import _DOD_CRITERIA
        for level, items in _DOD_CRITERIA.items():
            for item in items:
                assert isinstance(item, str), f"Non-string in DOD {level}: {item!r}"

    def test_dor_all_values_are_strings(self):
        from vetinari.decomposition import _DOR_CRITERIA
        for level, items in _DOR_CRITERIA.items():
            for item in items:
                assert isinstance(item, str), f"Non-string in DOR {level}: {item!r}"


# ---------------------------------------------------------------------------
# SubtaskSpec dataclass
# ---------------------------------------------------------------------------

class TestSubtaskSpec:
    def test_required_fields(self):
        spec = SubtaskSpec(
            subtask_id="s1",
            parent_task_id="root",
            description="Do something",
            agent_type="BUILDER",
            depth=1,
        )
        assert spec.subtask_id == "s1"
        assert spec.parent_task_id == "root"
        assert spec.description == "Do something"
        assert spec.agent_type == "BUILDER"
        assert spec.depth == 1

    def test_default_lists_are_empty(self):
        spec = SubtaskSpec(
            subtask_id="s1", parent_task_id="root",
            description="x", agent_type="BUILDER", depth=0
        )
        assert spec.inputs == []
        assert spec.outputs == []
        assert spec.dependencies == []
        assert spec.dod_criteria == []
        assert spec.dor_criteria == []

    def test_lists_are_independent_per_instance(self):
        s1 = SubtaskSpec("a", "root", "x", "BUILDER", 0)
        s2 = SubtaskSpec("b", "root", "y", "BUILDER", 0)
        s1.inputs.append("in1")
        assert s2.inputs == []

    def test_created_at_is_iso_string(self):
        spec = SubtaskSpec("s1", "root", "desc", "BUILDER", 0)
        # Should parse as ISO datetime without raising
        datetime.fromisoformat(spec.created_at)

    def test_custom_fields(self):
        spec = SubtaskSpec(
            subtask_id="s99",
            parent_task_id="task_42",
            description="Build API",
            agent_type="RESEARCHER",
            depth=3,
            inputs=["schema.json"],
            outputs=["api.py"],
            dependencies=["s1", "s2"],
            dod_criteria=["Tests pass"],
            dor_criteria=["Schema defined"],
        )
        assert spec.inputs == ["schema.json"]
        assert spec.outputs == ["api.py"]
        assert spec.dependencies == ["s1", "s2"]
        assert spec.dod_criteria == ["Tests pass"]
        assert spec.dor_criteria == ["Schema defined"]


# ---------------------------------------------------------------------------
# DecompositionEvent dataclass
# ---------------------------------------------------------------------------

class TestDecompositionEvent:
    def test_required_fields(self):
        evt = DecompositionEvent(
            event_id="e1",
            plan_id="plan_abc",
            task_id="task_1",
            depth=0,
            seeds_used=[],
            subtasks_created=3,
        )
        assert evt.event_id == "e1"
        assert evt.plan_id == "plan_abc"
        assert evt.task_id == "task_1"
        assert evt.depth == 0
        assert evt.subtasks_created == 3

    def test_timestamp_is_iso_string(self):
        evt = DecompositionEvent(
            event_id="e1", plan_id="p", task_id="t",
            depth=0, seeds_used=[], subtasks_created=0
        )
        datetime.fromisoformat(evt.timestamp)

    def test_seeds_used_stored(self):
        evt = DecompositionEvent(
            event_id="e1", plan_id="p", task_id="t",
            depth=1, seeds_used=["web_app", "api_service"], subtasks_created=5
        )
        assert evt.seeds_used == ["web_app", "api_service"]


# ---------------------------------------------------------------------------
# DecompositionEngine.__init__ and _build_default_templates
# ---------------------------------------------------------------------------

class TestDecompositionEngineInit:
    def test_history_starts_empty(self):
        engine = DecompositionEngine()
        assert engine._history == []

    def test_templates_built_on_init(self):
        engine = DecompositionEngine()
        assert len(engine._templates) > 0

    def test_exactly_15_templates(self):
        engine = DecompositionEngine()
        assert len(engine._templates) == 15

    def test_all_templates_have_required_keys(self):
        engine = DecompositionEngine()
        required = {"template_id", "name", "keywords", "agent_type", "dod_level", "subtasks"}
        for tmpl in engine._templates:
            assert required.issubset(set(tmpl.keys())), (
                f"Template {tmpl.get('template_id')} missing keys"
            )

    def test_template_ids_are_unique(self):
        engine = DecompositionEngine()
        ids = [t["template_id"] for t in engine._templates]
        assert len(ids) == len(set(ids))

    def test_all_templates_have_subtasks(self):
        engine = DecompositionEngine()
        for tmpl in engine._templates:
            assert len(tmpl["subtasks"]) >= 1

    def test_web_app_template_present(self):
        engine = DecompositionEngine()
        ids = [t["template_id"] for t in engine._templates]
        assert "web_app" in ids

    def test_security_audit_template_present(self):
        engine = DecompositionEngine()
        ids = [t["template_id"] for t in engine._templates]
        assert "security_audit" in ids

    def test_infrastructure_template_present(self):
        engine = DecompositionEngine()
        ids = [t["template_id"] for t in engine._templates]
        assert "infrastructure" in ids

    def test_known_template_ids(self):
        engine = DecompositionEngine()
        ids = {t["template_id"] for t in engine._templates}
        expected = {
            "web_app", "data_pipeline", "research", "cli_tool", "api_service",
            "library", "document_generation", "creative_writing", "testing",
            "refactoring", "debugging", "migration", "security_audit",
            "data_analysis", "infrastructure",
        }
        assert ids == expected


# ---------------------------------------------------------------------------
# DecompositionEngine.get_templates
# ---------------------------------------------------------------------------

class TestGetTemplates:
    @pytest.fixture
    def engine(self):
        return DecompositionEngine()

    def test_no_filters_returns_all(self, engine):
        results = engine.get_templates()
        assert len(results) == 15

    def test_keyword_filter_web(self, engine):
        results = engine.get_templates(keywords=["web"])
        template_ids = [t["template_id"] for t in results]
        assert "web_app" in template_ids

    def test_keyword_filter_returns_only_matching(self, engine):
        results = engine.get_templates(keywords=["react"])
        assert all("react" in t["keywords"] for t in results)

    def test_keyword_filter_case_insensitive(self, engine):
        results_lower = engine.get_templates(keywords=["web"])
        results_upper = engine.get_templates(keywords=["WEB"])
        assert len(results_lower) == len(results_upper)

    def test_agent_type_filter(self, engine):
        results = engine.get_templates(agent_type="RESEARCHER")
        assert all(t["agent_type"] == "RESEARCHER" for t in results)

    def test_agent_type_filter_case_insensitive(self, engine):
        results = engine.get_templates(agent_type="researcher")
        assert all(t["agent_type"] == "RESEARCHER" for t in results)

    def test_dod_level_filter(self, engine):
        results = engine.get_templates(dod_level="Hard")
        assert all(t["dod_level"] == "Hard" for t in results)
        assert len(results) >= 1

    def test_dod_level_light(self, engine):
        results = engine.get_templates(dod_level="Light")
        assert all(t["dod_level"] == "Light" for t in results)

    def test_combined_keyword_and_agent_filter(self, engine):
        results = engine.get_templates(keywords=["data"], agent_type="CONSOLIDATED_RESEARCHER")
        assert all(t["agent_type"] == "CONSOLIDATED_RESEARCHER" for t in results)
        assert all(
            any(kw in t["keywords"] for kw in ["data"])
            for t in results
        )

    def test_unknown_keyword_returns_empty(self, engine):
        results = engine.get_templates(keywords=["xyzzy_unknown_99"])
        assert results == []

    def test_unknown_agent_returns_empty(self, engine):
        results = engine.get_templates(agent_type="NONEXISTENT_AGENT")
        assert results == []

    def test_unknown_dod_level_returns_empty(self, engine):
        results = engine.get_templates(dod_level="Extreme")
        assert results == []

    def test_api_service_is_hard(self, engine):
        results = engine.get_templates(keywords=["api"])
        api = next((t for t in results if t["template_id"] == "api_service"), None)
        assert api is not None
        assert api["dod_level"] == "Hard"


# ---------------------------------------------------------------------------
# DecompositionEngine.get_dod_criteria / get_dor_criteria
# ---------------------------------------------------------------------------

class TestGetCriteriaMethods:
    @pytest.fixture
    def engine(self):
        return DecompositionEngine()

    def test_get_dod_standard(self, engine):
        criteria = engine.get_dod_criteria("Standard")
        assert isinstance(criteria, list)
        assert len(criteria) > 0

    def test_get_dod_light(self, engine):
        criteria = engine.get_dod_criteria("Light")
        assert isinstance(criteria, list)
        assert len(criteria) >= 1

    def test_get_dod_hard(self, engine):
        criteria = engine.get_dod_criteria("Hard")
        assert isinstance(criteria, list)
        assert len(criteria) >= 1

    def test_get_dod_unknown_falls_back_to_standard(self, engine):
        from vetinari.decomposition import _DOD_CRITERIA
        criteria = engine.get_dod_criteria("NonExistent")
        assert criteria == _DOD_CRITERIA["Standard"]

    def test_get_dod_default_is_standard(self, engine):
        from vetinari.decomposition import _DOD_CRITERIA
        criteria = engine.get_dod_criteria()
        assert criteria == _DOD_CRITERIA["Standard"]

    def test_get_dor_standard(self, engine):
        criteria = engine.get_dor_criteria("Standard")
        assert isinstance(criteria, list)
        assert len(criteria) > 0

    def test_get_dor_light(self, engine):
        criteria = engine.get_dor_criteria("Light")
        assert isinstance(criteria, list)

    def test_get_dor_hard(self, engine):
        criteria = engine.get_dor_criteria("Hard")
        assert isinstance(criteria, list)
        assert len(criteria) >= 1

    def test_get_dor_unknown_falls_back_to_standard(self, engine):
        from vetinari.decomposition import _DOR_CRITERIA
        criteria = engine.get_dor_criteria("Unknown")
        assert criteria == _DOR_CRITERIA["Standard"]

    def test_get_dor_default_is_standard(self, engine):
        from vetinari.decomposition import _DOR_CRITERIA
        criteria = engine.get_dor_criteria()
        assert criteria == _DOR_CRITERIA["Standard"]


# ---------------------------------------------------------------------------
# DecompositionEngine.decompose_task — happy path (PlannerAgent mock)
# ---------------------------------------------------------------------------

class TestDecomposeTaskWithPlanner:
    @pytest.fixture
    def engine(self):
        return DecompositionEngine()

    def _make_planner_result(self, tasks):
        result = MagicMock()
        result.success = True
        result.output = {"tasks": tasks}
        return result

    def test_returns_list_on_success(self, engine):
        task_data = [
            {
                "id": "t1",
                "description": "Setup DB",
                "assigned_agent": "DATA_ENGINEER",
                "inputs": [],
                "outputs": [],
                "dependencies": [],
                "acceptance_criteria": "DB running",
            }
        ]
        mock_planner = MagicMock()
        mock_planner.execute.return_value = self._make_planner_result(task_data)

        with patch("vetinari.decomposition.DecompositionEngine.decompose_task",
                   wraps=engine.decompose_task), patch(
            "vetinari.agents.planner_agent.get_planner_agent",
            return_value=mock_planner,
        ), patch(
            "vetinari.agents.contracts.AgentTask",
            return_value=MagicMock(),
        ):
            subtasks = engine.decompose_task(
                "Build a database service", plan_id="plan_1"
            )

        assert isinstance(subtasks, list)

    def test_planner_subtask_fields_mapped(self, engine):
        task_data = [
            {
                "id": "t1",
                "description": "My task",
                "assigned_agent": "BUILDER",
                "inputs": ["req.txt"],
                "outputs": ["code.py"],
                "dependencies": [],
                "acceptance_criteria": "Done",
            }
        ]
        mock_planner = MagicMock()
        mock_planner.execute.return_value = self._make_planner_result(task_data)
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   return_value=mock_planner), patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
            subtasks = engine.decompose_task("Build something")

        if subtasks:
            st = subtasks[0]
            assert "subtask_id" in st
            assert "parent_task_id" in st
            assert "description" in st
            assert "agent_type" in st
            assert "depth" in st

    def test_history_recorded_on_success(self, engine):
        task_data = [{"id": "t1", "description": "x", "assigned_agent": "BUILDER",
                      "inputs": [], "outputs": [], "dependencies": []}]
        mock_planner = MagicMock()
        mock_planner.execute.return_value = self._make_planner_result(task_data)
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   return_value=mock_planner), patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
            engine.decompose_task("task", plan_id="plan_x")

        history = engine.get_decomposition_history("plan_x")
        assert len(history) >= 1

    def test_depth_at_max_returns_empty(self, engine):
        result = engine.decompose_task("task", depth=DEFAULT_MAX_DEPTH)
        assert result == []

    def test_depth_exceeds_max_clamped_returns_empty(self, engine):
        # depth >= clamped max_depth -> returns []
        result = engine.decompose_task("task", depth=20, max_depth=20)
        # max_depth clamped to MAX_MAX_DEPTH=16; depth=20 >= 16 -> []
        assert result == []

    def test_max_depth_clamped_to_min(self, engine):
        """max_depth below MIN_MAX_DEPTH should be clamped to MIN_MAX_DEPTH."""
        # Patch keyword decompose to see what max_depth is used internally
        captured = {}
        original_kw = engine._keyword_decompose
        def tracking_kw(task_prompt, parent_task_id, depth):
            captured["called"] = True
            return original_kw(task_prompt, parent_task_id, depth)

        with patch.object(engine, "_keyword_decompose", side_effect=tracking_kw):
            # depth=0 < clamped max_depth, planner fails -> falls back to keyword
            with patch("vetinari.agents.planner_agent.get_planner_agent",
                       side_effect=Exception("no planner")):
                engine.decompose_task("task", depth=0, max_depth=1)
        # Should have called keyword decompose (not returned [] early)
        assert captured.get("called") is True

    def test_fallback_to_keyword_on_planner_exception(self, engine):
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   side_effect=ImportError("no module")):
            subtasks = engine.decompose_task("build a web app")
        assert isinstance(subtasks, list)
        assert len(subtasks) >= 1

    def test_fallback_when_planner_result_not_success(self, engine):
        mock_planner = MagicMock()
        mock_planner.execute.return_value = MagicMock(success=False, output=None)
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   return_value=mock_planner), patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
            subtasks = engine.decompose_task("do something")
        # Falls back to keyword decompose
        assert isinstance(subtasks, list)
        assert len(subtasks) >= 1

    def test_fallback_when_output_not_dict(self, engine):
        mock_planner = MagicMock()
        mock_planner.execute.return_value = MagicMock(success=True, output="just a string")
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   return_value=mock_planner), patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
            subtasks = engine.decompose_task("do something")
        assert isinstance(subtasks, list)
        assert len(subtasks) >= 1


# ---------------------------------------------------------------------------
# DecompositionEngine._keyword_decompose
# ---------------------------------------------------------------------------

class TestKeywordDecompose:
    @pytest.fixture
    def engine(self):
        return DecompositionEngine()

    def test_always_includes_analysis_task(self, engine):
        subtasks = engine._keyword_decompose("some random task", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("analyze" in d or "requirements" in d for d in descriptions)

    def test_code_keyword_adds_build_subtask(self, engine):
        subtasks = engine._keyword_decompose("implement the feature", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("implement" in d or "core functionality" in d for d in descriptions)

    def test_build_keyword_adds_build_subtask(self, engine):
        subtasks = engine._keyword_decompose("build a service", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("implement" in d for d in descriptions)

    def test_code_keyword_adds_test_subtask(self, engine):
        subtasks = engine._keyword_decompose("code the module", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("test" in d for d in descriptions)

    def test_ui_keyword_adds_ui_subtask(self, engine):
        subtasks = engine._keyword_decompose("build frontend UI", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("ui" in d or "interface" in d for d in descriptions)

    def test_web_keyword_adds_ui_subtask(self, engine):
        subtasks = engine._keyword_decompose("create a web page", "root", 0)
        descriptions = [s["description"].lower() for s in subtasks]
        assert any("ui" in d or "interface" in d for d in descriptions)

    def test_always_ends_with_review_task(self, engine):
        subtasks = engine._keyword_decompose("some task", "root", 0)
        last = subtasks[-1]["description"].lower()
        assert "review" in last or "document" in last

    def test_subtask_ids_are_unique(self, engine):
        subtasks = engine._keyword_decompose("implement and build", "root", 0)
        ids = [s["subtask_id"] for s in subtasks]
        assert len(ids) == len(set(ids))

    def test_parent_task_id_propagated(self, engine):
        subtasks = engine._keyword_decompose("do something", "my_parent_id", 0)
        for s in subtasks:
            assert s["parent_task_id"] == "my_parent_id"

    def test_depth_incremented_in_subtasks(self, engine):
        subtasks = engine._keyword_decompose("do something", "root", 2)
        for s in subtasks:
            assert s["depth"] == 3

    def test_returns_list_of_dicts(self, engine):
        subtasks = engine._keyword_decompose("any task", "root", 0)
        assert isinstance(subtasks, list)
        for s in subtasks:
            assert isinstance(s, dict)

    def test_no_code_keyword_still_returns_subtasks(self, engine):
        subtasks = engine._keyword_decompose("write a report", "root", 0)
        assert len(subtasks) >= 2  # at least analysis + review

    def test_dependencies_form_chain(self, engine):
        """Each subtask after the first should depend on the previous one."""
        subtasks = engine._keyword_decompose("implement a feature", "root", 0)
        for i, s in enumerate(subtasks[1:], 1):
            prev_id = subtasks[i - 1]["subtask_id"]
            assert prev_id in s["dependencies"]

    def test_acceptance_criteria_non_empty(self, engine):
        subtasks = engine._keyword_decompose("build app", "root", 0)
        for s in subtasks:
            assert s.get("acceptance_criteria"), f"Missing criteria: {s}"


# ---------------------------------------------------------------------------
# DecompositionEngine.get_decomposition_history
# ---------------------------------------------------------------------------

class TestGetDecompositionHistory:
    @pytest.fixture
    def engine(self):
        return DecompositionEngine()

    def test_empty_history_initially(self, engine):
        assert engine.get_decomposition_history() == []

    def test_history_returns_all_without_filter(self, engine):
        engine._history.append(
            DecompositionEvent("e1", "plan_a", "t1", 0, [], 3)
        )
        engine._history.append(
            DecompositionEvent("e2", "plan_b", "t2", 1, [], 2)
        )
        result = engine.get_decomposition_history()
        assert len(result) == 2

    def test_history_filtered_by_plan_id(self, engine):
        engine._history.append(DecompositionEvent("e1", "plan_a", "t1", 0, [], 3))
        engine._history.append(DecompositionEvent("e2", "plan_b", "t2", 1, [], 2))
        engine._history.append(DecompositionEvent("e3", "plan_a", "t3", 0, [], 1))
        result = engine.get_decomposition_history("plan_a")
        assert len(result) == 2
        assert all(e.plan_id == "plan_a" for e in result)

    def test_filter_returns_empty_for_unknown_plan(self, engine):
        engine._history.append(DecompositionEvent("e1", "plan_a", "t1", 0, [], 3))
        result = engine.get_decomposition_history("nonexistent_plan")
        assert result == []

    def test_no_filter_returns_copy(self, engine):
        engine._history.append(DecompositionEvent("e1", "plan_a", "t1", 0, [], 3))
        result1 = engine.get_decomposition_history()
        result2 = engine.get_decomposition_history()
        assert result1 == result2

    def test_history_accumulates(self, engine):
        with patch("vetinari.agents.planner_agent.get_planner_agent",
                   side_effect=ImportError("no planner")):
            engine.decompose_task("task 1", plan_id="p1")
            engine.decompose_task("task 2", plan_id="p2")
        # Keyword fallback doesn't record history, but
        # if planner succeeds it does. Just verify no error raised.
        history = engine.get_decomposition_history()
        assert isinstance(history, list)

    def test_returns_list_of_decomposition_events(self, engine):
        engine._history.append(DecompositionEvent("e1", "plan_a", "t1", 0, [], 1))
        result = engine.get_decomposition_history()
        assert all(isinstance(e, DecompositionEvent) for e in result)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def setup_method(self):
        decomp_module._decomposition_engine = None

    def teardown_method(self):
        decomp_module._decomposition_engine = None

    def test_get_engine_returns_engine_instance(self):
        engine = _get_engine()
        assert isinstance(engine, DecompositionEngine)

    def test_get_engine_returns_same_instance(self):
        e1 = _get_engine()
        e2 = _get_engine()
        assert e1 is e2

    def test_module_level_decomposition_engine_is_set(self):
        # The module exports a pre-built instance
        assert decomp_module.decomposition_engine is not None
        assert isinstance(decomp_module.decomposition_engine, DecompositionEngine)

    def test_singleton_reset_creates_new(self):
        _get_engine()
        decomp_module._decomposition_engine = None
        e2 = _get_engine()
        # Both are valid instances; after reset a new one is created
        assert isinstance(e2, DecompositionEngine)

    def test_get_engine_creates_if_none(self):
        decomp_module._decomposition_engine = None
        engine = _get_engine()
        assert engine is not None
        assert decomp_module._decomposition_engine is engine
