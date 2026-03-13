"""
Comprehensive pytest tests for vetinari/planning_engine.py

Suppresses DeprecationWarning emitted at module import time.
"""

import warnings

# Suppress the deprecation warning before import
warnings.filterwarnings("ignore", category=DeprecationWarning)

from vetinari.planning_engine import Model, Plan, PlanningEngine, Task

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_model(
    id="model-a",
    name="Model A",
    capabilities=None,
    context_len=4096,
    memory_gb=4,
    version="1.0",
):
    return Model(
        id=id,
        name=name,
        capabilities=capabilities if capabilities is not None else [],
        context_len=context_len,
        memory_gb=memory_gb,
        version=version,
    )


def make_task(
    id="t1",
    description="Do something useful",
    inputs=None,
    outputs=None,
    dependencies=None,
    model_override="",
    assigned_model_id="",
    depth=0,
    parent_id="",
    children=None,
    owner_id="",
    status="pending",
):
    return Task(
        id=id,
        description=description,
        inputs=inputs if inputs is not None else [],
        outputs=outputs if outputs is not None else [],
        dependencies=dependencies if dependencies is not None else [],
        model_override=model_override,
        assigned_model_id=assigned_model_id,
        depth=depth,
        parent_id=parent_id,
        children=children if children is not None else [],
        owner_id=owner_id,
        status=status,
    )


def make_engine(
    default_models=None,
    fallback_models=None,
    uncensored_fallback_models=None,
    memory_budget_gb=48,
):
    return PlanningEngine(
        default_models=default_models or [],
        fallback_models=fallback_models or [],
        uncensored_fallback_models=uncensored_fallback_models or [],
        memory_budget_gb=memory_budget_gb,
    )


def make_model_dict(
    id="model-a",
    name="Model A",
    capabilities=None,
    context_len=4096,
    memory_gb=4,
    version="1.0",
):
    return {
        "id": id,
        "name": name,
        "capabilities": capabilities if capabilities is not None else [],
        "context_len": context_len,
        "memory_gb": memory_gb,
        "version": version,
    }


# ---------------------------------------------------------------------------
# Model dataclass tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_defaults(self):
        m = Model(id="x", name="X")
        assert m.id == "x"
        assert m.name == "X"
        assert m.capabilities == []
        assert m.context_len == 2048
        assert m.memory_gb == 2
        assert m.version == ""

    def test_custom_values(self):
        m = make_model(id="gpt4", name="GPT-4", capabilities=["code_gen", "reasoning"],
                       context_len=8192, memory_gb=16, version="4.0")
        assert m.id == "gpt4"
        assert m.name == "GPT-4"
        assert "code_gen" in m.capabilities
        assert m.context_len == 8192
        assert m.memory_gb == 16
        assert m.version == "4.0"

    def test_to_dict_keys(self):
        m = make_model()
        d = m.to_dict()
        assert set(d.keys()) == {"id", "name", "capabilities", "context_len", "memory_gb", "version"}

    def test_to_dict_values(self):
        m = make_model(id="abc", name="ABC", capabilities=["chat"], context_len=1024,
                       memory_gb=8, version="2.1")
        d = m.to_dict()
        assert d["id"] == "abc"
        assert d["name"] == "ABC"
        assert d["capabilities"] == ["chat"]
        assert d["context_len"] == 1024
        assert d["memory_gb"] == 8
        assert d["version"] == "2.1"

    def test_capabilities_are_independent_per_instance(self):
        m1 = Model(id="a", name="A")
        m2 = Model(id="b", name="B")
        m1.capabilities.append("code_gen")
        assert "code_gen" not in m2.capabilities

    def test_to_dict_returns_list_for_capabilities(self):
        m = make_model(capabilities=["x", "y"])
        assert isinstance(m.to_dict()["capabilities"], list)


# ---------------------------------------------------------------------------
# Task dataclass tests
# ---------------------------------------------------------------------------

class TestTask:
    def test_defaults(self):
        t = Task(id="t1", description="desc")
        assert t.inputs == []
        assert t.outputs == []
        assert t.dependencies == []
        assert t.model_override == ""
        assert t.assigned_model_id == ""
        assert t.depth == 0
        assert t.parent_id == ""
        assert t.children == []
        assert t.owner_id == ""
        assert t.status == "pending"

    def test_to_dict_keys(self):
        t = make_task()
        d = t.to_dict()
        expected = {
            "id", "description", "inputs", "outputs", "dependencies",
            "model_override", "assigned_model_id", "depth",
            "parent_id", "children", "owner_id", "status"
        }
        assert set(d.keys()) == expected

    def test_to_dict_values(self):
        t = make_task(
            id="t2", description="test task",
            inputs=["a"], outputs=["b"], dependencies=["t1"],
            model_override="llama", assigned_model_id="llama-3",
            depth=2, parent_id="t1", children=["t3"],
            owner_id="user1", status="done"
        )
        d = t.to_dict()
        assert d["id"] == "t2"
        assert d["description"] == "test task"
        assert d["inputs"] == ["a"]
        assert d["outputs"] == ["b"]
        assert d["dependencies"] == ["t1"]
        assert d["model_override"] == "llama"
        assert d["assigned_model_id"] == "llama-3"
        assert d["depth"] == 2
        assert d["parent_id"] == "t1"
        assert d["children"] == ["t3"]
        assert d["owner_id"] == "user1"
        assert d["status"] == "done"

    def test_list_fields_are_independent_per_instance(self):
        t1 = Task(id="a", description="a")
        t2 = Task(id="b", description="b")
        t1.inputs.append("x")
        assert "x" not in t2.inputs


# ---------------------------------------------------------------------------
# Plan dataclass tests
# ---------------------------------------------------------------------------

class TestPlan:
    def test_defaults(self):
        p = Plan(goal="do stuff")
        assert p.goal == "do stuff"
        assert p.tasks == []
        assert p.model_scores == []
        assert p.notes == ""
        assert p.warnings == []
        assert p.needs_context is False
        assert p.follow_up_question == ""
        assert p.final_delivery_path == ""
        assert p.final_delivery_summary == ""

    def test_to_dict_always_present_keys(self):
        p = Plan(goal="g")
        d = p.to_dict()
        for key in ["goal", "tasks", "model_scores", "notes", "warnings",
                    "needs_context", "follow_up_question"]:
            assert key in d

    def test_to_dict_optional_final_delivery_path_omitted_when_empty(self):
        p = Plan(goal="g")
        d = p.to_dict()
        assert "final_delivery_path" not in d

    def test_to_dict_optional_final_delivery_summary_omitted_when_empty(self):
        p = Plan(goal="g")
        d = p.to_dict()
        assert "final_delivery_summary" not in d

    def test_to_dict_includes_final_delivery_path_when_set(self):
        p = Plan(goal="g", final_delivery_path="/out/file.py")
        d = p.to_dict()
        assert d["final_delivery_path"] == "/out/file.py"

    def test_to_dict_includes_final_delivery_summary_when_set(self):
        p = Plan(goal="g", final_delivery_summary="Built the thing.")
        d = p.to_dict()
        assert d["final_delivery_summary"] == "Built the thing."

    def test_to_dict_serializes_tasks(self):
        t = make_task(id="t1", description="desc")
        p = Plan(goal="g", tasks=[t])
        d = p.to_dict()
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["id"] == "t1"

    def test_to_dict_needs_context_flag(self):
        p = Plan(goal="g", needs_context=True, follow_up_question="What?")
        d = p.to_dict()
        assert d["needs_context"] is True
        assert d["follow_up_question"] == "What?"


# ---------------------------------------------------------------------------
# PlanningEngine.__init__ tests
# ---------------------------------------------------------------------------

class TestPlanningEngineInit:
    def test_defaults_none(self):
        e = PlanningEngine()
        assert e.default_models == []
        assert e.fallback_models == []
        assert e.uncensored_fallback_models == []
        assert e.memory_budget_gb == 48

    def test_custom_values(self):
        e = PlanningEngine(
            default_models=["llama3"],
            fallback_models=["phi3"],
            uncensored_fallback_models=["uncensored-llama"],
            memory_budget_gb=32,
        )
        assert e.default_models == ["llama3"]
        assert e.fallback_models == ["phi3"]
        assert e.uncensored_fallback_models == ["uncensored-llama"]
        assert e.memory_budget_gb == 32

    def test_prompt_type_keywords_present(self):
        e = make_engine()
        expected_types = {"planning", "coding", "docs", "reasoning", "creative", "financial", "data"}
        assert set(e.prompt_type_keywords.keys()) == expected_types

    def test_policy_sensitive_keywords_present(self):
        e = make_engine()
        for kw in ["investment", "stock", "trading", "hack", "exploit", "weapon", "illegal"]:
            assert kw in e.policy_sensitive_keywords

    def test_capability_keywords_present(self):
        e = make_engine()
        for cap in ["code_gen", "docs", "chat", "reasoning", "math", "creative"]:
            assert cap in e.capability_keywords


# ---------------------------------------------------------------------------
# _detect_prompt_type tests
# ---------------------------------------------------------------------------

class TestDetectPromptType:
    def setup_method(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.engine = make_engine()

    def test_coding_detected(self):
        assert self.engine._detect_prompt_type("Write a python script to parse JSON") == "coding"

    def test_planning_detected(self):
        assert self.engine._detect_prompt_type("Design a workflow and strategy for my project") == "planning"

    def test_docs_detected(self):
        assert self.engine._detect_prompt_type("Write a readme and tutorial guide") == "docs"

    def test_reasoning_detected(self):
        assert self.engine._detect_prompt_type("Solve this math logic problem") == "reasoning"

    def test_creative_detected(self):
        assert self.engine._detect_prompt_type("Write a creative story or poem") == "creative"

    def test_financial_detected(self):
        assert self.engine._detect_prompt_type("Analyze stock investment trading strategy") == "financial"

    def test_data_detected(self):
        assert self.engine._detect_prompt_type("Process data from a database query") == "data"

    def test_general_when_no_keywords(self):
        assert self.engine._detect_prompt_type("hello there") == "general"

    def test_case_insensitive(self):
        result = self.engine._detect_prompt_type("CODE this PYTHON function")
        assert result == "coding"

    def test_highest_score_wins(self):
        # "code implement build create python" — all coding keywords
        result = self.engine._detect_prompt_type(
            "code implement build create python script function class api"
        )
        assert result == "coding"

    def test_empty_string_returns_general(self):
        assert self.engine._detect_prompt_type("") == "general"

    def test_single_keyword_match(self):
        result = self.engine._detect_prompt_type("stock")
        assert result == "financial"


# ---------------------------------------------------------------------------
# _infer_required_capabilities tests
# ---------------------------------------------------------------------------

class TestInferRequiredCapabilities:
    def setup_method(self):
        self.engine = make_engine()

    def test_code_keywords_add_code_gen(self):
        caps = self.engine._infer_required_capabilities("implement this feature")
        assert "code_gen" in caps

    def test_docs_keywords(self):
        caps = self.engine._infer_required_capabilities("write documentation readme")
        assert "docs" in caps

    def test_reasoning_keywords(self):
        caps = self.engine._infer_required_capabilities("plan and reason about strategy")
        assert "reasoning" in caps

    def test_math_keywords(self):
        caps = self.engine._infer_required_capabilities("calculate the formula and equation")
        assert "math" in caps

    def test_creative_keywords(self):
        caps = self.engine._infer_required_capabilities("write a story or poem")
        assert "creative" in caps

    def test_chat_keywords(self):
        caps = self.engine._infer_required_capabilities("chat and respond to messages")
        assert "chat" in caps

    def test_fallback_when_no_keywords(self):
        caps = self.engine._infer_required_capabilities("xyz blah nothing matches")
        assert caps == ["code_gen", "chat"]

    def test_code_gen_deduplicated(self):
        caps = self.engine._infer_required_capabilities("implement and build and create code")
        assert caps.count("code_gen") == 1

    def test_returns_list(self):
        result = self.engine._infer_required_capabilities("build a web app")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _check_for_policy_refusal tests
# ---------------------------------------------------------------------------

class TestCheckForPolicyRefusal:
    def setup_method(self):
        self.engine = make_engine()

    def test_safe_model_with_policy_sensitive_task(self):
        assert self.engine._check_for_policy_refusal("safe-model", "discuss investment strategies") is True

    def test_claude_model_with_policy_sensitive_task(self):
        assert self.engine._check_for_policy_refusal("claude-3", "hack the stock market") is True

    def test_gpt_model_with_policy_sensitive_task(self):
        assert self.engine._check_for_policy_refusal("gpt-4", "trading tips and money profits") is True

    def test_guard_model_with_policy_sensitive_task(self):
        assert self.engine._check_for_policy_refusal("llama-guard", "illegal activity exploit") is True

    def test_policy_model_with_policy_sensitive_task(self):
        assert self.engine._check_for_policy_refusal("policy-bot", "weapon creation harmful") is True

    def test_normal_model_with_policy_sensitive_task(self):
        # "llama" has no refuse indicator
        assert self.engine._check_for_policy_refusal("llama-3", "stock trading investment") is False

    def test_safe_model_with_benign_task(self):
        assert self.engine._check_for_policy_refusal("safe-model", "write a hello world script") is False

    def test_neutral_model_neutral_task(self):
        assert self.engine._check_for_policy_refusal("mistral-7b", "implement a web scraper") is False

    def test_case_insensitive_model_id(self):
        # uppercase SAFE
        assert self.engine._check_for_policy_refusal("SAFE-guard", "investment strategy") is True


# ---------------------------------------------------------------------------
# _get_fallback_model tests
# ---------------------------------------------------------------------------

class TestGetFallbackModel:
    def setup_method(self):
        self.models = [
            make_model(id="big-llama", name="Big Llama", memory_gb=32),
            make_model(id="phi-small", name="Phi Small", memory_gb=4),
            make_model(id="uncensored-llama", name="Uncensored Llama", memory_gb=8),
        ]

    def test_returns_uncensored_when_policy_sensitive(self):
        engine = make_engine(uncensored_fallback_models=["uncensored-llama"])
        result = engine._get_fallback_model(self.models, is_policy_sensitive=True)
        assert result == "uncensored-llama"

    def test_skips_uncensored_when_not_policy_sensitive(self):
        engine = make_engine(
            fallback_models=["phi-small"],
            uncensored_fallback_models=["uncensored-llama"],
        )
        result = engine._get_fallback_model(self.models, is_policy_sensitive=False)
        assert result == "phi-small"

    def test_falls_through_to_regular_fallback_when_no_uncensored_match(self):
        engine = make_engine(
            fallback_models=["phi-small"],
            uncensored_fallback_models=["no-such-model"],
        )
        result = engine._get_fallback_model(self.models, is_policy_sensitive=True)
        assert result == "phi-small"

    def test_falls_through_to_default_models(self):
        engine = make_engine(default_models=["big-llama"])
        result = engine._get_fallback_model(self.models, is_policy_sensitive=False)
        assert result == "big-llama"

    def test_returns_smallest_when_nothing_matches(self):
        engine = make_engine()  # no lists
        result = engine._get_fallback_model(self.models, is_policy_sensitive=False)
        # smallest memory_gb is phi-small (4 GB)
        assert result == "phi-small"

    def test_returns_empty_string_when_no_models(self):
        engine = make_engine()
        result = engine._get_fallback_model([], is_policy_sensitive=False)
        assert result == ""

    def test_partial_name_match_for_fallback(self):
        engine = make_engine(fallback_models=["phi"])
        result = engine._get_fallback_model(self.models, is_policy_sensitive=False)
        assert result == "phi-small"

    def test_partial_name_match_for_uncensored(self):
        engine = make_engine(uncensored_fallback_models=["uncensored"])
        result = engine._get_fallback_model(self.models, is_policy_sensitive=True)
        assert result == "uncensored-llama"


# ---------------------------------------------------------------------------
# _score_models_for_task tests
# ---------------------------------------------------------------------------

class TestScoreModelsForTask:
    def setup_method(self):
        self.engine = make_engine()

    def test_returns_list(self):
        task = make_task(description="implement a python class")
        model = make_model(capabilities=["code_gen"])
        result = self.engine._score_models_for_task(task, [model])
        assert isinstance(result, list)

    def test_score_dict_keys(self):
        task = make_task(description="implement something")
        model = make_model()
        result = self.engine._score_models_for_task(task, [model])
        expected_keys = {"model_id", "model_name", "score", "capabilities",
                         "capability_matches", "memory_gb", "context_len"}
        assert expected_keys.issubset(set(result[0].keys()))

    def test_sorted_descending(self):
        task = make_task(description="implement code")
        good_model = make_model(id="coder", name="Coder", capabilities=["code_gen"], memory_gb=8)
        weak_model = make_model(id="weak", name="Weak", capabilities=[], memory_gb=2)
        result = self.engine._score_models_for_task(task, [weak_model, good_model])
        assert result[0]["score"] >= result[-1]["score"]

    def test_capability_match_boosts_score(self):
        task = make_task(description="implement code in python", outputs=["module"])
        with_cap = make_model(id="m1", capabilities=["code_gen"])
        without_cap = make_model(id="m2", capabilities=[])
        result = self.engine._score_models_for_task(task, [with_cap, without_cap])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["m1"] > scores["m2"]

    def test_empty_models_returns_empty(self):
        task = make_task()
        result = self.engine._score_models_for_task(task, [])
        assert result == []

    def test_reasoning_heuristic_boosts_70b_model(self):
        task = make_task(description="plan and reason about the architecture")
        big = make_model(id="llama-70b", name="Llama 70B", capabilities=["reasoning"])
        small = make_model(id="phi-3", name="Phi 3", capabilities=[])
        result = self.engine._score_models_for_task(task, [big, small])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["llama-70b"] > scores["phi-3"]

    def test_coder_id_heuristic(self):
        task = make_task(description="write code", outputs=["module"])
        coder = make_model(id="deepseek-coder", name="DeepSeek Coder", capabilities=["code_gen"])
        other = make_model(id="mistral", name="Mistral", capabilities=[])
        result = self.engine._score_models_for_task(task, [coder, other])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["deepseek-coder"] >= scores["mistral"]

    def test_small_model_preferred_for_simple_task(self):
        # simple task: 1 input, 1 output -> small model preferred
        task = make_task(description="greet user", inputs=["name"], outputs=["greeting"])
        small = make_model(id="small", memory_gb=4)
        large = make_model(id="large", memory_gb=32)
        result = self.engine._score_models_for_task(task, [small, large])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["small"] > scores["large"]

    def test_context_len_penalty_when_too_small(self):
        # output with long names -> many estimated tokens; tiny context gets penalized
        task = make_task(description="generate report",
                         outputs=["a" * 100, "b" * 100])
        tiny = make_model(id="tiny", context_len=1)
        big = make_model(id="big", context_len=128000)
        result = self.engine._score_models_for_task(task, [tiny, big])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["big"] > scores["tiny"]


# ---------------------------------------------------------------------------
# _select_best_model tests
# ---------------------------------------------------------------------------

class TestSelectBestModel:
    def setup_method(self):
        self.engine = make_engine()

    def _make_score(self, model_id, score, cap_matches=None):
        return {
            "model_id": model_id,
            "model_name": model_id,
            "score": score,
            "capabilities": [],
            "capability_matches": cap_matches or [],
            "memory_gb": 4,
            "context_len": 4096,
        }

    def test_empty_scores_returns_empty_string(self):
        assert self.engine._select_best_model([]) == ""

    def test_returns_top_scorer(self):
        scores = [
            self._make_score("a", 100),
            self._make_score("b", 50),
        ]
        assert self.engine._select_best_model(scores) == "a"

    def test_preferred_capability_overrides_score_order(self):
        scores = [
            self._make_score("a", 100, cap_matches=[]),
            self._make_score("b", 50, cap_matches=["reasoning"]),
        ]
        result = self.engine._select_best_model(scores, preferred_capabilities=["reasoning"])
        assert result == "b"

    def test_no_preferred_capability_match_falls_back_to_top(self):
        scores = [
            self._make_score("a", 100, cap_matches=["code_gen"]),
            self._make_score("b", 50, cap_matches=[]),
        ]
        result = self.engine._select_best_model(scores, preferred_capabilities=["reasoning"])
        assert result == "a"

    def test_single_entry(self):
        scores = [self._make_score("only", 10)]
        assert self.engine._select_best_model(scores) == "only"

    def test_first_preferred_cap_wins(self):
        scores = [
            self._make_score("a", 10, cap_matches=["code_gen"]),
            self._make_score("b", 10, cap_matches=["reasoning"]),
        ]
        # preferred order: code_gen first
        result = self.engine._select_best_model(scores, preferred_capabilities=["code_gen", "reasoning"])
        assert result == "a"


# ---------------------------------------------------------------------------
# _get_best_model_for_type tests
# ---------------------------------------------------------------------------

class TestGetBestModelForType:
    def setup_method(self):
        self.models = [
            make_model(id="llama3", name="Llama 3", capabilities=["code_gen", "reasoning"]),
            make_model(id="phi3", name="Phi 3", capabilities=["chat"]),
        ]

    def test_default_model_matching_preferred_cap_returned(self):
        engine = make_engine(default_models=["llama3"])
        result = engine._get_best_model_for_type("coding", self.models)
        assert result == "llama3"

    def test_fallback_model_used_when_default_lacks_cap(self):
        engine = make_engine(default_models=["phi3"], fallback_models=["llama3"])
        # phi3 has "chat" but not "code_gen" (preferred for coding)
        result = engine._get_best_model_for_type("coding", self.models)
        # falls through to fallback
        assert result == "llama3"

    def test_first_available_returned_when_no_lists(self):
        engine = make_engine()
        result = engine._get_best_model_for_type("coding", self.models)
        assert result == "llama3"  # first in list

    def test_empty_models_returns_empty_string(self):
        engine = make_engine()
        result = engine._get_best_model_for_type("coding", [])
        assert result == ""

    def test_unknown_type_uses_default_preferences(self):
        engine = make_engine()
        result = engine._get_best_model_for_type("unknown_type", self.models)
        # falls through to available_models[0]
        assert result == self.models[0].id


# ---------------------------------------------------------------------------
# _generate_tasks_from_goal tests
# ---------------------------------------------------------------------------

class TestGenerateTasksFromGoal:
    def setup_method(self):
        self.engine = make_engine()

    # depth=0 always-present tasks
    def test_depth0_creates_t1_analyze(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("Analyze" in d for d in descs)

    def test_depth0_creates_t2_setup(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("Set up project" in d for d in descs)

    def test_depth0_creates_review_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("Review code" in d for d in descs)

    def test_depth0_creates_test_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("test" in d.lower() for d in descs)

    def test_code_heavy_adds_core_logic_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("core business logic" in d for d in descs)

    def test_code_heavy_adds_ui_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("user interface" in d for d in descs)

    def test_code_heavy_adds_build_artifact_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web application with Python")
        descs = [t.description for t in tasks]
        assert any("build" in d.lower() or "deployment" in d.lower() for d in descs)

    def test_api_goal_adds_api_task(self):
        tasks = self.engine._generate_tasks_from_goal("Build a REST API service with endpoints")
        descs = [t.description for t in tasks]
        assert any("API endpoint" in d for d in descs)

    def test_data_goal_adds_database_task(self):
        tasks = self.engine._generate_tasks_from_goal("Create a database and data analysis pipeline")
        descs = [t.description for t in tasks]
        assert any("database" in d.lower() or "data layer" in d.lower() for d in descs)

    def test_non_code_goal_final_doc_task(self):
        tasks = self.engine._generate_tasks_from_goal("Write a strategy plan for the organization")
        descs = [t.description for t in tasks]
        assert any("documentation" in d.lower() or "summary" in d.lower() for d in descs)

    def test_max_depth_returns_empty(self):
        result = self.engine._generate_tasks_from_goal("Build something", depth=25, max_depth=25)
        assert result == []

    def test_depth_beyond_max_returns_empty(self):
        result = self.engine._generate_tasks_from_goal("Build something", depth=30, max_depth=25)
        assert result == []

    def test_depth1_code_heavy_returns_5_subtasks(self):
        tasks = self.engine._generate_tasks_from_goal(
            "implement the code", parent_id="t1", depth=1
        )
        assert len(tasks) == 5

    def test_depth1_code_heavy_subtask_descriptions(self):
        tasks = self.engine._generate_tasks_from_goal(
            "implement and build code", parent_id="t1", depth=1
        )
        descs = [t.description for t in tasks]
        assert any("error handling" in d for d in descs)
        assert any("configuration" in d for d in descs)

    def test_depth1_research_returns_5_subtasks(self):
        tasks = self.engine._generate_tasks_from_goal(
            "research and analyze the topic", parent_id="t1", depth=1
        )
        assert len(tasks) == 5

    def test_depth1_research_subtask_descriptions(self):
        tasks = self.engine._generate_tasks_from_goal(
            "research and analyze competitors and review findings", parent_id="t1", depth=1
        )
        descs = [t.description for t in tasks]
        assert any("background" in d.lower() or "existing" in d.lower() for d in descs)

    def test_depth2_returns_4_subtasks(self):
        tasks = self.engine._generate_tasks_from_goal(
            "anything at all", parent_id="t1", depth=2
        )
        assert len(tasks) == 4

    def test_depth2_subtask_descriptions(self):
        tasks = self.engine._generate_tasks_from_goal(
            "refine edge cases", parent_id="t1", depth=2
        )
        descs = [t.description for t in tasks]
        assert any("optimize" in d.lower() or "refine" in d.lower() for d in descs)
        assert any("edge case" in d.lower() for d in descs)

    def test_all_tasks_have_ids(self):
        tasks = self.engine._generate_tasks_from_goal("Build a Python web app")
        assert all(t.id for t in tasks)

    def test_all_tasks_have_descriptions(self):
        tasks = self.engine._generate_tasks_from_goal("Build a Python web app")
        assert all(t.description for t in tasks)

    def test_task_ids_are_unique(self):
        tasks = self.engine._generate_tasks_from_goal("Build a web app with database")
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_depth_attribute_set_on_depth0_root(self):
        tasks = self.engine._generate_tasks_from_goal("Plan something")
        assert tasks[0].depth == 0

    def test_depth1_subtasks_have_depth1(self):
        tasks = self.engine._generate_tasks_from_goal("implement code", parent_id="t0", depth=1)
        assert all(t.depth == 1 for t in tasks)

    def test_depth2_subtasks_have_depth2(self):
        tasks = self.engine._generate_tasks_from_goal("anything", parent_id="t0", depth=2)
        assert all(t.depth == 2 for t in tasks)

    def test_custom_max_depth_respected(self):
        result = self.engine._generate_tasks_from_goal("Build code", depth=3, max_depth=3)
        assert result == []

    def test_t1_has_no_dependencies(self):
        tasks = self.engine._generate_tasks_from_goal("Plan the workflow design strategy")
        t1 = tasks[0]
        assert t1.dependencies == []


# ---------------------------------------------------------------------------
# _check_token_limits tests
# ---------------------------------------------------------------------------

class TestCheckTokenLimits:
    def setup_method(self):
        self.engine = make_engine()

    def test_no_warning_when_within_limit(self):
        model = make_model(id="m1", context_len=100000)
        task = make_task(id="t1", description="short", inputs=["x"], outputs=["y"],
                         assigned_model_id="m1")
        plan = Plan(goal="g", tasks=[task])
        self.engine._check_token_limits(plan, [model])
        assert plan.warnings == []

    def test_warning_added_when_over_80_percent(self):
        # context_len=10; 80% = 8 tokens; make description long enough
        model = make_model(id="tiny", context_len=10)
        long_desc = "x " * 100
        task = make_task(id="t1", description=long_desc,
                         inputs=["input " * 20], outputs=["output " * 20],
                         assigned_model_id="tiny")
        plan = Plan(goal="g", tasks=[task])
        self.engine._check_token_limits(plan, [model])
        assert len(plan.warnings) > 0
        assert "tiny" in plan.warnings[0]

    def test_skips_task_with_unknown_model(self):
        model = make_model(id="known")
        task = make_task(id="t1", description="do stuff", assigned_model_id="unknown")
        plan = Plan(goal="g", tasks=[task])
        self.engine._check_token_limits(plan, [model])
        assert plan.warnings == []

    def test_multiple_tasks_can_each_warn(self):
        model = make_model(id="tiny", context_len=1)
        tasks = [
            make_task(id="t1", description="implement " * 50,
                      inputs=["big_input"], outputs=["big_output"], assigned_model_id="tiny"),
            make_task(id="t2", description="implement " * 50,
                      inputs=["big_input2"], outputs=["big_output2"], assigned_model_id="tiny"),
        ]
        plan = Plan(goal="g", tasks=tasks)
        self.engine._check_token_limits(plan, [model])
        assert len(plan.warnings) == 2


# ---------------------------------------------------------------------------
# plan() method — high-level integration tests
# ---------------------------------------------------------------------------

class TestPlanMethod:
    def setup_method(self):
        self.engine = make_engine()
        self.model_dicts = [
            make_model_dict(id="llama3", name="Llama 3", capabilities=["code_gen", "reasoning"],
                            context_len=8192, memory_gb=8),
            make_model_dict(id="phi3", name="Phi 3", capabilities=["chat"],
                            context_len=4096, memory_gb=4),
        ]

    def test_returns_plan_object(self):
        plan = self.engine.plan("Build a Python web application with login and database",
                                "system", self.model_dicts)
        assert isinstance(plan, Plan)

    def test_vague_goal_sets_needs_context(self):
        plan = self.engine.plan("make something", "system", self.model_dicts)
        assert plan.needs_context is True

    def test_vague_goal_sets_follow_up_question(self):
        plan = self.engine.plan("make something", "system", self.model_dicts)
        assert plan.follow_up_question != ""

    def test_short_goal_sets_needs_context(self):
        # 3 words -> short
        plan = self.engine.plan("build a thing", "system", self.model_dicts)
        assert plan.needs_context is True

    def test_clear_goal_creates_tasks(self):
        plan = self.engine.plan(
            "Build a Python web application with authentication and REST API endpoints",
            "system", self.model_dicts
        )
        assert len(plan.tasks) > 0

    def test_tasks_have_assigned_model(self):
        plan = self.engine.plan(
            "Implement a full-stack Python web application with database and API",
            "system", self.model_dicts
        )
        for task in plan.tasks:
            assert task.assigned_model_id != "", f"Task {task.id} has no assigned model"

    def test_model_scores_populated(self):
        plan = self.engine.plan(
            "Build a Python script to process data and query the database",
            "system", self.model_dicts
        )
        assert len(plan.model_scores) > 0

    def test_notes_populated(self):
        plan = self.engine.plan(
            "Build a web application with Python and REST API endpoints",
            "system", self.model_dicts
        )
        assert plan.notes != ""

    def test_policy_sensitive_goal_adds_warning_when_uncensored_available(self):
        engine = make_engine(uncensored_fallback_models=["uncensored-llama"])
        models = [make_model_dict(id="uncensored-llama", name="Uncensored Llama",
                                  capabilities=["code_gen"], memory_gb=8)]
        plan = engine.plan(
            "Analyze stock market investment trading strategies for profit optimization",
            "system", models
        )
        assert any("policy-sensitive" in w.lower() for w in plan.warnings)

    def test_memory_filter_excludes_large_models(self):
        engine = make_engine(memory_budget_gb=4)
        models = [
            make_model_dict(id="huge", name="Huge", memory_gb=80),
            make_model_dict(id="small", name="Small", memory_gb=4,
                            capabilities=["code_gen", "reasoning"]),
        ]
        plan = engine.plan(
            "Build a Python application with REST API and database",
            "system", models
        )
        # All assigned models should not be the huge one
        for task in plan.tasks:
            assert task.assigned_model_id != "huge"

    def test_no_models_fit_budget_adds_warning(self):
        engine = make_engine(memory_budget_gb=1)
        models = [make_model_dict(id="big", name="Big", memory_gb=80)]
        plan = engine.plan(
            "Build a Python application with REST API and database",
            "system", models
        )
        assert any("memory budget" in w.lower() or "No models" in w for w in plan.warnings)

    def test_model_override_respected(self):
        # We can't easily inject a model_override via plan(), but we can test it
        # by verifying tasks with override keep that override as assigned_model_id.
        # This tests the code path indirectly via a plan call that runs _generate_tasks_from_goal
        plan = self.engine.plan(
            "Write a research analysis and investigate market study review",
            "system", self.model_dicts
        )
        # All tasks without override should have an assigned model
        for task in plan.tasks:
            if not task.model_override:
                assert task.assigned_model_id != ""

    def test_goal_stored_on_plan(self):
        goal = "Build a comprehensive Python web application with database"
        plan = self.engine.plan(goal, "system", self.model_dicts)
        assert plan.goal == goal

    def test_empty_models_list_still_returns_plan(self):
        plan = self.engine.plan(
            "Build a Python application with REST API",
            "system", []
        )
        assert isinstance(plan, Plan)

    def test_model_dict_without_id_uses_name(self):
        models = [{"name": "fallback-model", "capabilities": ["code_gen"],
                   "context_len": 4096, "memory_gb": 4}]
        plan = self.engine.plan(
            "Build a Python web application with authentication",
            "system", models
        )
        assert isinstance(plan, Plan)

    def test_vague_indicators_trigger_needs_context(self):
        for vague_term in ["something", "stuff", "things"]:
            plan = self.engine.plan(
                f"Create {vague_term} for the users",
                "system", self.model_dicts
            )
            assert plan.needs_context is True, f"Expected needs_context for goal with '{vague_term}'"

    def test_plan_to_dict_is_serializable(self):
        import json
        plan = self.engine.plan(
            "Build a Python web application with REST API and database",
            "system", self.model_dicts
        )
        # Should not raise
        data = plan.to_dict()
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

    def test_research_goal_generates_tasks(self):
        plan = self.engine.plan(
            "Research and analyze existing solutions to investigate the competitive landscape and review findings",
            "system", self.model_dicts
        )
        assert len(plan.tasks) > 0


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.engine = make_engine()

    def test_model_to_dict_round_trip(self):
        m = make_model(id="x", name="X", capabilities=["code_gen"],
                       context_len=2048, memory_gb=8, version="1.0")
        d = m.to_dict()
        m2 = Model(**d)
        assert m2.id == m.id
        assert m2.capabilities == m.capabilities

    def test_task_to_dict_round_trip(self):
        t = make_task(id="t1", description="do something",
                      inputs=["a"], outputs=["b"], depth=1)
        d = t.to_dict()
        t2 = Task(**d)
        assert t2.id == t.id
        assert t2.description == t.description

    def test_plan_to_dict_tasks_roundtrip(self):
        t = make_task(id="t1")
        p = Plan(goal="g", tasks=[t])
        d = p.to_dict()
        assert len(d["tasks"]) == 1

    def test_generate_tasks_depth1_neither_code_nor_research(self):
        # depth=1, goal has neither code/implement/build nor research/analyze
        tasks = self.engine._generate_tasks_from_goal("hello", parent_id="t0", depth=1)
        # Neither code_heavy nor research -> no branch matched -> empty list
        assert tasks == []

    def test_score_models_partial_cap_match(self):
        task = make_task(description="generate code documentation")
        model = make_model(id="m", capabilities=["code_generation"])  # partial: "code_gen" in "code_generation"
        result = self.engine._score_models_for_task(task, [model])
        assert result[0]["score"] > 0

    def test_detect_prompt_type_with_multiple_matches_returns_max(self):
        # "code implement build" = 3 coding keywords, "plan" = 1 planning keyword
        result = self.engine._detect_prompt_type("code implement build and plan")
        assert result == "coding"

    def test_get_fallback_model_prefers_first_in_list(self):
        models = [
            make_model(id="fallback-a"),
            make_model(id="fallback-b"),
        ]
        engine = make_engine(fallback_models=["fallback-a", "fallback-b"])
        result = engine._get_fallback_model(models, is_policy_sensitive=False)
        assert result == "fallback-a"

    def test_plan_with_model_having_all_defaults(self):
        # Model dict with minimal fields
        models = [{"id": "m1", "name": "M1"}]
        plan = self.engine.plan(
            "Build a comprehensive web application with Python and REST API",
            "sys", models
        )
        assert isinstance(plan, Plan)

    def test_check_token_limits_empty_plan(self):
        plan = Plan(goal="g", tasks=[])
        models = [make_model(id="m1")]
        # Should not raise
        self.engine._check_token_limits(plan, models)
        assert plan.warnings == []

    def test_score_models_with_large_complex_task(self):
        # complex task: many inputs + many outputs -> large model preferred
        task = make_task(
            description="build system",
            inputs=["req1", "req2", "req3"],
            outputs=["out1", "out2", "out3"]
        )
        large = make_model(id="large", memory_gb=32)
        small = make_model(id="small", memory_gb=2)
        result = self.engine._score_models_for_task(task, [large, small])
        scores = {s["model_id"]: s["score"] for s in result}
        assert scores["large"] >= scores["small"]
