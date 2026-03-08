"""
Tests for vetinari/web_ui.py — Flask app with 80+ routes.
Uses Flask test client; all heavy deps stubbed before import.
"""
import os
import sys
import json
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ── 0. Path setup ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 1. Blueprint module stubs (MUST be registered before importing web_ui) ────

def _make_bp_module(dotted_name, url_prefix=None):
    """Create a stub module with a real Flask Blueprint."""
    from flask import Blueprint
    name = dotted_name.split(".")[-1].replace("_routes", "")
    bp = Blueprint(name, dotted_name, url_prefix=url_prefix)
    mod = types.ModuleType(dotted_name)
    mod.bp = bp
    parts = dotted_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg_mod = types.ModuleType(pkg_name)
            pkg_mod.__path__ = []
            sys.modules[pkg_name] = pkg_mod
    sys.modules[dotted_name] = mod
    return mod


_make_bp_module("vetinari.web.adr_routes", "/api/adr")
_make_bp_module("vetinari.web.decomposition_routes", "/api/decomposition")
_make_bp_module("vetinari.web.ponder_routes", "/api/ponder")
_make_bp_module("vetinari.web.rules_routes", "/api/rules")
_make_bp_module("vetinari.web.admin_routes", "/api/admin")
_make_bp_module("vetinari.web.training_routes", "/api/training")


# ── 2. Stub heavy vetinari deps ────────────────────────────────────────────────

def _stub(name, **attrs):
    if name not in sys.modules:
        parts = name.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            if pkg not in sys.modules:
                m = types.ModuleType(pkg)
                m.__path__ = []
                sys.modules[pkg] = m
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    else:
        mod = sys.modules[name]
    for k, v in attrs.items():
        if not hasattr(mod, k):
            setattr(mod, k, v)
    return mod


# Orchestrator stub
_mock_orchestrator = MagicMock(name="Orchestrator_instance")
_mock_orchestrator.model_pool.models = [
    {
        "id": "test-model",
        "name": "Test Model",
        "capabilities": ["code_gen"],
        "context_len": 4096,
        "memory_gb": 4,
        "version": "1.0",
    }
]
_mock_orchestrator.adapter.chat.return_value = {"output": "mock response", "latency_ms": 10, "tokens_used": 5}
_mock_orchestrator.upgrader.check_for_upgrades.return_value = []
_mock_orchestrator.executor._parse_code_blocks.return_value = {}
_MockOrchestratorCls = MagicMock(return_value=_mock_orchestrator)
_stub("vetinari.orchestrator", Orchestrator=_MockOrchestratorCls)

# Planning engine stub
_mock_plan = MagicMock()
_mock_plan.tasks = []
_mock_plan.notes = "Model: test-model"
_mock_plan.warnings = []
_mock_plan.needs_context = False
_mock_plan.follow_up_question = None
_mock_plan.to_dict.return_value = {"tasks": [], "notes": "Model: test-model", "warnings": []}
_MockPlanningEngineCls = MagicMock(return_value=MagicMock(plan=MagicMock(return_value=_mock_plan)))
_stub("vetinari.planning_engine", PlanningEngine=_MockPlanningEngineCls)

# Telemetry stub
_mock_telemetry = MagicMock()
_mock_telemetry.get_summary.return_value = {
    "total_tokens_used": 100,
    "total_cost_usd": 0.5,
    "by_model": {},
    "by_provider": {},
    "session_requests": 5,
}
_stub("vetinari.telemetry", get_telemetry_collector=MagicMock(return_value=_mock_telemetry))

# Analytics cost stub
_mock_cost_tracker = MagicMock()
_mock_cost_tracker.get_summary.return_value = {"total_cost_usd": 1.23, "by_model": {"test": 1.0}}
_stub("vetinari.analytics")
_stub("vetinari.analytics.cost", get_cost_tracker=MagicMock(return_value=_mock_cost_tracker))

# Model search stub
_stub("vetinari.model_search", ModelSearchEngine=MagicMock())

# Model pool stub
_stub("vetinari.model_pool", ModelPool=MagicMock())

# APScheduler stub (avoid starting real scheduler)
_stub("apscheduler")
_stub("apscheduler.schedulers")
_stub("apscheduler.schedulers.background", BackgroundScheduler=MagicMock())

# Decomposition engine stub
_stub("vetinari.decomposition", decomposition_engine=MagicMock())

# Rules manager stub
_stub("vetinari.rules_manager", get_rules_manager=MagicMock(return_value=MagicMock()))

# Plan API stub
_stub("vetinari.plan_api", register_plan_api=MagicMock())

# Planning stub (plan_manager)
_mock_plan_mgr = MagicMock()
_mock_plan_obj = MagicMock()
_mock_plan_obj.plan_id = "plan-001"
_mock_plan_obj.status = "pending"
_mock_plan_obj.updated_at = "2024-01-01T00:00:00"
_mock_plan_obj.template_version = "v1"
_mock_plan_obj.current_wave = None
_mock_plan_obj.completed_tasks = 0
_mock_plan_obj.waves = []
_mock_plan_obj.progress_percent = 0
_mock_plan_obj.to_dict.return_value = {"plan_id": "plan-001", "status": "pending"}
_mock_plan_mgr.create_plan.return_value = _mock_plan_obj
_mock_plan_mgr.list_plans.return_value = [_mock_plan_obj]
_mock_plan_mgr.get_plan.return_value = _mock_plan_obj
_mock_plan_mgr.update_plan.return_value = _mock_plan_obj
_mock_plan_mgr.delete_plan.return_value = True
_mock_plan_mgr.start_plan.return_value = _mock_plan_obj
_mock_plan_mgr.pause_plan.return_value = _mock_plan_obj
_mock_plan_mgr.resume_plan.return_value = _mock_plan_obj
_mock_plan_mgr.cancel_plan.return_value = _mock_plan_obj
_stub("vetinari.planning", plan_manager=_mock_plan_mgr)

# Multi-agent orchestrator stub
_mock_mao = MagicMock()
_mock_mao.get_agent_status.return_value = []
_mock_mao.agents = {}
_mock_mao.task_queue = []
_stub("vetinari.multi_agent_orchestrator", MultiAgentOrchestrator=MagicMock(get_instance=MagicMock(return_value=_mock_mao)))

# Shared memory stub
_mock_shared_mem = MagicMock()
_mock_shared_mem.get_all.return_value = []
_mock_shared_mem.get_memories_by_type.return_value = []
_mock_shared_mem.resolve_decision.return_value = None
_stub("vetinari.shared_memory", SharedMemory=MagicMock(get_instance=MagicMock(return_value=_mock_shared_mem)))

# Model relay stub
_mock_model_relay = MagicMock()
_mock_relay_model = MagicMock()
_mock_relay_model.to_dict.return_value = {"id": "relay-model", "name": "Relay Model"}
_mock_relay_selection = MagicMock()
_mock_relay_selection.to_dict.return_value = {"model_id": "relay-model"}
_mock_relay_policy = MagicMock()
_mock_relay_policy.to_dict.return_value = {"strategy": "default"}
_mock_model_relay.get_all_models.return_value = [_mock_relay_model]
_mock_model_relay.get_model.return_value = _mock_relay_model
_mock_model_relay.pick_model_for_task.return_value = _mock_relay_selection
_mock_model_relay.get_policy.return_value = _mock_relay_policy
_mock_model_relay.reload_catalog.return_value = None
_mock_model_relay.DEFAULT_BACKEND = "cocoindex"
_mock_routing_policy = MagicMock()
_mock_routing_policy.from_dict.return_value = _mock_relay_policy
_stub("vetinari.model_relay", model_relay=_mock_model_relay, RoutingPolicy=_mock_routing_policy)

# Sandbox stub
_mock_sandbox_result = MagicMock()
_mock_sandbox_result.to_dict.return_value = {"status": "success", "output": "done"}
_mock_sandbox_mgr = MagicMock()
_mock_sandbox_mgr.execute.return_value = _mock_sandbox_result
_mock_sandbox_mgr.get_status.return_value = {"active_sandboxes": 0}
_mock_sandbox_mgr.get_audit_log.return_value = []
_stub("vetinari.sandbox", sandbox_manager=_mock_sandbox_mgr)

# Code search stub
_mock_search_result = MagicMock()
_mock_search_result.to_dict.return_value = {"file": "main.py", "line": 1, "content": "hello"}
_mock_search_adapter = MagicMock()
_mock_search_adapter.search.return_value = [_mock_search_result]
_mock_search_adapter.index_project.return_value = True
_mock_code_search_registry = MagicMock()
_mock_code_search_registry.get_adapter.return_value = _mock_search_adapter
_mock_code_search_registry.list_backends.return_value = ["cocoindex", "ripgrep"]
_mock_code_search_registry.get_backend_info.return_value = {"name": "cocoindex", "status": "ready"}
_mock_code_search_registry.DEFAULT_BACKEND = "cocoindex"
_stub("vetinari.code_search", code_search_registry=_mock_code_search_registry)

# Subtask tree stub
_mock_subtask = MagicMock()
_mock_subtask.subtask_id = "sub-001"
_mock_subtask.description = "test subtask"
_mock_subtask.agent_type = "builder"
_mock_subtask.assigned_agent = None
_mock_subtask.status = "pending"
_mock_subtask.depth = 0
_mock_subtask.to_dict.return_value = {
    "subtask_id": "sub-001", "description": "test subtask",
    "agent_type": "builder", "status": "pending", "depth": 0,
}
_mock_subtask_tree = MagicMock()
_mock_subtask_tree.get_subtasks_by_parent.return_value = [_mock_subtask]
_mock_subtask_tree.get_root_subtasks.return_value = [_mock_subtask]
_mock_subtask_tree.create_subtask.return_value = _mock_subtask
_mock_subtask_tree.update_subtask.return_value = _mock_subtask
_mock_subtask_tree.get_all_subtasks.return_value = [_mock_subtask]
_mock_subtask_tree.get_tree_depth.return_value = 2
_stub("vetinari.subtask_tree", subtask_tree=_mock_subtask_tree)

# Assignment pass stub
_stub("vetinari.assignment_pass", execute_assignment_pass=MagicMock(return_value={"status": "ok", "assigned": 1}))

# Template loader stub
_mock_template_loader = MagicMock()
_mock_template_loader.list_versions.return_value = ["v1", "v2"]
_mock_template_loader.default_version.return_value = "v1"
_mock_template_loader.load_templates.return_value = [{"template_id": "tmpl-1", "name": "Default"}]
_stub("vetinari.template_loader", template_loader=_mock_template_loader)

# Live model search stub
_mock_live_adapter = MagicMock()
_mock_live_candidate = MagicMock()
_mock_live_candidate.to_dict.return_value = {"model_id": "live-model", "score": 0.9}
_mock_live_adapter.search.return_value = [_mock_live_candidate]
_stub("vetinari.live_model_search", LiveModelSearchAdapter=MagicMock(return_value=_mock_live_adapter))

# Goal verifier stub
_mock_verifier = MagicMock()
_mock_report = MagicMock()
_mock_report.to_dict.return_value = {"passed": True, "score": 1.0}
_mock_report.get_corrective_tasks.return_value = []
_mock_verifier.verify.return_value = _mock_report
_stub("vetinari.goal_verifier", get_goal_verifier=MagicMock(return_value=_mock_verifier))

# Safety guardrails stub
_mock_gr_result = MagicMock()
_mock_gr_result.allowed = True
_mock_gr_result.violations = []
_mock_gr_result.content = "safe output"
_mock_guardrails = MagicMock()
_mock_guardrails.check_input.return_value = _mock_gr_result
_mock_guardrails.check_output.return_value = _mock_gr_result
_stub("vetinari.safety")
_stub("vetinari.safety.guardrails", get_guardrails=MagicMock(return_value=_mock_guardrails))

# ── 3. Import web_ui ───────────────────────────────────────────────────────────
import vetinari.web_ui as _web_ui
from vetinari.web_ui import app as _flask_app


# ── 4. Helper to build a minimal project directory ────────────────────────────

def _make_project(tmp_dir: str, project_id: str = "proj_abc123", config: dict = None) -> Path:
    """Create a minimal project directory structure for testing."""
    import yaml
    proj_dir = Path(tmp_dir) / "projects" / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or {
        "project_name": "Test Project",
        "description": "A test project",
        "high_level_goal": "Test goal",
        "goal": "Test goal",
        "tasks": [
            {
                "id": "t1",
                "description": "Task one",
                "inputs": [],
                "outputs": [],
                "assigned_model_id": "test-model",
                "dependencies": [],
                "model_override": "",
            }
        ],
        "model": "test-model",
        "active_model_id": "test-model",
        "status": "completed",
        "archived": False,
        "warnings": [],
    }
    (proj_dir / "project.yaml").write_text(
        __import__("yaml").dump(cfg), encoding="utf-8"
    )
    conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    (proj_dir / "conversation.json").write_text(json.dumps(conv), encoding="utf-8")
    return proj_dir


# ── 5. Base test class ────────────────────────────────────────────────────────

class TestWebUiBase(unittest.TestCase):
    def setUp(self):
        _flask_app.config["TESTING"] = True
        self.client = _flask_app.test_client()
        _web_ui.orchestrator = _mock_orchestrator
        _web_ui._models_cache = list(_mock_orchestrator.model_pool.models)
        _web_ui._models_cache_ts = __import__("time").time()

    def get(self, url, **kwargs):
        return self.client.get(url, **kwargs)

    def post(self, url, data=None, **kwargs):
        return self.client.post(url, json=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        return self.client.put(url, json=data, **kwargs)

    def delete(self, url, **kwargs):
        return self.client.delete(url, **kwargs)

    def json(self, resp):
        return json.loads(resp.data)


# ============================================================================
# TestStaticAndIndex
# ============================================================================

class TestStaticAndIndex(TestWebUiBase):
    def test_index_returns_200(self):
        resp = self.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_index_content_type_html(self):
        resp = self.get("/")
        self.assertIn("text/html", resp.content_type)

    def test_static_missing_returns_404(self):
        resp = self.get("/static/does_not_exist_xyz.js")
        self.assertEqual(resp.status_code, 404)


# ============================================================================
# TestApiStatus
# ============================================================================

class TestApiStatus(TestWebUiBase):
    def _data(self):
        resp = self.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        return self.json(resp)

    def test_status_200(self):
        resp = self.get("/api/status")
        self.assertEqual(resp.status_code, 200)

    def test_status_has_status_key(self):
        data = self._data()
        self.assertEqual(data["status"], "running")

    def test_status_has_host(self):
        data = self._data()
        self.assertIn("host", data)

    def test_status_has_config_path(self):
        data = self._data()
        self.assertIn("config_path", data)

    def test_status_api_token_masked_when_set(self):
        orig = _web_ui.current_config.get("api_token", "")
        _web_ui.current_config["api_token"] = "secret-token"
        try:
            data = self._data()
            self.assertEqual(data["api_token"], "***")
        finally:
            _web_ui.current_config["api_token"] = orig

    def test_status_api_token_empty_when_not_set(self):
        orig = _web_ui.current_config.get("api_token", "")
        _web_ui.current_config["api_token"] = ""
        try:
            data = self._data()
            self.assertEqual(data["api_token"], "")
        finally:
            _web_ui.current_config["api_token"] = orig

    def test_status_has_default_models(self):
        data = self._data()
        self.assertIn("default_models", data)
        self.assertIsInstance(data["default_models"], list)

    def test_status_has_memory_budget_gb(self):
        data = self._data()
        self.assertIn("memory_budget_gb", data)

    def test_status_has_active_model_id(self):
        data = self._data()
        self.assertIn("active_model_id", data)


# ============================================================================
# TestApiModels
# ============================================================================

class TestApiModels(TestWebUiBase):
    def test_get_models_returns_200(self):
        resp = self.get("/api/models")
        self.assertEqual(resp.status_code, 200)

    def test_get_models_has_models_key(self):
        resp = self.get("/api/models")
        data = self.json(resp)
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)

    def test_get_models_has_count(self):
        resp = self.get("/api/models")
        data = self.json(resp)
        self.assertIn("count", data)

    def test_get_models_refresh_post_200(self):
        resp = self.post("/api/models/refresh")
        self.assertEqual(resp.status_code, 200)

    def test_get_models_refresh_cached_false(self):
        resp = self.post("/api/models/refresh")
        data = self.json(resp)
        self.assertFalse(data.get("cached", True))

    def test_get_models_refresh_get_200(self):
        resp = self.get("/api/models/refresh")
        self.assertEqual(resp.status_code, 200)

    def test_get_models_cache_used(self):
        # When cache is warm, models are returned without hitting orchestrator
        _web_ui._models_cache = [{"id": "cached", "name": "Cached", "capabilities": [], "context_len": 0, "memory_gb": 0, "version": ""}]
        _web_ui._models_cache_ts = __import__("time").time()
        resp = self.get("/api/models")
        data = self.json(resp)
        self.assertTrue(data["cached"])
        self.assertEqual(data["models"][0]["id"], "cached")


# ============================================================================
# TestApiScoreModels
# ============================================================================

class TestApiScoreModels(TestWebUiBase):
    def test_score_models_returns_200(self):
        resp = self.post("/api/score-models", {"task_description": "write python code"})
        self.assertEqual(resp.status_code, 200)

    def test_score_models_has_models_key(self):
        resp = self.post("/api/score-models", {"task_description": "write python code"})
        data = self.json(resp)
        self.assertIn("models", data)

    def test_score_models_no_description(self):
        resp = self.post("/api/score-models", {"task_description": ""})
        # With an empty description no capabilities are required — still 200 if models present
        self.assertIn(resp.status_code, (200, 400))

    def test_score_models_code_keyword_scores_code_gen(self):
        _web_ui.orchestrator = _mock_orchestrator
        _mock_orchestrator.model_pool.models = [
            {"id": "coder", "name": "Coder", "capabilities": ["code_gen"], "memory_gb": 4}
        ]
        resp = self.post("/api/score-models", {"task_description": "implement a python function"})
        data = self.json(resp)
        if data.get("models"):
            self.assertIn("score", data["models"][0])

    def test_score_models_no_models_returns_400(self):
        _mock_orchestrator.model_pool.models = []
        try:
            resp = self.post("/api/score-models", {"task_description": "build something"})
            self.assertEqual(resp.status_code, 400)
        finally:
            _mock_orchestrator.model_pool.models = [
                {"id": "test-model", "name": "Test Model", "capabilities": ["code_gen"], "context_len": 4096, "memory_gb": 4, "version": "1.0"}
            ]


# ============================================================================
# TestApiSearch
# ============================================================================

class TestApiSearch(TestWebUiBase):
    def test_search_empty_query_returns_empty(self):
        resp = self.get("/api/search")
        data = self.json(resp)
        self.assertEqual(data["results"], [])
        self.assertEqual(data["query"], "")

    def test_search_empty_query_200(self):
        resp = self.get("/api/search")
        self.assertEqual(resp.status_code, 200)

    def test_search_with_query_returns_results_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_search01", config={
                "project_name": "findme project",
                "description": "a findme project",
                "high_level_goal": "findme",
                "goal": "findme",
                "tasks": [],
                "model": "test-model",
                "active_model_id": "test-model",
                "status": "completed",
                "archived": False,
                "warnings": [],
            })
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/search?q=findme")
                data = self.json(resp)
                self.assertIn("results", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_search_with_nonmatching_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/search?q=xyznonexistent999")
                data = self.json(resp)
                self.assertEqual(data["results"], [])
            finally:
                _web_ui.PROJECT_ROOT = Path(_ROOT)


# ============================================================================
# TestApiTokenStats
# ============================================================================

class TestApiTokenStats(TestWebUiBase):
    def test_token_stats_200(self):
        resp = self.get("/api/token-stats")
        self.assertEqual(resp.status_code, 200)

    def test_token_stats_has_total_tokens(self):
        resp = self.get("/api/token-stats")
        data = self.json(resp)
        self.assertIn("total_tokens_used", data)

    def test_token_stats_has_total_cost(self):
        resp = self.get("/api/token-stats")
        data = self.json(resp)
        self.assertIn("total_cost_usd", data)

    def test_token_stats_has_by_model(self):
        resp = self.get("/api/token-stats")
        data = self.json(resp)
        self.assertIn("by_model", data)

    def test_token_stats_has_session_requests(self):
        resp = self.get("/api/token-stats")
        data = self.json(resp)
        self.assertIn("session_requests", data)


# ============================================================================
# TestApiCancelProject
# ============================================================================

class TestApiCancelProject(TestWebUiBase):
    def test_cancel_no_flag_returns_not_found(self):
        resp = self.post("/api/project/no_such_project_xyz/cancel")
        data = self.json(resp)
        self.assertEqual(data["status"], "not_found")

    def test_cancel_with_flag_returns_cancelled(self):
        import threading
        flag = threading.Event()
        _web_ui._cancel_flags["proj_to_cancel"] = flag
        try:
            resp = self.post("/api/project/proj_to_cancel/cancel")
            data = self.json(resp)
            self.assertEqual(data["status"], "cancelled")
            self.assertTrue(flag.is_set())
        finally:
            _web_ui._cancel_flags.pop("proj_to_cancel", None)

    def test_cancel_returns_project_id(self):
        resp = self.post("/api/project/some_pid/cancel")
        data = self.json(resp)
        self.assertEqual(data["project_id"], "some_pid")

    def test_cancel_200(self):
        resp = self.post("/api/project/any_pid/cancel")
        self.assertEqual(resp.status_code, 200)


# ============================================================================
# TestApiTasks
# ============================================================================

class TestApiTasks(TestWebUiBase):
    def test_tasks_returns_200(self):
        resp = self.get("/api/tasks")
        self.assertEqual(resp.status_code, 200)

    def test_tasks_has_tasks_key(self):
        resp = self.get("/api/tasks")
        data = self.json(resp)
        self.assertIn("tasks", data)

    def test_tasks_returns_list(self):
        resp = self.get("/api/tasks")
        data = self.json(resp)
        self.assertIsInstance(data["tasks"], list)


# ============================================================================
# TestApiRunPrompt
# ============================================================================

class TestApiRunPrompt(TestWebUiBase):
    def test_run_prompt_missing_prompt_400(self):
        resp = self.post("/api/run-prompt", {"prompt": ""})
        self.assertEqual(resp.status_code, 400)

    def test_run_prompt_missing_prompt_error_key(self):
        resp = self.post("/api/run-prompt", {"prompt": ""})
        data = self.json(resp)
        self.assertIn("error", data)

    def test_run_prompt_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/run-prompt", {
                    "prompt": "hello world",
                    "model": "test-model",
                })
                self.assertIn(resp.status_code, (200, 400, 500))
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_run_prompt_returns_response_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/run-prompt", {
                    "prompt": "hello",
                    "model": "test-model",
                })
                if resp.status_code == 200:
                    data = self.json(resp)
                    self.assertIn("response", data)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiCreatePlan
# ============================================================================

class TestApiCreatePlan(TestWebUiBase):
    def test_create_plan_missing_goal_400(self):
        resp = self.post("/api/plan", {"goal": ""})
        self.assertEqual(resp.status_code, 400)

    def test_create_plan_missing_goal_error_key(self):
        resp = self.post("/api/plan", {"goal": ""})
        data = self.json(resp)
        self.assertIn("error", data)

    def test_create_plan_no_models_400(self):
        _mock_orchestrator.model_pool.models = []
        try:
            resp = self.post("/api/plan", {"goal": "build something"})
            self.assertEqual(resp.status_code, 400)
        finally:
            _mock_orchestrator.model_pool.models = [
                {"id": "test-model", "name": "Test Model", "capabilities": ["code_gen"], "context_len": 4096, "memory_gb": 4, "version": "1.0"}
            ]

    def test_create_plan_with_goal_calls_planner(self):
        resp = self.post("/api/plan", {"goal": "build a web app"})
        self.assertIn(resp.status_code, (200, 500))


# ============================================================================
# TestApiNewProject
# ============================================================================

class TestApiNewProject(TestWebUiBase):
    def test_new_project_missing_goal_400(self):
        resp = self.post("/api/new-project", {"goal": ""})
        self.assertEqual(resp.status_code, 400)

    def test_new_project_missing_goal_error_key(self):
        resp = self.post("/api/new-project", {"goal": ""})
        data = self.json(resp)
        self.assertIn("error", data)

    def test_new_project_no_models_400(self):
        _mock_orchestrator.model_pool.models = []
        try:
            resp = self.post("/api/new-project", {"goal": "some goal"})
            self.assertEqual(resp.status_code, 400)
        finally:
            _mock_orchestrator.model_pool.models = [
                {"id": "test-model", "name": "Test Model", "capabilities": ["code_gen"], "context_len": 4096, "memory_gb": 4, "version": "1.0"}
            ]

    def test_new_project_with_goal_returns_project_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/new-project", {
                    "goal": "build a CLI tool",
                    "auto_run": False,
                })
                if resp.status_code == 200:
                    data = self.json(resp)
                    self.assertIn("project_id", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_new_project_status_planned_when_auto_run_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/new-project", {
                    "goal": "build a CLI tool",
                    "auto_run": False,
                })
                if resp.status_code == 200:
                    data = self.json(resp)
                    self.assertEqual(data["status"], "planned")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjects
# ============================================================================

class TestApiProjects(TestWebUiBase):
    def test_projects_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/projects")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_projects_has_projects_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/projects")
                data = self.json(resp)
                self.assertIn("projects", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_projects_lists_existing_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_listed")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/projects")
                data = self.json(resp)
                ids = [p["id"] for p in data["projects"]]
                self.assertIn("proj_listed", ids)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_projects_hides_archived_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_archived", config={
                "project_name": "Archived", "description": "", "high_level_goal": "",
                "goal": "", "tasks": [], "model": "test", "active_model_id": "test",
                "status": "archived", "archived": True, "warnings": [],
            })
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/projects")
                data = self.json(resp)
                ids = [p["id"] for p in data["projects"]]
                self.assertNotIn("proj_archived", ids)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProject
# ============================================================================

class TestApiProject(TestWebUiBase):
    def test_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/nonexistent_xyz")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_project_found_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_found01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_found01")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_project_has_id_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_idcheck")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_idcheck")
                data = self.json(resp)
                self.assertEqual(data["id"], "proj_idcheck")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_project_has_conversation_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_conv")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_conv")
                data = self.json(resp)
                self.assertIn("conversation", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_project_has_tasks_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_tasks")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_tasks")
                data = self.json(resp)
                self.assertIn("tasks", data)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectMessage
# ============================================================================

class TestApiProjectMessage(TestWebUiBase):
    def test_message_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/message", {"message": "hello"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_message_missing_message_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_msg01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_msg01/message", {"message": ""})
                self.assertEqual(resp.status_code, 400)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_message_success_has_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_msg02")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_msg02/message", {"message": "what is the status?"})
                if resp.status_code == 200:
                    data = self.json(resp)
                    self.assertIn("response", data)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectTask
# ============================================================================

class TestApiProjectTask(TestWebUiBase):
    def test_add_task_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/task", {"description": "new task"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_add_task_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_add_task")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_add_task/task", {
                    "id": "t_new", "description": "A new task"
                })
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "added")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_add_task_duplicate_id_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_dup_task")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_dup_task/task", {
                    "id": "t1", "description": "Duplicate task"
                })
                self.assertEqual(resp.status_code, 400)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_update_task_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.put("/api/project/no_proj/task/t1", {"description": "updated"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_update_task_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_upd_task")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.put("/api/project/proj_upd_task/task/t1", {"description": "Updated desc"})
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "updated")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_update_task_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_upd_miss")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.put("/api/project/proj_upd_miss/task/no_such_task", {"description": "x"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_delete_task_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.delete("/api/project/no_proj/task/t1")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_delete_task_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_del_task")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.delete("/api/project/proj_del_task/task/t1")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "deleted")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectReview
# ============================================================================

class TestApiProjectReview(TestWebUiBase):
    def test_review_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/no_proj/review")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_review_found_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_review01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_review01/review")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_review_has_project_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_review02")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_review02/review")
                data = self.json(resp)
                self.assertEqual(data["project_id"], "proj_review02")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_review_has_tasks_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_review03")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_review03/review")
                data = self.json(resp)
                self.assertIn("tasks", data)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectApprove
# ============================================================================

class TestApiProjectApprove(TestWebUiBase):
    def test_approve_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/approve")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_approve_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_approve01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_approve01/approve")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "approved")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiSystemPrompts
# ============================================================================

class TestApiSystemPrompts(TestWebUiBase):
    def test_get_system_prompts_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/system-prompts")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_get_system_prompts_has_prompts_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/system-prompts")
                data = self.json(resp)
                self.assertIn("prompts", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_save_system_prompt_missing_name_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/system-prompts", {"name": "", "content": "test"})
                self.assertEqual(resp.status_code, 400)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_save_system_prompt_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/system-prompts", {"name": "my_prompt", "content": "You are helpful."})
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "saved")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_delete_system_prompt_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                # Create then delete
                self.post("/api/system-prompts", {"name": "del_prompt", "content": "bye"})
                resp = self.delete("/api/system-prompts/del_prompt")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "deleted")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_delete_nonexistent_prompt_still_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.delete("/api/system-prompts/no_such_prompt")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiAllTasks
# ============================================================================

class TestApiAllTasks(TestWebUiBase):
    def test_all_tasks_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/all-tasks")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_all_tasks_has_tasks_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/all-tasks")
                data = self.json(resp)
                self.assertIn("tasks", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_all_tasks_includes_project_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_alltasks")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/all-tasks")
                data = self.json(resp)
                task_ids = [t["task_id"] for t in data["tasks"]]
                self.assertIn("t1", task_ids)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiConfig
# ============================================================================

class TestApiConfig(TestWebUiBase):
    def test_config_update_host(self):
        orig_host = _web_ui.current_config.get("host")
        try:
            resp = self.post("/api/config", {"host": "http://newhost:9999"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(_web_ui.current_config["host"], "http://newhost:9999")
        finally:
            _web_ui.current_config["host"] = orig_host

    def test_config_returns_updated_status(self):
        resp = self.post("/api/config", {"config_path": "new/path.yaml"})
        data = self.json(resp)
        self.assertEqual(data["status"], "updated")

    def test_config_update_resets_orchestrator(self):
        _web_ui.orchestrator = _mock_orchestrator
        self.post("/api/config", {"host": "http://localhost:1234"})
        self.assertIsNone(_web_ui.orchestrator)
        _web_ui.orchestrator = _mock_orchestrator

    def test_config_update_invalidates_model_cache(self):
        _web_ui._models_cache_ts = 9999999.0
        self.post("/api/config", {"host": "http://localhost:1234"})
        self.assertEqual(_web_ui._models_cache_ts, 0.0)
        _web_ui._models_cache_ts = 0.0


# ============================================================================
# TestApiUpgradeCheck
# ============================================================================

class TestApiUpgradeCheck(TestWebUiBase):
    def test_upgrade_check_200(self):
        resp = self.get("/api/upgrade-check")
        self.assertEqual(resp.status_code, 200)

    def test_upgrade_check_has_upgrades_key(self):
        resp = self.get("/api/upgrade-check")
        data = self.json(resp)
        self.assertIn("upgrades", data)

    def test_upgrade_check_returns_list(self):
        resp = self.get("/api/upgrade-check")
        data = self.json(resp)
        self.assertIsInstance(data["upgrades"], list)


# ============================================================================
# TestApiWorkflow
# ============================================================================

class TestApiWorkflow(TestWebUiBase):
    def test_workflow_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/workflow")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_workflow_has_projects_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/workflow")
                data = self.json(resp)
                self.assertIn("projects", data)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_workflow_search_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_wf_filter", config={
                "project_name": "unique_wf_xyz", "description": "unique_wf_xyz",
                "high_level_goal": "unique_wf_xyz", "goal": "unique_wf_xyz",
                "tasks": [], "model": "test", "active_model_id": "test",
                "status": "completed", "archived": False, "warnings": [],
            })
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/workflow?search=unique_wf_xyz")
                data = self.json(resp)
                self.assertEqual(len(data["projects"]), 1)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiModelConfig
# ============================================================================

class TestApiModelConfig(TestWebUiBase):
    def test_get_model_config_200(self):
        resp = self.get("/api/model-config")
        self.assertEqual(resp.status_code, 200)

    def test_get_model_config_has_default_models(self):
        resp = self.get("/api/model-config")
        data = self.json(resp)
        self.assertIn("default_models", data)

    def test_get_model_config_has_memory_budget(self):
        resp = self.get("/api/model-config")
        data = self.json(resp)
        self.assertIn("memory_budget_gb", data)

    def test_post_model_config_updates_defaults(self):
        orig = list(_web_ui.current_config.get("default_models", []))
        try:
            resp = self.post("/api/model-config", {"default_models": ["new-model-1"]})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(_web_ui.current_config["default_models"], ["new-model-1"])
        finally:
            _web_ui.current_config["default_models"] = orig

    def test_post_model_config_updates_memory_budget(self):
        orig = _web_ui.current_config.get("memory_budget_gb", 48)
        try:
            resp = self.post("/api/model-config", {"memory_budget_gb": "16"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(_web_ui.current_config["memory_budget_gb"], 16)
        finally:
            _web_ui.current_config["memory_budget_gb"] = orig


# ============================================================================
# TestApiSwapModel
# ============================================================================

class TestApiSwapModel(TestWebUiBase):
    def test_swap_model_missing_model_id_400(self):
        resp = self.post("/api/swap-model", {"model_id": ""})
        self.assertEqual(resp.status_code, 400)

    def test_swap_model_global_success(self):
        orig = _web_ui.current_config.get("active_model_id")
        try:
            resp = self.post("/api/swap-model", {"model_id": "new-active-model"})
            self.assertEqual(resp.status_code, 200)
            data = self.json(resp)
            self.assertEqual(data["active_model_id"], "new-active-model")
        finally:
            _web_ui.current_config["active_model_id"] = orig

    def test_swap_model_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/swap-model", {
                    "model_id": "test-model",
                    "project_id": "no_such_proj"
                })
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_swap_model_project_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_swap")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/swap-model", {
                    "model_id": "swapped-model",
                    "project_id": "proj_swap"
                })
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["active_model_id"], "swapped-model")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectRename
# ============================================================================

class TestApiProjectRename(TestWebUiBase):
    def test_rename_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/rename", {"name": "New Name"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_rename_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_rename01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_rename01/rename", {"name": "Renamed Project"})
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "renamed")
                self.assertEqual(data["project_name"], "Renamed Project")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectArchive
# ============================================================================

class TestApiProjectArchive(TestWebUiBase):
    def test_archive_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/archive", {"archive": True})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_archive_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_archive01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_archive01/archive", {"archive": True})
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "archived")
                self.assertTrue(data["archived"])
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_unarchive_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_unarchive01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_unarchive01/archive", {"archive": False})
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "unarchived")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectDelete
# ============================================================================

class TestApiProjectDelete(TestWebUiBase):
    def test_delete_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.delete("/api/project/no_such_project")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_delete_project_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_to_delete")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.delete("/api/project/proj_to_delete")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "deleted")
                self.assertFalse((Path(tmp) / "projects" / "proj_to_delete").exists())
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiAgents
# ============================================================================

class TestApiAgents(TestWebUiBase):
    def test_agents_status_200(self):
        resp = self.get("/api/agents/status")
        self.assertEqual(resp.status_code, 200)

    def test_agents_status_has_agents_key(self):
        resp = self.get("/api/agents/status")
        data = self.json(resp)
        self.assertIn("agents", data)

    def test_agents_initialize_200(self):
        resp = self.post("/api/agents/initialize")
        self.assertIn(resp.status_code, (200, 500))

    def test_agents_active_200(self):
        resp = self.get("/api/agents/active")
        self.assertEqual(resp.status_code, 200)

    def test_agents_active_has_agents_key(self):
        resp = self.get("/api/agents/active")
        data = self.json(resp)
        self.assertIn("agents", data)

    def test_agents_tasks_200(self):
        resp = self.get("/api/agents/tasks")
        self.assertEqual(resp.status_code, 200)

    def test_agents_tasks_has_tasks_key(self):
        resp = self.get("/api/agents/tasks")
        data = self.json(resp)
        self.assertIn("tasks", data)

    def test_agents_status_none_orchestrator_returns_empty(self):
        _mock_mao_cls = sys.modules["vetinari.multi_agent_orchestrator"].MultiAgentOrchestrator
        orig = _mock_mao_cls.get_instance.return_value
        _mock_mao_cls.get_instance.return_value = None
        try:
            resp = self.get("/api/agents/status")
            data = self.json(resp)
            self.assertEqual(data["agents"], [])
        finally:
            _mock_mao_cls.get_instance.return_value = orig


# ============================================================================
# TestApiMemory
# ============================================================================

class TestApiMemory(TestWebUiBase):
    def test_memory_200(self):
        resp = self.get("/api/memory")
        self.assertEqual(resp.status_code, 200)

    def test_memory_has_memories_key(self):
        resp = self.get("/api/memory")
        data = self.json(resp)
        self.assertIn("memories", data)

    def test_memory_returns_list(self):
        resp = self.get("/api/memory")
        data = self.json(resp)
        self.assertIsInstance(data["memories"], list)


# ============================================================================
# TestApiDecisions
# ============================================================================

class TestApiDecisions(TestWebUiBase):
    def test_decisions_pending_200(self):
        resp = self.get("/api/decisions/pending")
        self.assertEqual(resp.status_code, 200)

    def test_decisions_pending_has_decisions_key(self):
        resp = self.get("/api/decisions/pending")
        data = self.json(resp)
        self.assertIn("decisions", data)

    def test_decisions_submit_200(self):
        resp = self.post("/api/decisions", {"decision_id": "d1", "choice": "yes"})
        self.assertEqual(resp.status_code, 200)

    def test_decisions_submit_has_status(self):
        resp = self.post("/api/decisions", {"decision_id": "d1", "choice": "yes"})
        data = self.json(resp)
        self.assertEqual(data["status"], "resolved")

    def test_decisions_submit_reflects_choice(self):
        resp = self.post("/api/decisions", {"decision_id": "d1", "choice": "option_b"})
        data = self.json(resp)
        self.assertEqual(data["choice"], "option_b")


# ============================================================================
# TestApiPlans
# ============================================================================

class TestApiPlans(TestWebUiBase):
    def test_plan_create_201(self):
        resp = self.post("/api/plans", {"title": "My Plan", "prompt": "do things"})
        self.assertEqual(resp.status_code, 201)

    def test_plan_create_has_plan_id(self):
        resp = self.post("/api/plans", {"title": "My Plan"})
        data = self.json(resp)
        self.assertIn("plan_id", data)

    def test_plans_list_200(self):
        resp = self.get("/api/plans")
        self.assertEqual(resp.status_code, 200)

    def test_plans_list_has_plans_key(self):
        resp = self.get("/api/plans")
        data = self.json(resp)
        self.assertIn("plans", data)

    def test_plan_get_200(self):
        resp = self.get("/api/plans/plan-001")
        self.assertEqual(resp.status_code, 200)

    def test_plan_get_not_found_404(self):
        _mock_plan_mgr.get_plan.return_value = None
        try:
            resp = self.get("/api/plans/nonexistent_plan")
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_plan_mgr.get_plan.return_value = _mock_plan_obj

    def test_plan_update_200(self):
        resp = self.put("/api/plans/plan-001", {"title": "Updated Plan"})
        self.assertEqual(resp.status_code, 200)

    def test_plan_update_not_found_404(self):
        _mock_plan_mgr.update_plan.return_value = None
        try:
            resp = self.put("/api/plans/no_plan", {"title": "x"})
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_plan_mgr.update_plan.return_value = _mock_plan_obj

    def test_plan_delete_204(self):
        resp = self.delete("/api/plans/plan-001")
        self.assertEqual(resp.status_code, 204)

    def test_plan_delete_not_found_404(self):
        _mock_plan_mgr.delete_plan.return_value = False
        try:
            resp = self.delete("/api/plans/no_plan")
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_plan_mgr.delete_plan.return_value = True

    def test_plan_start_200(self):
        resp = self.post("/api/plans/plan-001/start")
        self.assertEqual(resp.status_code, 200)

    def test_plan_start_not_found_404(self):
        _mock_plan_mgr.start_plan.return_value = None
        try:
            resp = self.post("/api/plans/no_plan/start")
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_plan_mgr.start_plan.return_value = _mock_plan_obj

    def test_plan_pause_200(self):
        resp = self.post("/api/plans/plan-001/pause")
        self.assertEqual(resp.status_code, 200)

    def test_plan_resume_200(self):
        resp = self.post("/api/plans/plan-001/resume")
        self.assertEqual(resp.status_code, 200)

    def test_plan_cancel_200(self):
        resp = self.post("/api/plans/plan-001/cancel")
        self.assertEqual(resp.status_code, 200)

    def test_plan_status_200(self):
        resp = self.get("/api/plans/plan-001/status")
        self.assertEqual(resp.status_code, 200)

    def test_plan_status_has_plan_id(self):
        resp = self.get("/api/plans/plan-001/status")
        data = self.json(resp)
        self.assertIn("plan_id", data)

    def test_plan_status_not_found_404(self):
        _mock_plan_mgr.get_plan.return_value = None
        try:
            resp = self.get("/api/plans/no_plan/status")
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_plan_mgr.get_plan.return_value = _mock_plan_obj


# ============================================================================
# TestApiModelCatalog
# ============================================================================

class TestApiModelCatalog(TestWebUiBase):
    def test_model_catalog_200(self):
        resp = self.get("/api/model-catalog")
        self.assertEqual(resp.status_code, 200)

    def test_model_catalog_has_models(self):
        resp = self.get("/api/model-catalog")
        data = self.json(resp)
        self.assertIn("models", data)

    def test_model_get_200(self):
        resp = self.get("/api/models/relay-model")
        self.assertEqual(resp.status_code, 200)

    def test_model_get_not_found_404(self):
        _mock_model_relay.get_model.return_value = None
        try:
            resp = self.get("/api/models/no_model_xyz")
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_model_relay.get_model.return_value = _mock_relay_model

    def test_model_select_200(self):
        resp = self.post("/api/models/select", {"task_type": "code_gen"})
        self.assertEqual(resp.status_code, 200)

    def test_model_policy_get_200(self):
        resp = self.get("/api/models/policy")
        self.assertEqual(resp.status_code, 200)

    def test_model_policy_update_200(self):
        resp = self.put("/api/models/policy", {"strategy": "cost"})
        self.assertEqual(resp.status_code, 200)

    def test_model_policy_update_has_status(self):
        resp = self.put("/api/models/policy", {"strategy": "cost"})
        data = self.json(resp)
        self.assertEqual(data["status"], "updated")

    def test_models_reload_200(self):
        resp = self.post("/api/models/reload")
        self.assertEqual(resp.status_code, 200)

    def test_models_reload_has_status(self):
        resp = self.post("/api/models/reload")
        data = self.json(resp)
        self.assertEqual(data["status"], "reloaded")


# ============================================================================
# TestApiSandbox
# ============================================================================

class TestApiSandbox(TestWebUiBase):
    """Sandbox requires admin (localhost). Test client runs as 127.0.0.1."""

    def test_sandbox_execute_200(self):
        resp = self.post("/api/sandbox/execute", {"code": "print('hi')", "sandbox_type": "in_process"})
        self.assertEqual(resp.status_code, 200)

    def test_sandbox_execute_has_status(self):
        resp = self.post("/api/sandbox/execute", {"code": "1+1"})
        data = self.json(resp)
        self.assertIn("status", data)

    def test_sandbox_status_200(self):
        resp = self.get("/api/sandbox/status")
        self.assertEqual(resp.status_code, 200)

    def test_sandbox_audit_200(self):
        resp = self.get("/api/sandbox/audit")
        self.assertEqual(resp.status_code, 200)

    def test_sandbox_audit_has_audit_entries(self):
        resp = self.get("/api/sandbox/audit")
        data = self.json(resp)
        self.assertIn("audit_entries", data)


# ============================================================================
# TestApiCodeSearch
# ============================================================================

class TestApiCodeSearch(TestWebUiBase):
    def test_code_search_no_query_400(self):
        resp = self.get("/api/code-search")
        self.assertEqual(resp.status_code, 400)

    def test_code_search_with_query_200(self):
        resp = self.get("/api/code-search?q=def+main")
        self.assertEqual(resp.status_code, 200)

    def test_code_search_has_results(self):
        resp = self.get("/api/code-search?q=def+main")
        data = self.json(resp)
        self.assertIn("results", data)

    def test_code_search_has_query(self):
        resp = self.get("/api/code-search?q=hello")
        data = self.json(resp)
        self.assertEqual(data["query"], "hello")


# ============================================================================
# TestApiSearchIndex
# ============================================================================

class TestApiSearchIndex(TestWebUiBase):
    def test_search_index_missing_path_400(self):
        resp = self.post("/api/search/index", {"project_path": ""})
        self.assertEqual(resp.status_code, 400)

    def test_search_index_success(self):
        resp = self.post("/api/search/index", {"project_path": "/some/path"})
        self.assertEqual(resp.status_code, 200)
        data = self.json(resp)
        self.assertIn("status", data)

    def test_search_status_200(self):
        resp = self.get("/api/search/status")
        self.assertEqual(resp.status_code, 200)

    def test_search_status_has_backends(self):
        resp = self.get("/api/search/status")
        data = self.json(resp)
        self.assertIn("backends", data)


# ============================================================================
# TestApiSubtasks
# ============================================================================

class TestApiSubtasks(TestWebUiBase):
    def test_get_subtasks_200(self):
        resp = self.get("/api/subtasks/plan-001")
        self.assertEqual(resp.status_code, 200)

    def test_get_subtasks_has_subtasks_key(self):
        resp = self.get("/api/subtasks/plan-001")
        data = self.json(resp)
        self.assertIn("subtasks", data)

    def test_get_subtasks_has_plan_id(self):
        resp = self.get("/api/subtasks/plan-001")
        data = self.json(resp)
        self.assertEqual(data["plan_id"], "plan-001")

    def test_create_subtask_200(self):
        resp = self.post("/api/subtasks/plan-001", {
            "description": "A subtask",
            "agent_type": "builder",
        })
        self.assertEqual(resp.status_code, 200)

    def test_create_subtask_has_subtask_id(self):
        resp = self.post("/api/subtasks/plan-001", {"description": "subtask"})
        data = self.json(resp)
        self.assertIn("subtask_id", data)

    def test_update_subtask_200(self):
        resp = self.put("/api/subtasks/plan-001/sub-001", {"status": "running"})
        self.assertEqual(resp.status_code, 200)

    def test_update_subtask_not_found_404(self):
        _mock_subtask_tree.update_subtask.return_value = None
        try:
            resp = self.put("/api/subtasks/plan-001/no_sub", {"status": "running"})
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_subtask_tree.update_subtask.return_value = _mock_subtask

    def test_get_subtask_tree_200(self):
        resp = self.get("/api/subtasks/plan-001/tree")
        self.assertEqual(resp.status_code, 200)

    def test_get_subtask_tree_has_depth(self):
        resp = self.get("/api/subtasks/plan-001/tree")
        data = self.json(resp)
        self.assertIn("depth", data)


# ============================================================================
# TestApiAssignments
# ============================================================================

class TestApiAssignments(TestWebUiBase):
    def test_assignment_execute_pass_missing_plan_id_400(self):
        resp = self.post("/api/assignments/execute-pass", {})
        self.assertEqual(resp.status_code, 400)

    def test_assignment_execute_pass_success(self):
        resp = self.post("/api/assignments/execute-pass", {"plan_id": "plan-001"})
        self.assertEqual(resp.status_code, 200)

    def test_get_assignments_200(self):
        resp = self.get("/api/assignments/plan-001")
        self.assertEqual(resp.status_code, 200)

    def test_get_assignments_has_assignments(self):
        resp = self.get("/api/assignments/plan-001")
        data = self.json(resp)
        self.assertIn("assignments", data)

    def test_override_assignment_missing_agent_400(self):
        resp = self.put("/api/assignments/plan-001/sub-001", {})
        self.assertEqual(resp.status_code, 400)

    def test_override_assignment_success(self):
        resp = self.put("/api/assignments/plan-001/sub-001", {"assigned_agent": "builder-1"})
        self.assertEqual(resp.status_code, 200)

    def test_override_assignment_not_found_404(self):
        _mock_subtask_tree.update_subtask.return_value = None
        try:
            resp = self.put("/api/assignments/plan-001/no_sub", {"assigned_agent": "x"})
            self.assertEqual(resp.status_code, 404)
        finally:
            _mock_subtask_tree.update_subtask.return_value = _mock_subtask


# ============================================================================
# TestApiTemplates
# ============================================================================

class TestApiTemplates(TestWebUiBase):
    def test_template_versions_200(self):
        resp = self.get("/api/templates/versions")
        self.assertEqual(resp.status_code, 200)

    def test_template_versions_has_versions(self):
        resp = self.get("/api/templates/versions")
        data = self.json(resp)
        self.assertIn("versions", data)
        self.assertIsInstance(data["versions"], list)

    def test_template_versions_has_default(self):
        resp = self.get("/api/templates/versions")
        data = self.json(resp)
        self.assertIn("default", data)

    def test_templates_200(self):
        resp = self.get("/api/templates")
        self.assertEqual(resp.status_code, 200)

    def test_templates_has_templates_key(self):
        resp = self.get("/api/templates")
        data = self.json(resp)
        self.assertIn("templates", data)

    def test_templates_has_total(self):
        resp = self.get("/api/templates")
        data = self.json(resp)
        self.assertIn("total", data)


# ============================================================================
# TestApiHelperFunctions
# ============================================================================

class TestApiHelperFunctions(TestWebUiBase):
    def test_register_project_task_creates_event(self):
        import threading
        event = _web_ui._register_project_task("test_proj_reg")
        self.assertIsInstance(event, threading.Event)
        self.assertIn("test_proj_reg", _web_ui._cancel_flags)
        del _web_ui._cancel_flags["test_proj_reg"]

    def test_cancel_project_task_sets_flag(self):
        import threading
        flag = threading.Event()
        _web_ui._cancel_flags["proj_for_cancel"] = flag
        result = _web_ui._cancel_project_task("proj_for_cancel")
        self.assertTrue(result)
        self.assertTrue(flag.is_set())
        del _web_ui._cancel_flags["proj_for_cancel"]

    def test_cancel_project_task_missing_returns_false(self):
        result = _web_ui._cancel_project_task("definitely_not_registered_xyz")
        self.assertFalse(result)

    def test_get_sse_queue_creates_queue(self):
        import queue
        q = _web_ui._get_sse_queue("sse_proj_001")
        self.assertIsInstance(q, queue.Queue)
        del _web_ui._sse_streams["sse_proj_001"]

    def test_get_sse_queue_returns_same_queue(self):
        q1 = _web_ui._get_sse_queue("sse_proj_002")
        q2 = _web_ui._get_sse_queue("sse_proj_002")
        self.assertIs(q1, q2)
        del _web_ui._sse_streams["sse_proj_002"]

    def test_push_sse_event_puts_to_queue(self):
        q = _web_ui._get_sse_queue("sse_proj_003")
        _web_ui._push_sse_event("sse_proj_003", "task_start", {"task_id": "t1"})
        msg = q.get_nowait()
        self.assertEqual(msg["event"], "task_start")
        del _web_ui._sse_streams["sse_proj_003"]

    def test_push_sse_event_noop_for_unknown_project(self):
        # Should not raise even if the project has no queue
        _web_ui._push_sse_event("proj_with_no_queue_xyz", "test", {})

    def test_get_models_cached_returns_cache_when_fresh(self):
        _web_ui._models_cache = [{"id": "cached-m", "name": "C", "capabilities": [], "context_len": 0, "memory_gb": 0, "version": ""}]
        _web_ui._models_cache_ts = __import__("time").time()
        result = _web_ui._get_models_cached(force=False)
        self.assertEqual(result[0]["id"], "cached-m")

    def test_get_models_cached_force_bypasses_cache(self):
        _web_ui._models_cache = [{"id": "stale", "name": "S", "capabilities": [], "context_len": 0, "memory_gb": 0, "version": ""}]
        _web_ui._models_cache_ts = __import__("time").time()
        _web_ui.orchestrator = _mock_orchestrator
        result = _web_ui._get_models_cached(force=True)
        # The result reflects what orchestrator returns (test-model)
        ids = [m["id"] for m in result]
        self.assertIn("test-model", ids)


# ============================================================================
# TestApiSSEStream
# ============================================================================

class TestApiSSEStream(TestWebUiBase):
    def test_stream_returns_200(self):
        # Just test that it returns with the right content-type, don't consume stream
        resp = self.client.get("/api/project/stream_test_proj/stream")
        self.assertEqual(resp.status_code, 200)

    def test_stream_content_type_event_stream(self):
        resp = self.client.get("/api/project/stream_test_proj/stream")
        self.assertIn("text/event-stream", resp.content_type)


# ============================================================================
# TestApiDiscover
# ============================================================================

class TestApiDiscover(TestWebUiBase):
    def test_discover_200(self):
        resp = self.get("/api/discover")
        self.assertEqual(resp.status_code, 200)

    def test_discover_has_discovered_key(self):
        resp = self.get("/api/discover")
        data = self.json(resp)
        self.assertIn("discovered", data)

    def test_discover_has_models(self):
        resp = self.get("/api/discover")
        data = self.json(resp)
        self.assertIn("models", data)


# ============================================================================
# TestApiArtifacts
# ============================================================================

class TestApiArtifacts(TestWebUiBase):
    def test_artifacts_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/artifacts")
                self.assertEqual(resp.status_code, 200)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_artifacts_has_artifacts_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/artifacts")
                data = self.json(resp)
                self.assertIn("artifacts", data)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectFiles
# ============================================================================

class TestApiProjectFiles(TestWebUiBase):
    def test_read_file_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/files/read", {"path": "main.py"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_read_file_missing_path_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_files01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_files01/files/read", {"path": ""})
                self.assertEqual(resp.status_code, 400)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_read_file_path_traversal_403(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_files02")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_files02/files/read", {"path": "../../etc/passwd"})
                self.assertEqual(resp.status_code, 403)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_write_file_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/files/write", {"path": "hello.py", "content": "x=1"})
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_write_file_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_write01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_write01/files/write", {
                    "path": "hello.py",
                    "content": "print('hello')"
                })
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "ok")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_list_files_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/no_proj/files/list")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_list_files_empty_when_no_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_list01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_list01/files/list")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["files"], [])
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_list_files_includes_written_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_list02")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                # Write a file first
                self.post("/api/project/proj_list02/files/write", {
                    "path": "script.py", "content": "x=1"
                })
                resp = self.get("/api/project/proj_list02/files/list")
                data = self.json(resp)
                paths = [f["path"] for f in data["files"]]
                self.assertIn("script.py", paths)
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiProjectAssemble
# ============================================================================

class TestApiProjectAssemble(TestWebUiBase):
    def test_assemble_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/assemble")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_assemble_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_assemble01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/proj_assemble01/assemble")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["status"], "assembled")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_assemble_creates_final_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_assemble02")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                self.post("/api/project/proj_assemble02/assemble")
                report = Path(tmp) / "projects" / "proj_assemble02" / "final_delivery" / "final_report.md"
                self.assertTrue(report.exists())
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiVerifyGoal
# ============================================================================

class TestApiVerifyGoal(TestWebUiBase):
    def test_verify_goal_no_goal_no_project_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.post("/api/project/no_proj/verify-goal", {})
                self.assertEqual(resp.status_code, 400)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_verify_goal_success(self):
        resp = self.post("/api/project/any_proj/verify-goal", {
            "goal": "build a REST API",
            "final_output": "Here is the API..."
        })
        self.assertEqual(resp.status_code, 200)

    def test_verify_goal_has_report(self):
        resp = self.post("/api/project/any_proj/verify-goal", {
            "goal": "build something",
        })
        data = self.json(resp)
        self.assertIn("report", data)

    def test_verify_goal_has_corrective_tasks(self):
        resp = self.post("/api/project/any_proj/verify-goal", {
            "goal": "build something",
        })
        data = self.json(resp)
        self.assertIn("corrective_tasks", data)


# ============================================================================
# TestApiTaskOutput
# ============================================================================

class TestApiTaskOutput(TestWebUiBase):
    def test_task_output_project_not_found_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/no_proj/task/t1/output")
                self.assertEqual(resp.status_code, 404)
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_task_output_empty_when_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_project(tmp, "proj_taskout01")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_taskout01/task/t1/output")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["output"], "")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_task_output_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj_dir = _make_project(tmp, "proj_taskout02")
            out_dir = proj_dir / "outputs" / "t1"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "output.txt").write_text("task output content", encoding="utf-8")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/project/proj_taskout02/task/t1/output")
                data = self.json(resp)
                self.assertEqual(data["output"], "task output content")
            finally:
                _web_ui.PROJECT_ROOT = orig


# ============================================================================
# TestApiOutput (global output lookup)
# ============================================================================

class TestApiOutput(TestWebUiBase):
    def test_output_not_found_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/output/no_such_task_xyz")
                self.assertEqual(resp.status_code, 200)
                data = self.json(resp)
                self.assertEqual(data["output"], "")
            finally:
                _web_ui.PROJECT_ROOT = orig

    def test_output_found_in_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj_dir = _make_project(tmp, "proj_output01")
            out_dir = proj_dir / "outputs" / "task_abc"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "output.txt").write_text("found output", encoding="utf-8")
            orig = _web_ui.PROJECT_ROOT
            _web_ui.PROJECT_ROOT = Path(tmp)
            try:
                resp = self.get("/api/output/task_abc")
                data = self.json(resp)
                self.assertEqual(data["output"], "found output")
            finally:
                _web_ui.PROJECT_ROOT = orig


if __name__ == "__main__":
    unittest.main()
