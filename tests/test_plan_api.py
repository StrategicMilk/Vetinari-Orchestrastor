"""
Comprehensive tests for vetinari/plan_api.py

All heavy / side-effectful dependencies are stubbed in sys.modules before any
vetinari import takes place.  The test file is self-contained and does not
write to disk or open network connections.
"""
import json
import sys
import types
import unittest
from enum import Enum
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# ---- Minimal real enum we need from vetinari.types ----
# We import vetinari.types directly (no heavy deps) so PlanStatus is genuine.
# ---------------------------------------------------------------------------
import vetinari.types as _vt  # noqa – tiny file, no side effects

# ---------------------------------------------------------------------------
# Build enum stubs used throughout the stubs
# ---------------------------------------------------------------------------
class _MemoryEntryType(str, Enum):
    APPROVAL = "approval"
    FEATURE  = "feature"
    GENERIC  = "generic"


class _MemoryEntry:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _ApprovalDetails:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    def to_dict(self):
        return self.__dict__


class _CodingTaskType(str, Enum):
    SCAFFOLD  = "scaffold"
    IMPLEMENT = "implement"
    TEST      = "test"
    REVIEW    = "review"


class _CodeTask:
    _counter = 0
    def __init__(self, **kw):
        _CodeTask._counter += 1
        self.task_id = f"task_{_CodeTask._counter:04d}"
        for k, v in kw.items():
            setattr(self, k, v)


class _PlanExplanation:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
    def to_dict(self):
        return {"summary": getattr(self, "summary", ""), "blocks": []}
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---------------------------------------------------------------------------
# Install ALL stubs before importing any vetinari sub-package
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    """Register a stub module in sys.modules; return it."""
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# vetinari.memory.interfaces
_stub("vetinari.memory.interfaces",
    IMemoryStore=object,
    MemoryStats=MagicMock,
    content_hash=lambda x: x,
    MEMORY_BACKEND_MODE="oc",
    OC_MEMORY_PATH="./oc_mem",
    MNEMOSYNE_PATH="./mn_mem",
    MEMORY_PRIMARY_READ="oc",
    MEMORY_DEDUP_ENABLED=False,
    MEMORY_MERGE_LIMIT=100,
    MemoryEntryType=_MemoryEntryType,
    MemoryEntry=_MemoryEntry,
    ApprovalDetails=_ApprovalDetails,
)

# vetinari.memory.oc_memory
_stub("vetinari.memory.oc_memory", OcMemoryStore=MagicMock)

# vetinari.memory.mnemosyne_memory
_stub("vetinari.memory.mnemosyne_memory", MnemosyneMemoryStore=MagicMock)

_mock_dual_store = MagicMock()
# vetinari.memory.dual_memory
_stub("vetinari.memory.dual_memory",
    DualMemoryStore=MagicMock,
    get_dual_memory_store=MagicMock(return_value=_mock_dual_store),
    init_dual_memory_store=MagicMock(),
)

# vetinari.explain_agent  – must include ExplainAgent so plan_mode.py can import it
_mock_explain_agent_instance = MagicMock()
_mock_explain_agent_instance.sanitize_explanation.return_value = MagicMock(
    to_dict=lambda: {"summary": "sanitized", "blocks": []}
)
_stub("vetinari.explain_agent",
    is_explainability_enabled=MagicMock(return_value=False),
    EXPLAINABILITY_ENABLED=False,
    get_explain_agent=MagicMock(return_value=_mock_explain_agent_instance),
    ExplainAgent=MagicMock,
    PlanExplanation=_PlanExplanation,
    SubtaskExplanation=MagicMock,
    ExplanationBlock=MagicMock,
)

# vetinari.learning.*
_stub("vetinari.learning")
_stub("vetinari.learning.workflow_learner",
    get_workflow_learner=MagicMock(return_value=MagicMock(
        get_recommendations=MagicMock(return_value={})
    ))
)

# vetinari.coding_agent
_stub("vetinari.coding_agent",
    CodingTaskType=_CodingTaskType,
    CodeTask=_CodeTask,
    CodeAgentEngine=MagicMock,
    get_coding_agent=MagicMock(),
)

# ---------------------------------------------------------------------------
# NOW import the real vetinari.memory package (needs stubs above already in place)
# ---------------------------------------------------------------------------
import vetinari.memory as _mem_pkg  # noqa

# Patch package-level names that plan_api.py pulls from it at import time
_mem_pkg.PLAN_ADMIN_TOKEN       = ""
_mem_pkg.DUAL_MEMORY_AVAILABLE  = True
_mem_pkg.MemoryEntry            = _MemoryEntry
_mem_pkg.MemoryEntryType        = _MemoryEntryType
_mem_pkg.ApprovalDetails        = _ApprovalDetails
_mem_pkg.get_dual_memory_store  = MagicMock(return_value=_mock_dual_store)

# ---------------------------------------------------------------------------
# Import plan_api (which pulls in plan_mode, plan_types, memory)
# ---------------------------------------------------------------------------
import vetinari.plan_api as plan_api_module  # noqa

from flask import Flask  # noqa

# ---------------------------------------------------------------------------
# Test-level constants
# ---------------------------------------------------------------------------
VALID_TOKEN = "secret-admin-token"
AUTH_HEADER = {"Authorization": f"Bearer {VALID_TOKEN}"}
JSON_CT     = {"Content-Type": "application/json"}
AUTH_JSON   = {**AUTH_HEADER, **JSON_CT}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def create_test_app():
    app = Flask(__name__)
    app.testing = True
    plan_api_module.register_plan_api(app)
    return app


def _mock_plan(plan_id="plan_abc123", status="draft", goal="Test goal"):
    from vetinari.plan_types import Plan, PlanRiskLevel
    from vetinari.types import PlanStatus

    plan = Plan()
    plan.plan_id          = plan_id
    plan.plan_version     = 1
    plan.goal             = goal
    plan.constraints      = ""
    plan.status           = PlanStatus(status)
    plan.risk_score       = 0.1
    plan.risk_level       = PlanRiskLevel.LOW
    plan.dry_run          = False
    plan.auto_approved    = False
    plan.approved_by      = None
    plan.approved_at      = None
    plan.subtasks         = []
    plan.dependencies     = {}
    plan.plan_candidates  = []
    plan.chosen_plan_id   = None
    plan.plan_justification  = ""
    plan.plan_explanation_json = ""
    plan.created_at       = "2025-01-01T00:00:00"
    plan.updated_at       = "2025-01-01T00:00:00"
    plan.completed_at     = None
    return plan


def _mock_subtask(subtask_id="subtask_001", plan_id="plan_abc123"):
    from vetinari.plan_types import Subtask, TaskDomain
    from vetinari.types import SubtaskStatus

    st = Subtask()
    st.subtask_id              = subtask_id
    st.plan_id                 = plan_id
    st.description             = "Test subtask"
    st.domain                  = TaskDomain.CODING
    st.status                  = SubtaskStatus.PENDING
    st.subtask_explanation_json = ""
    st.definition_of_done      = MagicMock(criteria=[])
    st.definition_of_ready     = MagicMock(prerequisites=[])
    st.to_dict = lambda: {
        "subtask_id": st.subtask_id,
        "plan_id":    st.plan_id,
        "description": st.description,
        "domain":     st.domain.value,
        "status":     st.status.value,
    }
    return st


def _make_engine(plan=None, subtasks=None, plans_history=None):
    engine = MagicMock()
    engine.generate_plan.return_value   = plan or _mock_plan()
    engine.get_plan.return_value        = plan
    engine.approve_plan.return_value    = plan or _mock_plan(status="approved")
    engine.get_subtasks.return_value    = subtasks or []
    engine.get_plan_history.return_value = plans_history or []
    engine.check_subtask_approval_required.return_value = {
        "requires_approval": True,
        "subtask_id": "subtask_001",
        "domain":     "coding",
        "plan_mode":  True,
        "description": "desc",
        "status":      "pending",
    }
    engine.log_approval_decision.return_value = True
    engine._domain_templates = {}
    return engine


# Convenience context-manager: patch the three key knobs for protected routes
def _auth_ctx(token=VALID_TOKEN, plan_mode=True, engine=None):
    """Return a list of patch objects to use as context managers."""
    return [
        patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
        patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
        patch("vetinari.plan_api.get_plan_engine",   return_value=engine or _make_engine()),
    ]


# ---------------------------------------------------------------------------
# Test: require_admin_token decorator
# ---------------------------------------------------------------------------

class TestRequireAdminToken(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _call(self, headers, token_value=VALID_TOKEN, plan=None):
        e = _make_engine(plan=plan or _mock_plan())
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token_value),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE", True),
            patch("vetinari.plan_api.get_plan_engine",  return_value=e),
        ):
            return self.client.get("/api/plan/plan_abc123", headers=headers)

    def test_no_header_with_token_set_returns_401(self):
        resp = self._call(headers={}, token_value=VALID_TOKEN)
        self.assertEqual(resp.status_code, 401)

    def test_wrong_bearer_token_returns_401(self):
        resp = self._call(headers={"Authorization": "Bearer wrong"})
        self.assertEqual(resp.status_code, 401)

    def test_non_bearer_wrong_token_returns_401(self):
        resp = self._call(headers={"Authorization": "wrong"})
        self.assertEqual(resp.status_code, 401)

    def test_correct_bearer_token_passes(self):
        resp = self._call(headers=AUTH_HEADER)
        self.assertNotEqual(resp.status_code, 401)

    def test_no_plan_admin_token_set_bypasses_check(self):
        resp = self._call(headers={}, token_value="")
        self.assertNotEqual(resp.status_code, 401)

    def test_non_bearer_correct_token_passes(self):
        resp = self._call(headers={"Authorization": VALID_TOKEN})
        self.assertNotEqual(resp.status_code, 401)

    def test_401_response_has_error_and_message_fields(self):
        resp = self._call(headers={}, token_value=VALID_TOKEN)
        data = resp.get_json()
        self.assertIn("error",   data)
        self.assertIn("message", data)


# ---------------------------------------------------------------------------
# Test: check_plan_mode_enabled helper
# ---------------------------------------------------------------------------

class TestCheckPlanModeEnabled(unittest.TestCase):

    def test_enabled_returns_true_none(self):
        with patch("vetinari.plan_api.PLAN_MODE_ENABLE", True):
            ok, err = plan_api_module.check_plan_mode_enabled()
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_disabled_returns_false_message(self):
        with patch("vetinari.plan_api.PLAN_MODE_ENABLE", False):
            ok, err = plan_api_module.check_plan_mode_enabled()
        self.assertFalse(ok)
        self.assertIn("disabled", err.lower())


# ---------------------------------------------------------------------------
# Test: POST /api/plan/generate
# ---------------------------------------------------------------------------

class TestGeneratePlan(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _post(self, body, token=VALID_TOKEN, plan_mode=True, engine=None):
        e = engine or _make_engine(plan=_mock_plan())
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.post(
                "/api/plan/generate",
                data=json.dumps(body),
                headers=AUTH_JSON,
                content_type="application/json",
            )

    def test_missing_goal_returns_400(self):
        resp = self._post({})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("goal", resp.get_json()["error"].lower())

    def test_empty_goal_returns_400(self):
        resp = self._post({"goal": ""})
        self.assertEqual(resp.status_code, 400)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._post({"goal": "build"}, plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.post(
                "/api/plan/generate",
                data=json.dumps({"goal": "build"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 401)

    def test_success_returns_200_with_plan_metadata(self):
        plan = _mock_plan()
        resp = self._post({"goal": "build a feature"}, engine=_make_engine(plan=plan))
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["plan_id"], plan.plan_id)

    def test_success_response_includes_all_required_fields(self):
        plan = _mock_plan()
        resp = self._post({"goal": "do something"}, engine=_make_engine(plan=plan))
        data = resp.get_json()
        for f in ["plan_id", "version", "goal", "status", "risk_score",
                  "risk_level", "subtask_count", "dry_run", "auto_approved",
                  "plan_candidates", "chosen_plan_id", "plan_justification", "created_at"]:
            self.assertIn(f, data, f"Missing field: {f}")

    def test_engine_exception_returns_500(self):
        e = _make_engine()
        e.generate_plan.side_effect = RuntimeError("engine failure")
        resp = self._post({"goal": "broken"}, engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_invalid_domain_hint_value_error_returns_500(self):
        e = _make_engine()
        e.generate_plan.side_effect = ValueError("bad domain")
        resp = self._post({"goal": "test", "domain_hint": "not_a_domain"}, engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_optional_fields_accepted_without_error(self):
        resp = self._post({
            "goal": "build thing",
            "constraints": "fast",
            "plan_depth_cap": 8,
            "max_candidates": 2,
            "dry_run": True,
            "risk_threshold": 0.3,
        })
        self.assertEqual(resp.status_code, 200)

    def test_valid_domain_hint_coding(self):
        resp = self._post({"goal": "build api", "domain_hint": "coding"})
        self.assertEqual(resp.status_code, 200)

    def test_engine_generate_plan_called_once(self):
        e = _make_engine(plan=_mock_plan())
        self._post({"goal": "build thing"}, engine=e)
        e.generate_plan.assert_called_once()


# ---------------------------------------------------------------------------
# Test: GET /api/plan/<plan_id>
# ---------------------------------------------------------------------------

class TestGetPlan(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_id="plan_abc123", plan=None, plan_mode=True, token=VALID_TOKEN):
        e = _make_engine(plan=plan)
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(f"/api/plan/{plan_id}", headers=AUTH_HEADER)

    def test_found_returns_200_with_plan(self):
        plan = _mock_plan()
        resp = self._get(plan=plan)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["plan"]["plan_id"], plan.plan_id)

    def test_not_found_returns_404(self):
        resp = self._get(plan=None)
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/plan/plan_abc123")
        self.assertEqual(resp.status_code, 401)

    def test_response_plan_contains_subtasks_and_candidates(self):
        plan = _mock_plan()
        resp = self._get(plan=plan)
        plan_data = resp.get_json()["plan"]
        self.assertIn("subtasks",       plan_data)
        self.assertIn("plan_candidates", plan_data)
        self.assertIn("dependencies",   plan_data)

    def test_engine_exception_returns_500(self):
        e = _make_engine()
        e.get_plan.side_effect = RuntimeError("DB failure")
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  True),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            resp = self.client.get("/api/plan/bad", headers=AUTH_HEADER)
        self.assertEqual(resp.status_code, 500)


# ---------------------------------------------------------------------------
# Test: POST /api/plan/<plan_id>/approve
# ---------------------------------------------------------------------------

class TestApprovePlan(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _post(self, plan_id="plan_abc123", body=None, plan=None,
              plan_mode=True, token=VALID_TOKEN, engine=None):
        approved_plan = plan or _mock_plan(status="approved")
        e = engine or _make_engine(plan=approved_plan)
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",      token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",       plan_mode),
            patch("vetinari.plan_api.get_plan_engine",        return_value=e),
            # DUAL_MEMORY_AVAILABLE is imported from vetinari.memory inside the route function
            patch("vetinari.memory.DUAL_MEMORY_AVAILABLE",    False),
        ):
            return self.client.post(
                f"/api/plan/{plan_id}/approve",
                data=json.dumps(body or {}),
                headers=AUTH_JSON,
                content_type="application/json",
            )

    def test_missing_approved_field_returns_400(self):
        resp = self._post(body={"approver": "admin"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("approved", resp.get_json()["error"])

    def test_missing_approver_field_returns_400(self):
        resp = self._post(body={"approved": True})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("approver", resp.get_json()["error"])

    def test_plan_mode_disabled_returns_403(self):
        resp = self._post(body={"approved": True, "approver": "admin"}, plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_successful_approval_returns_200(self):
        plan = _mock_plan(status="approved")
        resp = self._post(body={"approved": True, "approver": "admin"}, plan=plan)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("plan_id", data)
        self.assertIn("status",  data)

    def test_successful_rejection_returns_200(self):
        plan = _mock_plan(status="rejected")
        resp = self._post(
            body={"approved": False, "approver": "admin", "reason": "risky"},
            plan=plan,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])

    def test_plan_not_found_value_error_returns_404(self):
        e = _make_engine()
        e.approve_plan.side_effect = ValueError("Plan not found: plan_xyz")
        resp = self._post(body={"approved": True, "approver": "admin"}, engine=e)
        self.assertEqual(resp.status_code, 404)

    def test_general_exception_returns_500(self):
        e = _make_engine()
        e.approve_plan.side_effect = RuntimeError("unexpected")
        resp = self._post(body={"approved": True, "approver": "admin"}, engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.post(
                "/api/plan/plan_abc123/approve",
                data=json.dumps({"approved": True, "approver": "admin"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 401)

    def test_optional_audit_id_returned(self):
        plan = _mock_plan(status="approved")
        resp = self._post(
            body={"approved": True, "approver": "admin", "audit_id": "audit-001"},
            plan=plan,
        )
        self.assertEqual(resp.get_json()["audit_id"], "audit-001")

    def test_memory_logging_failure_does_not_cause_500(self):
        """Memory logging failure inside approve_plan is swallowed; 200 still returned."""
        plan = _mock_plan(status="approved")
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",   VALID_TOKEN),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",   True),
            patch("vetinari.plan_api.get_plan_engine",    return_value=_make_engine(plan=plan)),
            # DUAL_MEMORY_AVAILABLE lives in vetinari.memory, imported locally inside the route
            patch("vetinari.memory.DUAL_MEMORY_AVAILABLE", True),
            patch("vetinari.memory.get_dual_memory_store", side_effect=RuntimeError("mem down")),
        ):
            resp = self.client.post(
                "/api/plan/plan_abc123/approve",
                data=json.dumps({"approved": True, "approver": "admin"}),
                headers=AUTH_JSON,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)

    def test_dual_memory_logging_called_when_available(self):
        plan = _mock_plan(status="approved")
        mock_store = MagicMock()
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",    VALID_TOKEN),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",    True),
            patch("vetinari.plan_api.get_plan_engine",     return_value=_make_engine(plan=plan)),
            patch("vetinari.memory.DUAL_MEMORY_AVAILABLE", True),
            patch("vetinari.memory.get_dual_memory_store", return_value=mock_store),
        ):
            resp = self.client.post(
                "/api/plan/plan_abc123/approve",
                data=json.dumps({"approved": True, "approver": "admin"}),
                headers=AUTH_JSON,
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# Test: POST /api/plan/<plan_id>/subtasks/<subtask_id>/approve
# ---------------------------------------------------------------------------

class TestApproveSubtask(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _post(self, plan_id="plan_abc123", subtask_id="subtask_001",
              body=None, plan=None, plan_mode=True, token=VALID_TOKEN, engine=None):
        e = engine or _make_engine(plan=plan or _mock_plan())
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.post(
                f"/api/plan/{plan_id}/subtasks/{subtask_id}/approve",
                data=json.dumps(body or {}),
                headers=AUTH_JSON,
                content_type="application/json",
            )

    def test_success_returns_200(self):
        resp = self._post(body={"approved": True, "approver": "admin"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])

    def test_missing_approved_field_returns_400(self):
        resp = self._post(body={"approver": "admin"})
        self.assertEqual(resp.status_code, 400)

    def test_missing_approver_field_returns_400(self):
        resp = self._post(body={"approved": True})
        self.assertEqual(resp.status_code, 400)

    def test_plan_not_found_returns_404(self):
        e = _make_engine(plan=None)
        resp = self._post(body={"approved": True, "approver": "admin"}, engine=e)
        self.assertEqual(resp.status_code, 404)

    def test_subtask_not_found_returns_404(self):
        e = _make_engine(plan=_mock_plan())
        e.check_subtask_approval_required.return_value = {"error": "Subtask not found"}
        resp = self._post(body={"approved": True, "approver": "admin"}, engine=e)
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._post(body={"approved": True, "approver": "admin"}, plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.post(
                "/api/plan/plan_abc123/subtasks/subtask_001/approve",
                data=json.dumps({"approved": True, "approver": "admin"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 401)

    def test_engine_exception_returns_500(self):
        e = _make_engine(plan=_mock_plan())
        e.log_approval_decision.side_effect = RuntimeError("boom")
        resp = self._post(body={"approved": True, "approver": "admin"}, engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_response_includes_all_expected_fields(self):
        resp = self._post(body={"approved": True, "approver": "admin"})
        data = resp.get_json()
        for f in ["success", "plan_id", "subtask_id", "approved", "approver"]:
            self.assertIn(f, data)

    def test_rejection_also_returns_200(self):
        resp = self._post(body={"approved": False, "approver": "admin", "reason": "not ready"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["approved"])


# ---------------------------------------------------------------------------
# Test: GET /api/plan/<plan_id>/history
# ---------------------------------------------------------------------------

class TestGetPlanHistory(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_id="plan_abc123", plan=None, subtasks=None,
             plan_mode=True, token=VALID_TOKEN, engine=None):
        e = engine or _make_engine(plan=plan, subtasks=subtasks or [])
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(f"/api/plan/{plan_id}/history", headers=AUTH_HEADER)

    def test_success_returns_200(self):
        resp = self._get(plan=_mock_plan())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("plan_id",  data)
        self.assertIn("subtasks", data)

    def test_not_found_returns_404(self):
        resp = self._get(plan=None)
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/plan/plan_abc123/history")
        self.assertEqual(resp.status_code, 401)

    def test_response_includes_risk_and_timestamps(self):
        resp = self._get(plan=_mock_plan())
        data = resp.get_json()
        for f in ["risk_score", "risk_level", "created_at", "updated_at"]:
            self.assertIn(f, data)

    def test_engine_exception_returns_500(self):
        e = _make_engine(plan=_mock_plan())
        e.get_subtasks.side_effect = RuntimeError("subtask fail")
        resp = self._get(engine=e)
        self.assertEqual(resp.status_code, 500)


# ---------------------------------------------------------------------------
# Test: GET /api/plan/history
# ---------------------------------------------------------------------------

class TestGetAllPlanHistory(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, params="", plan_mode=True, token=VALID_TOKEN, plans=None, engine=None):
        e = engine or _make_engine(plans_history=plans or [])
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(f"/api/plan/history{params}", headers=AUTH_HEADER)

    def test_success_default_limit_returns_200(self):
        resp = self._get(plans=[{"plan_id": "p1", "goal": "build"}])
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("plans", data)
        self.assertIn("count", data)

    def test_goal_contains_filter_forwarded_to_engine(self):
        e = _make_engine(plans_history=[])
        self._get(params="?goal_contains=test", engine=e)
        e.get_plan_history.assert_called_once_with(goal_contains="test", limit=10)

    def test_custom_limit_forwarded_to_engine(self):
        e = _make_engine(plans_history=[])
        self._get(params="?limit=5", engine=e)
        e.get_plan_history.assert_called_once_with(goal_contains=None, limit=5)

    def test_both_params_together(self):
        e = _make_engine(plans_history=[])
        self._get(params="?goal_contains=build&limit=3", engine=e)
        e.get_plan_history.assert_called_once_with(goal_contains="build", limit=3)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/plan/history")
        self.assertEqual(resp.status_code, 401)

    def test_engine_exception_returns_500(self):
        e = _make_engine()
        e.get_plan_history.side_effect = RuntimeError("history fail")
        resp = self._get(engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_empty_result_returns_count_zero(self):
        resp = self._get(plans=[])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["count"], 0)


# ---------------------------------------------------------------------------
# Test: GET /api/plan/<plan_id>/subtasks
# ---------------------------------------------------------------------------

class TestGetPlanSubtasks(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_id="plan_abc123", plan=None, subtasks=None,
             plan_mode=True, token=VALID_TOKEN, engine=None):
        e = engine or _make_engine(plan=plan, subtasks=subtasks or [])
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(f"/api/plan/{plan_id}/subtasks", headers=AUTH_HEADER)

    def test_success_returns_200_with_subtask_list(self):
        st = _mock_subtask()
        resp = self._get(plan=_mock_plan(), subtasks=[st])
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 1)

    def test_plan_not_found_returns_404(self):
        resp = self._get(plan=None)
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/plan/plan_abc123/subtasks")
        self.assertEqual(resp.status_code, 401)

    def test_empty_subtasks_count_zero(self):
        resp = self._get(plan=_mock_plan(), subtasks=[])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["count"], 0)

    def test_engine_exception_returns_500(self):
        e = _make_engine(plan=_mock_plan())
        e.get_subtasks.side_effect = RuntimeError("subtask err")
        resp = self._get(engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_response_includes_plan_id_field(self):
        resp = self._get(plan=_mock_plan())
        self.assertIn("plan_id", resp.get_json())


# ---------------------------------------------------------------------------
# Test: GET /api/plan/status  (no auth)
# ---------------------------------------------------------------------------

class TestGetPlanModeStatus(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_mode=True, token_set=True, memory_raises=False):
        mock_mem = MagicMock()
        if memory_raises:
            mock_mem.get_memory_stats.side_effect = RuntimeError("db down")
        else:
            mock_mem.get_memory_stats.return_value = {"total_plans": 5}
        with (
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.PLAN_MODE_DEFAULT", True),
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",  "tok" if token_set else ""),
            patch("vetinari.memory.get_memory_store",    return_value=mock_mem),
        ):
            return self.client.get("/api/plan/status")

    def test_no_auth_required_returns_200(self):
        resp = self._get()
        self.assertEqual(resp.status_code, 200)

    def test_response_contains_plan_mode_enabled(self):
        resp = self._get(plan_mode=True)
        data = resp.get_json()
        self.assertIn("plan_mode_enabled", data)
        self.assertTrue(data["plan_mode_enabled"])

    def test_response_contains_config_block(self):
        resp = self._get()
        data = resp.get_json()
        self.assertIn("config", data)
        cfg = data["config"]
        self.assertIn("PLAN_MODE_ENABLE",     cfg)
        self.assertIn("PLAN_ADMIN_TOKEN_SET", cfg)

    def test_memory_error_handled_gracefully(self):
        resp = self._get(memory_raises=True)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("memory_stats", {}), {})

    def test_plan_admin_token_set_flag_true(self):
        resp = self._get(token_set=True)
        self.assertTrue(resp.get_json()["config"]["PLAN_ADMIN_TOKEN_SET"])

    def test_plan_admin_token_set_flag_false(self):
        resp = self._get(token_set=False)
        self.assertFalse(resp.get_json()["config"]["PLAN_ADMIN_TOKEN_SET"])

    def test_success_field_true(self):
        resp = self._get()
        self.assertTrue(resp.get_json()["success"])


# ---------------------------------------------------------------------------
# Test: GET /api/plan/templates  (no auth)
# ---------------------------------------------------------------------------

class TestGetPlanTemplates(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _make_template_engine(self):
        from vetinari.plan_types import TaskDomain, DefinitionOfDone, DefinitionOfReady
        e = MagicMock()
        e._domain_templates = {
            TaskDomain.CODING: [
                {
                    "description": "Define API",
                    "definition_of_done":  DefinitionOfDone(criteria=["api done"]),
                    "definition_of_ready": DefinitionOfReady(prerequisites=["req met"]),
                }
            ]
        }
        return e

    def test_no_auth_required_returns_200(self):
        with patch("vetinari.plan_api.get_plan_engine", return_value=self._make_template_engine()):
            resp = self.client.get("/api/plan/templates")
        self.assertEqual(resp.status_code, 200)

    def test_response_contains_templates_and_domains(self):
        with patch("vetinari.plan_api.get_plan_engine", return_value=self._make_template_engine()):
            resp = self.client.get("/api/plan/templates")
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("templates", data)
        self.assertIn("domains",   data)

    def test_exception_returns_500(self):
        # Make get_plan_engine raise so the outer try/except catches it
        with patch("vetinari.plan_api.get_plan_engine", side_effect=RuntimeError("boom")):
            resp = self.client.get("/api/plan/templates")
        self.assertEqual(resp.status_code, 500)


# ---------------------------------------------------------------------------
# Test: GET /api/plan/<plan_id>/explanations
# ---------------------------------------------------------------------------

class TestGetPlanExplanations(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_id="plan_abc123", plan=None, plan_mode=True,
             token=VALID_TOKEN, params="", engine=None):
        e = engine or _make_engine(plan=plan)
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(
                f"/api/plan/{plan_id}/explanations{params}",
                headers=AUTH_HEADER,
            )

    def test_plan_not_found_returns_404(self):
        resp = self._get(plan=None)
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/plan/plan_abc123/explanations")
        self.assertEqual(resp.status_code, 401)

    def test_plan_with_no_explanation_returns_null(self):
        plan = _mock_plan()
        plan.plan_explanation_json = ""
        resp = self._get(plan=plan)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.get_json()["explanation"])

    def test_plan_with_valid_explanation_returns_dict(self):
        plan = _mock_plan()
        plan.plan_explanation_json = json.dumps({"summary": "hello", "blocks": []})
        resp = self._get(plan=plan)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.get_json()["explanation"])

    def test_sanitized_true_invokes_explain_agent(self):
        plan = _mock_plan()
        plan.plan_explanation_json = json.dumps({"summary": "test", "blocks": []})
        mock_agent = MagicMock()
        mock_sanitized = MagicMock()
        mock_sanitized.to_dict.return_value = {"summary": "sanitized", "blocks": []}
        mock_agent.sanitize_explanation.return_value = mock_sanitized
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",    VALID_TOKEN),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",     True),
            patch("vetinari.plan_api.get_plan_engine",      return_value=_make_engine(plan=plan)),
            patch("vetinari.explain_agent.get_explain_agent", return_value=mock_agent),
            patch("vetinari.explain_agent.PlanExplanation",   _PlanExplanation),
        ):
            resp = self.client.get(
                "/api/plan/plan_abc123/explanations?sanitized=true",
                headers=AUTH_HEADER,
            )
        self.assertEqual(resp.status_code, 200)

    def test_sanitized_false_does_not_invoke_explain_agent(self):
        plan = _mock_plan()
        plan.plan_explanation_json = json.dumps({"summary": "x", "blocks": []})
        resp = self._get(plan=plan, params="?sanitized=false")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("explanation", resp.get_json())

    def test_malformed_explanation_json_returns_null(self):
        plan = _mock_plan()
        plan.plan_explanation_json = "not-valid-json{"
        resp = self._get(plan=plan)
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.get_json()["explanation"])

    def test_sanitized_param_accepts_1_as_true(self):
        plan = _mock_plan()
        plan.plan_explanation_json = json.dumps({"summary": "x", "blocks": []})
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",    VALID_TOKEN),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",     True),
            patch("vetinari.plan_api.get_plan_engine",      return_value=_make_engine(plan=plan)),
            patch("vetinari.explain_agent.get_explain_agent",
                  return_value=_mock_explain_agent_instance),
            patch("vetinari.explain_agent.PlanExplanation", _PlanExplanation),
        ):
            resp = self.client.get(
                "/api/plan/plan_abc123/explanations?sanitized=1",
                headers=AUTH_HEADER,
            )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# Test: GET /api/plan/<plan_id>/subtasks/<subtask_id>/explanation
# ---------------------------------------------------------------------------

class TestGetSubtaskExplanation(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, plan_id="plan_abc123", subtask_id="subtask_001",
             subtasks=None, plan_mode=True, token=VALID_TOKEN, engine=None):
        e = engine or _make_engine(subtasks=subtasks or [])
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", token),
            patch("vetinari.plan_api.PLAN_MODE_ENABLE",  plan_mode),
            patch("vetinari.plan_api.get_plan_engine",   return_value=e),
        ):
            return self.client.get(
                f"/api/plan/{plan_id}/subtasks/{subtask_id}/explanation",
                headers=AUTH_HEADER,
            )

    def test_subtask_found_with_explanation_returns_200(self):
        st = _mock_subtask()
        st.subtask_explanation_json = json.dumps({"summary": "subtask exp"})
        resp = self._get(subtasks=[st])
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["explanation"])

    def test_subtask_not_found_returns_404(self):
        resp = self._get(subtasks=[], subtask_id="nonexistent_subtask")
        self.assertEqual(resp.status_code, 404)

    def test_plan_mode_disabled_returns_403(self):
        resp = self._get(plan_mode=False)
        self.assertEqual(resp.status_code, 403)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get(
                "/api/plan/plan_abc123/subtasks/subtask_001/explanation"
            )
        self.assertEqual(resp.status_code, 401)

    def test_subtask_with_no_explanation_returns_null(self):
        st = _mock_subtask()
        st.subtask_explanation_json = ""
        resp = self._get(subtasks=[st])
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.get_json()["explanation"])

    def test_malformed_subtask_explanation_json_returns_null(self):
        st = _mock_subtask()
        st.subtask_explanation_json = "bad-json{"
        resp = self._get(subtasks=[st])
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(resp.get_json()["explanation"])

    def test_engine_exception_returns_500(self):
        e = _make_engine()
        e.get_subtasks.side_effect = RuntimeError("fail")
        resp = self._get(engine=e)
        self.assertEqual(resp.status_code, 500)

    def test_response_includes_plan_id_and_subtask_id(self):
        st = _mock_subtask()
        resp = self._get(subtasks=[st])
        data = resp.get_json()
        self.assertIn("plan_id",    data)
        self.assertIn("subtask_id", data)


# ---------------------------------------------------------------------------
# Test: POST /api/coding/task
# ---------------------------------------------------------------------------

class TestCreateCodingTask(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _post(self, body=None, token=VALID_TOKEN, agent_available=True,
              run_raises=None):
        mock_agent = MagicMock()
        mock_agent.is_available.return_value = agent_available
        if run_raises:
            mock_agent.run_task.side_effect = run_raises
        else:
            artifact = MagicMock()
            artifact.to_dict.return_value = {"files": [], "status": "ok"}
            mock_agent.run_task.return_value = artifact

        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",      token),
            patch("vetinari.coding_agent.get_coding_agent",  return_value=mock_agent),
            patch("vetinari.coding_agent.CodingTaskType",    _CodingTaskType),
            patch("vetinari.coding_agent.CodeTask",          _CodeTask),
        ):
            return self.client.post(
                "/api/coding/task",
                data=json.dumps(body or {}),
                headers=AUTH_JSON,
                content_type="application/json",
            )

    def test_success_returns_200(self):
        resp = self._post(body={"type": "implement", "description": "build thing"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("task_id",  data)
        self.assertIn("artifact", data)

    def test_invalid_task_type_returns_400(self):
        resp = self._post(body={"type": "nonexistent_type"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid task type", resp.get_json()["error"])

    def test_agent_unavailable_returns_503(self):
        resp = self._post(body={"type": "implement"}, agent_available=False)
        self.assertEqual(resp.status_code, 503)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.post(
                "/api/coding/task",
                data=json.dumps({"type": "implement"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 401)

    def test_scaffold_type_valid(self):
        self.assertEqual(self._post(body={"type": "scaffold"}).status_code, 200)

    def test_test_type_valid(self):
        self.assertEqual(self._post(body={"type": "test"}).status_code, 200)

    def test_review_type_valid(self):
        self.assertEqual(self._post(body={"type": "review"}).status_code, 200)

    def test_run_task_exception_returns_500(self):
        resp = self._post(body={"type": "implement"}, run_raises=RuntimeError("exec fail"))
        self.assertEqual(resp.status_code, 500)

    def test_default_type_implement_when_missing(self):
        resp = self._post(body={})
        self.assertEqual(resp.status_code, 200)

    def test_extra_optional_fields_accepted(self):
        resp = self._post(body={
            "type": "implement",
            "language": "python",
            "description": "test",
            "repo_path": "./src",
            "target_files": ["file.py"],
            "constraints": "no globals",
        })
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# Test: GET /api/coding/task/<task_id>
# ---------------------------------------------------------------------------

class TestGetCodingTask(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _get(self, task_id="task_0001", token=VALID_TOKEN, agent_available=True):
        mock_agent = MagicMock()
        mock_agent.is_available.return_value = agent_available
        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",     token),
            patch("vetinari.coding_agent.get_coding_agent", return_value=mock_agent),
        ):
            return self.client.get(f"/api/coding/task/{task_id}", headers=AUTH_HEADER)

    def test_success_returns_200_with_status_completed(self):
        resp = self._get()
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "completed")

    def test_agent_unavailable_returns_503(self):
        resp = self._get(agent_available=False)
        self.assertEqual(resp.status_code, 503)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.get("/api/coding/task/task_0001")
        self.assertEqual(resp.status_code, 401)

    def test_task_id_echoed_in_response(self):
        resp = self._get(task_id="task_custom_99")
        self.assertEqual(resp.get_json()["task_id"], "task_custom_99")


# ---------------------------------------------------------------------------
# Test: POST /api/coding/multi-step
# ---------------------------------------------------------------------------

class TestCreateMultiStepCoding(unittest.TestCase):

    def setUp(self):
        self.app    = create_test_app()
        self.client = self.app.test_client()

    def _post(self, body=None, token=VALID_TOKEN, agent_available=True,
              fail_on_calls=None):
        mock_agent = MagicMock()
        mock_agent.is_available.return_value = agent_available

        call_count = [0]

        def _run_task(task):
            call_count[0] += 1
            if fail_on_calls and call_count[0] in fail_on_calls:
                raise RuntimeError(f"task {call_count[0]} failed")
            artifact = MagicMock()
            artifact.to_dict.return_value = {"files": [], "status": "ok"}
            return artifact

        mock_agent.run_task.side_effect = _run_task

        with (
            patch("vetinari.plan_api.PLAN_ADMIN_TOKEN",     token),
            patch("vetinari.coding_agent.get_coding_agent", return_value=mock_agent),
            patch("vetinari.coding_agent.CodingTaskType",   _CodingTaskType),
            patch("vetinari.coding_agent.CodeTask",         _CodeTask),
        ):
            return self.client.post(
                "/api/coding/multi-step",
                data=json.dumps(body or {}),
                headers=AUTH_JSON,
                content_type="application/json",
            )

    def test_success_all_subtasks_returns_200(self):
        body = {
            "plan_id": "plan_abc",
            "subtasks": [
                {"subtask_id": "s1", "type": "scaffold",  "description": "sc"},
                {"subtask_id": "s2", "type": "implement", "description": "impl"},
                {"subtask_id": "s3", "type": "test",      "description": "test"},
            ],
        }
        resp = self._post(body=body)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["results"]), 3)
        self.assertTrue(all(r["success"] for r in data["results"]))

    def test_partial_failure_still_returns_200(self):
        body = {
            "plan_id": "plan_abc",
            "subtasks": [
                {"subtask_id": "s1", "type": "scaffold"},
                {"subtask_id": "s2", "type": "implement"},
            ],
        }
        resp = self._post(body=body, fail_on_calls={2})
        self.assertEqual(resp.status_code, 200)
        results = resp.get_json()["results"]
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertIn("error", results[1])

    def test_agent_unavailable_returns_503(self):
        resp = self._post(agent_available=False)
        self.assertEqual(resp.status_code, 503)

    def test_no_auth_returns_401(self):
        with patch("vetinari.plan_api.PLAN_ADMIN_TOKEN", VALID_TOKEN):
            resp = self.client.post(
                "/api/coding/multi-step",
                data=json.dumps({"plan_id": "p1", "subtasks": []}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 401)

    def test_empty_subtasks_returns_empty_results(self):
        body = {"plan_id": "plan_abc", "subtasks": []}
        resp = self._post(body=body)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["results"], [])

    def test_invalid_task_type_falls_back_to_implement(self):
        body = {
            "plan_id": "plan_abc",
            "subtasks": [{"subtask_id": "s1", "type": "unknown_type"}],
        }
        resp = self._post(body=body)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["results"][0]["success"])

    def test_plan_id_echoed_in_response(self):
        body = {"plan_id": "plan_xyz", "subtasks": []}
        resp = self._post(body=body)
        self.assertEqual(resp.get_json()["plan_id"], "plan_xyz")

    def test_all_fail_returns_200_with_all_failed_results(self):
        body = {
            "plan_id": "plan_abc",
            "subtasks": [
                {"subtask_id": "s1", "type": "implement"},
                {"subtask_id": "s2", "type": "implement"},
            ],
        }
        resp = self._post(body=body, fail_on_calls={1, 2})
        self.assertEqual(resp.status_code, 200)
        results = resp.get_json()["results"]
        self.assertTrue(all(not r["success"] for r in results))


# ---------------------------------------------------------------------------
# Test: register_plan_api()
# ---------------------------------------------------------------------------

class TestRegisterPlanApi(unittest.TestCase):

    def test_blueprint_registered_successfully(self):
        app = Flask(__name__)
        app.testing = True
        plan_api_module.register_plan_api(app)
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertTrue(any("/api/plan/status"  in r for r in rules))
        self.assertTrue(any("/api/plan/history" in r for r in rules))
        self.assertTrue(any("/api/coding/task"  in r for r in rules))

    def test_separate_apps_both_get_routes(self):
        app1 = Flask("app_alpha")
        app2 = Flask("app_beta")
        plan_api_module.register_plan_api(app1)
        plan_api_module.register_plan_api(app2)
        rules1 = [r.rule for r in app1.url_map.iter_rules()]
        rules2 = [r.rule for r in app2.url_map.iter_rules()]
        self.assertTrue(any("/api/plan/status" in r for r in rules1))
        self.assertTrue(any("/api/plan/status" in r for r in rules2))

    def test_generate_route_exists(self):
        app = Flask("app_gen")
        plan_api_module.register_plan_api(app)
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertTrue(any("/api/plan/generate" in r for r in rules))

    def test_templates_route_exists(self):
        app = Flask("app_tmpl")
        plan_api_module.register_plan_api(app)
        rules = [r.rule for r in app.url_map.iter_rules()]
        self.assertTrue(any("/api/plan/templates" in r for r in rules))


if __name__ == "__main__":
    unittest.main()
