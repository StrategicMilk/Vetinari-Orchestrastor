"""Governance tests for training and learning read routes.

Proves that all training and learning GET routes return bounded 503 on
subsystem failure -- not raw 500, not a green-gauge/idle/empty 200 that
hides the error from the caller.

Special assertions:
- ``test_training_status_subsystem_failure_returns_503_not_ok``: proves
  the status route no longer returns ``{"status":"ok"}`` when subsystems fail.
- ``test_training_experiment_unavailable_returns_503_not_404``: proves
  the experiment detail route no longer misclassifies trainer unavailability
  as a 404 "not found".

All tests go through the full Litestar HTTP stack via TestClient so that
framework-level serialization and exception handlers are exercised.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


# Module references captured immediately after create_app() while all vetinari
# modules are still present in sys.modules.  The autouse function-scoped fixture
# _isolate_vetinari_modules wipes sys.modules before every test, so any fixture
# that calls sys.modules.get() lazily (i.e. when a test first requests it) will
# find None.  Storing the references here at collection/fixture-setup time
# (inside the app fixture, which is module-scoped and runs once before the first
# test's function-scoped fixtures fire) ensures the values are stable for the
# whole test module regardless of which test class first requests them.
_web_module_refs: dict[str, object] = {}


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is only built once; each test
    creates its own TestClient context so connection state does not leak.

    Also populates the module-level ``_web_module_refs`` dict with live vetinari
    web module objects captured while they are still in ``sys.modules``.  This
    must happen here (inside the module-scoped ``app`` fixture) because the
    autouse function-scoped ``_isolate_vetinari_modules`` fixture will wipe
    ``sys.modules`` before any test body runs  -  including before any module-scoped
    fixture that is evaluated lazily on first request.

    Side effects:
        - Populates ``_web_module_refs["experiments"]`` and
          ``_web_module_refs["training_api"]`` for use by dependent fixtures.

    Returns:
        A Litestar application instance.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        litestar_app = create_app(debug=True)

    # Capture NOW  -  before _isolate_vetinari_modules runs for the first test.
    _web_module_refs["experiments"] = sys.modules.get("vetinari.web.litestar_training_experiments_api")
    _web_module_refs["training_api"] = sys.modules.get("vetinari.web.litestar_training_api")
    return litestar_app


@pytest.fixture(scope="module")
def training_experiments_mod(app):
    """Return the training experiments module as loaded by the app.

    The conftest ``_isolate_vetinari_modules`` fixture (function-scoped,
    autouse) wipes all vetinari modules from ``sys.modules`` before each test.
    When a test then calls ``patch("vetinari.web.litestar_training_experiments_api
    ._get_automation_rules", ...)`` the string-based patch imports a fresh
    module object that is different from the one the app's route handlers
    reference.  The handlers use ``fn.__globals__`` which was bound at
    ``create_app()`` time  -  patching a new module object has no effect.

    The reference is captured inside the ``app`` fixture (before any
    function-scoped fixture can wipe ``sys.modules``) and stored in the
    module-level ``_web_module_refs`` dict, making it immune to test ordering.

    Args:
        app: The Litestar application instance (module-scoped).

    Returns:
        The ``litestar_training_experiments_api`` module object used by the
        app's route handlers.
    """
    return _web_module_refs.get("experiments")


@pytest.fixture(scope="module")
def training_api_mod(app):
    """Return the training API module as loaded by the app.

    Same isolation problem as ``training_experiments_mod``: the autouse
    ``_isolate_vetinari_modules`` fixture wipes vetinari modules from
    ``sys.modules`` before each test.  String patches on
    ``vetinari.web.litestar_training_api._get_scheduler`` resolve a fresh
    module object that the route handlers do not reference.

    The reference is captured inside the ``app`` fixture and stored in
    ``_web_module_refs`` so this fixture is immune to test ordering.

    Args:
        app: The Litestar application instance (module-scoped).

    Returns:
        The ``litestar_training_api`` module object used by the app's route
        handlers.
    """
    return _web_module_refs.get("training_api")


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert that *response* is a bounded 503 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


# ---------------------------------------------------------------------------
# Learning routes (4)
# ---------------------------------------------------------------------------


class TestLearningThompsonArms:
    """GET /api/v1/learning/thompson -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_thompson_selector to raise; endpoint must return 503."""
        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            side_effect=RuntimeError("thompson selector down"),
        ):
            response = client.get("/api/v1/learning/thompson")
        _assert_503_error(response)

    def test_arms_access_raises_returns_503(self, client: object) -> None:
        """When _arms attribute access raises, endpoint must return 503."""
        mock_selector = MagicMock()
        type(mock_selector)._arms = property(lambda self: (_ for _ in ()).throw(RuntimeError("arms broken")))
        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            return_value=mock_selector,
        ):
            response = client.get("/api/v1/learning/thompson")
        _assert_503_error(response)


class TestLearningQualityHistory:
    """GET /api/v1/learning/quality-history -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_quality_scorer to raise; endpoint must return 503."""
        with patch(
            "vetinari.learning.quality_scorer.get_quality_scorer",
            side_effect=RuntimeError("quality scorer down"),
        ):
            response = client.get("/api/v1/learning/quality-history")
        _assert_503_error(response)

    def test_get_history_raises_returns_503(self, client: object) -> None:
        """When get_history() raises after import, endpoint must return 503."""
        mock_scorer = MagicMock()
        mock_scorer.get_history.side_effect = RuntimeError("history DB gone")
        with patch(
            "vetinari.learning.quality_scorer.get_quality_scorer",
            return_value=mock_scorer,
        ):
            response = client.get("/api/v1/learning/quality-history")
        _assert_503_error(response)


class TestLearningTrainingStats:
    """GET /api/v1/learning/training-stats -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_training_collector to raise; endpoint must return 503."""
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            side_effect=RuntimeError("collector down"),
        ):
            response = client.get("/api/v1/learning/training-stats")
        _assert_503_error(response)

    def test_get_stats_raises_returns_503(self, client: object) -> None:
        """When get_stats() raises after import, endpoint must return 503."""
        mock_collector = MagicMock()
        mock_collector.get_stats.side_effect = RuntimeError("stats table gone")
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            response = client.get("/api/v1/learning/training-stats")
        _assert_503_error(response)


class TestLearningWorkflowPatterns:
    """GET /api/v1/learning/workflow-patterns -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch WorkflowLearner to raise on instantiation; endpoint must return 503."""
        with patch(
            "vetinari.learning.workflow_learner.WorkflowLearner",
            side_effect=RuntimeError("workflow learner init failed"),
        ):
            response = client.get("/api/v1/learning/workflow-patterns")
        _assert_503_error(response)

    def test_get_all_patterns_raises_returns_503(self, client: object) -> None:
        """When get_all_patterns() raises, endpoint must return 503."""
        mock_learner = MagicMock()
        mock_learner.get_all_patterns.side_effect = RuntimeError("patterns table gone")
        with patch(
            "vetinari.learning.workflow_learner.WorkflowLearner",
            return_value=mock_learner,
        ):
            response = client.get("/api/v1/learning/workflow-patterns")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training status (route 1) -- special assertion: no longer returns status:ok
# ---------------------------------------------------------------------------


class TestTrainingStatus:
    """GET /api/v1/training/status -- all-subsystem failure must return 503, not status:ok."""

    def test_training_status_subsystem_failure_returns_503_not_ok(
        self, client: object, training_api_mod: object
    ) -> None:
        """Prove status route returns 503, not {status:ok,phase:unknown}, when all subsystems fail.

        This is the key regression guard: the old implementation swallowed all
        subsystem errors and returned a 200 with phase='unknown', giving callers
        no signal that the training system was down.

        Uses ``patch.object`` on ``training_api_mod`` (the live module object
        bound at app-creation time) rather than a string-based patch, because
        the autouse ``_isolate_vetinari_modules`` fixture wipes vetinari modules
        from sys.modules before each test.  String patches on
        ``vetinari.web.litestar_training_api._get_scheduler`` would resolve a
        fresh module object that the route handlers do not reference.
        """
        with (
            patch(
                "vetinari.training.pipeline.TrainingPipeline",
                side_effect=RuntimeError("pipeline unavailable"),
            ),
            patch(
                "vetinari.training.curriculum.TrainingCurriculum",
                side_effect=RuntimeError("curriculum unavailable"),
            ),
            patch.object(training_api_mod, "_get_scheduler", return_value=None),
            patch.object(training_api_mod, "_scheduler_singleton", None),
        ):
            response = client.get("/api/v1/training/status")

        assert response.status_code == 503, (
            f"Expected 503 when all subsystems fail, got {response.status_code}. Body: {response.text[:400]}"
        )
        body = response.json()
        assert body.get("status") != "ok", (
            f"Training status must not return status='ok' when all subsystems fail: {body}"
        )

    def test_partial_subsystem_available_returns_200(self, client: object, training_api_mod: object) -> None:
        """When at least one subsystem responds, status route should not return 503.

        Verifies that the 503 gate is only triggered when ALL subsystems fail,
        not when one or two are unavailable but the system is partially up.
        """
        mock_pipeline_reqs = {"ready_for_training": False, "libraries": {}}
        with (
            patch(
                "vetinari.training.pipeline.TrainingPipeline",
            ) as mock_pipeline_cls,
            patch(
                "vetinari.training.curriculum.TrainingCurriculum",
                side_effect=RuntimeError("curriculum unavailable"),
            ),
            patch.object(training_api_mod, "_get_scheduler", return_value=None),
        ):
            mock_pipeline_cls.return_value.check_requirements.return_value = mock_pipeline_reqs
            response = client.get("/api/v1/training/status")

        assert response.status_code == 200, (
            f"Expected 200 when pipeline subsystem responds, got {response.status_code}. Body: {response.text[:400]}"
        )


# ---------------------------------------------------------------------------
# Training data stats (route 2)
# ---------------------------------------------------------------------------


class TestTrainingDataStats:
    """GET /api/v1/training/data/stats -- both-subsystem failure must return 503."""

    def test_both_subsystems_unavailable_returns_503(self, client: object) -> None:
        """Patch seeder and collector to raise; endpoint must return 503."""
        with (
            patch(
                "vetinari.training.data_seeder.get_training_data_seeder",
                side_effect=RuntimeError("seeder down"),
            ),
            patch(
                "vetinari.learning.training_data.get_training_collector",
                side_effect=RuntimeError("collector down"),
            ),
        ):
            response = client.get("/api/v1/training/data/stats")
        _assert_503_error(response)

    def test_both_unavailable_not_status_ok(self, client: object) -> None:
        """Response must not be status:ok when both seeder and collector fail."""
        with (
            patch(
                "vetinari.training.data_seeder.get_training_data_seeder",
                side_effect=RuntimeError("seeder gone"),
            ),
            patch(
                "vetinari.learning.training_data.get_training_collector",
                side_effect=RuntimeError("collector gone"),
            ),
        ):
            response = client.get("/api/v1/training/data/stats")
        if response.status_code == 200:
            body = response.json()
            assert body.get("status") != "ok", (
                f"data/stats must not return status='ok' when both subsystems fail: {body}"
            )


# ---------------------------------------------------------------------------
# Training data browse (route 3)
# ---------------------------------------------------------------------------


class TestTrainingDataBrowse:
    """GET /api/v1/training/data/browse -- collector failure must return 503."""

    def test_collector_unavailable_returns_503(self, client: object) -> None:
        """Patch get_training_collector to raise; endpoint must return 503."""
        with patch(
            "vetinari.learning.training_collector.get_training_collector",
            side_effect=RuntimeError("collector unavailable"),
        ):
            response = client.get("/api/v1/training/data/browse")
        _assert_503_error(response)

    def test_collector_unavailable_not_empty_200(self, client: object) -> None:
        """Response must not be items:[] / total:0 masking a real failure."""
        with patch(
            "vetinari.learning.training_collector.get_training_collector",
            side_effect=RuntimeError("collector gone"),
        ):
            response = client.get("/api/v1/training/data/browse")
        assert response.status_code != 200, (
            "data/browse must not return empty 200 when collector is unavailable -- "
            "this hides the failure behind an empty result set"
        )


# ---------------------------------------------------------------------------
# Automation rules (route 4)
# ---------------------------------------------------------------------------


class TestTrainingAutomationRules:
    """GET /api/v1/training/automation/rules -- load failure must return 503."""

    def test_load_failure_returns_503(self, client: object, training_experiments_mod: object) -> None:
        """Patch _get_automation_rules to raise; endpoint must return 503.

        Uses ``patch.object`` on the module reference captured at app-creation
        time (``training_experiments_mod``) rather than a string-based patch.
        The string-based approach fails here because the autouse
        ``_isolate_vetinari_modules`` fixture wipes all vetinari modules from
        ``sys.modules`` before each test, causing a string patch to resolve a
        different module object than the one the route handlers reference.
        """
        with patch.object(
            training_experiments_mod,
            "_get_automation_rules",
            side_effect=OSError("disk read failed"),
        ):
            response = client.get("/api/v1/training/automation/rules")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training curriculum (route 5)
# ---------------------------------------------------------------------------


class TestTrainingCurriculum:
    """GET /api/v1/training/curriculum -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_training_curriculum to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.curriculum.get_training_curriculum",
            side_effect=RuntimeError("curriculum down"),
        ):
            response = client.get("/api/v1/training/curriculum")
        _assert_503_error(response)

    def test_get_status_raises_returns_503(self, client: object) -> None:
        """When get_status() raises after import, endpoint must return 503."""
        mock_curriculum = MagicMock()
        mock_curriculum.get_status.side_effect = RuntimeError("status table gone")
        with patch(
            "vetinari.training.curriculum.get_training_curriculum",
            return_value=mock_curriculum,
        ):
            response = client.get("/api/v1/training/curriculum")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training curriculum/next (route 6)
# ---------------------------------------------------------------------------


class TestTrainingCurriculumNext:
    """GET /api/v1/training/curriculum/next -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_training_curriculum to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.curriculum.get_training_curriculum",
            side_effect=RuntimeError("curriculum unavailable"),
        ):
            response = client.get("/api/v1/training/curriculum/next")
        _assert_503_error(response)

    def test_next_activity_raises_returns_503(self, client: object) -> None:
        """When next_activity() raises after import, endpoint must return 503."""
        mock_curriculum = MagicMock()
        mock_curriculum.next_activity.side_effect = RuntimeError("no activities")
        with patch(
            "vetinari.training.curriculum.get_training_curriculum",
            return_value=mock_curriculum,
        ):
            response = client.get("/api/v1/training/curriculum/next")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training history (route 7)
# ---------------------------------------------------------------------------


class TestTrainingHistory:
    """GET /api/v1/training/history -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_agent_trainer to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.agent_trainer.get_agent_trainer",
            side_effect=RuntimeError("agent trainer down"),
        ):
            response = client.get("/api/v1/training/history")
        _assert_503_error(response)

    def test_get_stats_raises_returns_503(self, client: object) -> None:
        """When get_stats() raises after import, endpoint must return 503."""
        mock_trainer = MagicMock()
        mock_trainer.get_stats.side_effect = RuntimeError("stats unavailable")
        with patch(
            "vetinari.training.agent_trainer.get_agent_trainer",
            return_value=mock_trainer,
        ):
            response = client.get("/api/v1/training/history")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training jobs (route 8)
# ---------------------------------------------------------------------------


class TestTrainingJobs:
    """GET /api/v1/training/jobs -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_training_manager to raise; endpoint must return 503."""
        with patch(
            "vetinari.learning.training_manager.get_training_manager",
            side_effect=RuntimeError("training manager down"),
        ):
            response = client.get("/api/v1/training/jobs")
        _assert_503_error(response)

    def test_list_jobs_raises_returns_503(self, client: object) -> None:
        """When list_jobs() raises after import, endpoint must return 503."""
        mock_manager = MagicMock()
        mock_manager.list_jobs.side_effect = RuntimeError("jobs table gone")
        with patch(
            "vetinari.learning.training_manager.get_training_manager",
            return_value=mock_manager,
        ):
            response = client.get("/api/v1/training/jobs")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training summary (route 9) -- special assertion: no longer returns sentinel
# ---------------------------------------------------------------------------


class TestTrainingSummary:
    """GET /api/v1/training/summary -- subsystem failure must return 503, not sentinel."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch training subsystems to raise; endpoint must return 503."""
        with (
            patch(
                "vetinari.learning.training_data.get_training_collector",
                side_effect=RuntimeError("collector down"),
            ),
            patch(
                "vetinari.training.idle_scheduler.get_idle_detector",
                side_effect=RuntimeError("idle detector down"),
            ),
            patch(
                "vetinari.training.curriculum.TrainingCurriculum",
                side_effect=RuntimeError("curriculum down"),
            ),
        ):
            response = client.get("/api/v1/training/summary")
        _assert_503_error(response)

    def test_subsystem_failure_does_not_return_idle_sentinel(self, client: object) -> None:
        """Response must not be {status:idle,current_job:null} masking a real failure."""
        with (
            patch(
                "vetinari.learning.training_data.get_training_collector",
                side_effect=RuntimeError("collector unavailable"),
            ),
            patch(
                "vetinari.training.idle_scheduler.get_idle_detector",
                side_effect=RuntimeError("idle detector unavailable"),
            ),
            patch(
                "vetinari.training.curriculum.TrainingCurriculum",
                side_effect=RuntimeError("curriculum unavailable"),
            ),
        ):
            response = client.get("/api/v1/training/summary")
        if response.status_code == 200:
            body = response.json()
            assert body.get("status") != "idle", (
                "training/summary must not return status='idle' as a sentinel "
                "when all subsystems fail -- this hides failures as normal idle state"
            )


# ---------------------------------------------------------------------------
# Training quality (route 10) -- special assertion: no longer returns sentinel
# ---------------------------------------------------------------------------


class TestTrainingQuality:
    """GET /api/v1/training/quality -- subsystem failure must return 503, not sentinel."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_training_quality_gate to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.quality_gate.get_training_quality_gate",
            side_effect=RuntimeError("quality gate down"),
        ):
            response = client.get("/api/v1/training/quality")
        _assert_503_error(response)

    def test_subsystem_failure_does_not_return_no_data_sentinel(self, client: object) -> None:
        """Response must not be {baseline_quality:0.0,decision:no_data} on failure."""
        with patch(
            "vetinari.training.quality_gate.get_training_quality_gate",
            side_effect=RuntimeError("quality gate unavailable"),
        ):
            response = client.get("/api/v1/training/quality")
        if response.status_code == 200:
            body = response.json()
            assert body.get("decision") != "no_data", (
                "training/quality must not return decision='no_data' as a sentinel "
                "when the quality gate is unavailable -- callers cannot distinguish "
                "a genuine 'no history yet' from a subsystem failure"
            )


# ---------------------------------------------------------------------------
# Training models (route 11)
# ---------------------------------------------------------------------------


class TestTrainingModels:
    """GET /api/v1/training/models -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_adapter_registry to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.adapter_registry.get_adapter_registry",
            side_effect=RuntimeError("adapter registry down"),
        ):
            response = client.get("/api/v1/training/models")
        _assert_503_error(response)

    def test_list_all_raises_returns_503(self, client: object) -> None:
        """When list_all() raises after import, endpoint must return 503."""
        mock_registry = MagicMock()
        mock_registry.list_all.side_effect = RuntimeError("registry table gone")
        with patch(
            "vetinari.training.adapter_registry.get_adapter_registry",
            return_value=mock_registry,
        ):
            response = client.get("/api/v1/training/models")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training adapters (route 12)
# ---------------------------------------------------------------------------


class TestTrainingAdapters:
    """GET /api/v1/training/adapters -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_adapter_registry to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.adapter_registry.get_adapter_registry",
            side_effect=RuntimeError("adapter registry down"),
        ):
            response = client.get("/api/v1/training/adapters?task_type=coding")
        _assert_503_error(response)

    def test_missing_task_type_returns_400_not_503(self, client: object) -> None:
        """Missing task_type query param is a client error (400), not server error."""
        response = client.get("/api/v1/training/adapters")
        assert response.status_code == 400, f"Expected 400 for missing task_type, got {response.status_code}"


# ---------------------------------------------------------------------------
# Training adapters/deployed (route 13)
# ---------------------------------------------------------------------------


class TestTrainingAdaptersDeployed:
    """GET /api/v1/training/adapters/deployed -- subsystem failure must return 503."""

    def test_subsystem_failure_returns_503(self, client: object) -> None:
        """Patch get_adapter_registry to raise; endpoint must return 503."""
        with patch(
            "vetinari.training.adapter_registry.get_adapter_registry",
            side_effect=RuntimeError("adapter registry down"),
        ):
            response = client.get("/api/v1/training/adapters/deployed")
        _assert_503_error(response)

    def test_list_deployed_raises_returns_503(self, client: object) -> None:
        """When list_deployed() raises after import, endpoint must return 503."""
        mock_registry = MagicMock()
        mock_registry.list_deployed.side_effect = RuntimeError("deployed table gone")
        with patch(
            "vetinari.training.adapter_registry.get_adapter_registry",
            return_value=mock_registry,
        ):
            response = client.get("/api/v1/training/adapters/deployed")
        _assert_503_error(response)


# ---------------------------------------------------------------------------
# Training experiments/{id} (route 14) -- special: 503 vs 404 distinction
# ---------------------------------------------------------------------------


class TestTrainingExperiment:
    """GET /api/v1/training/experiments/{id} -- trainer unavailable must return 503 not 404."""

    def test_training_experiment_unavailable_returns_503_not_404(self, client: object) -> None:
        """Prove trainer unavailability returns 503, not 404.

        The old implementation raised FileNotFoundError even when the trainer
        could not be imported, mapping to a 404.  Callers cannot distinguish
        "trainer is down" from "experiment does not exist".
        """
        with patch(
            "vetinari.training.agent_trainer.get_agent_trainer",
            side_effect=RuntimeError("agent trainer module unavailable"),
        ):
            response = client.get("/api/v1/training/experiments/exp-999")

        assert response.status_code == 503, (
            f"Expected 503 when trainer is unavailable, got {response.status_code}. Body: {response.text[:400]}"
        )
        assert response.status_code != 404, (
            "training/experiments/{id} must not return 404 when the trainer subsystem "
            "is unavailable -- 404 implies the trainer is up but the experiment was "
            "not found, which is misleading when the whole subsystem is down"
        )

    def test_experiment_not_found_returns_404_when_trainer_available(self, client: object) -> None:
        """When trainer is reachable but experiment ID is absent, return 404.

        Validates the happy-path distinction: trainer up + missing ID = 404,
        not 503.
        """
        mock_trainer = MagicMock()
        mock_trainer.get_stats.return_value = {}
        mock_trainer.get_training_priority.return_value = []
        with patch(
            "vetinari.training.agent_trainer.get_agent_trainer",
            return_value=mock_trainer,
        ):
            response = client.get("/api/v1/training/experiments/no-such-experiment")

        assert response.status_code == 404, (
            f"Expected 404 for unknown experiment when trainer is available, "
            f"got {response.status_code}. Body: {response.text[:400]}"
        )
