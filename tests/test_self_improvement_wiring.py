"""Tests for self-improvement feature wiring (Dept 4.3 #36-42)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSelfRefinementWiring:
    """Tests for self-refinement wiring into pipeline."""

    def test_custom_tier_triggers_refinement(self):
        """Custom tier execution runs the real self-refinement hook."""
        from vetinari.orchestration.pipeline_events import PipelineStage
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)
        orch._emit = MagicMock()
        orch._execute_via_agent_graph_or_fallback = MagicMock(
            return_value={
                "completed": 1,
                "failed": 0,
                "task_results": {
                    "task-1": {
                        "status": "completed",
                        "output": "draft output",
                    },
                },
            },
        )
        orch._validate_stage_boundary = MagicMock(return_value=(True, []))
        orch._check_stage_constraints = MagicMock(return_value=(True, []))
        orch.is_paused = MagicMock(return_value=True)

        graph = MagicMock()
        graph.plan_id = "plan-123"
        graph.nodes = {}
        stages: dict[str, object] = {}

        refined = MagicMock(improved=True, output="refined output", rounds_used=2)
        refiner = MagicMock()
        refiner.refine.return_value = refined

        with (
            patch("vetinari.orchestration.pipeline_stages.get_event_bus", return_value=MagicMock()),
            patch("vetinari.learning.self_refinement.get_self_refiner", return_value=refiner),
        ):
            result = TwoLayerOrchestrator._run_execution_stages(
                orch,
                goal="Write a migration",
                graph=graph,
                context={"intake_tier": "custom", "mode": "coding", "model_id": "test-model"},
                stages=stages,
                start_time=0.0,
                _corr_ctx=None,
                _pipeline_span=None,
                task_handler=None,
                project_id=None,
                _intake_tier=None,
                _intake_features=None,
            )

        refiner.refine.assert_called_once_with(
            task_description="Write a migration",
            initial_output="draft output",
            task_type="coding",
            model_id="test-model",
            importance=0.8,
        )
        assert stages["self_refinement"] == {"applied": True, "tier": "custom"}
        assert stages["execution"]["task_results"]["task-1"]["output"] == "refined output"
        assert "Andon signal after execution stage" in result["error"]
        assert any(call.args[0] == PipelineStage.REFINEMENT for call in orch._emit.call_args_list)

    def test_self_refinement_module_exists(self):
        """self_refinement module is importable with refine() method."""
        from vetinari.learning.self_refinement import SelfRefinementLoop

        assert hasattr(SelfRefinementLoop, "refine")


class TestVRAMFiltering:
    """Tests for VRAM-aware model routing."""

    def test_select_model_uses_vram_manager_budget(self):
        """DynamicModelRouter filters candidates using live free VRAM when available."""
        from tests.factories import make_router_model_info
        from vetinari.models.dynamic_model_router import DynamicModelRouter, TaskType

        router = DynamicModelRouter(max_memory_gb=64)
        router.register_model(make_router_model_info("fit-model", code_gen=True, memory_gb=8.0))
        router.register_model(make_router_model_info("too-big", code_gen=True, memory_gb=40.0))

        vram_manager = MagicMock()
        vram_manager.get_free_vram_gb.return_value = 12.0

        with patch("vetinari.models.vram_manager.get_vram_manager", return_value=vram_manager):
            result = router.select_model(TaskType.CODE)

        assert result is not None
        assert result.model.id == "fit-model"

    def test_select_model_vram_fallback(self):
        """When VRAMManager unavailable, falls back to max_memory_gb."""
        from vetinari.models.dynamic_model_router import DynamicModelRouter

        router = DynamicModelRouter.__new__(DynamicModelRouter)
        router.models = {}
        router.max_memory_gb = 16.0
        router.max_latency_ms = 5000
        router.cost_weight = 0.0
        router.latency_weight = 0.0
        router._history = {}
        router._task_defaults = {}

        # With no models, should return None (no models available)
        # The import is lazy inside select_model, so patch at source
        with patch("vetinari.models.vram_manager.get_vram_manager", side_effect=ImportError):
            from vetinari.models.dynamic_model_router import TaskType

            result = router.select_model(TaskType.GENERAL)
            # None because no models registered
            assert result is None


class TestConfigurablePaths:
    """Tests for VETINARI_STATE_DIR configurable paths."""

    def test_prompt_evolver_uses_env_var(self, tmp_path):
        """PromptEvolver._get_state_path() respects VETINARI_STATE_DIR."""
        from vetinari.learning.prompt_evolver import PromptEvolver

        with patch.dict(os.environ, {"VETINARI_STATE_DIR": str(tmp_path)}):
            path = PromptEvolver._get_state_path()
            assert str(tmp_path) in str(path)
            assert path.name == "prompt_variants.json"

    def test_prompt_evolver_fallback(self):
        """PromptEvolver falls back to .vetinari/ when env var not set."""
        from vetinari.learning.prompt_evolver import PromptEvolver

        with patch.dict(os.environ, {}, clear=True):
            # Remove VETINARI_STATE_DIR if present
            os.environ.pop("VETINARI_STATE_DIR", None)
            path = PromptEvolver._get_state_path()
            assert ".vetinari" in str(path)

    def test_workflow_learner_uses_env_var(self, tmp_path):
        """WorkflowLearner._get_state_path() respects VETINARI_STATE_DIR."""
        from vetinari.learning.workflow_learner import WorkflowLearner

        with patch.dict(os.environ, {"VETINARI_STATE_DIR": str(tmp_path)}):
            path = WorkflowLearner._get_state_path()
            assert str(tmp_path) in str(path)
            assert path.name == "workflow_patterns.json"

    def test_workflow_learner_fallback(self):
        """WorkflowLearner falls back to .vetinari/ when env var not set."""
        from vetinari.learning.workflow_learner import WorkflowLearner

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VETINARI_STATE_DIR", None)
            path = WorkflowLearner._get_state_path()
            assert ".vetinari" in str(path)

    def test_training_pipeline_uses_vetinari_models_dir_env(self):
        """Training pipeline resolves its model directory from the env-backed constants chain."""
        import vetinari.constants as constants
        import vetinari.training.pipeline as pipeline_module

        original_default = constants.DEFAULT_MODELS_DIR
        try:
            with patch.dict(os.environ, {"VETINARI_MODELS_DIR": str(Path("C:/tmp/custom-models"))}):
                importlib.reload(constants)
                reloaded_pipeline = importlib.reload(pipeline_module)
                assert Path("C:/tmp/custom-models") == reloaded_pipeline._MODELS_DIR
        finally:
            with patch.dict(os.environ, {}, clear=False):
                if "VETINARI_MODELS_DIR" in os.environ:
                    del os.environ["VETINARI_MODELS_DIR"]
            importlib.reload(constants)
            importlib.reload(pipeline_module)
            assert original_default == constants.DEFAULT_MODELS_DIR

    def test_model_selector_persistence_uses_vetinari_state_dir(self, tmp_path):
        """Thompson persistence helper resolves state_dir from VETINARI_STATE_DIR."""
        import vetinari.learning.model_selector_persistence as persistence

        with patch.dict(os.environ, {"VETINARI_STATE_DIR": str(tmp_path)}):
            assert persistence.get_state_dir(selector=object()) == str(tmp_path)
