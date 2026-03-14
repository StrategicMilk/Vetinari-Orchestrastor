"""Orchestrator — LEGACY manifest-based task execution engine.

.. deprecated::
    This module handles YAML-manifest execution. For goal-based orchestration,
    use ``vetinari.orchestration.two_layer.TwoLayerOrchestrator``.
    A future release will unify all orchestrators into a single module.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

warnings.warn(
    "vetinari.orchestrator is deprecated. Use vetinari.orchestration.two_layer.TwoLayerOrchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from vetinari.adapter_manager import get_adapter_manager  # noqa: E402
from vetinari.builder import Builder  # noqa: E402

# Phase 2: OpenCode Integration
from vetinari.execution_context import (  # noqa: E402
    ExecutionMode,
    ToolPermission,
    get_context_manager,
)
from vetinari.executor import TaskExecutor  # noqa: E402
from vetinari.lmstudio_adapter import LMStudioAdapter  # noqa: E402
from vetinari.model_pool import ModelPool  # noqa: E402
from vetinari.scheduler import Scheduler  # noqa: E402
from vetinari.tool_interface import get_tool_registry  # noqa: E402
from vetinari.upgrader import Upgrader  # noqa: E402
from vetinari.validator import Validator  # noqa: E402
from vetinari.verification import get_verifier_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

# Plan Mode integration
PLAN_MODE_ENABLE = os.environ.get("PLAN_MODE_ENABLE", "true").lower() in ("1", "true", "yes")
PLAN_MODE_DEFAULT = os.environ.get("PLAN_MODE_DEFAULT", "true").lower() in ("1", "true", "yes")

# Phase 2: Execution mode from environment
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", "execution").lower()
VERIFICATION_LEVEL = os.environ.get("VERIFICATION_LEVEL", "standard").lower()


class Orchestrator:
    """Central coordinator that drives the agent execution pipeline."""
    def __init__(
        self,
        manifest_path: str,
        host: str | None = None,
        api_token: str | None = None,
        max_concurrent: int = 4,
        execution_mode: str | None = None,
    ):
        # Resolve host from env if not provided
        if host is None:
            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
        # Auto-read API token from environment when not explicitly provided
        if api_token is None:
            api_token = os.environ.get("LM_STUDIO_API_TOKEN", "")

        # Make manifest_path absolute if it's relative
        manifest_path_obj = Path(manifest_path)
        if not manifest_path_obj.is_absolute():
            # Assume it's relative to the project root (parent of vetinari package)
            project_root = Path(__file__).resolve().parents[1]
            self.manifest_path = project_root / manifest_path_obj
        else:
            self.manifest_path = manifest_path_obj
        self.project_root = self.manifest_path.parent.parent
        self.config = self._load_manifest()
        self.config["project_root"] = str(self.project_root)
        self.api_token = api_token
        self.max_concurrent = max_concurrent
        self.adapter = LMStudioAdapter(host=host, api_token=api_token)
        self.model_pool = ModelPool(self.config, host, api_token=api_token)
        self.scheduler = Scheduler(self.config, max_concurrent=max_concurrent)
        self.validator = Validator()
        self.executor = TaskExecutor(self.adapter, self.validator, self.config)
        self.upgrader = Upgrader(self.config)
        self.builder = Builder(self.config)

        # Phase 2: Initialize OpenCode integration
        self.context_manager = get_context_manager()
        self.adapter_manager = get_adapter_manager()
        self.tool_registry = get_tool_registry()
        self.verifier_pipeline = get_verifier_pipeline()

        # Set execution mode
        mode_str = execution_mode or EXECUTION_MODE
        try:
            self.execution_mode = ExecutionMode(mode_str)
            self.context_manager.switch_mode(self.execution_mode)
            logger.info("Execution mode set to: %s", self.execution_mode.value)
        except ValueError:
            logger.warning("Invalid execution mode: %s, using default EXECUTION", mode_str)
            self.execution_mode = ExecutionMode.EXECUTION

        # Plan Mode initialization
        self.plan_mode_enabled = PLAN_MODE_ENABLE and PLAN_MODE_DEFAULT
        self.plan_engine = None
        if self.plan_mode_enabled:
            try:
                from vetinari.plan_mode import get_plan_engine

                self.plan_engine = get_plan_engine()
                logger.info("Plan Mode initialized successfully")
            except Exception as e:
                logger.warning("Plan Mode initialization failed: %s. Continuing without Plan Mode.", e)
                self.plan_mode_enabled = False

        # Phase 2+: Initialize agent system and wire context
        self._agent_context = {
            "adapter_manager": self.adapter_manager,
            "tool_registry": self.tool_registry,
        }
        # Lazily initialize web search tool into agent context
        try:
            from vetinari.tools.web_search_tool import get_search_tool

            self._agent_context["web_search"] = get_search_tool()
            logger.info("Web search tool registered in agent context")
        except Exception as e:
            logger.warning("Web search tool unavailable: %s", e)

        # Register LM Studio as a provider in the adapter manager
        self._register_lmstudio_adapter(host, api_token)

        # Phase 5: Register default SLO targets for analytics
        self._register_default_slos()

        logger.info("Vetinari orchestrator initialized with Phase 2 OpenCode integration.")

    def _register_default_slos(self) -> None:
        """Register default SLO targets for the analytics SLA tracker."""
        try:
            from vetinari.analytics.sla import SLOTarget, SLOType, get_sla_tracker

            tracker = get_sla_tracker()
            defaults = [
                SLOTarget(
                    name="latency-p95",
                    slo_type=SLOType.LATENCY_P95,
                    budget=2000.0,
                    window_seconds=3600,
                    description="95th percentile latency under 2s",
                ),
                SLOTarget(
                    name="success-rate",
                    slo_type=SLOType.SUCCESS_RATE,
                    budget=95.0,
                    window_seconds=3600,
                    description="At least 95% successful requests",
                ),
                SLOTarget(
                    name="error-rate",
                    slo_type=SLOType.ERROR_RATE,
                    budget=5.0,
                    window_seconds=3600,
                    description="Error rate below 5%",
                ),
                SLOTarget(
                    name="approval-rate",
                    slo_type=SLOType.APPROVAL_RATE,
                    budget=80.0,
                    window_seconds=86400,
                    description="Plan approval rate above 80%",
                ),
            ]
            for slo in defaults:
                tracker.register_slo(slo)
            logger.info("Registered %d default SLO targets", len(defaults))
        except Exception as e:
            logger.warning("Failed to register default SLOs: %s", e)

    def _register_lmstudio_adapter(self, host: str, api_token: str | None = None):
        """Register LM Studio as a provider in the AdapterManager.

        This bridges the legacy LMStudioAdapter with the new adapter system,
        ensuring that AdapterManager.infer() works via LM Studio.
        """
        try:
            from vetinari.adapters.base import ProviderConfig, ProviderType

            config = ProviderConfig(
                provider_type=ProviderType.LM_STUDIO,
                name="lmstudio",
                endpoint=host,
                api_key=api_token or "",
                timeout_seconds=120,
            )
            self.adapter_manager.register_provider(config, "lmstudio")
            logger.info("LM Studio registered in AdapterManager at %s", host)
        except Exception as e:
            logger.warning("Could not register LM Studio in AdapterManager: %s", e)

    def _initialize_agent(self, agent_instance) -> None:
        """Initialize a single agent with the shared context."""
        try:
            agent_instance.initialize(self._agent_context)
        except Exception as e:
            logger.warning("Agent initialization failed for %s: %s", agent_instance, e)

    def update_settings(self, host: str | None = None, api_token: str | None = None):
        """Update settings including host and API token.

        Args:
            host: The host.
            api_token: The api token.
        """
        if host:
            self.adapter.host = host.rstrip("/")
            self.model_pool.host = host.rstrip("/")

        if api_token is not None:
            self.api_token = api_token
            self.adapter.set_api_token(api_token)
            self.model_pool.set_api_token(api_token)

        logger.info(f"Settings updated: host={host}, api_token={'***' if api_token else 'None'}")  # noqa: VET051 — complex expression

    def _load_manifest(self):
        with open(self.manifest_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_all(self):
        """Run all for the current context.

        Returns:
            The computed result.
        """
        logger.info("Starting full workflow: plan -> allocate -> execute -> validate -> build.")

        # 0) Plan Mode: Generate plan before execution (if enabled)
        if self.plan_mode_enabled and self.plan_engine:
            try:
                goal = self.config.get("goal", "Execute tasks from manifest")
                constraints = self.config.get("constraints", "")

                from vetinari.plan_types import PlanGenerationRequest

                req = PlanGenerationRequest(
                    goal=goal,
                    constraints=constraints,
                    plan_depth_cap=int(os.environ.get("PLAN_DEPTH_CAP", 16)),
                    max_candidates=int(os.environ.get("PLAN_MAX_CANDIDATES", 3)),
                    dry_run=os.environ.get("DRY_RUN_ENABLED", "false").lower() in ("1", "true", "yes"),
                )

                plan = self.plan_engine.generate_plan(req)
                self.config["_plan_id"] = plan.plan_id
                self.config["_plan_risk_score"] = plan.risk_score

                if plan.auto_approved:
                    logger.info("Plan %s auto-approved (risk_score=%.2f)", plan.risk_score, plan.plan_id)
                else:
                    logger.info("Plan %s generated (risk_score=%.2f), awaiting approval", plan.risk_score, plan.plan_id)

            except Exception as e:
                logger.warning("Plan Mode failed: %s. Continuing without plan generation.", e)

        # Phase 5: Apply AutoTuner configuration before execution
        try:
            from vetinari.learning.auto_tuner import get_auto_tuner

            tuner_config = get_auto_tuner().get_config()
            if "max_concurrent" in tuner_config:
                self.max_concurrent = tuner_config["max_concurrent"]
                logger.info("AutoTuner applied max_concurrent=%s", self.max_concurrent)
        except Exception as e:
            logger.debug("AutoTuner config not applied: %s", e)

        # 1) Auto-discover models
        try:
            self.model_pool.discover_models()
        except Exception as e:
            logger.error("Model discovery failed: %s", e)
            # Continue with empty model pool if discovery fails

        # 2) Assign tasks to models
        self.model_pool.assign_tasks_to_models(self.config)

        # 3) Build schedule (layers for parallel execution)
        layers = self.scheduler.build_schedule_layers(self.config)

        logger.info("Built %s execution layers", len(layers))

        # 4) Execute tasks layer by layer
        completed_tasks = set()
        all_results = []

        for layer_idx, layer in enumerate(layers):
            logger.info("Executing layer %s/%s with %s tasks in parallel", layer_idx + 1, len(layers), len(layer))

            # Execute tasks in this layer concurrently
            layer_results = self._execute_layer_parallel(layer)
            all_results.extend(layer_results)

            # Mark tasks as completed
            for result in layer_results:
                if result.get("status") == "completed":
                    completed_tasks.add(result["task_id"])
                else:
                    logger.warning("Task %s did not complete successfully", result.get("task_id"))

        logger.info("All layers completed. %s tasks finished successfully.", len(completed_tasks))

        # 5) Build final artifact
        self.builder.build_final_artifact(all_results)

        return all_results

    def _execute_layer_parallel(self, layer: list) -> list:
        """Execute a layer of tasks in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=len(layer)) as executor:
            # Submit all tasks in this layer
            future_to_task = {executor.submit(self.run_task, task["id"]): task for task in layer}

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info("Task %s completed with status: %s", task["id"], result.get("status"))
                except Exception as e:
                    logger.error("Task %s failed with exception: %s", task["id"], e)
                    results.append({"status": "failed", "task_id": task["id"], "error": str(e)})

        return results

    def run_task(self, task_id: str):
        """Run task for the current context.

        Returns:
            The computed result.
        """
        logger.info("Running task %s", task_id)

        # Check permission in current execution context
        try:
            self.context_manager.enforce_permission(ToolPermission.MODEL_INFERENCE, f"task {task_id}")
        except PermissionError as e:
            logger.error("Task %s blocked by permission: %s", task_id, e)
            return {
                "status": "failed",
                "task_id": task_id,
                "error": str(e),
            }

        result = self.executor.execute_task(task_id)

        # Verify task output
        task_output = result.get("output", "")
        if task_output:
            try:
                logger.info("Running verification pipeline for task %s", task_id)
                verification_results = self.verifier_pipeline.verify(task_output)
                verification_summary = self.verifier_pipeline.get_summary(verification_results)

                # Log verification results
                logger.info("Verification summary for %s: %s", task_id, verification_summary["overall_status"])
                if verification_summary["total_issues"] > 0:
                    logger.warning("Task %s has %s verification issues", task_id, verification_summary["total_issues"])
                    if verification_summary["error_count"] > 0:
                        logger.error("Task %s has %s verification errors", task_id, verification_summary["error_count"])

                # Attach verification results to task result
                result["verification"] = verification_summary
            except Exception as e:
                logger.warning("Verification pipeline failed for task %s: %s", task_id, e)

        # Record operation in audit trail
        self.context_manager.current_context.record_operation(
            f"task_{task_id}",
            {"task_id": task_id},
            result,
        )

        if result.get("status") != "completed":
            logger.warning("Task %s did not complete successfully. See logs for details.", task_id)
        else:
            logger.info("Task %s completed.", task_id)
        return result

    def check_and_upgrade_models(self):
        """Check for and install model upgrades.

        Supports non-interactive mode via VETINARI_UPGRADE_AUTO_APPROVE environment variable.
        Phase 2: Check permissions before upgrading.
        """
        # Check permission in current execution context
        try:
            self.context_manager.enforce_permission(ToolPermission.MODEL_DISCOVERY, "model upgrade check")
        except PermissionError as e:
            logger.error("Model upgrade blocked by permission: %s", e)
            return

        # Check if running in non-interactive mode
        auto_approve = os.environ.get("VETINARI_UPGRADE_AUTO_APPROVE", "false").lower() in ("1", "true", "yes")
        is_interactive = not auto_approve and sys.stdin.isatty()

        upgrades = self.upgrader.check_for_upgrades()
        if not upgrades:
            logger.info("No upgrades available.")
            return

        for u in upgrades:
            upgrade_policy = self.config.get("upgrade_policy", {})
            require_approval = upgrade_policy.get("require_approval", True)

            if require_approval and not auto_approve:
                if is_interactive:
                    # Interactive mode: prompt user
                    try:
                        user_input = input(f"Upgrade available: {u['name']} (version {u['version']}). Install? (y/n): ")
                        if user_input.strip().lower() != "y":
                            logger.info("Upgrade skipped by user: %s", u["name"])
                            continue
                    except EOFError:
                        # No TTY available, skip this upgrade
                        logger.warning("No TTY available - skipping upgrade prompt for %s", u["name"])
                        continue
                else:
                    # Non-interactive mode: skip unless auto_approve is set
                    logger.warning(
                        f"Non-interactive mode: Skipping upgrade {u['name']} (set VETINARI_UPGRADE_AUTO_APPROVE=true to auto-approve)"
                    )
                    continue
            elif auto_approve:
                logger.info("Auto-approving upgrade: %s (VETINARI_UPGRADE_AUTO_APPROVE=true)", u["name"])

            try:
                self.upgrader.install_upgrade(u)
                logger.info("Upgrade installed: %s v%s", u["name"], u["version"])
            except Exception as e:
                logger.error("Failed to install upgrade %s: %s", u["name"], e)

        logger.info("Upgrade process complete.")

    def get_execution_status(self) -> dict:
        """Get current execution status (Phase 2 enhancement).

        Returns:
            Dictionary of results.
        """
        context_status = self.context_manager.get_status()
        adapter_status = self.adapter_manager.get_status()

        return {
            "execution_context": context_status,
            "adapters": adapter_status,
            "timestamp": time.time(),
        }


# Convenience when running as script
if __name__ == "__main__":
    Orchestrator("manifest/vetinari.yaml").run_all()
