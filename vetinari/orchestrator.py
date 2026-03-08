import os
import logging
import sys
import time
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

from vetinari.constants import (
    DEFAULT_LM_STUDIO_HOST,
    DEFAULT_API_TOKEN,
    DEFAULT_PLAN_DEPTH_CAP,
    DRY_RUN_ENABLED,
    EXECUTION_MODE as _DEFAULT_EXECUTION_MODE,
    PLAN_MAX_CANDIDATES,
    PLAN_MODE_DEFAULT,
    PLAN_MODE_ENABLE,
    UPGRADE_AUTO_APPROVE,
    VERIFICATION_LEVEL as _DEFAULT_VERIFICATION_LEVEL,
)
from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.model_pool import ModelPool
from vetinari.scheduler import Scheduler
from vetinari.executor import TaskExecutor
from vetinari.upgrader import Upgrader
from vetinari.validator import Validator
from vetinari.builder import Builder

# Phase 2: OpenCode Integration
from vetinari.execution_context import (
    get_context_manager,
    ExecutionMode,
    ToolPermission,
)
from vetinari.adapter_manager import get_adapter_manager
from vetinari.tool_interface import get_tool_registry
from vetinari.verification import get_verifier_pipeline, VerificationLevel


class Orchestrator:
    def __init__(self, manifest_path: str, host: str = None, api_token: str = None, max_concurrent: int = 4, execution_mode: str = None):
        # Resolve host from env if not provided
        if host is None:
            host = DEFAULT_LM_STUDIO_HOST
        # Auto-read API token from environment when not explicitly provided
        if api_token is None:
            api_token = DEFAULT_API_TOKEN

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
        mode_str = execution_mode or _DEFAULT_EXECUTION_MODE
        try:
            self.execution_mode = ExecutionMode(mode_str)
            self.context_manager.switch_mode(self.execution_mode)
            logger.info(f"Execution mode set to: {self.execution_mode.value}")
        except ValueError:
            logger.warning(f"Invalid execution mode: {mode_str}, using default EXECUTION")
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
                logger.warning(f"Plan Mode initialization failed: {e}. Continuing without Plan Mode.")
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
            logger.warning(f"Web search tool unavailable: {e}")

        # Register LM Studio as a provider in the adapter manager
        self._register_lmstudio_adapter(host, api_token)

        logger.info("Vetinari orchestrator initialized with Phase 2 OpenCode integration.")

    def _register_lmstudio_adapter(self, host: str, api_token: Optional[str] = None):
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
            logger.info(f"LM Studio registered in AdapterManager at {host}")
        except Exception as e:
            logger.warning(f"Could not register LM Studio in AdapterManager: {e}")

    def _initialize_agent(self, agent_instance) -> None:
        """Initialize a single agent with the shared context."""
        try:
            agent_instance.initialize(self._agent_context)
        except Exception as e:
            logger.warning(f"Agent initialization failed for {agent_instance}: {e}")

    def update_settings(self, host: str = None, api_token: str = None):
        """Update settings including host and API token."""
        if host:
            self.adapter.host = host.rstrip("/")
            self.model_pool.host = host.rstrip("/")
        
        if api_token is not None:
            self.api_token = api_token
            self.adapter.set_api_token(api_token)
            self.model_pool.set_api_token(api_token)
        
        logger.info(f"Settings updated: host={host}, api_token={'***' if api_token else 'None'}")

    def _load_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_all(self):
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
                    plan_depth_cap=DEFAULT_PLAN_DEPTH_CAP,
                    max_candidates=PLAN_MAX_CANDIDATES,
                    dry_run=DRY_RUN_ENABLED,
                )
                
                plan = self.plan_engine.generate_plan(req)
                self.config["_plan_id"] = plan.plan_id
                self.config["_plan_risk_score"] = plan.risk_score
                
                if plan.auto_approved:
                    logger.info(f"Plan {plan.plan_id} auto-approved (risk_score={plan.risk_score:.2f})")
                else:
                    logger.info(f"Plan {plan.plan_id} generated (risk_score={plan.risk_score:.2f}), awaiting approval")
                    
            except Exception as e:
                logger.warning(f"Plan Mode failed: {e}. Continuing without plan generation.")
        
        # 1) Auto-discover models
        try:
            self.model_pool.discover_models()
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            # Continue with empty model pool if discovery fails
        
        # 2) Assign tasks to models
        self.model_pool.assign_tasks_to_models(self.config)
        
        # 3) Build schedule (layers for parallel execution)
        layers = self.scheduler.build_schedule_layers(self.config)
        
        logger.info(f"Built {len(layers)} execution layers")
        
        # 4) Execute tasks layer by layer
        completed_tasks = set()
        all_results = []
        
        for layer_idx, layer in enumerate(layers):
            logger.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks in parallel")
            
            # Execute tasks in this layer concurrently
            layer_results = self._execute_layer_parallel(layer)
            all_results.extend(layer_results)
            
            # Mark tasks as completed
            for result in layer_results:
                if result.get("status") == "completed":
                    completed_tasks.add(result["task_id"])
                else:
                    logger.warning(f"Task {result.get('task_id')} did not complete successfully")
        
        logger.info(f"All layers completed. {len(completed_tasks)} tasks finished successfully.")
        
        # 5) Build final artifact
        self.builder.build_final_artifact(all_results)
        
        return all_results

    def _execute_layer_parallel(self, layer: list) -> list:
        """Execute a layer of tasks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(layer)) as executor:
            # Submit all tasks in this layer
            future_to_task = {
                executor.submit(self.run_task, task["id"]): task 
                for task in layer
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Task {task['id']} completed with status: {result.get('status')}")
                except Exception as e:
                    logger.error(f"Task {task['id']} failed with exception: {e}")
                    results.append({
                        "status": "failed",
                        "task_id": task["id"],
                        "error": str(e)
                    })
        
        return results

    def run_task(self, task_id: str):
        logger.info(f"Running task {task_id}")
        
        # Check permission in current execution context
        try:
            self.context_manager.enforce_permission(
                ToolPermission.MODEL_INFERENCE,
                f"task {task_id}"
            )
        except PermissionError as e:
            logger.error(f"Task {task_id} blocked by permission: {e}")
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
                logger.info(f"Running verification pipeline for task {task_id}")
                verification_results = self.verifier_pipeline.verify(task_output)
                verification_summary = self.verifier_pipeline.get_summary(verification_results)
                
                # Log verification results
                logger.info(f"Verification summary for {task_id}: {verification_summary['overall_status']}")
                if verification_summary['total_issues'] > 0:
                    logger.warning(f"Task {task_id} has {verification_summary['total_issues']} verification issues")
                    if verification_summary['error_count'] > 0:
                        logger.error(f"Task {task_id} has {verification_summary['error_count']} verification errors")
                
                # Attach verification results to task result
                result["verification"] = verification_summary
            except Exception as e:
                logger.warning(f"Verification pipeline failed for task {task_id}: {e}")
        
        # Record operation in audit trail
        self.context_manager.current_context.record_operation(
            f"task_{task_id}",
            {"task_id": task_id},
            result,
        )
        
        if result.get("status") != "completed":
            logger.warning(f"Task {task_id} did not complete successfully. See logs for details.")
        else:
            logger.info(f"Task {task_id} completed.")
        return result

    def check_and_upgrade_models(self):
        """
        Check for and install model upgrades.
        Supports non-interactive mode via VETINARI_UPGRADE_AUTO_APPROVE environment variable.
        Phase 2: Check permissions before upgrading.
        """
        # Check permission in current execution context
        try:
            self.context_manager.enforce_permission(
                ToolPermission.MODEL_DISCOVERY,
                "model upgrade check"
            )
        except PermissionError as e:
            logger.error(f"Model upgrade blocked by permission: {e}")
            return
        
        # Check if running in non-interactive mode
        auto_approve = UPGRADE_AUTO_APPROVE
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
                            logger.info(f"Upgrade skipped by user: {u['name']}")
                            continue
                    except EOFError:
                        # No TTY available, skip this upgrade
                        logger.warning(f"No TTY available - skipping upgrade prompt for {u['name']}")
                        continue
                else:
                    # Non-interactive mode: skip unless auto_approve is set
                    logger.warning(f"Non-interactive mode: Skipping upgrade {u['name']} (set VETINARI_UPGRADE_AUTO_APPROVE=true to auto-approve)")
                    continue
            elif auto_approve:
                logger.info(f"Auto-approving upgrade: {u['name']} (VETINARI_UPGRADE_AUTO_APPROVE=true)")
            
            try:
                self.upgrader.install_upgrade(u)
                logger.info(f"Upgrade installed: {u['name']} v{u['version']}")
            except Exception as e:
                logger.error(f"Failed to install upgrade {u['name']}: {str(e)}")
        
        logger.info("Upgrade process complete.")
    
    def get_execution_status(self) -> dict:
        """Get current execution status (Phase 2 enhancement)."""
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
