import os
import json
import logging
import time
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set

from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.model_pool import ModelPool
from vetinari.scheduler import Scheduler
from vetinari.executor import TaskExecutor
from vetinari.upgrader import Upgrader
from vetinari.validator import Validator
from vetinari.builder import Builder


class Orchestrator:
    def __init__(self, manifest_path: str, host: str = "http://100.78.30.7:1234", api_token: str = None, max_concurrent: int = 4):
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

        logging.info("Vetinari orchestrator initialized.")

    def update_settings(self, host: str = None, api_token: str = None):
        """Update settings including host and API token."""
        if host:
            self.adapter.host = host.rstrip("/")
            self.model_pool.host = host.rstrip("/")
        
        if api_token is not None:
            self.api_token = api_token
            self.adapter.set_api_token(api_token)
            self.model_pool.set_api_token(api_token)
        
        logging.info(f"Settings updated: host={host}, api_token={'***' if api_token else 'None'}")

    def _load_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_all(self):
        logging.info("Starting full workflow: plan -> allocate -> execute -> validate -> build.")
        
        # 1) Auto-discover models
        try:
            self.model_pool.discover_models()
        except Exception as e:
            logging.error(f"Model discovery failed: {e}")
            # Continue with empty model pool if discovery fails
        
        # 2) Assign tasks to models
        self.model_pool.assign_tasks_to_models(self.config)
        
        # 3) Build schedule (layers for parallel execution)
        layers = self.scheduler.build_schedule_layers(self.config)
        
        logging.info(f"Built {len(layers)} execution layers")
        
        # 4) Execute tasks layer by layer
        completed_tasks = set()
        all_results = []
        
        for layer_idx, layer in enumerate(layers):
            logging.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks in parallel")
            
            # Execute tasks in this layer concurrently
            layer_results = self._execute_layer_parallel(layer)
            all_results.extend(layer_results)
            
            # Mark tasks as completed
            for result in layer_results:
                if result.get("status") == "completed":
                    completed_tasks.add(result["task_id"])
                else:
                    logging.warning(f"Task {result.get('task_id')} did not complete successfully")
        
        logging.info(f"All layers completed. {len(completed_tasks)} tasks finished successfully.")
        
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
                    logging.info(f"Task {task['id']} completed with status: {result.get('status')}")
                except Exception as e:
                    logging.error(f"Task {task['id']} failed with exception: {e}")
                    results.append({
                        "status": "failed",
                        "task_id": task["id"],
                        "error": str(e)
                    })
        
        return results

    def run_task(self, task_id: str):
        logging.info(f"Running task {task_id}")
        result = self.executor.execute_task(task_id)
        if result.get("status") != "completed":
            logging.warning(f"Task {task_id} did not complete successfully. See logs for details.")
        else:
            logging.info(f"Task {task_id} completed.")
        return result

    def check_and_upgrade_models(self):
        upgrades = self.upgrader.check_for_upgrades()
        if not upgrades:
            logging.info("No upgrades available.")
            return
        
        for u in upgrades:
            if self.config.get("upgrade_policy", {}).get("require_approval", True):
                user_input = input(f"Upgrade available: {u['name']} (version {u['version']}). Install? (y/n): ")
                if user_input.strip().lower() != "y":
                    logging.info("Upgrade skipped by user.")
                    continue
            self.upgrader.install_upgrade(u)
        logging.info("Upgrade process complete.")


# Convenience when running as script
if __name__ == "__main__":
    Orchestrator("manifest/vetinari.yaml").run_all()
