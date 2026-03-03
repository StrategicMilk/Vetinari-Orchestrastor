import os
import json
import logging
import time
from pathlib import Path
import yaml
from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.model_pool import ModelPool
from vetinari.scheduler import Scheduler
from vetinari.executor import TaskExecutor
from vetinari.upgrader import Upgrader
from vetinari.validator import Validator
from vetinari.builder import Builder

class Orchestrator:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.config = self._load_manifest()
        self.adapter = LMStudioAdapter(host="http://10.0.0.96:1234")
        self.model_pool = ModelPool(self.config)
        self.scheduler = Scheduler(self.config)
        self.validator = Validator()
        self.executor = TaskExecutor(self.adapter, self.validator, self.config)
        self.upgrader = Upgrader(self.config)
        self.builder = Builder(self.config)

        logging.info("Vetinari orchestrator initialized.")

    def _load_manifest(self):
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_all(self):
        logging.info("Starting full workflow: plan → allocate → execute → validate → build.")
        
        # 1) Auto-discover models
        self.model_pool.discover_models()
        
        # 2) Assign tasks to models
        self.model_pool.assign_tasks_to_models(self.config)
        
        # 3) Build schedule
        schedule = self.scheduler.build_schedule(self.config)
        
        # 4) Execute tasks in schedule order
        for task in schedule:
            self.run_task(task["id"])
        
        # 5) Build final artifact
        self.builder.build_final_artifact()

    def run_task(self, task_id: str):
        logging.info(f"Running task {task_id}")
        result = self.executor.execute_task(task_id)
        if result.get("status") != "completed":
            logging.warning(f"Task {task_id} did not complete successfully. See logs for details.")
        else:
            logging.info(f"Task {task_id} completed.")

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