import os
from pathlib import Path
from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.validator import Validator

class TaskExecutor:
    def __init__(self, adapter: LMStudioAdapter, validator: "Validator", config: dict):
        self.adapter = adapter
        self.validator = validator
        self.config = config

    def _load_prompt(self, task_id: str, stage: str) -> str:
        prompts_dir = Path(self.config.get("prompts_dir", "prompts"))
        fname = f"{task_id}_{stage}.txt" if stage != "run" else f"{task_id}_run.txt"
        p = prompts_dir / fname
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        # Fallback to embedded YAML prompts
        tasks = self.config.get("tasks", [])
        task = next((t for t in tasks if t["id"] == task_id), None)
        return task.get("prompts", {}).get(stage, f"Prompt for {task_id}:{stage}")

    def execute_task(self, task_id: str) -> dict:
        # Locate task
        tasks = self.config.get("tasks", [])
        task = next((t for t in tasks if t["id"] == task_id), None)
        if not task:
            return {"status": "failed", "reason": "task not found"}

        model_endpoint = None
        model_id = task.get("assigned_model_id")
        # Find model from config/models
        models = self.config.get("models", [])
        model = next((m for m in models if m.get("id") == model_id), None)
        if model:
            model_endpoint = model.get("endpoint")

        if not model_endpoint:
            return {"status": "failed", "reason": "no model endpoint"}

        # Run inference
        prompt_to_send = self._load_prompt(task_id, "run")
        result = self.adapter.infer(model_endpoint, prompt_to_send)

        # Save outputs
        outs_dir = Path(self.config.get("outputs_dir", "outputs")) / task_id
        outs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outs_dir / "output.txt"
        output_path.write_text(str(result.get("output", "")), encoding="utf-8")

        # Validate
        valid = True
        if not self.validator.is_valid_text(result.get("output", "")):
            valid = False

        status = "completed" if result.get("status") == "ok" and valid else "failed"
        return {
            "status": status,
            "task_id": task_id,
            "model_id": model_id,
            "latency_ms": result.get("latency_ms"),
            "output_path": str(output_path),
            "error": result.get("error")
        }