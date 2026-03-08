import re
import time
from pathlib import Path
from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.validator import Validator

# Import structured logging
try:
    from vetinari.structured_logging import get_logger, log_task_start, log_task_complete, log_task_error
    STRUCTURED_LOGGING = True
except ImportError:
    STRUCTURED_LOGGING = False
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class TaskExecutor:
    def __init__(self, adapter: LMStudioAdapter, validator: "Validator", config: dict):
        self.adapter = adapter
        self.validator = validator
        self.config = config
        self.project_root = Path(config.get("project_root", "."))

    def _load_prompt(self, task_id: str, stage: str) -> str:
        prompts_dir = Path(self.config.get("prompts_dir", "prompts"))
        fname = f"{task_id}_{stage}.txt"
        p = prompts_dir / fname
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        # Fallback to embedded YAML prompts
        tasks = self.config.get("tasks", [])
        task = next((t for t in tasks if t["id"] == task_id), None)
        return task.get("prompts", {}).get(stage, f"Prompt for {task_id}:{stage}")

    def _parse_code_blocks(self, text: str) -> dict:
        code_blocks = {}
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            if 'cli.py' in code[:100].lower() or 'def ' in code[:100]:
                code_blocks['cli.py'] = code.strip()
            elif 'api.py' in code[:100].lower() or 'app' in code[:100].lower():
                code_blocks['api.py'] = code.strip()
            elif 'readme' in code[:100].lower():
                code_blocks['README.md'] = code.strip()
            elif 'requirements' in code[:100].lower():
                code_blocks['requirements.txt'] = code.strip()
            elif 'test' in code[:100].lower():
                code_blocks['test_main.py'] = code.strip()
            else:
                code_blocks[f'output_{len(code_blocks)}.py'] = code.strip()
        return code_blocks

    def _write_files(self, code_blocks: dict, task_id: str):
        output_dir = self.project_root / "outputs" / task_id / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        written = []
        for filename, code in code_blocks.items():
            filepath = output_dir / filename
            filepath.write_text(code, encoding="utf-8")
            written.append(str(filepath))
            logger.info(f"[Vetinari] Written: {filepath}")
        return written

    def execute_task(self, task_id: str) -> dict:
        # Locate task
        tasks = self.config.get("tasks", [])
        task = next((t for t in tasks if t["id"] == task_id), None)
        if not task:
            if STRUCTURED_LOGGING:
                log_task_error(task_id, "Task not found")
            return {"status": "failed", "reason": "task not found"}
        
        # Log task start
        task_type = task.get("type", "generic")
        if STRUCTURED_LOGGING:
            log_task_start(task_id, task_type=task_type)
        
        start_time = time.time()

        model_id = task.get("assigned_model_id")
        
        # Find model from config/models to get the model identifier
        models = self.config.get("models", [])
        model = next((m for m in models if m.get("id") == model_id), None)
        
        # If not found in static models, check discovered models
        if not model:
            # Get the model pool from adapter to find discovered models
            # For now, use the model_id directly as it should be the LM Studio model name
            model = {"name": model_id, "id": model_id}
        
        # Get the actual model name that LM Studio expects
        lm_model_name = model.get("name", model_id)
        if not lm_model_name:
            lm_model_name = model_id

        # Run inference using chat API
        prompt_to_send = self._load_prompt(task_id, "run")
        system_prompt = self._load_prompt(task_id, "init")
        
        # Use the chat API
        result = self.adapter.chat(lm_model_name, system_prompt, prompt_to_send)

        # Save outputs
        outs_dir = Path(self.config.get("outputs_dir", "outputs")) / task_id
        outs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outs_dir / "output.txt"
        output_text = str(result.get("output", ""))
        output_path.write_text(output_text, encoding="utf-8")

        # Parse and write code blocks to files
        written_files = []
        code_blocks = self._parse_code_blocks(output_text)
        if code_blocks:
            written_files = self._write_files(code_blocks, task_id)

        # Validate
        valid = True
        if not self.validator.is_valid_text(output_text):
            valid = False

        # Run verification pipeline if available
        verification_result = None
        try:
            from vetinari.validation.verification import get_verifier_pipeline, VerificationStatus
            pipeline = get_verifier_pipeline()
            vr_dict = pipeline.verify(output_text)
            # pipeline.verify() returns Dict[str, VerificationResult]
            if isinstance(vr_dict, dict) and vr_dict:
                all_passed = all(
                    v.status == VerificationStatus.PASSED for v in vr_dict.values()
                )
                # Compute score from issue counts: 1.0 = no issues, lower with more errors
                total_checks = len(vr_dict)
                scores = []
                all_issues = []
                for v in vr_dict.values():
                    issue_total = v.error_count + v.warning_count + v.info_count
                    score = 1.0 - (v.error_count / max(issue_total, 1)) if issue_total > 0 else 1.0
                    scores.append(score)
                    all_issues.extend(v.issues)
                avg_score = sum(scores) / total_checks
                verification_result = {"passed": all_passed, "score": avg_score, "issues": all_issues}
                if not all_passed:
                    logger.warning(f"Verification failed for task {task_id}: {all_issues}")
        except Exception as e:
            logger.debug(f"Verification pipeline skipped: {e}")

        # Record quality score in learning system
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer
            from vetinari.learning.feedback_loop import get_feedback_loop
            scorer = get_quality_scorer()
            score = scorer.score(
                task_id=task_id, model_id=model_id or "default",
                task_type=task.get("type", "generic"),
                task_description=task.get("description", ""),
                output=output_text, use_llm=False,
            )
            get_feedback_loop().record_outcome(
                task_id=task_id, model_id=model_id or "default",
                task_type=task.get("type", "generic"),
                quality_score=score.overall_score,
                latency_ms=int(result.get("latency_ms") or 0),
                success=(result.get("status") == "ok"),
            )
        except Exception as e:
            logger.debug(f"[TaskExecutor] Quality scoring skipped: {e}")

        status = "completed" if result.get("status") == "ok" and valid else "failed"
        
        # Log task completion
        execution_time_ms = (time.time() - start_time) * 1000
        if STRUCTURED_LOGGING:
            if status == "completed":
                log_task_complete(task_id, execution_time_ms, model_id=model_id)
            else:
                log_task_error(task_id, result.get("error", "Unknown error"))
        
        return {
            "status": status,
            "task_id": task_id,
            "model_id": model_id,
            "latency_ms": result.get("latency_ms"),
            "output_path": str(output_path),
            "generated_files": written_files,
            "error": result.get("error"),
            "verification": verification_result,
        }
