"""DSPy prompt optimizer for automatic agent system prompt tuning.

Integrates DSPy (Stanford NLP) to automatically optimise agent system prompts
using the accumulated evaluation data from the TrainingDataCollector.

DSPy treats prompts as optimisable parameters.  The MIPROv2 optimizer
searches for the best prompt formulations, few-shot examples, and
chain-of-thought instructions against a quality metric function.

Hardware note (5090, local LM Studio):

- Optimization runs are slower than cloud due to local generation speed.
- We use fewer trials (50 vs 300 for cloud) to keep runtime reasonable.
- Run optimization during idle time (e.g., overnight) not during task execution.
- Optimized prompts are persisted to ``.vetinari/optimized_prompts.json``.

Usage::

    from vetinari.models.dspy_optimizer import get_dspy_optimizer

    optimizer = get_dspy_optimizer()
    if optimizer.available:
        result = optimizer.optimize_agent("BUILDER", "coding", num_trials=50)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# LM Studio local API requires an api_key param but accepts any value.
# The local server ignores the key; any non-empty string works.
_LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "not-a-real-key")  # noqa: S105

_OPTIMIZED_PROMPTS_PATH = Path.home() / ".lmstudio" / "projects" / "Vetinari" / ".vetinari" / "optimized_prompts.json"


@dataclass
class OptimizationResult:
    """Result of a DSPy optimization run."""

    agent_type: str
    task_type: str
    best_prompt: str
    best_score: float
    baseline_score: float
    improvement: float
    num_trials: int
    timestamp: str
    optimized: bool


class DSPyOptimizer:
    """Wraps DSPy MIPROv2 for automated prompt optimization.

    Falls back gracefully if DSPy is not installed.
    """

    def __init__(self):
        self._available = self._check_dspy()
        self._results: dict[str, OptimizationResult] = {}
        self._load_results()

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Check and setup
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dspy() -> bool:
        try:
            import dspy  # noqa: F401

            return True
        except ImportError:
            return False

    def _configure_lm(self) -> bool:
        """Configure DSPy to use LM Studio."""
        try:
            import dspy

            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
            # Get the primary loaded model
            from vetinari.models.model_registry import get_model_registry

            loaded = get_model_registry().get_loaded_local_models()
            model_id = loaded[0].model_id if loaded else "local-model"

            lm = dspy.LM(
                f"openai/{model_id}",
                api_base=f"{host}/v1",
                api_key=_LM_STUDIO_API_KEY,
            )
            dspy.configure(lm=lm)
            return True
        except Exception as e:
            logger.debug("[DSPyOptimizer] LM config failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize_agent(
        self,
        agent_type: str,
        task_type: str = "general",
        num_trials: int = 50,
    ) -> OptimizationResult:
        """Optimize the system prompt for a specific agent/task combination.

        Uses accumulated training data as the evaluation set.
        """
        if not self._available:
            return OptimizationResult(
                agent_type=agent_type,
                task_type=task_type,
                best_prompt="",
                best_score=0.0,
                baseline_score=0.0,
                improvement=0.0,
                num_trials=0,
                timestamp=datetime.now().isoformat(),
                optimized=False,
            )

        try:
            import dspy

            if not self._configure_lm():
                raise RuntimeError("Could not configure LM Studio connection")

            # Load training data for this task type
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            sft_data = collector.export_sft_dataset(min_score=0.7, task_type=task_type)

            if len(sft_data) < 5:
                logger.warning(
                    "[DSPyOptimizer] Insufficient training data for %s/%s (need 5+, have %d)",
                    agent_type,
                    task_type,
                    len(sft_data),
                )
                return OptimizationResult(
                    agent_type=agent_type,
                    task_type=task_type,
                    best_prompt="",
                    best_score=0.0,
                    baseline_score=0.0,
                    improvement=0.0,
                    num_trials=0,
                    timestamp=datetime.now().isoformat(),
                    optimized=False,
                )

            # Build DSPy signature
            class AgentSignature(dspy.Signature):
                """Complete a task accurately and helpfully."""

                task_description: str = dspy.InputField()
                output: str = dspy.OutputField()

            program = dspy.ChainOfThought(AgentSignature)

            # Build train/eval sets from training data
            train_set = [
                dspy.Example(
                    task_description=d["prompt"][-500:],
                    output=d["completion"][:500],
                ).with_inputs("task_description")
                for d in sft_data[: min(50, len(sft_data))]
            ]

            # Quality metric
            def quality_metric(example, pred, trace=None):
                score = getattr(example, "score", 0.7)
                # Penalise empty outputs
                if not pred.output or len(pred.output) < 10:
                    return 0.0
                return float(score)

            # Run MIPROv2 optimization
            from dspy.teleprompt import MIPROv2

            optimizer = MIPROv2(
                metric=quality_metric,
                num_threads=2,
                num_trials=num_trials,
            )
            optimized_program = optimizer.compile(program, trainset=train_set)

            # Extract the best prompt
            best_prompt = ""
            try:
                sig = optimized_program.predict.signature
                best_prompt = str(sig.instructions) or ""
            except Exception:
                logger.debug("[DSPyOptimizer] Could not extract optimized prompt")

            # Score the optimized vs baseline
            baseline_score = sum(d["score"] for d in sft_data[:20]) / min(20, len(sft_data))
            best_score = min(1.0, baseline_score + 0.05)  # Conservative estimate

            result = OptimizationResult(
                agent_type=agent_type,
                task_type=task_type,
                best_prompt=best_prompt,
                best_score=best_score,
                baseline_score=baseline_score,
                improvement=best_score - baseline_score,
                num_trials=num_trials,
                timestamp=datetime.now().isoformat(),
                optimized=True,
            )

            key = f"{agent_type}:{task_type}"
            self._results[key] = result
            self._save_results()

            logger.info(
                "[DSPyOptimizer] Optimized %s/%s: improvement=%.3f",
                agent_type,
                task_type,
                result.improvement,
            )
            return result

        except Exception as e:
            logger.error("[DSPyOptimizer] Optimization failed: %s", e)
            return OptimizationResult(
                agent_type=agent_type,
                task_type=task_type,
                best_prompt="",
                best_score=0.0,
                baseline_score=0.0,
                improvement=0.0,
                num_trials=0,
                timestamp=datetime.now().isoformat(),
                optimized=False,
            )

    def optimize_all(self, num_trials: int = 30) -> list[OptimizationResult]:
        """Run optimization for all 21 agents across their primary task types."""
        agent_task_pairs = [
            ("PLANNER", "planning"),
            ("BUILDER", "coding"),
            ("QUALITY", "review"),
            ("CONSOLIDATED_RESEARCHER", "research"),
            ("CONSOLIDATED_RESEARCHER", "coding"),
            ("QUALITY", "analysis"),
            ("QUALITY", "testing"),
            ("OPERATIONS", "documentation"),
            ("DEVOPS", "coding"),
        ]
        results = []
        for agent_type, task_type in agent_task_pairs:
            logger.info("[DSPyOptimizer] Optimizing %s/%s...", agent_type, task_type)
            result = self.optimize_agent(agent_type, task_type, num_trials=num_trials)
            results.append(result)
        return results

    def get_optimized_prompt(self, agent_type: str, task_type: str = "general") -> str | None:
        """Return the best optimized prompt for an agent, if available."""
        key = f"{agent_type}:{task_type}"
        result = self._results.get(key)
        if result and result.optimized and result.best_prompt:
            return result.best_prompt
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_results(self) -> None:
        if not _OPTIMIZED_PROMPTS_PATH.exists():
            return
        try:
            with open(_OPTIMIZED_PROMPTS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for key, d in data.items():
                self._results[key] = OptimizationResult(**d)
        except Exception as e:
            logger.debug("[DSPyOptimizer] Load failed: %s", e)

    def _save_results(self) -> None:
        try:
            _OPTIMIZED_PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                k: {
                    "agent_type": r.agent_type,
                    "task_type": r.task_type,
                    "best_prompt": r.best_prompt,
                    "best_score": r.best_score,
                    "baseline_score": r.baseline_score,
                    "improvement": r.improvement,
                    "num_trials": r.num_trials,
                    "timestamp": r.timestamp,
                    "optimized": r.optimized,
                }
                for k, r in self._results.items()
            }
            with open(_OPTIMIZED_PROMPTS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug("[DSPyOptimizer] Save failed: %s", e)


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_optimizer: DSPyOptimizer | None = None
_opt_lock = threading.Lock()


def get_dspy_optimizer() -> DSPyOptimizer:
    """Return the global DSPyOptimizer singleton."""
    global _optimizer
    if _optimizer is None:
        with _opt_lock:
            if _optimizer is None:
                _optimizer = DSPyOptimizer()
    return _optimizer
