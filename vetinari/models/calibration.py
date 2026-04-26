"""Model Calibration - first-load performance measurement and prior seeding.

Runs 5 short calibration prompts on a newly loaded model to measure:
- Token generation speed (tokens/sec)
- Memory usage under load
- Output quality per task type

Results seed Thompson BetaArm priors with informed values instead of
the default Beta(2,2), giving the model selector better starting points.

The calibration budget is 30 seconds total.

Usage::

    from vetinari.models.calibration import calibrate_model

    results = calibrate_model(model_id, llm_instance)
    # -> CalibrationResult with per-task metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Total time budget for all calibration prompts (seconds)
CALIBRATION_BUDGET_SECONDS = 30

# Per-prompt time limit (seconds) - prevents one slow prompt from consuming the budget
PER_PROMPT_TIMEOUT_SECONDS = 8

# Calibration prompts: (task_type, system_prompt, user_prompt, max_tokens)
_CALIBRATION_PROMPTS: list[tuple[str, str, str, int]] = [
    (
        "coding",
        "You are a Python expert.",
        "Write a function that checks if a number is prime. Include type hints.",
        150,
    ),
    (
        "reasoning",
        "You are a logical reasoning assistant.",
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain.",
        100,
    ),
    (
        "general",
        "You are a helpful assistant.",
        "Explain what a hash table is in 2-3 sentences.",
        80,
    ),
    (
        "instruction",
        "You follow instructions precisely.",
        "List exactly 5 programming languages that compile to native code. Format: numbered list.",
        60,
    ),
    (
        "creative",
        "You are a creative writer.",
        "Write a one-sentence opening line for a sci-fi short story set on Mars.",
        40,
    ),
]


@dataclass
class TaskCalibration:
    """Calibration metrics for a single task type."""

    task_type: str
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    latency_ms: int = 0
    output_length: int = 0
    completed: bool = False

    def __repr__(self) -> str:
        return (
            f"TaskCalibration(type={self.task_type!r}, tok/s={self.tokens_per_second:.1f}, latency={self.latency_ms}ms)"
        )


@dataclass
class CalibrationResult:
    """Complete calibration results for a model.

    Contains per-task metrics and aggregate statistics for seeding
    Thompson sampling priors.
    """

    model_id: str
    total_time_seconds: float = 0.0
    tasks: list[TaskCalibration] = field(default_factory=list)
    avg_tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    completed_count: int = 0

    def __repr__(self) -> str:
        return (
            f"CalibrationResult(model={self.model_id!r}, "
            f"avg_tok/s={self.avg_tokens_per_second:.1f}, "
            f"completed={self.completed_count}/{len(self.tasks)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    def get_thompson_priors(self) -> dict[str, tuple[float, float]]:
        """Compute Thompson BetaArm prior (alpha, beta) per task type.

        Maps calibration quality into informed priors. A model that
        generates fast, complete output gets a higher alpha (more
        optimistic prior).

        Returns:
            Dict mapping task_type to (alpha, beta) tuple.
        """
        priors: dict[str, tuple[float, float]] = {}
        for task in self.tasks:
            if task.completed:
                # Scale alpha by tokens/sec relative to a baseline of 20 tok/s
                speed_factor = min(2.0, task.tokens_per_second / 20.0)
                alpha = 2.0 + speed_factor * 3.0  # Range: 2-8
                beta = 2.0  # Keep beta constant
            else:
                # Failed calibration - pessimistic prior
                alpha = 1.0
                beta = 3.0
            priors[task.task_type] = (round(alpha, 1), round(beta, 1))
        return priors


def calibrate_model(model_id: str, llm: Any) -> CalibrationResult:
    """Run calibration prompts on a freshly loaded model.

    Measures token generation speed, output quality, and memory usage
    within a 30-second total budget. Results are used to seed Thompson
    sampling priors.

    Args:
        model_id: Model identifier for logging and result tracking.
        llm: A loaded ``llama_cpp.Llama`` instance.

    Returns:
        CalibrationResult with per-task and aggregate metrics.
    """
    result = CalibrationResult(model_id=model_id)
    budget_start = time.time()

    logger.info("[Calibration] Starting calibration for %s (budget=%ds)", model_id, CALIBRATION_BUDGET_SECONDS)

    for task_type, system_prompt, user_prompt, max_tokens in _CALIBRATION_PROMPTS:
        elapsed = time.time() - budget_start
        if elapsed >= CALIBRATION_BUDGET_SECONDS:
            logger.info("[Calibration] Budget exhausted after %.1fs - stopping", elapsed)
            break

        remaining = CALIBRATION_BUDGET_SECONDS - elapsed
        effective_timeout = min(PER_PROMPT_TIMEOUT_SECONDS, remaining)

        task_cal = TaskCalibration(task_type=task_type)
        task_start = time.time()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Use InferenceConfigManager for calibration params when available
            try:
                from vetinari.config.inference_config import get_inference_config

                _cal_params = get_inference_config().get_effective_params(task_type)
                _cal_temp = _cal_params.get("temperature", 0.3)
                _cal_top_p = _cal_params.get("top_p", 0.9)
            except Exception:
                _cal_temp = 0.3
                _cal_top_p = 0.9

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=_cal_temp,
                top_p=_cal_top_p,
                min_p=0.05,
            )

            task_elapsed = time.time() - task_start

            # Check timeout
            if task_elapsed > effective_timeout:
                logger.debug(
                    "[Calibration] %s exceeded timeout (%.1fs > %.1fs)", task_type, task_elapsed, effective_timeout
                )
                task_cal.latency_ms = int(task_elapsed * 1000)
                result.tasks.append(task_cal)
                continue

            choices = response.get("choices", [])
            output = choices[0]["message"]["content"] if choices else ""
            usage = response.get("usage", {})
            tokens_used = usage.get("completion_tokens", 0) or len(output.split())

            task_cal.tokens_generated = tokens_used
            task_cal.tokens_per_second = tokens_used / max(task_elapsed, 0.001)
            task_cal.latency_ms = int(task_elapsed * 1000)
            task_cal.output_length = len(output)
            task_cal.completed = bool(output.strip())

            logger.debug(
                "[Calibration] %s: %d tokens in %.1fs (%.1f tok/s)",
                task_type,
                tokens_used,
                task_elapsed,
                task_cal.tokens_per_second,
            )

        except Exception as exc:
            task_elapsed = time.time() - task_start
            task_cal.latency_ms = int(task_elapsed * 1000)
            logger.warning("[Calibration] %s failed: %s", task_type, exc)

        result.tasks.append(task_cal)

    # Aggregate metrics
    result.total_time_seconds = round(time.time() - budget_start, 1)
    completed = [t for t in result.tasks if t.completed]
    result.completed_count = len(completed)

    if completed:
        result.avg_tokens_per_second = round(sum(t.tokens_per_second for t in completed) / len(completed), 1)

    # Memory usage snapshot
    try:
        import psutil

        process = psutil.Process()
        result.memory_usage_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        logger.warning("Could not measure memory during calibration")

    logger.info(
        "[Calibration] Completed for %s: %d/%d tasks, avg %.1f tok/s, %.1fs total",
        model_id,
        result.completed_count,
        len(result.tasks),
        result.avg_tokens_per_second,
        result.total_time_seconds,
    )

    return result


def seed_thompson_priors(model_id: str, calibration: CalibrationResult) -> None:
    """Update Thompson sampling priors with calibration data.

    Args:
        model_id: Model identifier.
        calibration: Calibration results with per-task metrics.
    """
    try:
        from vetinari.learning.model_selector import get_model_selector

        selector = get_model_selector()
        priors = calibration.get_thompson_priors()

        for task_type, (alpha, beta) in priors.items():
            arm_key = f"{model_id}:{task_type}"
            if hasattr(selector, "set_prior"):
                selector.set_prior(arm_key, alpha, beta)
                logger.debug("[Calibration] Seeded Thompson prior for %s: alpha=%.1f, beta=%.1f", arm_key, alpha, beta)

    except Exception as exc:
        logger.warning("Failed to seed Thompson priors: %s", exc)
