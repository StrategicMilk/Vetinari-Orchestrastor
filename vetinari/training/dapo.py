"""DAPO training stage orchestrator.

Orchestrates training stages using the reward system defined in dapo_rewards.py
and the stage execution functions in dapo_stages.py.
Re-exports all public types for backward compatibility.

Reward tiers (additive, sum to 1.0):
  Tier 1: Execution pass/fail     — 0.10 (free, instant)
  Tier 2: Test pass rate           — 0.25 (free, ~seconds)
  Tier 3: Static analysis score    — 0.15 (free, instant)
  Tier 4: Heuristic pre-screen     — 0.20 (~ms, cheap ML)
  Tier 5: LLM-as-judge score      — 0.20 (expensive, tokens)
  Efficiency bonus (no rework)     — 0.10
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any

from vetinari.training.dapo_rewards import (
    TIER_EFFICIENCY_WEIGHT,
    TIER_EXECUTION_WEIGHT,
    TIER_HEURISTIC_WEIGHT,
    TIER_LLM_JUDGE_WEIGHT,
    TIER_STATIC_WEIGHT,
    TIER_TEST_WEIGHT,
    DapoExecutionResult,
    DapoTrainingResult,
    RewardBreakdown,
    StageResult,
    compute_dapo_reward,
)
from vetinari.training.dapo_stages import run_dapo_stage, run_sft_stage, run_simpo_stage
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)

__all__ = [
    "TIER_EFFICIENCY_WEIGHT",
    "TIER_EXECUTION_WEIGHT",
    "TIER_HEURISTIC_WEIGHT",
    "TIER_LLM_JUDGE_WEIGHT",
    "TIER_STATIC_WEIGHT",
    "TIER_TEST_WEIGHT",
    "DapoExecutionResult",
    "DapoTrainingResult",
    "RewardBreakdown",
    "StageResult",
    "TrainingStageOrchestrator",
    "compute_dapo_reward",
]


class TrainingStageOrchestrator:
    """Manages the SFT -> SimPO -> DAPO training pipeline with validation gates.

    Runs stages sequentially with inter-stage validation to catch
    regressions early. Each stage produces a model checkpoint that
    feeds into the next.
    """

    STAGES = ["sft", "simpo", "dapo"]

    def run_pipeline(
        self,
        base_model: str,
        dataset_path: Path,
        config: dict[str, Any] | None = None,
    ) -> DapoTrainingResult:
        """Run the full training pipeline with inter-stage validation.

        Args:
            base_model: Path or identifier of the base model.
            dataset_path: Path to the training dataset.
            config: Optional stage-specific configuration overrides.

        Returns:
            TrainingResult with success/failure and stage details.
        """
        config = config or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract
        current_model = base_model
        stage_results: list[StageResult] = []

        for stage in self.STAGES:
            # Pre-stage validation
            if not self._validate_stage_readiness(stage, current_model, dataset_path):
                logger.warning("Stage %s prerequisites not met — skipping", stage)
                stage_results.append(
                    StageResult(
                        success=True,
                        stage_name=stage,
                        output_model=current_model,
                        metrics={StatusEnum.SKIPPED.value: True, "reason": "prerequisites_not_met"},
                    )
                )
                continue

            # Run stage
            result = self._run_stage(stage, current_model, dataset_path, config)
            stage_results.append(result)

            if not result.success:
                logger.error("Stage %s failed: %s", stage, result.error)
                return DapoTrainingResult(
                    success=False,
                    stage_failed=stage,
                    error=result.error,
                    stage_results=stage_results,
                )

            # Post-stage validation
            if result.output_model and not self._validate_stage_output(
                stage,
                current_model,
                result.output_model,
            ):
                logger.warning("Stage %s output failed validation — reverting", stage)
                return DapoTrainingResult(
                    success=False,
                    stage_failed=stage,
                    error="regression_detected",
                    stage_results=stage_results,
                )

            if result.output_model:
                current_model = result.output_model

        return DapoTrainingResult(
            success=True,
            final_model=current_model,
            stage_results=stage_results,
        )

    def _validate_stage_readiness(
        self,
        stage: str,
        model: str,
        dataset_path: Path,
    ) -> bool:
        """Check if a stage's prerequisites are met.

        Args:
            stage: Stage name (sft, simpo, dapo).
            model: Current model path.
            dataset_path: Training data path.

        Returns:
            True if ready to proceed.
        """
        if stage == "sft":
            if not bool(model):
                return False
            # Defect 207: require dataset_path to exist, not just be truthy
            if isinstance(dataset_path, Path):
                return dataset_path.exists()
            elif dataset_path:
                return Path(dataset_path).exists()
            return False

        if stage == "simpo":
            # SimPO needs SFT output and a dataset to curate preference pairs from
            # (defect 207: also check dataset_path exists).
            if not bool(model):
                return False
            # Defect 207: require dataset_path to exist, not just be truthy
            if isinstance(dataset_path, Path):
                return dataset_path.exists()
            elif dataset_path:
                return Path(dataset_path).exists()
            return False

        if stage == "dapo":
            # DAPO needs group data (min 5 groups of K=4); in cold start, skip DAPO
            # (defect 207: also check dataset_path exists).
            if not bool(model):
                return False
            # Defect 207: require dataset_path to exist, not just be truthy
            if isinstance(dataset_path, Path):
                return dataset_path.exists()
            elif dataset_path:
                return Path(dataset_path).exists()
            return False

        return True

    def _run_stage(
        self,
        stage: str,
        model: str,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> StageResult:
        """Dispatch a single training stage to its execution function.

        Args:
            stage: Stage name.
            model: Input model path.
            dataset_path: Training data path.
            config: Configuration overrides.

        Returns:
            StageResult with output model path.
        """
        logger.info("Running training stage: %s (model=%s)", stage, model)

        try:
            if stage == "sft":
                return run_sft_stage(model, dataset_path, config)
            if stage == "simpo":
                return run_simpo_stage(model, dataset_path, config)
            if stage == "dapo":
                return run_dapo_stage(model, dataset_path, config)
            return StageResult(
                success=False,
                stage_name=stage,
                error=f"Unknown stage: {stage}",
            )
        except Exception as exc:
            logger.exception("Stage %s raised exception", stage)
            return StageResult(
                success=False,
                stage_name=stage,
                error=str(exc),
            )

    def _validate_stage_output(
        self,
        stage: str,
        input_model: str,
        output_model: str,
    ) -> bool:
        """Validate that stage output did not regress model quality.

        Checks that the output model path exists and, if a trainer state
        file was written, that the final eval loss is not catastrophic.

        Args:
            stage: Stage name (sft, simpo, dapo).
            input_model: Model path before stage.
            output_model: Model path after stage.

        Returns:
            True if the output passes validation checks.
        """
        if output_model == input_model:
            # Stage was skipped or produced no change — trivially valid
            return True

        output_path = Path(output_model)
        if not output_path.exists():
            logger.warning(
                "Stage %s output path does not exist: %s",
                stage,
                output_model,
            )
            return False

        # Check for trainer_state.json with eval_loss
        state_file = output_path / "trainer_state.json"
        if not state_file.exists():
            # Also check parent dir (adapter may be in a subdirectory)
            state_file = output_path.parent / "trainer_state.json"

        if state_file.exists():
            try:
                state = _json.loads(state_file.read_text(encoding="utf-8"))
                log_history = state.get("log_history", [])
                if log_history:
                    last_entry = log_history[-1]
                    eval_loss = last_entry.get("eval_loss")
                    if eval_loss is not None and eval_loss > 5.0:
                        logger.warning(
                            "Stage %s: eval_loss %.2f exceeds catastrophic threshold (5.0)",
                            stage,
                            eval_loss,
                        )
                        return False
                    train_loss = last_entry.get("loss")
                    if train_loss is not None and train_loss > 10.0:
                        logger.warning(
                            "Stage %s: train_loss %.2f exceeds catastrophic threshold (10.0)",
                            stage,
                            train_loss,
                        )
                        return False
            except _json.JSONDecodeError as exc:
                # Corrupt trainer_state.json means the training run is suspect —
                # fail validation rather than silently accepting bad state (defect 16 fix).
                logger.warning(
                    "Stage %s: trainer_state.json is corrupt (%s) — stage validation failed",
                    stage,
                    exc,
                )
                return False
            except (OSError, KeyError):
                logger.warning("Stage %s: could not read trainer state; assuming valid", stage)

        logger.info("Stage %s: validation passed for %s", stage, output_model)
        return True

    def _run_simpo(
        self,
        model: str,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> StageResult:
        """Run the SimPO alignment stage directly.

        Convenience wrapper around ``_run_stage("simpo", ...)`` for
        programmatic and test access.

        Args:
            model: Input model path or identifier.
            dataset_path: Path to the DPO preference dataset.
            config: Stage configuration overrides.

        Returns:
            StageResult from the SimPO training stage.
        """
        return self._run_stage("simpo", model, dataset_path, config)

    def _run_dapo(
        self,
        model: str,
        dataset_path: Path,
        config: dict[str, Any],
    ) -> StageResult:
        """Run the DAPO RL stage directly.

        Convenience wrapper around ``_run_stage("dapo", ...)`` for
        programmatic and test access.

        Args:
            model: Input model path or identifier.
            dataset_path: Path to the ranking dataset.
            config: Stage configuration overrides.

        Returns:
            StageResult from the DAPO training stage.
        """
        return self._run_stage("dapo", model, dataset_path, config)
