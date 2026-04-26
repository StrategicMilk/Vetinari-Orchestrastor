"""Training pipeline core helpers.

Contains the support classes used by the full training pipeline:
BenchmarkTracker, CloudOutputStore, TrainingRun, DataCurator,
and the ``_ensure_packages`` utility.  These are separated from the
heavier trainer and converter classes to keep each module focused.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vetinari.constants import TRUNCATE_OUTPUT_PREVIEW, TRUNCATE_OUTPUT_SUMMARY, get_user_dir
from vetinari.exceptions import ExecutionError

logger = logging.getLogger(__name__)
_AUTO_INSTALL_PACKAGE_ALLOWLIST: frozenset[str] = frozenset({
    "bitsandbytes",
    "datasets",
    "peft",
    "transformers",
    "trl",
})


class BenchmarkTracker:
    """Track benchmark run history to detect staleness.

    Stores the timestamp of the last benchmark run in a simple JSON file
    so the curriculum can decide when to re-run benchmarks.
    """

    def _state_path(self) -> Path:
        """Return the path to the benchmark tracker state file.

        Uses get_user_dir() to store state in a consistent, writable location
        across different working directory contexts.

        Returns:
            Path to benchmark_tracker.json under the user directory.
        """
        return get_user_dir() / "benchmark_tracker.json"

    def days_since_last_run(self) -> int:
        """Return the number of days since the last benchmark run.

        Returns:
            Days since last run, or 999 if no run has been recorded.
        """
        try:
            with self._state_path().open(encoding="utf-8") as f:
                data = json.load(f)
            last = datetime.fromisoformat(data.get("last_run", ""))
            return (datetime.now(timezone.utc) - last).days
        except Exception:
            logger.warning(
                "Could not read training pipeline state file %s — treating last run as very stale (999 days), retraining may trigger",
                self._state_path(),
            )
            return 999  # No record — treat as very stale

    def record_run(self) -> None:
        """Record that a benchmark run happened now."""
        self._state_path().parent.mkdir(parents=True, exist_ok=True)
        with self._state_path().open("w", encoding="utf-8") as f:
            json.dump({"last_run": datetime.now(timezone.utc).isoformat()}, f)


class CloudOutputStore:
    """Store high-quality cloud model outputs for knowledge distillation.

    Collects outputs from cloud API calls (e.g., Claude, GPT-4) that score
    above a quality threshold, so they can be used to train local models.

    The store file is resolved through ``get_user_dir()`` so the path is
    stable regardless of the process working directory (defect 4 fix).
    """

    # Filename within the user dir — resolved lazily via _store_path().
    _STORE_FILENAME = "cloud_outputs.jsonl"

    def _store_path(self) -> Path:
        """Return the absolute path to the cloud output store file.

        Returns:
            Path inside the canonical Vetinari user directory.
        """
        return get_user_dir() / self._STORE_FILENAME

    def get_high_quality_outputs(self, min_score: float = 0.8) -> list[dict]:
        """Return stored cloud outputs above the quality threshold.

        Args:
            min_score: Minimum quality score to include.

        Returns:
            List of output dicts with prompt, response, score, and model fields.
        """
        outputs = []
        store_path = self._store_path()
        if not store_path.exists():
            return []
        with store_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("score", 0) >= min_score:
                        outputs.append(entry)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON line in cloud output store %s — entry ignored",
                        store_path,
                    )
                    continue
        return outputs

    def record(self, prompt: str, response: str, score: float, model: str) -> None:
        """Store a cloud model output for potential distillation.

        Args:
            prompt: The input prompt.
            response: The model's output.
            score: Quality score (0.0-1.0).
            model: The model that produced this output.
        """
        store_path = self._store_path()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "prompt": prompt,
            "response": response,
            "score": score,
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


@dataclass
class TrainingRun:
    """Result of a training pipeline run."""

    run_id: str
    timestamp: str
    base_model: str
    task_type: str
    training_examples: int
    epochs: int
    success: bool
    output_model_path: str = ""
    adapter_path: str = ""
    backend: str = "llama_cpp"
    model_format: str = "gguf"
    model_revision: str = ""
    model_manifest_path: str = ""
    eval_score: float = 0.0
    baseline_score: float = 0.0
    error: str = ""

    def __repr__(self) -> str:
        return (
            f"TrainingRun(run_id={self.run_id!r}, base_model={self.base_model!r}, "
            f"task_type={self.task_type!r}, success={self.success!r}, "
            f"eval_score={self.eval_score!r})"
        )


class DataCurator:
    """Curates high-quality training data from the TrainingDataCollector."""

    def curate(
        self,
        task_type: str | None = None,
        min_score: float = 0.8,
        max_examples: int = 5000,
        output_dir: str = ".",
    ) -> str:
        """Curate SFT training data and write to a JSONL file.

        Returns the path to the curated dataset file.

        Args:
            task_type: The task type.
            min_score: The min score.
            max_examples: The max examples.
            output_dir: The output dir.

        Returns:
            Absolute path to the written JSONL file, named
            ``sft_<task_type>_<YYYYMMDD_HHMM>.jsonl``.

        Raises:
            ExecutionError: If no training data meets criteria.
        """
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        data = collector.export_sft_dataset(
            min_score=min_score,
            task_type=task_type,
            max_records=max_examples,
        )

        if not data:
            raise ExecutionError(f"No training data meets criteria (score>={min_score}, type={task_type})")

        # Format for Alpaca-style fine-tuning
        formatted = [
            {
                "instruction": d["prompt"][:TRUNCATE_OUTPUT_SUMMARY],
                "input": "",
                "output": d["completion"][:TRUNCATE_OUTPUT_PREVIEW],
            }
            for d in data
        ]

        out_path = (
            Path(output_dir)
            / f"sft_{task_type or 'general'}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.jsonl"
        )
        with Path(out_path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(item) + "\n" for item in formatted)

        logger.info("[DataCurator] Wrote %d examples to %s", len(formatted), out_path)
        return str(out_path)

    def curate_dpo(
        self,
        task_type: str | None = None,
        min_score_gap: float = 0.2,
        output_dir: str = ".",
    ) -> str:
        """Curate DPO preference pairs and write to a JSONL file.

        Args:
            task_type: The task type.
            min_score_gap: The min score gap.
            output_dir: The output dir.

        Returns:
            Absolute path to the written JSONL file, named
            ``dpo_<task_type>_<YYYYMMDD_HHMM>.jsonl``.

        Raises:
            ExecutionError: If no preference pairs are available.
        """
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        pairs = collector.export_dpo_dataset(
            task_type=task_type,
            min_score_gap=min_score_gap,
        )

        if not pairs:
            raise ExecutionError("No preference pairs available for DPO training")

        out_path = (
            Path(output_dir)
            / f"dpo_{task_type or 'general'}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.jsonl"
        )
        with Path(out_path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(pair) + "\n" for pair in pairs)

        logger.info("[DataCurator] Wrote %d DPO pairs to %s", len(pairs), out_path)
        return str(out_path)


def _ensure_packages(packages: list[str]) -> dict[str, bool]:
    """Auto-install missing Python packages via pip subprocess.

    Checks each package with importlib and installs missing ones via pip.

    Args:
        packages: List of package names to ensure are installed.

    Returns:
        Dict mapping package name to True if installed (already or newly),
        False if installation failed.
    """
    import importlib.util

    results: dict[str, bool] = {}
    for pkg in packages:
        if pkg not in _AUTO_INSTALL_PACKAGE_ALLOWLIST:
            logger.warning("_ensure_packages: refusing unapproved package %s", pkg)
            results[pkg] = False
            continue
        if importlib.util.find_spec(pkg) is not None:
            results[pkg] = True
            continue

        logger.info("_ensure_packages: %s not found, attempting pip install", pkg)
        try:
            proc = subprocess.run(  # noqa: S603 - package names are constrained by _AUTO_INSTALL_PACKAGE_ALLOWLIST.
                [sys.executable, "-m", "pip", "install", pkg],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode == 0:
                logger.info("_ensure_packages: successfully installed %s", pkg)
                results[pkg] = True
            else:
                logger.warning(
                    "_ensure_packages: pip install %s failed (rc=%d): %s",
                    pkg,
                    proc.returncode,
                    proc.stderr[-500:] if proc.stderr else "",
                )
                results[pkg] = False
        except subprocess.TimeoutExpired:
            logger.warning("_ensure_packages: pip install %s timed out", pkg)
            results[pkg] = False
        except Exception as exc:
            logger.warning("_ensure_packages: failed to install %s: %s", pkg, exc)
            results[pkg] = False

    return results
