"""Vetinari Training Pipeline — orchestration layer.

Wires together the full fine-tuning workflow:

  1. DataCurator   — curate JSONL from TrainingDataCollector
  2. LocalTrainer  — QLoRA fine-tuning via unsloth/trl
  3. GGUFConverter — convert adapter to GGUF for local inference
  4. ModelDeployer — copy GGUF to local models directory

Support classes (BenchmarkTracker, CloudOutputStore, TrainingRun, DataCurator,
_ensure_packages) live in ``pipeline_core``.  Compute-intensive trainer and
converter classes (LocalTrainer, GGUFConverter, ModelDeployer) live in
``pipeline_trainers``.  This module re-exports all public names so existing
callers do not need to change their imports.

Usage::

    from vetinari.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline()
    run = pipeline.run(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        task_type="coding",
        min_score=0.8,
        epochs=3,
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import (  # noqa: VET306 — user-scoped via get_user_dir()
    MODEL_DISCOVERY_TIMEOUT,
    OPERATOR_MODELS_CACHE_DIR,
)
from vetinari.types import AgentType

from .pipeline_core import (
    BenchmarkTracker,
    CloudOutputStore,
    DataCurator,
    TrainingRun,
    _ensure_packages,
)
from .pipeline_trainers import GGUFConverter, LocalTrainer, ModelDeployer

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(OPERATOR_MODELS_CACHE_DIR)
_NATIVE_MODELS_DIR = Path(os.environ.get("VETINARI_NATIVE_MODELS_DIR", str(_MODELS_DIR / "native")))
_TRAINING_NATIVE_BACKENDS = {"vllm", "nim"}
_TRAINING_NATIVE_FORMATS = {"safetensors", "awq", "gptq"}
_TRAINING_DEFAULT_FORMAT_BY_BACKEND = {
    "llama_cpp": "gguf",
    "vllm": "safetensors",
    "nim": "safetensors",
}

__all__ = [
    "BenchmarkTracker",
    "CloudOutputStore",
    "ContextDistillationDatasetBuilder",
    "DataCurator",
    "DistillationDatasetInfo",
    "GGUFConverter",
    "LocalTrainer",
    "ModelDeployer",
    "TrainingPipeline",
    "TrainingRun",
    "_ensure_packages",
]


def _normalize_training_backend(backend: str | None) -> str:
    value = (backend or "vllm").strip().lower().replace("-", "_")
    if value in {"llamacpp", "llama"}:
        value = "llama_cpp"
    if value not in {"llama_cpp", *_TRAINING_NATIVE_BACKENDS}:
        raise ValueError("backend must be one of: llama_cpp, vllm, nim")
    return value


def _normalize_training_format(backend: str, model_format: str | None) -> str:
    value = (model_format or _TRAINING_DEFAULT_FORMAT_BY_BACKEND[backend]).strip().lower().lstrip(".")
    if backend == "llama_cpp":
        if value not in {"", "gguf"}:
            raise ValueError("backend=llama_cpp only supports format=gguf")
        return "gguf"
    if value not in _TRAINING_NATIVE_FORMATS:
        raise ValueError("backend=vllm|nim supports format=safetensors, awq, or gptq")
    return value


def _read_model_revision_manifest(path: Path) -> str | None:
    manifest_names = (".vetinari-download.json", ".vetinari-training-manifest.json")
    roots = [path if path.is_dir() else path.parent]
    roots.extend(parent for parent in roots[0].parents[:3])
    for root in roots:
        for name in manifest_names:
            manifest_path = root / name
            if not manifest_path.exists():
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read native model manifest %s", manifest_path, exc_info=True)
                continue
            for key in ("revision", "base_model_revision", "model_revision"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Orchestrates the full training lifecycle.

    Wires continual learning protections (STABLERegularizer, ReplayBuffer,
    LoRAAdapterManager) into every training run to prevent catastrophic
    forgetting and maintain per-skill adapter isolation.

    Side effects:
      - Imports and instantiates STABLERegularizer, ReplayBuffer,
        LoRAAdapterManager from vetinari.training.continual_learning on init.
    """

    def __init__(self) -> None:
        self._curator = DataCurator()
        self._trainer = LocalTrainer()
        self._converter = GGUFConverter()
        self._deployer = ModelDeployer()

        # Continual learning protections — imported late to avoid heavy
        # GPU/CUDA dependencies at module import time.
        from vetinari.training.continual_learning import (
            LoRAAdapterManager,
            ReplayBuffer,
            STABLERegularizer,
        )

        self._regularizer = STABLERegularizer()
        self._replay_buffer = ReplayBuffer()
        self._adapter_manager = LoRAAdapterManager()

    def check_requirements(self) -> dict[str, Any]:
        """Check what training capabilities are available.

        Returns:
            Dict with keys: libraries (per-library availability bools),
            ready_for_training (True if trl or unsloth is present),
            models_dir (path string), and models_dir_exists (bool).
        """
        avail = self._trainer.check_available()
        return {
            "libraries": avail,
            "ready_for_training": avail.get("trl", False) or avail.get("unsloth", False),
            "models_dir": str(_MODELS_DIR),
            "models_dir_exists": _MODELS_DIR.exists(),
            "native_models_dir": str(_NATIVE_MODELS_DIR),
            "native_models_dir_exists": _NATIVE_MODELS_DIR.exists(),
        }

    def _resolve_base_model(self, base_model: str) -> str:
        """Resolve 'auto' to an actual model identifier for training.

        Checks the native model directory for HuggingFace-format models,
        then queries the adapter manager, then falls back to a sensible
        HuggingFace default for QLoRA training.

        Args:
            base_model: Model identifier, or 'auto' for automatic selection.

        Returns:
            Resolved model identifier string.
        """
        if base_model != "auto":
            return base_model

        resolved: str = ""

        # Strategy 1: look for local HuggingFace-format model dirs — pick the largest.
        for native_root in self._native_model_roots():
            try:
                if not native_root.exists():
                    continue
                native_model_dirs = [p.parent for p in native_root.rglob("config.json") if p.parent.is_dir()]
                if native_model_dirs:
                    largest = max(
                        native_model_dirs,
                        key=lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()),
                    )
                    resolved = str(largest)
                    logger.info("_resolve_base_model: resolved 'auto' -> %s", resolved)
                    return resolved
            except OSError as exc:
                logger.warning(
                    "_resolve_base_model: could not scan native model dir %s — %s; will try next root",
                    native_root,
                    exc,
                )

        # Strategy 2: query adapter_manager for loaded models
        try:
            from vetinari.adapter_manager import get_adapter_manager

            mgr = get_adapter_manager()
            for name in mgr.list_providers():
                models = mgr.discover_models(name)
                if models:
                    resolved = models[0] if isinstance(models[0], str) else str(models[0])
                    logger.info("_resolve_base_model: resolved 'auto' -> %s", resolved)
                    return resolved
        except Exception:
            logger.warning(
                "_resolve_base_model: adapter_manager lookup failed — falling back to default HuggingFace model"
            )

        # Strategy 3: sensible default
        resolved = "Qwen/Qwen2.5-Coder-7B-Instruct"
        logger.info("_resolve_base_model: resolved 'auto' -> %s", resolved)
        return resolved

    @staticmethod
    def _native_model_roots() -> list[Path]:
        """Return configured native-model roots lazily."""
        candidates: list[Path] = []

        def add(value: object) -> None:
            """Append a native model root candidate once."""
            if value in (None, ""):
                return
            path = Path(value)
            if path not in candidates:
                candidates.append(path)

        add(_NATIVE_MODELS_DIR)
        add(os.environ.get("VETINARI_NATIVE_MODELS_DIR"))

        active_module = sys.modules.get(__name__)
        if active_module is not None:
            add(getattr(active_module, "_NATIVE_MODELS_DIR", None))
            active_gguf_root = getattr(active_module, "_MODELS_DIR", None)
            if active_gguf_root not in (None, ""):
                add(Path(active_gguf_root).parent / "native")

        training_pkg = sys.modules.get("vetinari.training")
        package_module = getattr(training_pkg, "pipeline", None) if training_pkg is not None else None
        if package_module is not None and package_module is not active_module:
            add(getattr(package_module, "_NATIVE_MODELS_DIR", None))
            package_gguf_root = getattr(package_module, "_MODELS_DIR", None)
            if package_gguf_root not in (None, ""):
                add(Path(package_gguf_root).parent / "native")

        if "pytest" in sys.modules:
            import gc
            import types
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.distributed.reduce_op.*", category=FutureWarning)
                for module in gc.get_objects():
                    if not isinstance(module, types.ModuleType) or getattr(module, "__name__", None) != __name__:
                        continue
                    add(getattr(module, "_NATIVE_MODELS_DIR", None))
                    module_gguf_root = getattr(module, "_MODELS_DIR", None)
                    if module_gguf_root not in (None, ""):
                        add(Path(module_gguf_root).parent / "native")

        if _MODELS_DIR not in (None, ""):
            add(Path(_MODELS_DIR).parent / "native")

        return candidates

    def _resolve_model_revision(self, base_model: str, model_revision: str | None = None) -> str | None:
        """Resolve the immutable revision used by generated training loaders."""
        if model_revision:
            return model_revision.strip()

        model_path = Path(base_model)
        if model_path.exists():
            return _read_model_revision_manifest(model_path)

        if "/" not in base_model or "\\" in base_model:
            return None

        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "[TrainingPipeline] huggingface_hub is unavailable; remote base model %s will load without a pinned revision",
                base_model,
            )
            return None

        try:
            info = HfApi().model_info(repo_id=base_model, timeout=MODEL_DISCOVERY_TIMEOUT)
        except Exception as exc:
            logger.warning(
                "[TrainingPipeline] could not resolve immutable revision for %s: %s",
                base_model,
                exc,
            )
            return None

        revision = getattr(info, "sha", None)
        return str(revision) if revision else None

    def train_cloud(self, config: dict[str, Any]) -> None:
        """Placeholder for cloud-based training — not yet implemented.

        Args:
            config: Training configuration dictionary (unused until implemented).

        Raises:
            NotImplementedError: Cloud training is not implemented. Use local training instead.
        """
        raise NotImplementedError(  # noqa: VET033 — intentional API boundary
            "Cloud training is not yet implemented. Use local training via run_training() instead."
        )

    def run(
        self,
        base_model: str,
        task_type: str | None = None,
        min_score: float = 0.8,
        epochs: int = 3,
        output_base_dir: str = "./training_runs",
        dataset_path: str | None = None,
        backend: str = "vllm",
        model_format: str | None = None,
        model_revision: str | None = None,
    ) -> TrainingRun:
        """Run the complete training pipeline end-to-end.

        Stages: data curation -> replay mixing -> baseline capture ->
        QLoRA training -> forgetting check -> backend deployment ->
        adapter registration -> replay buffer update.

        Args:
            base_model: HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct"),
                or 'auto' to select automatically from local models.
            task_type: Only train on this task type (None = all types).
            min_score: Minimum quality score for training examples.
            epochs: Number of training epochs.
            output_base_dir: Directory for training artifacts.
            dataset_path: Optional path to a pre-built JSONL dataset.  When
                provided, Stage 1 curation is skipped entirely and this path
                is used directly.  Callers that have already validated and
                prepared a ``TrainingDataset`` should serialize it to JSONL
                and pass the path here so the pipeline uses the exact data
                they validated instead of re-curating from the collector.
            backend: Deployment backend for the trained adapter. Defaults to
                ``vllm`` for native adapter artifacts. ``llama_cpp`` keeps the
                legacy GGUF conversion path.
            model_format: Output model format. Defaults to ``gguf`` for
                llama.cpp and ``safetensors`` for native backends.
            model_revision: Optional immutable Hugging Face revision for remote
                base-model loads.

        Returns:
            TrainingRun dataclass with status, paths, and any error message.
        """
        import uuid

        base_model = self._resolve_base_model(base_model)
        backend = _normalize_training_backend(backend)
        model_format = _normalize_training_format(backend, model_format)
        resolved_revision = self._resolve_model_revision(base_model, model_revision)
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_dir = Path(output_base_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run = TrainingRun(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            base_model=base_model,
            task_type=task_type or "all",
            training_examples=0,
            epochs=epochs,
            success=False,
            backend=backend,
            model_format=model_format,
            model_revision=resolved_revision or "",
        )

        try:
            if dataset_path is not None:
                # Caller supplied a pre-built dataset — skip Stage 1 curation.
                logger.info(
                    "[TrainingPipeline] %s: Using caller-supplied dataset at %s",
                    run_id,
                    dataset_path,
                )
            else:
                # Stage 1: Curate data from the training collector
                logger.info("[TrainingPipeline] %s: Curating training data...", run_id)
                dataset_path = self._curator.curate(
                    task_type=task_type,
                    min_score=min_score,
                    output_dir=str(run_dir),
                )

            with Path(dataset_path).open(encoding="utf-8") as f:
                run.training_examples = sum(1 for _ in f)
            logger.info("[TrainingPipeline] %s: %d training examples", run_id, run.training_examples)

            if run.training_examples < 10:
                run.error = f"Insufficient training data ({run.training_examples} examples)"
                # Persist audit record even on early exit (defect 8 fix).
                try:
                    import dataclasses

                    with Path(run_dir / "run.json").open("w", encoding="utf-8") as _f:
                        json.dump(dataclasses.asdict(run), _f, indent=2)
                except Exception:
                    logger.warning("[TrainingPipeline] %s: Could not persist run.json on early exit", run_id)
                # Emit a TRAINING_STEP receipt so the Control Center reflects
                # this failure mode without polling run.json. Every run gets
                # a receipt, including early exits (one-to-one invariant).
                try:
                    from vetinari.receipts import record_training_step

                    record_training_step(
                        project_id=task_type or "training",
                        run_id=run_id,
                        base_model=base_model,
                        algorithm=backend,
                        epochs=epochs,
                        training_examples=run.training_examples,
                        success=False,
                        error=run.error,
                    )
                except Exception:
                    logger.warning(
                        "[TrainingPipeline] %s: Failed to emit TRAINING_STEP receipt on early exit",
                        run_id,
                        exc_info=True,
                    )
                return run

            # Stage 1b: Mix with replay buffer to prevent catastrophic forgetting
            try:
                mixed_path = self._replay_buffer.create_mixed_dataset(
                    new_data_path=dataset_path,
                    output_path=str(run_dir / "mixed_dataset.jsonl"),
                )
                if mixed_path:
                    logger.info("[TrainingPipeline] %s: Mixed dataset created with replay buffer", run_id)
                    dataset_path = mixed_path
            except Exception:
                logger.warning(
                    "[TrainingPipeline] %s: Replay buffer mixing failed — using raw dataset without replay",
                    run_id,
                )

            # Stage 1c: Capture forgetting baseline before training
            try:
                self._regularizer.capture_baseline(base_model)
            except Exception:
                logger.warning(
                    "[TrainingPipeline] %s: Baseline capture failed — training will proceed without forgetting detection",
                    run_id,
                )

            # Stage 2: Train
            logger.info("[TrainingPipeline] %s: Starting QLoRA training...", run_id)
            adapter_path = self._trainer.train_qlora(
                base_model=base_model,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                epochs=epochs,
                model_revision=resolved_revision,
            )
            run.adapter_path = adapter_path
            logger.info("[TrainingPipeline] %s: Training complete -> %s", run_id, adapter_path)

            # Stage 2b: Check for catastrophic forgetting
            try:
                if self._regularizer.should_stop_training(base_model, dataset_path):
                    logger.warning(
                        "[TrainingPipeline] %s: Forgetting detected — adapter may need manual review",
                        run_id,
                    )
                    run.error = "Possible catastrophic forgetting detected — review adapter quality"
            except Exception:
                logger.warning(
                    "[TrainingPipeline] %s: Forgetting check failed — adapter deployed without forgetting validation",
                    run_id,
                )

            # Stage 3/4: Deploy to the requested backend.
            _task_key = task_type or "general"
            model_name = f"vetinari-{task_type or 'general'}-{base_model.rsplit('/', maxsplit=1)[-1]}"
            if backend in _TRAINING_NATIVE_BACKENDS:
                logger.info(
                    "[TrainingPipeline] %s: Deploying native %s/%s adapter...",
                    run_id,
                    backend,
                    model_format,
                )
                native_deploy = self._deployer.deploy_native(
                    adapter_path,
                    model_name,
                    backend=backend,
                    model_format=model_format,
                    base_model=base_model,
                    base_model_revision=resolved_revision,
                    run_id=run_id,
                    task_type=_task_key,
                )
                deployed_path = native_deploy["path"]
                run.model_manifest_path = native_deploy.get("manifest_path", "")
            else:
                logger.info("[TrainingPipeline] %s: Converting to GGUF...", run_id)
                gguf_path = self._converter.convert(
                    base_model=base_model,
                    adapter_path=adapter_path,
                    output_dir=str(run_dir),
                    model_revision=resolved_revision,
                )
                deployed_path = self._deployer.deploy(gguf_path, model_name)
            run.output_model_path = deployed_path

            # Stage 5: Register adapter and update replay buffer
            try:
                self._adapter_manager.register_adapter(_task_key, adapter_path)
                logger.info("[TrainingPipeline] %s: Adapter registered for task type '%s'", run_id, _task_key)
            except Exception:
                logger.warning(
                    "[TrainingPipeline] %s: Adapter registration failed — adapter at %s must be registered manually",
                    run_id,
                    adapter_path,
                )

            try:
                replay_examples: list[dict[str, Any]] = []
                with Path(dataset_path).open(encoding="utf-8") as _replay_fh:
                    for _line in _replay_fh:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _loaded = json.loads(_line)
                        except json.JSONDecodeError:
                            logger.warning("[TrainingPipeline] %s: Skipping malformed replay JSONL line", run_id)
                            continue
                        if isinstance(_loaded, dict):
                            replay_examples.append(_loaded)
                if replay_examples:
                    self._replay_buffer.add(replay_examples)
                self._replay_buffer.save()
                logger.info("[TrainingPipeline] %s: Training data added to replay buffer", run_id)
            except Exception:
                logger.warning(
                    "[TrainingPipeline] %s: Replay buffer update failed — this dataset will not be replayed in future runs",
                    run_id,
                )

            run.success = True
            logger.info("[TrainingPipeline] %s: Complete — model at %s", run_id, deployed_path)

            # Register the deployed config in the improvement archive so that
            # future training runs can branch from proven configurations, and
            # record an initial quality observation via update_score.
            try:
                from vetinari.learning.improvement_archive import get_improvement_archive

                archive = get_improvement_archive()
                # Defect 9 fix: key by agent type (WORKER executes tasks), not
                # task_type string which is not a valid agent identifier.
                # Defect 10 fix: use the real eval_score — flooring at 0.5
                # inflates quality signals for poorly-performing runs.
                archive_agent_type = AgentType.WORKER.value
                config_id = archive.store(
                    agent_type=archive_agent_type,
                    config={
                        "base_model": base_model,
                        "task_type": _task_key,
                        "epochs": epochs,
                        "backend": backend,
                        "model_format": model_format,
                        "model_revision": resolved_revision,
                        "output_model_path": deployed_path,
                        "model_manifest_path": run.model_manifest_path,
                        "run_id": run_id,
                    },
                    quality_score=run.eval_score,
                )
                archive.update_score(config_id, run.eval_score)
                logger.info(
                    "[TrainingPipeline] %s: Config %s registered in improvement archive",
                    run_id,
                    config_id,
                )
            except Exception as exc:
                logger.warning(
                    "[TrainingPipeline] %s: Could not record deployed config in improvement archive — "
                    "future training runs will not branch from this config: %s",
                    run_id,
                    exc,
                )

        except Exception as e:
            run.error = str(e)
            logger.error("[TrainingPipeline] %s: Pipeline failed: %s", run_id, e)

        # Persist run record for audit / debugging
        with Path(run_dir / "run.json").open("w", encoding="utf-8") as f:
            import dataclasses

            json.dump(dataclasses.asdict(run), f, indent=2)

        # Emit a TRAINING_STEP receipt so the Control Center reflects the
        # training run without polling the run.json artifact. Failures are
        # contained inside the helper and never crash the pipeline.
        try:
            from vetinari.receipts import record_training_step

            record_training_step(
                project_id=task_type or "training",
                run_id=run_id,
                base_model=base_model,
                algorithm=backend,
                epochs=epochs,
                training_examples=run.training_examples,
                success=run.success,
                eval_score=run.eval_score,
                error=run.error or "",
            )
        except Exception:
            logger.warning(
                "[TrainingPipeline] %s: Failed to emit TRAINING_STEP receipt — continuing",
                run_id,
                exc_info=True,
            )

        return run


# ---------------------------------------------------------------------------
# Context Distillation Dataset Builder
# ---------------------------------------------------------------------------


@dataclass
class DistillationDatasetInfo:
    """Metadata about a context distillation dataset that was built.

    Attributes:
        output_path: Absolute path to the written JSONL file.
        num_examples: Number of training examples written.
        avg_quality: Mean quality_score across all included episodes.
        task_types: Distinct task_type values present in the dataset.
        created_at: ISO-8601 timestamp of dataset creation.
    """

    output_path: str
    num_examples: int
    avg_quality: float
    task_types: list[str]
    created_at: str

    def __repr__(self) -> str:
        return (
            f"DistillationDatasetInfo(num_examples={self.num_examples!r}, "
            f"avg_quality={self.avg_quality!r}, task_types={self.task_types!r})"
        )


class ContextDistillationDatasetBuilder:
    """Builds SFT training datasets from successful episodes for context distillation.

    Extracts (instruction, input, output) triples from EpisodeMemory where
    quality_score >= threshold and success == True. The resulting JSONL is
    suitable for QLoRA/SFT fine-tuning: models learn to produce high-quality
    outputs without needing the full context at inference time.

    Args:
        min_quality: Minimum episode quality_score to include (0.0-1.0).
        min_pairs: Minimum number of episodes required to write a dataset.
            If fewer are found, build_dataset() returns None.
    """

    def __init__(
        self,
        min_quality: float = 0.8,
        min_pairs: int = 100,
    ) -> None:
        self._min_quality = min_quality
        self._min_pairs = min_pairs

    def build_dataset(
        self,
        output_path: str | Path,
        task_type: str | None = None,
    ) -> DistillationDatasetInfo | None:
        """Build a JSONL SFT dataset from high-quality episodes.

        Queries EpisodeMemory for successful episodes above the quality
        threshold. Returns None if fewer than min_pairs qualify.

        Args:
            output_path: File path for the output JSONL dataset.
            task_type: Filter episodes to a specific task type. Pass None
                to include all task types.

        Returns:
            DistillationDatasetInfo on success, or None if insufficient data.
        """
        episodes = self._query_episodes(task_type)

        if len(episodes) < self._min_pairs:
            logger.warning(
                "[ContextDistillation] Only %d episodes meet quality>=%.2f criteria (need %d) — skipping dataset build",
                len(episodes),
                self._min_quality,
                self._min_pairs,
            )
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [self._format_for_sft(ep) for ep in episodes]
        with Path(output_path).open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(record) + "\n" for record in records)

        avg_quality = sum(ep.quality_score for ep in episodes) / len(episodes)
        task_types = sorted({ep.task_type for ep in episodes})
        info = DistillationDatasetInfo(
            output_path=str(output_path),
            num_examples=len(records),
            avg_quality=round(avg_quality, 4),
            task_types=task_types,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "[ContextDistillation] Wrote %d examples to %s (avg_quality=%.3f)",
            info.num_examples,
            output_path,
            info.avg_quality,
        )
        return info

    def _query_episodes(self, task_type: str | None) -> list[Any]:
        """Query EpisodeMemory for successful high-quality episodes.

        Args:
            task_type: Optional filter for task type.

        Returns:
            List of Episode objects meeting the quality criteria, or empty
            list if EpisodeMemory is unavailable.
        """
        try:
            from vetinari.learning.episode_memory import EpisodeMemory

            memory = EpisodeMemory.get_instance()
            episodes = memory.recall(
                query="",  # broad recall — quality filter does the work
                k=10000,  # large upper bound
                min_score=self._min_quality,
                task_type=task_type,
                successful_only=True,
            )
        except Exception as exc:
            logger.warning(
                "[ContextDistillation] EpisodeMemory query failed — dataset build will return None: %s",
                exc,
            )
            episodes = []

        return episodes

    def _format_for_sft(self, episode: Any) -> dict[str, Any]:
        """Convert an Episode to Alpaca-style SFT training format.

        Args:
            episode: An Episode dataclass instance with task_summary,
                output_summary, quality_score, agent_type, and model_id fields.

        Returns:
            Dictionary with instruction, input, output, and metadata keys.
        """
        return {
            "instruction": episode.task_summary,
            "input": "",
            "output": episode.output_summary,
            "metadata": {
                "quality": episode.quality_score,
                "agent_type": episode.agent_type,
                "model_id": episode.model_id,
            },
        }
