"""Training data seeding — bootstrap from day one.

Downloads a curated set of seed datasets on first run so that the training
pipeline has data to work with before any real task history has accumulated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


def _training_data_dir() -> Path:
    """Return the configured training-data cache root lazily."""
    return get_user_dir() / "training_data"


def _seed_marker() -> Path:
    """Return the seed-completion marker under the configured cache root."""
    return _training_data_dir() / ".seeded"


@dataclass
class SeedDataset:
    """Specification for a seed dataset to bootstrap training.

    Attributes:
        name: HuggingFace dataset identifier (e.g. ``"mbpp"``).
        domain: Broad topic area covered by the dataset.
        size: Target number of examples to retain.
        description: Human-readable summary of what this dataset provides.
        subsample: When True, randomly subsample the dataset to ``size``.
    """

    name: str
    domain: str
    size: int
    description: str
    subsample: bool = False

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"SeedDataset(name={self.name!r}, domain={self.domain!r}, size={self.size!r})"


class TrainingDataSeeder:
    """Seeds the local training data store with curated external datasets.

    Checks whether data already exists before downloading anything.  Individual
    dataset failures are caught and logged so the remaining seeds can proceed.
    Uses late imports for optional ``vetinari`` training dependencies.
    """

    SEED_DATASETS: list[SeedDataset] = [
        SeedDataset(
            name="codeparrot/apps",
            domain="coding_eval",
            size=5000,
            description="Coding problems from competitive programming for evaluation",
        ),
        SeedDataset(
            name="mbpp",
            domain="coding",
            size=1000,
            description="Basic Python problems — foundational Python coding coverage",
        ),
        SeedDataset(
            name="hendrycks/competition_math",
            domain="reasoning",
            size=5000,
            description="Competition math — chain-of-thought reasoning from structured problem sets",
            subsample=True,
        ),
        SeedDataset(
            name="tatsu-lab/alpaca",
            domain="instruction",
            size=10000,
            description="Instruction following — broad generalist instruction-tuning data",
            subsample=True,
        ),
    ]

    def __init__(self) -> None:
        """Initialise the seeder with lazy-loaded dependencies."""
        self._manager: Any = None

    def seed_if_empty(self) -> int:
        """Download seed datasets if no training data exists yet.

        Checks for existing data first and skips entirely when data is found.
        For each seed dataset, logs progress and handles download failures
        gracefully — a failure on one dataset does not abort the remaining seeds.

        Returns:
            The number of datasets successfully seeded on this call (0 when
            data already existed or all downloads failed).
        """
        if self._training_data_exists():
            logger.info("Training data already exists — skipping seed download")
            return 0

        logger.info(
            "No training data found — seeding %d dataset(s)",
            len(self.SEED_DATASETS),
        )

        manager = self._get_manager()
        if manager is None:
            # Attempt auto-install of datasets library
            from vetinari.training.pipeline import _ensure_packages

            install_results = _ensure_packages(["datasets"])
            logger.info("seed_if_empty: auto-install results: %s", install_results)
            # Re-initialize the manager after install
            self._manager = None
            manager = self._get_manager()

        if manager is None:
            logger.warning(
                "ExternalDataManager unavailable even after install attempt; cannot seed training data",
            )
            return 0

        # Re-check availability after potential install
        if hasattr(manager, "is_available") and not manager.is_available():
            from vetinari.training.pipeline import _ensure_packages

            _ensure_packages(["datasets"])
            # Re-initialize manager to pick up newly installed library
            self._manager = None
            manager = self._get_manager()
            if manager is None or (hasattr(manager, "is_available") and not manager.is_available()):
                logger.info("seed_if_empty: 'datasets' library still unavailable after install attempt")
                return 0

        training_data_dir = _training_data_dir()
        training_data_dir.mkdir(parents=True, exist_ok=True)
        seeded_count = 0

        for seed in self.SEED_DATASETS:
            logger.info(
                "Seeding dataset '%s' (domain=%s, size=%d) ...",
                seed.name,
                seed.domain,
                seed.size,
            )
            try:
                from vetinari.training.external_data import DatasetSpec

                spec = DatasetSpec(
                    name=seed.name,
                    domain=seed.domain,
                    format="sft",
                    description=seed.description,
                    max_examples=seed.size,
                )
                path = manager.download_dataset(spec)
                logger.info("Seeded '%s' -> %s", seed.name, path)
                seeded_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to seed dataset '%s': %s — continuing with remaining datasets",
                    seed.name,
                    exc,
                )

        if seeded_count > 0:
            # Write marker so subsequent runs skip the seed phase
            try:
                _seed_marker().write_text(
                    f"seeded={seeded_count} datasets\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not write seed marker: %s", exc)

        logger.info(
            "Seed phase complete: %d/%d dataset(s) downloaded",
            seeded_count,
            len(self.SEED_DATASETS),
        )
        return seeded_count

    def seed_with_progress(self) -> Iterator[dict[str, Any]]:
        """Download seed datasets, yielding progress events as a generator.

        Each yielded dict contains progress information suitable for SSE
        streaming to a frontend progress indicator.  Events:

        - ``start``: total count and dataset names
        - ``progress``: per-dataset status with percent and ETA
        - ``done``: final summary

        Yields:
            Progress event dicts with keys ``event``, ``percent``,
            ``eta_seconds``, ``dataset``, ``status``, etc.
        """
        datasets = self.SEED_DATASETS
        total = len(datasets)

        if self._training_data_exists():
            yield {
                "event": "done",
                "seeded": 0,
                StatusEnum.FAILED.value: 0,
                "total_examples": 0,
                "message": "Training data already exists",
            }
            return

        manager = self._get_manager()

        training_data_dir = _training_data_dir()
        training_data_dir.mkdir(parents=True, exist_ok=True)

        # Auto-install required packages before proceeding
        yield {"event": "installing", "message": "Checking and installing required packages..."}

        from vetinari.training.pipeline import _ensure_packages

        install_results = _ensure_packages(["datasets"])
        yield {"event": "installing", "message": "Installed datasets library", "results": install_results}

        training_results = _ensure_packages(["trl", "peft", "bitsandbytes", "transformers"])
        yield {"event": "installing", "message": "Installed training libraries", "results": training_results}

        # Re-initialize manager after install to pick up newly available libraries
        self._manager = None
        manager = self._get_manager()

        if manager is None:
            yield {
                "event": "error",
                "error": "ExternalDataManager unavailable even after install attempt",
            }
            return

        if hasattr(manager, "is_available") and not manager.is_available():
            yield {
                "event": "error",
                "error": "The 'datasets' library is not available even after install attempt",
            }
            return

        yield {
            "event": "start",
            "total": total,
            "datasets": [s.name for s in datasets],
        }

        seeded = 0
        failed = 0
        total_examples = 0
        elapsed_times: list[float] = []

        for idx, seed in enumerate(datasets):
            # Compute ETA from average of completed downloads
            avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
            remaining = total - idx
            eta_seconds = round(avg_time * remaining, 1) if elapsed_times else None

            yield {
                "event": "progress",
                "dataset": seed.name,
                "index": idx + 1,
                "total": total,
                "percent": round(idx / total * 100),
                "status": "downloading",
                "eta_seconds": eta_seconds,
            }

            t0 = time.monotonic()
            try:
                from vetinari.training.external_data import DatasetSpec

                spec = DatasetSpec(
                    name=seed.name,
                    domain=seed.domain,
                    format="sft",
                    description=seed.description,
                    max_examples=seed.size,
                )
                path = manager.download_dataset(spec)
                elapsed = time.monotonic() - t0
                elapsed_times.append(elapsed)

                # Count examples in the downloaded file
                examples = 0
                try:
                    with Path(path).open(encoding="utf-8") as fh:
                        examples = sum(1 for ln in fh if ln.strip())
                except OSError:
                    logger.warning("Could not read seed file: %s", path)

                seeded += 1
                total_examples += examples
                logger.info("Seeded '%s' -> %s (%d examples)", seed.name, path, examples)

                yield {
                    "event": "progress",
                    "dataset": seed.name,
                    "index": idx + 1,
                    "total": total,
                    "percent": round((idx + 1) / total * 100),
                    "status": "complete",
                    "examples": examples,
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": (
                        round(sum(elapsed_times) / len(elapsed_times) * (total - idx - 1), 1) if idx + 1 < total else 0
                    ),
                }
            except Exception as exc:
                elapsed = time.monotonic() - t0
                elapsed_times.append(elapsed)
                failed += 1
                logger.warning("Failed to seed '%s': %s", seed.name, exc)

                yield {
                    "event": "progress",
                    "dataset": seed.name,
                    "index": idx + 1,
                    "total": total,
                    "percent": round((idx + 1) / total * 100),
                    "status": StatusEnum.FAILED.value,
                    "error": str(exc),
                }

        if seeded > 0:
            try:
                _seed_marker().write_text(
                    f"seeded={seeded} datasets\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Could not write seed marker: %s", exc)

        yield {
            "event": "done",
            "seeded": seeded,
            StatusEnum.FAILED.value: failed,
            "total_examples": total_examples,
        }

    def get_seed_status(self) -> dict[str, Any]:
        """Return the current seed status for each configured dataset.

        Returns:
            Dictionary with keys:
            - ``total_seed_datasets``: total number of configured seed datasets
            - ``downloaded``: list of dataset names present in the local cache
            - ``pending``: list of dataset names not yet downloaded
            - ``total_examples``: count of training examples across downloaded files
            - ``data_dir``: path to the training data directory
        """
        manager = self._get_manager()
        downloaded: list[str] = []
        pending: list[str] = []
        total_examples = 0

        for seed in self.SEED_DATASETS:
            # Derive expected file path using the same slug logic as ExternalDataManager
            safe_name = seed.name.replace("/", "__")
            training_data_dir = _training_data_dir()
            expected_path = training_data_dir / f"{safe_name}.jsonl"

            if expected_path.exists() and expected_path.stat().st_size > 0:
                downloaded.append(seed.name)
                try:
                    with expected_path.open(encoding="utf-8") as fh:
                        total_examples += sum(1 for ln in fh if ln.strip())
                except OSError as exc:
                    logger.warning(
                        "Could not count examples in %s: %s",
                        expected_path,
                        exc,
                    )
            else:
                pending.append(seed.name)

        # If manager available, cross-reference its view
        if manager:
            try:
                available = manager.get_available_datasets()
                seed_names = {s.name for s in self.SEED_DATASETS}
                extra_downloaded = [d.name for d in available if d.downloaded and d.name in seed_names]
                # Merge — prefer manager's authoritative path check
                downloaded = list(set(downloaded) | set(extra_downloaded))
                pending = [s.name for s in self.SEED_DATASETS if s.name not in set(downloaded)]
            except Exception as exc:
                logger.warning("Could not query manager for seed status: %s", exc)

        return {
            "total_seed_datasets": len(self.SEED_DATASETS),
            "downloaded": sorted(downloaded),
            StatusEnum.PENDING.value: sorted(pending),
            "total_examples": total_examples,
            "data_dir": str(_training_data_dir()),
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _training_data_exists(self) -> bool:
        """Check whether any training data files already exist.

        Returns:
            True if the seed marker file is present or at least one non-empty
            JSONL file exists in the training data directory.
        """
        training_data_dir = _training_data_dir()
        if _seed_marker().exists():
            return True

        if not training_data_dir.exists():
            return False

        jsonl_files = list(training_data_dir.glob("*.jsonl"))
        return any(f.stat().st_size > 0 for f in jsonl_files)

    def _get_manager(self) -> Any:
        """Return a lazily initialised ExternalDataManager, or None.

        Returns:
            Configured ExternalDataManager instance, or None if the
            ``vetinari.training.external_data`` module cannot be imported.
        """
        if self._manager is not None:
            return self._manager

        try:
            from vetinari.training.external_data import ExternalDataManager

            self._manager = ExternalDataManager(cache_dir=_training_data_dir())
            return self._manager
        except ImportError as exc:
            logger.debug("ExternalDataManager not available: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level get_training_data_seeder singleton
# ---------------------------------------------------------------------------
# Exposes the canonical TrainingDataSeeder so all callers share one instance —
# avoids re-creating the seeder (and its lazy ExternalDataManager) per request.

_seeder_instance: TrainingDataSeeder | None = None
_seeder_instance_lock: threading.Lock = threading.Lock()


def get_training_data_seeder() -> TrainingDataSeeder:
    """Return the canonical TrainingDataSeeder singleton.

    Uses double-checked locking so the first call creates the instance and
    all subsequent calls return the same object with no lock contention.

    Returns:
        The shared TrainingDataSeeder instance.
    """
    global _seeder_instance
    if _seeder_instance is not None:
        return _seeder_instance
    with _seeder_instance_lock:
        if _seeder_instance is not None:
            return _seeder_instance
        _seeder_instance = TrainingDataSeeder()
    logger.debug("get_training_data_seeder: created new singleton")
    return _seeder_instance
