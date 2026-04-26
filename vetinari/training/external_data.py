"""External training data manager for acquiring datasets from HuggingFace Hub.

This module provides utilities to discover, download, and mix external datasets
for fine-tuning Vetinari's models. It supports SFT and DPO formats, converts
all datasets to Alpaca-style JSONL, and manages a local cache.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir

logger = logging.getLogger(__name__)

# Default fraction of own data when mixing datasets
DEFAULT_OWN_DATA_RATIO = 0.6

# Default upper bound on mixed dataset size
DEFAULT_MAX_MIXED_TOTAL = 10000

# Subdirectory name used within the home-based cache
CACHE_SUBDIR = ".vetinari/training_data"


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a HuggingFace dataset to acquire.

    Args:
        name: HuggingFace dataset identifier, e.g. "mbpp".
        domain: Broad topic area, e.g. "coding", "reasoning", "instruction", "alignment".
        format: Training format — "sft" or "dpo".
        description: Human-readable summary of the dataset.
        max_examples: Maximum number of examples to retain after download.
        subset: Optional dataset subset / config name on HuggingFace Hub.
    """

    name: str
    domain: str
    format: str
    description: str
    max_examples: int
    subset: str | None = None

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"DatasetSpec(name={self.name!r}, domain={self.domain!r}, format={self.format!r})"


@dataclass
class DatasetInfo:
    """Runtime information about a dataset, including download status.

    Args:
        name: HuggingFace dataset identifier.
        domain: Broad topic area.
        size: Number of examples available (or expected).
        estimated_train_minutes: Rough training-time estimate in minutes.
        downloaded: Whether the dataset is present in the local cache.
        path: Local path to the JSONL file if downloaded, else None.
    """

    name: str
    domain: str
    size: int
    estimated_train_minutes: int
    downloaded: bool = False
    path: Path | None = None

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"DatasetInfo(name={self.name!r}, domain={self.domain!r}, downloaded={self.downloaded!r})"


class ExternalDataManager:
    """Manages acquisition and preparation of external training datasets.

    Downloads datasets from HuggingFace Hub, converts them to Alpaca-style
    JSONL, and supports mixing with locally-generated data for fine-tuning.

    Attributes:
        DATASET_CATALOG: Class-level registry of curated datasets by category.
    """

    DATASET_CATALOG: dict[str, list[DatasetSpec]] = {
        "code_sft": [
            DatasetSpec(
                name="bigcode/the-stack-v2-dedup",
                domain="coding",
                format="sft",
                description="Deduplicated source-code corpus from Software Heritage (Python subset).",
                max_examples=50000,
                subset="python",
            ),
            DatasetSpec(
                name="deepmind/code_contests",
                domain="coding",
                format="sft",
                description="Competitive programming problems and solutions from DeepMind.",
                max_examples=10000,
            ),
            DatasetSpec(
                name="mbpp",
                domain="coding",
                format="sft",
                description="Mostly Basic Python Problems benchmark dataset.",
                max_examples=1000,
            ),
            DatasetSpec(
                name="codeparrot/apps",
                domain="coding",
                format="sft",
                description="APPS coding challenge dataset with test cases.",
                max_examples=10000,
            ),
        ],
        "reasoning_sft": [
            DatasetSpec(
                name="codeparrot/apps",
                domain="reasoning",
                format="sft",
                description="Live competitive programming problems for reasoning evaluation.",
                max_examples=5000,
            ),
            DatasetSpec(
                name="hendrycks/competition_math",
                domain="reasoning",
                format="sft",
                description="Competition-level mathematics problems with step-by-step solutions.",
                max_examples=12500,
            ),
            DatasetSpec(
                name="codeparrot/codecontests",
                domain="reasoning",
                format="sft",
                description="Code contest problems collected by CodeParrot.",
                max_examples=10000,
            ),
        ],
        "instruction_sft": [
            DatasetSpec(
                name="HuggingFaceTB/smoltalk",
                domain="instruction",
                format="sft",
                description="SmolTalk instruction-following dataset from HuggingFace.",
                max_examples=50000,
            ),
            DatasetSpec(
                name="tatsu-lab/alpaca",
                domain="instruction",
                format="sft",
                description="Stanford Alpaca instruction-following dataset.",
                max_examples=52000,
            ),
            DatasetSpec(
                name="Open-Orca/OpenOrca",
                domain="instruction",
                format="sft",
                description="OpenOrca augmented instruction dataset.",
                max_examples=50000,
            ),
        ],
        "preference_dpo": [
            DatasetSpec(
                name="Anthropic/hh-rlhf",
                domain="alignment",
                format="dpo",
                description="Anthropic human-preference data for RLHF/DPO training.",
                max_examples=50000,
            ),
            DatasetSpec(
                name="argilla/ultrafeedback-binarized-preferences",
                domain="alignment",
                format="dpo",
                description="UltraFeedback binarised preference pairs from Argilla.",
                max_examples=60000,
            ),
        ],
    }

    def __init__(
        self,
        cache_dir: Path | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Initialise the manager with a cache directory and optional HF token.

        Args:
            cache_dir: Directory for storing downloaded datasets.
                Defaults to ``~/.vetinari/training_data``.
            hf_token: HuggingFace API token. Falls back to the ``HF_TOKEN``
                environment variable if not provided.
        """
        self.cache_dir: Path = cache_dir or (get_user_dir() / "training_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hf_token: str | None = hf_token or os.environ.get("HF_TOKEN")

        logger.info("ExternalDataManager initialised with cache_dir=%s", self.cache_dir)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_available_datasets(self, domain: str | None = None) -> list[DatasetInfo]:
        """Return a prioritised list of available datasets, optionally filtered by domain.

        Checks which datasets are already present in the local cache and marks
        them accordingly. Downloaded datasets are listed first.

        Args:
            domain: If given, only return datasets whose domain matches this string.

        Returns:
            List of DatasetInfo objects sorted so downloaded datasets appear first.
        """
        infos: list[DatasetInfo] = []

        for _category, specs in self.DATASET_CATALOG.items():
            for spec in specs:
                if domain is not None and spec.domain != domain:
                    continue

                local_path = self._expected_path(spec)
                is_downloaded = local_path.exists()

                # Rough heuristic: 1 minute per 500 examples at typical GPU speed
                est_minutes = max(1, spec.max_examples // 500)

                infos.append(
                    DatasetInfo(
                        name=spec.name,
                        domain=spec.domain,
                        size=spec.max_examples,
                        estimated_train_minutes=est_minutes,
                        downloaded=is_downloaded,
                        path=local_path if is_downloaded else None,
                    ),
                )

        # Prioritise already-downloaded datasets
        infos.sort(key=lambda d: (not d.downloaded, d.name))
        logger.debug(
            "get_available_datasets(domain=%s) -> %d entries",
            domain,
            len(infos),
        )
        return infos

    def download_dataset(self, spec: DatasetSpec) -> Path:
        """Download a dataset from HuggingFace Hub and convert it to Alpaca JSONL.

        Uses a late import of ``datasets.load_dataset`` so that the ``datasets``
        library remains an optional dependency.  The dataset is subsampled to
        ``spec.max_examples`` before writing.

        Args:
            spec: Specification describing which dataset to download.

        Returns:
            Path to the resulting JSONL file in the local cache.

        Raises:
            ImportError: If the ``datasets`` library is not installed.
            RuntimeError: If the download or conversion fails.
        """
        if not self.is_available():
            raise ImportError(
                "The 'datasets' library is required for downloading external data. "
                "Install it with: pip install datasets",  # noqa: VET301 — user guidance string
            )

        from datasets import load_dataset

        output_path = self._expected_path(spec)
        if output_path.exists():
            logger.info("Dataset already cached at %s, skipping download", output_path)
            return output_path

        logger.info("Downloading dataset %s (subset=%s)", spec.name, spec.subset)

        load_kwargs: dict[str, Any] = {"path": spec.name, "split": "train"}
        if spec.subset:
            load_kwargs["name"] = spec.subset
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        try:
            # Curated training datasets intentionally float unless the operator pins a specific revision.
            ds = load_dataset(**load_kwargs)  # nosec B615
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{spec.name}' from HuggingFace: {exc}",
            ) from exc

        # Subsample if needed
        total = len(ds)
        if total > spec.max_examples:
            indices = random.sample(range(total), spec.max_examples)
            ds = ds.select(indices)
            logger.info(
                "Subsampled %s from %d -> %d examples",
                spec.name,
                total,
                spec.max_examples,
            )

        count = self._convert_to_training_format(ds, spec, output_path)
        logger.info(
            "Dataset %s written to %s (%d records)",
            spec.name,
            output_path,
            count,
        )
        return output_path

    def create_mixed_dataset(
        self,
        own_data_path: Path | None,
        external_specs: list[DatasetSpec],
        ratio: float = DEFAULT_OWN_DATA_RATIO,
        max_total: int = DEFAULT_MAX_MIXED_TOTAL,
    ) -> Path:
        """Create a mixed JSONL dataset from own data and external datasets.

        Blends locally-generated examples (``ratio`` fraction) with external
        data (``1 - ratio`` fraction), capped at ``max_total`` records total.

        Args:
            own_data_path: Path to an existing Alpaca-style JSONL file produced
                by Vetinari, or ``None`` if no own data is available.
            external_specs: List of DatasetSpec objects for external datasets to
                include. Each will be downloaded on demand if not cached.
            ratio: Fraction of the final dataset that should come from own data.
                Must be in ``[0, 1]``. Defaults to 0.6.
            max_total: Maximum number of records in the mixed dataset.

        Returns:
            Path to the mixed JSONL file inside the cache directory.

        Raises:
            ValueError: If ``ratio`` is outside ``[0, 1]``.
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")

        own_quota = int(max_total * ratio)
        ext_quota = max_total - own_quota

        own_rows: list[dict[str, Any]] = []
        if own_data_path is not None and own_data_path.exists():
            with own_data_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            own_rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed JSON line in %s",
                                own_data_path,
                            )
            if len(own_rows) > own_quota:
                own_rows = random.sample(own_rows, own_quota)
            logger.info("Own data: %d examples from %s", len(own_rows), own_data_path)

        # Backfill: if own data is absent or yielded fewer rows than the quota,
        # allow the external pool to fill up to max_total instead of leaving
        # the mixed dataset permanently underfilled (defect 3 fix).
        own_shortfall = own_quota - len(own_rows)
        effective_ext_quota = ext_quota + own_shortfall

        ext_rows: list[dict[str, Any]] = []
        for spec in external_specs:
            if len(ext_rows) >= effective_ext_quota:
                break
            try:
                path = self.download_dataset(spec)
            except Exception as exc:
                logger.warning(
                    "Could not obtain external dataset %s: %s",
                    spec.name,
                    exc,
                )
                continue

            remaining = effective_ext_quota - len(ext_rows)
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if remaining <= 0:
                        break
                    line = line.strip()
                    if line:
                        try:
                            ext_rows.append(json.loads(line))
                            remaining -= 1
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed JSON line in %s",
                                path,
                            )

        all_rows = own_rows + ext_rows
        random.shuffle(all_rows)

        mixed_path = self.cache_dir / "mixed_dataset.jsonl"
        with mixed_path.open("w", encoding="utf-8") as out:
            for row in all_rows:
                out.write(json.dumps(row) + "\n")

        logger.info(
            "Mixed dataset written to %s (%d own + %d external = %d total)",
            mixed_path,
            len(own_rows),
            len(ext_rows),
            len(all_rows),
        )
        return mixed_path

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the local dataset cache.

        Returns:
            Dictionary with the following keys:

            - ``total_datasets``: Total number of datasets in the catalog.
            - ``downloaded_count``: Number of datasets present in the cache.
            - ``total_examples``: Sum of examples across all downloaded datasets.
            - ``cache_dir_size_bytes``: Total size of the cache directory in bytes.
            - ``cache_dir``: Absolute path to the cache directory as a string.
        """
        all_specs: list[DatasetSpec] = [spec for specs in self.DATASET_CATALOG.values() for spec in specs]

        downloaded_count = 0
        total_examples = 0

        for spec in all_specs:
            path = self._expected_path(spec)
            if path.exists():
                downloaded_count += 1
                # Count lines as a proxy for example count
                try:
                    with path.open(encoding="utf-8") as fh:
                        total_examples += sum(1 for ln in fh if ln.strip())
                except OSError as exc:
                    logger.warning("Could not count examples in %s: %s", path, exc)

        cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())

        return {
            "total_datasets": len(all_specs),
            "downloaded_count": downloaded_count,
            "total_examples": total_examples,
            "cache_dir_size_bytes": cache_size,
            "cache_dir": str(self.cache_dir),
        }

    def is_available(self) -> bool:
        """Check whether the optional ``datasets`` library is installed.

        Returns:
            True if ``datasets`` can be imported, False otherwise.
        """
        try:
            import datasets  # noqa: F401 — intentional probe import

            return True
        except ImportError:
            logger.warning(
                "HuggingFace datasets library not installed — external dataset loading unavailable; install with: pip install datasets"  # noqa: VET301 — user guidance string
            )
            return False
        except Exception:
            # pyarrow ArrowKeyError on type extension re-registration,
            # RuntimeError from C extension crashes, etc.
            logger.warning(
                "datasets library import failed (possibly pyarrow C extension conflict) — external dataset loading unavailable",
                exc_info=True,
            )
            return False

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _convert_to_training_format(
        self,
        ds: Any,
        spec: DatasetSpec,
        output_path: Path,
    ) -> int:
        """Convert a HuggingFace dataset object to Alpaca-style JSONL.

        For SFT datasets the output schema is ``{instruction, input, output}``.
        For DPO datasets the output schema is ``{prompt, chosen, rejected}``.

        Args:
            ds: A HuggingFace ``Dataset`` object (or any iterable of row dicts).
            spec: Specification used to determine field mapping and format.
            output_path: Destination path for the JSONL file.

        Returns:
            Number of records successfully written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with output_path.open("w", encoding="utf-8") as out:
            for row in ds:
                mapped = self._map_row(dict(row), spec)
                if mapped is None:
                    continue
                out.write(json.dumps(mapped) + "\n")
                count += 1

        return count

    def _map_row(self, row: dict[str, Any], spec: DatasetSpec) -> dict[str, Any] | None:
        """Map a single HuggingFace dataset row to the Alpaca training format.

        SFT rows are mapped to ``{instruction, input, output}``.
        DPO rows are mapped to ``{prompt, chosen, rejected}``.

        The method attempts a best-effort field lookup using common column names
        found in popular HuggingFace datasets. Returns ``None`` when essential
        fields cannot be resolved so that the caller can skip the row.

        Args:
            row: Raw dictionary representing one dataset example.
            spec: Specification indicating the expected format and dataset name.

        Returns:
            Normalised dictionary ready for JSONL serialisation, or ``None``
            if the row cannot be mapped.
        """
        if spec.format == "sft":
            return self._map_sft_row(row)
        if spec.format == "dpo":
            return self._map_dpo_row(row)

        logger.warning(
            "Unknown format '%s' for dataset %s; skipping row",
            spec.format,
            spec.name,
        )
        return None

    def _map_sft_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Map a row to the SFT Alpaca format {instruction, input, output}.

        Args:
            row: Raw dataset row dictionary.

        Returns:
            Mapped dict or None if required fields are absent.
        """
        # instruction candidates (ordered by preference)
        instruction = (
            row.get("instruction") or row.get("prompt") or row.get("question") or row.get("problem") or row.get("text")
        )
        # output candidates
        output = (
            row.get("output")
            or row.get("response")
            or row.get("answer")
            or row.get("solution")
            or row.get("canonical_solution")
        )

        if not instruction or not output:
            return None

        # Optional secondary input context
        context = row.get("input") or row.get("context") or ""

        return {
            "instruction": str(instruction),
            "input": str(context),
            "output": str(output),
        }

    def _map_dpo_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Map a row to the DPO format {prompt, chosen, rejected}.

        Args:
            row: Raw dataset row dictionary.

        Returns:
            Mapped dict or None if required fields are absent.
        """
        prompt = row.get("prompt") or row.get("question") or row.get("instruction")
        chosen = row.get("chosen") or row.get("accepted") or row.get("preferred")
        rejected = row.get("rejected") or row.get("dispreferred")

        if not prompt or not chosen or not rejected:
            return None

        # HH-RLHF stores chosen/rejected as full conversation strings
        return {
            "prompt": str(prompt),
            "chosen": str(chosen),
            "rejected": str(rejected),
        }

    def _expected_path(self, spec: DatasetSpec) -> Path:
        """Return the canonical local cache path for a given DatasetSpec.

        Args:
            spec: The dataset specification.

        Returns:
            Path object pointing to the expected JSONL file location.
        """
        # Sanitise name to a filesystem-safe slug
        safe_name = spec.name.replace("/", "__")
        suffix = f"__{spec.subset}" if spec.subset else ""
        filename = f"{safe_name}{suffix}.jsonl"
        return self.cache_dir / filename
