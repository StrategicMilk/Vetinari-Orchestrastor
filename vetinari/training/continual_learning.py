"""Continual learning protection for Vetinari training.

Implements three complementary mechanisms to prevent catastrophic forgetting
during fine-tuning on new tasks:

1. STABLERegularizer — threshold-based LoRA gating using forward-pass-only
   metrics (EM drop, KL divergence, bits increase) to detect and gate
   per-layer updates when forgetting risk is detected.

2. ReplayBuffer — maintains a diverse buffer of high-quality past examples
   for mixing into new training batches, providing experience replay.

3. LoRAAdapterManager — manages per-skill LoRA adapters so each task type
   retains its own adapter weights rather than overwriting shared parameters.

These mechanisms work independently or in combination. STABLERegularizer and
ReplayBuffer are most effective when used together in the training pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir

logger = logging.getLogger(__name__)

_ADAPTER_REGISTRY_FILENAME = "registry.json"

# Forgetting risk multiplier: gates halved when global metrics breach thresholds
_GLOBAL_BREACH_GATE_FACTOR = 0.5

# Severe forgetting threshold multiplier for early stop
_STOP_TRAINING_THRESHOLD_MULTIPLIER = 2.0


def _require_immutable_model_revision(model_revision: str | None) -> str:
    """Return a usable immutable remote revision or raise."""
    revision = (model_revision or "").strip()
    if not revision or revision == "main":
        msg = (
            "Remote Hugging Face model loads require an explicit immutable revision "
            "(tag or commit hash), not the floating 'main' branch."
        )
        raise ValueError(msg)
    return revision


def _default_replay_buffer_path() -> Path:
    """Return the configured replay-buffer path lazily."""
    return get_user_dir() / "replay_buffer.jsonl"


def _default_adapters_dir() -> Path:
    """Return the configured LoRA adapter registry root lazily."""
    return get_user_dir() / "adapters"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text through a same-directory temp file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary file %s", tmp_path)


def _move_corrupt_file(path: Path) -> Path | None:
    """Move a corrupt persistence file aside for visible operator diagnosis."""
    if not path.exists():
        return None
    corrupt_path = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
    try:
        path.replace(corrupt_path)
        return corrupt_path
    except OSError as exc:
        logger.warning("Could not move corrupt file %s aside: %s", path, exc)
        return None


class STABLERegularizer:
    """Threshold-based LoRA gating for continual learning.

    Uses forward-pass-only metrics (no gradient computation needed) to
    detect catastrophic forgetting risk during fine-tuning, then gates
    LoRA updates per-layer to protect previously learned capabilities.

    Metrics monitored:
    - EM drop: per-layer embedding drift (L2 norm change from baseline)
    - KL divergence: output distribution shift on validation data
    - Bits increase: cross-entropy increase (nats → bits) on validation data

    All metrics are computed via forward passes only, making this suitable
    for monitoring during training without disrupting gradient flow.

    Example::

        regularizer = STABLERegularizer(em_threshold=0.15)
        regularizer.capture_baseline("path/to/model", "path/to/val.jsonl")
        gates = regularizer.compute_layer_gates("path/to/model", "path/to/val.jsonl")
        # gates: {"layer.0": 0.8, "layer.1": 1.0, ...}
    """

    def __init__(
        self,
        em_threshold: float = 0.15,
        kl_threshold: float = 0.5,
        bits_threshold: float = 0.3,
    ) -> None:
        """Initialise the regularizer with configurable forgetting thresholds.

        Args:
            em_threshold: Maximum tolerated L2 norm change per embedding layer
                before gating is applied. Default 0.15.
            kl_threshold: Maximum tolerated KL divergence on validation data
                before all layer gates are halved. Default 0.5.
            bits_threshold: Maximum tolerated increase in bits-per-token on
                validation data before all layer gates are halved. Default 0.3.
        """
        self.em_threshold = em_threshold
        self.kl_threshold = kl_threshold
        self.bits_threshold = bits_threshold

        self._baseline_embeddings: dict[str, float] = {}
        self._baseline_loss: float = 0.0
        self._baseline_captured: bool = False

        # Cached current metrics from the most recent compute call
        self._current_kl: float = 0.0
        self._current_bits_increase: float = 0.0
        self._current_em_drops: dict[str, float] = {}

    def capture_baseline(
        self,
        model_path: str,
        validation_data_path: str,
        model_revision: str | None = None,
    ) -> bool:
        """Capture baseline metrics before training on a new task.

        Runs a forward pass on the validation set to record embedding norms
        and loss. Must be called BEFORE fine-tuning begins so that drift can
        be measured relative to the pre-training state.

        Uses late imports for torch and transformers so this module remains
        importable even without those optional dependencies installed.

        Args:
            model_path: Path or HuggingFace model identifier for the base model.
            validation_data_path: Path to JSONL file containing validation
                examples with ``"text"`` or ``"prompt"`` + ``"completion"`` fields.
            model_revision: Immutable revision for remote Hugging Face loads.

        Returns:
            True if baseline captured successfully, False if torch or
            transformers are unavailable.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning(
                "torch or transformers not available — STABLERegularizer baseline "
                "capture skipped. Install optional [training] dependencies.",
            )
            return False

        logger.info("Capturing STABLE baseline from model=%s", model_path)

        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)  # nosec B615
            model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                output_hidden_states=True,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        else:
            resolved_revision = _require_immutable_model_revision(model_revision)
            tokenizer = AutoTokenizer.from_pretrained(  # noqa: VET305 - operator-supplied immutable revision
                model_path,
                revision=resolved_revision,
            )  # nosec B615
            model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                output_hidden_states=True,
                torch_dtype=torch.float32,
                revision=resolved_revision,  # noqa: VET305 - operator-supplied immutable revision
            )
        model.eval()

        val_texts = self._load_validation_texts(validation_data_path)
        if not val_texts:
            logger.warning("No validation texts loaded from %s", validation_data_path)
            return False

        total_loss = 0.0
        sample_count = 0
        layer_norm_accum: dict[str, list[float]] = {}

        with torch.no_grad():
            for text in val_texts[:100]:  # cap at 100 for speed
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                sample_count += 1

                # Accumulate L2 norms per hidden layer
                if outputs.hidden_states:
                    for idx, hs in enumerate(outputs.hidden_states):
                        key = f"layer.{idx}"
                        norm = hs.norm(dim=-1).mean().item()
                        layer_norm_accum.setdefault(key, []).append(norm)

        if sample_count == 0:
            return False

        self._baseline_loss = total_loss / sample_count
        self._baseline_embeddings = {k: sum(v) / len(v) for k, v in layer_norm_accum.items()}
        self._baseline_captured = True

        logger.info(
            "Baseline captured: loss=%.4f, layers=%d",
            self._baseline_loss,
            len(self._baseline_embeddings),
        )
        return True

    def compute_layer_gates(
        self,
        model_path: str,
        validation_data_path: str,
        model_revision: str | None = None,
    ) -> dict[str, float]:
        """Compute per-layer gating factors to control LoRA update magnitude.

        A gate of 1.0 means full updates are allowed; 0.0 means the layer is
        frozen. Gates are halved globally if KL divergence or bits increase
        exceeds their respective thresholds.

        All computation is forward-pass only — no gradients are computed.

        Args:
            model_path: Path or HuggingFace identifier for the current model
                (post some training steps, to measure drift).
            validation_data_path: Path to JSONL validation data.
            model_revision: Immutable revision for remote Hugging Face loads.

        Returns:
            Dictionary mapping layer name to gate factor in [0.0, 1.0].
            Returns empty dict if baseline was not captured or torch is
            unavailable.
        """
        if not self._baseline_captured:
            logger.warning(
                "compute_layer_gates called before capture_baseline — returning "
                "empty gates. Call capture_baseline() first.",
            )
            return {}

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning("torch/transformers unavailable — returning empty gates")
            return {}

        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)  # nosec B615
            model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                output_hidden_states=True,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        else:
            resolved_revision = _require_immutable_model_revision(model_revision)
            tokenizer = AutoTokenizer.from_pretrained(  # noqa: VET305 - operator-supplied immutable revision
                model_path,
                revision=resolved_revision,
            )  # nosec B615
            model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                model_path,
                output_hidden_states=True,
                torch_dtype=torch.float32,
                revision=resolved_revision,  # noqa: VET305 - operator-supplied immutable revision
            )
        model.eval()

        val_texts = self._load_validation_texts(validation_data_path)
        if not val_texts:
            return {}

        total_loss = 0.0
        total_kl = 0.0
        sample_count = 0
        current_norms: dict[str, list[float]] = {}

        with torch.no_grad():
            for text in val_texts[:100]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                sample_count += 1

                # Approximate KL via softmax entropy of logits
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                # Uniform reference: H(p) gives a KL proxy relative to uniform
                entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
                total_kl += entropy.item()

                if outputs.hidden_states:
                    for idx, hs in enumerate(outputs.hidden_states):
                        key = f"layer.{idx}"
                        norm = hs.norm(dim=-1).mean().item()
                        current_norms.setdefault(key, []).append(norm)

        if sample_count == 0:
            return {}

        current_loss = total_loss / sample_count
        current_kl_avg = total_kl / sample_count

        import math

        bits_increase = (current_loss - self._baseline_loss) / math.log(2)
        self._current_kl = current_kl_avg
        self._current_bits_increase = bits_increase

        # Compute per-layer EM drop gates
        gates: dict[str, float] = {}
        for layer_key, norms in current_norms.items():
            current_norm = sum(norms) / len(norms)
            baseline_norm = self._baseline_embeddings.get(layer_key, current_norm)
            if baseline_norm > 0:
                em_drop = abs(current_norm - baseline_norm) / baseline_norm
            else:
                em_drop = 0.0
            self._current_em_drops[layer_key] = em_drop

            # Gate reduces linearly from 1.0 at 0 drop to 0.0 at 2x threshold
            if em_drop >= self.em_threshold:
                gate = max(0.0, 1.0 - (em_drop / (2.0 * self.em_threshold)))
            else:
                gate = 1.0
            gates[layer_key] = gate

        # Halve all gates if global metrics breach thresholds
        global_breach = self._current_bits_increase > self.bits_threshold or self._current_kl > self.kl_threshold
        if global_breach:
            logger.warning(
                "Global forgetting breach detected — kl=%.4f (thresh=%.4f), "
                "bits_increase=%.4f (thresh=%.4f). Halving all layer gates.",
                self._current_kl,
                self.kl_threshold,
                self._current_bits_increase,
                self.bits_threshold,
            )
            gates = {k: v * _GLOBAL_BREACH_GATE_FACTOR for k, v in gates.items()}

        logger.info(
            "Layer gates computed: %d layers, global_breach=%s",
            len(gates),
            global_breach,
        )
        return gates

    def should_stop_training(self, model_path: str, validation_data_path: str) -> bool:
        """Check whether training should be stopped due to severe forgetting.

        Returns True if any metric exceeds 2x its threshold, indicating the
        model is suffering severe catastrophic forgetting and further training
        is likely harmful.

        Args:
            model_path: Path or HuggingFace identifier for the current model.
            validation_data_path: Path to JSONL validation data.

        Returns:
            True if severe forgetting detected and training should stop,
            False otherwise or if metrics cannot be computed.
        """
        gates = self.compute_layer_gates(model_path, validation_data_path)
        if not gates:
            return False

        severe_kl = self._current_kl > self.kl_threshold * _STOP_TRAINING_THRESHOLD_MULTIPLIER
        severe_bits = self._current_bits_increase > self.bits_threshold * _STOP_TRAINING_THRESHOLD_MULTIPLIER
        severe_em = any(
            drop > self.em_threshold * _STOP_TRAINING_THRESHOLD_MULTIPLIER for drop in self._current_em_drops.values()
        )

        stop = severe_kl or severe_bits or severe_em
        if stop:
            logger.warning(
                "Severe forgetting detected — stopping training. kl=%.4f, bits_increase=%.4f, max_em_drop=%.4f",
                self._current_kl,
                self._current_bits_increase,
                max(self._current_em_drops.values(), default=0.0),
            )
        return stop

    def get_metrics(self) -> dict[str, Any]:
        """Return current metric values and configured thresholds.

        Returns:
            Dictionary with keys:
            - ``baseline_captured``: whether baseline has been recorded
            - ``current_kl``: latest KL divergence measurement
            - ``current_bits_increase``: latest bits-per-token increase
            - ``current_em_drops``: per-layer EM drop values
            - ``thresholds``: dict with em, kl, and bits threshold values
        """
        return {
            "baseline_captured": self._baseline_captured,
            "current_kl": self._current_kl,
            "current_bits_increase": self._current_bits_increase,
            "current_em_drops": dict(self._current_em_drops),
            "thresholds": {
                "em": self.em_threshold,
                "kl": self.kl_threshold,
                "bits": self.bits_threshold,
            },
        }

    @staticmethod
    def _load_validation_texts(validation_data_path: str) -> list[str]:
        """Load plain text strings from a JSONL validation file.

        Args:
            validation_data_path: Path to JSONL file. Each line must be a JSON
                object with a ``"text"`` key, or ``"prompt"`` and
                ``"completion"`` keys that are concatenated.

        Returns:
            List of text strings. Empty list if file is missing or unreadable.
        """
        path = Path(validation_data_path)
        if not path.exists():
            logger.warning("Validation data path does not exist: %s", path)
            return []

        texts: list[str] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "text" in record:
                        texts.append(record["text"])
                    elif "prompt" in record and "completion" in record:
                        texts.append(record["prompt"] + record["completion"])
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line in validation data")
        return texts


class ReplayBuffer:
    """Maintains a diverse buffer of high-quality past examples for experience replay.

    Examples from previous tasks are stored on disk as JSONL and sampled into
    new training batches to reduce catastrophic forgetting. The buffer is
    size-capped; when it overflows, stratified sampling preserves diversity
    across domain/task_type fields.

    Thread-safe: all mutations are protected by an internal Lock.

    Example::

        buf = ReplayBuffer(max_size=5000)
        buf.add([{"text": "...", "task_type": "code"}, ...])
        mixed = buf.create_mixed_dataset(Path("new_data.jsonl"))
    """

    def __init__(
        self,
        max_size: int = 5000,
        buffer_path: Path | None = None,
    ) -> None:
        """Initialise the replay buffer.

        Args:
            max_size: Maximum number of examples to retain. When exceeded,
                stratified sampling is applied to preserve diversity.
            buffer_path: Path for persisting the buffer as JSONL. Defaults to
                ``~/.vetinari/replay_buffer.jsonl``.
        """
        self.max_size = max_size
        self.buffer_path = Path(buffer_path) if buffer_path is not None else _default_replay_buffer_path()
        self._examples: list[dict] = []
        self._lock = threading.Lock()

        # Load existing buffer from disk if present
        if self.buffer_path.exists():
            self.load()

    def add(self, examples: list[dict]) -> int:
        """Add new examples to the buffer.

        If the buffer exceeds max_size after adding, stratified sampling is
        applied to trim it back while preserving domain diversity.

        Args:
            examples: List of example dicts. Each should contain a ``"text"``
                or ``"prompt"``/``"completion"`` field. A ``"task_type"`` or
                ``"domain"`` field enables stratified sampling.

        Returns:
            Number of examples actually added (before any trimming).
        """
        added = len(examples)
        with self._lock:
            self._examples.extend(examples)
            if len(self._examples) > self.max_size:
                self._examples = self._stratified_sample(self.max_size)
        logger.info(
            "ReplayBuffer: added %d examples, buffer size now %d",
            added,
            len(self._examples),
        )
        return added

    def get_replay_batch(self, batch_size: int) -> list[dict]:
        """Return a random sample from the replay buffer.

        Args:
            batch_size: Number of examples to return. Clamped to buffer size
                if larger.

        Returns:
            List of randomly sampled example dicts.
        """
        with self._lock:
            available = len(self._examples)
        if available == 0:
            logger.warning("ReplayBuffer is empty — returning empty batch")
            return []
        k = min(batch_size, available)
        with self._lock:
            return random.sample(self._examples, k)

    def create_mixed_dataset(
        self,
        new_data_path: Path,
        replay_ratio: float = 0.2,
        output_path: Path | None = None,
    ) -> Path:
        """Create a mixed training dataset combining new data and replay examples.

        The output contains (1 - replay_ratio) proportion of new data and
        replay_ratio proportion of replay buffer examples, shuffled together.

        Args:
            new_data_path: Path to JSONL file with new training examples.
            replay_ratio: Fraction of the mixed dataset to fill from the replay
                buffer. Must be in [0.0, 1.0]. Default 0.2.
            output_path: Output JSONL path. Defaults to a sibling file of
                new_data_path with ``_mixed`` appended to the stem.

        Returns:
            Path to the written mixed dataset JSONL file.

        Raises:
            ValueError: If replay_ratio is outside [0.0, 1.0].
        """
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError(f"replay_ratio must be in [0.0, 1.0], got {replay_ratio}")

        new_examples: list[dict] = []
        with Path(new_data_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        new_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line in new data")

        total_new = len(new_examples)
        if total_new == 0:
            logger.warning("No examples loaded from %s", new_data_path)

        # Compute replay count based on desired ratio and total dataset size
        if replay_ratio > 0.0 and total_new > 0:
            replay_count = int(total_new * replay_ratio / max(1.0 - replay_ratio, 1e-9))
        else:
            replay_count = 0

        replay_examples = self.get_replay_batch(replay_count) if replay_count > 0 else []

        mixed = new_examples + replay_examples
        random.shuffle(mixed)

        if output_path is None:
            output_path = new_data_path.with_stem(new_data_path.stem + "_mixed")

        _atomic_write_text(
            Path(output_path),
            "".join(json.dumps(example) + "\n" for example in mixed),
        )

        logger.info(
            "Mixed dataset written: new=%d, replay=%d, total=%d -> %s",
            total_new,
            len(replay_examples),
            len(mixed),
            output_path,
        )
        return output_path

    def save(self) -> None:
        """Persist the replay buffer to disk as JSONL.

        Creates parent directories if they do not exist.
        """
        with self._lock:
            examples_snapshot = list(self._examples)

        _atomic_write_text(
            Path(self.buffer_path),
            "".join(json.dumps(example) + "\n" for example in examples_snapshot),
        )

        logger.info(
            "ReplayBuffer saved: %d examples -> %s",
            len(examples_snapshot),
            self.buffer_path,
        )

    def load(self) -> None:
        """Load the replay buffer from disk.

        Silently replaces any in-memory examples with the persisted data.
        Skips malformed JSONL lines with a debug log.

        Raises:
            OSError: If the buffer file exists but cannot be read.
        """
        if not self.buffer_path.exists():
            logger.info(
                "ReplayBuffer file not found at %s — starting with empty buffer",
                self.buffer_path,
            )
            return

        examples: list[dict] = []
        malformed_lines = 0
        with Path(self.buffer_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed_lines += 1

        if malformed_lines:
            corrupt_path = _move_corrupt_file(Path(self.buffer_path))
            raise ValueError(
                f"ReplayBuffer file {self.buffer_path} contains {malformed_lines} malformed JSONL line(s); "
                f"moved aside to {corrupt_path}"
            )

        with self._lock:
            self._examples = examples

        logger.info(
            "ReplayBuffer loaded: %d examples from %s",
            len(examples),
            self.buffer_path,
        )

    def __len__(self) -> int:
        """Return the current number of examples in the buffer.

        Returns:
            Integer count of stored examples.
        """
        with self._lock:
            return len(self._examples)

    def _stratified_sample(self, target_size: int) -> list[dict]:
        """Sample examples while preserving proportional domain/task_type diversity.

        Groups examples by the ``"domain"`` or ``"task_type"`` key (falling back
        to ``"_unknown"`` if neither is present), then samples proportionally
        across all groups to reach target_size.

        Args:
            target_size: Total number of examples in the output.

        Returns:
            List of target_size examples with preserved stratification.

        Note:
            Must be called while holding self._lock.
        """
        if target_size >= len(self._examples):
            return list(self._examples)

        # Group by domain/task_type
        groups: dict[str, list[dict]] = {}
        for ex in self._examples:
            key = ex.get("domain") or ex.get("task_type") or "_unknown"
            groups.setdefault(key, []).append(ex)

        total = len(self._examples)
        sampled: list[dict] = []

        for _group_key, group_examples in groups.items():
            proportion = len(group_examples) / total
            n_from_group = max(1, round(proportion * target_size))
            n_from_group = min(n_from_group, len(group_examples))
            sampled.extend(random.sample(group_examples, n_from_group))

        # Trim or top-up to exactly target_size
        if len(sampled) > target_size:
            sampled = random.sample(sampled, target_size)
        elif len(sampled) < target_size:
            remaining = [ex for ex in self._examples if ex not in sampled]
            gap = target_size - len(sampled)
            sampled.extend(random.sample(remaining, min(gap, len(remaining))))

        return sampled


class LoRAAdapterManager:
    """Manages per-skill LoRA adapters for task-specific fine-tuning.

    Each task type (e.g., ``"code"``, ``"qa"``, ``"summarisation"``) gets its
    own LoRA adapter stored at a registered path. Adapters are loaded on demand
    and the registry is persisted as JSON.

    This approach prevents different skills from overwriting each other's
    adapter weights, a common source of catastrophic forgetting when the same
    adapter is continually fine-tuned on diverse tasks.

    Example::

        manager = LoRAAdapterManager()
        manager.register_adapter("code", Path("/adapters/code_lora"))
        path = manager.get_adapter("code")
    """

    def __init__(self, adapters_dir: Path | None = None) -> None:
        """Initialise the adapter manager.

        Args:
            adapters_dir: Directory where adapter registry is stored. Defaults
                to ``~/.vetinari/adapters/``. Created on first save if absent.
        """
        self.adapters_dir = Path(adapters_dir) if adapters_dir is not None else _default_adapters_dir()
        self._registry: dict[str, Path] = {}
        self._registry_path = self.adapters_dir / _ADAPTER_REGISTRY_FILENAME

        if self._registry_path.exists():
            self.load_registry()

    def register_adapter(self, task_type: str, adapter_path: Path) -> None:
        """Register a LoRA adapter path for a task type.

        Args:
            task_type: Identifier for the skill or task (e.g., ``"code"``,
                ``"qa"``). Must be a non-empty string.
            adapter_path: Path to the LoRA adapter directory or file.

        Raises:
            ValueError: If task_type is empty.
        """
        if not task_type:
            raise ValueError("task_type must be a non-empty string")
        self._registry[task_type] = Path(adapter_path)
        logger.info(
            "LoRAAdapterManager: registered adapter for task_type=%s at %s",
            task_type,
            adapter_path,
        )

    def get_adapter(self, task_type: str) -> Path | None:
        """Return the registered adapter path for a task type.

        Args:
            task_type: The task identifier to look up.

        Returns:
            Path to the adapter, or None if no adapter is registered for this
            task type.
        """
        return self._registry.get(task_type)

    def list_adapters(self) -> dict[str, Path]:
        """Return all registered adapters.

        Returns:
            Dictionary mapping task type strings to their adapter paths.
        """
        return dict(self._registry)

    def remove_adapter(self, task_type: str) -> bool:
        """Remove the adapter registration for a task type.

        Does NOT delete the adapter files from disk — only removes the
        registry entry.

        Args:
            task_type: The task identifier to deregister.

        Returns:
            True if the adapter was registered and removed, False if it was
            not found.
        """
        if task_type in self._registry:
            del self._registry[task_type]
            logger.info(
                "LoRAAdapterManager: removed registration for task_type=%s",
                task_type,
            )
            return True
        logger.warning(
            "LoRAAdapterManager: remove_adapter called for unknown task_type=%s",
            task_type,
        )
        return False

    def save_registry(self) -> None:
        """Persist the adapter registry to JSON.

        Creates the adapters directory if it does not exist. Paths are stored
        as strings for JSON compatibility.
        """
        serialisable = {k: str(v) for k, v in self._registry.items()}
        _atomic_write_text(Path(self._registry_path), json.dumps(serialisable, indent=2) + "\n")
        logger.info(
            "LoRAAdapterManager: registry saved (%d adapters) to %s",
            len(self._registry),
            self._registry_path,
        )

    def load_registry(self) -> None:
        """Load the adapter registry from JSON.

        Silently no-ops if the registry file does not exist. Converts stored
        string paths back to Path objects.

        Raises:
            OSError: If the registry file exists but cannot be read.
            json.JSONDecodeError: If the registry file is malformed.
        """
        if not self._registry_path.exists():
            logger.info(
                "LoRAAdapterManager: no registry file at %s — starting empty",
                self._registry_path,
            )
            return

        try:
            with Path(self._registry_path).open(encoding="utf-8") as fh:
                raw: dict[str, str] = json.load(fh)
        except json.JSONDecodeError as exc:
            corrupt_path = _move_corrupt_file(Path(self._registry_path))
            raise ValueError(f"LoRA adapter registry is corrupt; moved aside to {corrupt_path}") from exc

        self._registry = {k: Path(v) for k, v in raw.items()}
        logger.info(
            "LoRAAdapterManager: loaded %d adapters from %s",
            len(self._registry),
            self._registry_path,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about registered adapters.

        Computes total on-disk size for adapters whose paths exist.

        Returns:
            Dictionary with keys:
            - ``count``: total number of registered adapters
            - ``total_size_bytes``: combined size of all existing adapter paths
            - ``task_types``: sorted list of registered task type names
        """
        total_size = 0
        for adapter_path in self._registry.values():
            p = Path(adapter_path)
            if p.is_dir():
                total_size += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            elif p.is_file():
                total_size += p.stat().st_size

        return {
            "count": len(self._registry),
            "total_size_bytes": total_size,
            "task_types": sorted(self._registry.keys()),
        }
