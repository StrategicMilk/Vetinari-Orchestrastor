"""GGUF metadata types, lookup tables, and calculation helpers for ModelProfiler.

Contains all constants, data classes, and pure functions used by
``ModelProfiler``:

- KV cache size tables and family-specific constants
- ``GGUFMetadata`` and ``ModelProfile`` dataclasses
- ``read_metadata()`` GGUF header reader
- ``detect_family()``, ``calculate_optimal_context()``,
  ``calculate_gpu_layers()``, ``get_temperature()``,
  ``estimate_kv_per_token()`` helper functions
- Disk cache helpers: ``_load_cached_profile()``, ``_save_profile()``

These are factored out of ``model_profiler.py`` to keep that module under the
550-line ceiling while allowing the data tables to grow without risk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from vetinari.constants import get_user_dir
from vetinari.models.model_profiler_schemas import (  # noqa: F401 (re-exported via model_profiler)
    CHAT_FORMAT_MAP as _CHAT_FORMAT_MAP,
)
from vetinari.models.model_profiler_schemas import (
    DEFAULT_TEMPERATURES as _DEFAULT_TEMPERATURES,
)
from vetinari.models.model_profiler_schemas import (
    FAMILY_PATTERNS as _FAMILY_PATTERNS,
)
from vetinari.models.model_profiler_schemas import (
    GPU_SAFETY_MARGIN as _GPU_SAFETY_MARGIN,
)
from vetinari.models.model_profiler_schemas import (
    KV_BYTES_PER_TOKEN as _KV_BYTES_PER_TOKEN,
)
from vetinari.models.model_profiler_schemas import (
    QUANT_TEMP_OFFSETS as _QUANT_TEMP_OFFSETS,
)
from vetinari.models.model_profiler_schemas import (  # noqa: F401 (re-exported via model_profiler)
    ROPE_FREQ_BASE_OVERRIDES as _ROPE_FREQ_BASE_OVERRIDES,
)
from vetinari.models.model_profiler_schemas import (
    RUNTIME_OVERHEAD_GB as _RUNTIME_OVERHEAD_GB,
)
from vetinari.models.model_profiler_schemas import (
    TEMPERATURE_MATRIX as _TEMPERATURE_MATRIX,
)
from vetinari.models.model_profiler_schemas import (  # noqa: F401 (re-exported via model_profiler)
    GGUFMetadata,
    ModelProfile,
)

logger = logging.getLogger(__name__)

_ARTIFACT_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_ARTIFACT_HASH_LOCK = threading.Lock()

# ── Config persistence ────────────────────────────────────────────────────────


def _get_config_dir() -> Path:
    """Return the model configs directory, re-reading the env on each call.

    Using ``get_user_dir()`` instead of a module-level cached constant ensures
    that test overrides via ``monkeypatch.setenv("VETINARI_USER_DIR", ...)``
    take effect without restarting the process.

    Returns:
        Path to the per-user model configs directory.
    """
    return get_user_dir() / "model_configs"


def _config_path(model_id: str) -> Path:
    """Return the per-model config file path.

    Args:
        model_id: Model identifier (used as filename stem).

    Returns:
        Path to ``~/.vetinari/model_configs/{model_id}.json``.
    """
    safe_id = re.sub(r"[^\w\-.]", "_", model_id)
    return _get_config_dir() / f"{safe_id}.json"


def compute_artifact_sha256(model_path: Path) -> str:
    """Return a memoized SHA-256 digest for a model artifact.

    Returns:
        Hex-encoded SHA-256 digest of the resolved model artifact.
    """
    resolved = model_path.resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    with _ARTIFACT_HASH_LOCK:
        cached = _ARTIFACT_HASH_CACHE.get(cache_key)
        if cached is not None:
            return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()

    with _ARTIFACT_HASH_LOCK:
        _ARTIFACT_HASH_CACHE[cache_key] = value
    return value


def build_model_artifact_identity(model_path: Path) -> dict[str, Any]:
    """Return identity fields for a local model artifact.

    Returns:
        Stable path, digest, size, and mtime fields for profile cache keys.
    """
    resolved = model_path.resolve()
    stat = resolved.stat()
    return {
        "artifact_path": str(resolved),
        "artifact_sha256": compute_artifact_sha256(resolved),
        "artifact_size_bytes": stat.st_size,
        "artifact_mtime_ns": stat.st_mtime_ns,
    }


def model_profile_cache_id(model_path: Path, artifact_sha256: str | None = None) -> str:
    """Return a collision-resistant cache ID for a GGUF model profile.

    Args:
        model_path: Local model artifact path.
        artifact_sha256: Optional precomputed artifact digest.

    Returns:
        Filename-safe profile cache identifier.
    """
    digest = artifact_sha256 or compute_artifact_sha256(model_path)
    return f"{model_path.stem}-{digest[:16]}"


def _load_cached_profile(model_id: str) -> dict[str, Any] | None:
    """Load a cached profile from disk if it exists.

    Args:
        model_id: Model identifier.

    Returns:
        Raw profile dict, or None if not cached.
    """
    path = _config_path(model_id)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        corrupt_path = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        try:
            path.replace(corrupt_path)
            logger.warning(
                "Corrupt cached profile for %s moved to %s before re-profile: %s",
                model_id,
                corrupt_path,
                exc,
            )
        except OSError as move_exc:
            logger.warning(
                "Corrupt cached profile for %s could not be moved aside; refusing silent overwrite: %s",
                model_id,
                move_exc,
            )
        return None
    except OSError as exc:
        logger.warning("Failed to load cached profile for %s — will re-profile: %s", model_id, exc)
        return None


def _save_profile(model_id: str, profile_data: dict[str, Any]) -> None:
    """Persist a profile to disk.

    Args:
        model_id: Model identifier.
        profile_data: Serialized profile dict.
    """
    path = _config_path(model_id)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, default=str)
            f.write("\n")
        tmp_path.replace(path)
        logger.debug("Saved model profile for %s to %s", model_id, path)
    except OSError as exc:
        logger.warning("Failed to save model profile for %s — results not cached: %s", model_id, exc)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            logger.warning("Could not remove temporary profile file %s", tmp_path, exc_info=True)


# ── GGUF metadata reader ──────────────────────────────────────────────────────


def read_metadata(model_path: Path) -> GGUFMetadata:
    """Read GGUF header metadata from a model file.

    Uses the ``gguf`` package for memory-mapped header-only reads (instant,
    no full-file load). Falls back to filename heuristics if the ``gguf``
    package is not installed.

    Args:
        model_path: Path to a ``.gguf`` file.

    Returns:
        Populated GGUFMetadata dataclass.
    """
    file_size_gb = model_path.stat().st_size / (1024**3) if model_path.exists() else 0.0

    try:
        from gguf import GGUFReader  # type: ignore[import-untyped]

        reader = GGUFReader(str(model_path))
        kv = {f.name: f.data for f in reader.fields.values() if hasattr(f, "data")}

        arch = _extract_kv_string(kv, "general.architecture")

        # Architecture-prefixed keys (e.g., "llama.block_count")
        block_count = _extract_kv_int(kv, f"{arch}.block_count")
        head_count = _extract_kv_int(kv, f"{arch}.attention.head_count")
        head_count_kv = _extract_kv_int(kv, f"{arch}.attention.head_count_kv")
        context_length = _extract_kv_int(kv, f"{arch}.context_length")
        expert_count = _extract_kv_int(kv, "general.expert_count") or _extract_kv_int(kv, f"{arch}.expert_count")
        expert_used = _extract_kv_int(kv, "general.expert_used_count") or _extract_kv_int(
            kv, f"{arch}.expert_used_count"
        )
        file_type = _extract_kv_int(kv, "general.file_type")
        embedding_length = _extract_kv_int(kv, f"{arch}.embedding_length")
        vocab_size = _extract_kv_int(kv, f"{arch}.vocab_size") or _extract_kv_int(kv, "tokenizer.ggml.vocab_size")

        quantization = _detect_quantization(file_type, model_path.name)

        return GGUFMetadata(
            architecture=arch,
            block_count=block_count,
            head_count=head_count,
            head_count_kv=head_count_kv,
            context_length=context_length,
            expert_count=expert_count,
            expert_used_count=expert_used,
            file_type=file_type,
            embedding_length=embedding_length,
            vocab_size=vocab_size,
            quantization=quantization,
            file_size_gb=round(file_size_gb, 2),
        )

    except ImportError:
        logger.info("gguf package not installed; falling back to filename heuristics for %s", model_path.name)
        return _metadata_from_filename(model_path)
    except Exception as exc:
        logger.warning("Failed to read GGUF metadata from %s — falling back to heuristics: %s", model_path.name, exc)
        return _metadata_from_filename(model_path)


def _extract_kv_string(kv: dict, key: str) -> str:
    """Extract a string value from GGUF key-value data.

    Args:
        kv: Key-value mapping from GGUF reader.
        key: The GGUF metadata key to extract.

    Returns:
        String value, or empty string if not found.
    """
    val = kv.get(key)
    if val is None:
        return ""
    # gguf library may return bytes, numpy arrays, or lists
    if hasattr(val, "tobytes"):
        return val.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(val, (list, tuple)) and val:
        first = val[0] if len(val) == 1 else val
        if isinstance(first, (bytes, bytearray)):
            return bytes(first).decode("utf-8", errors="replace").rstrip("\x00")
        return str(first)
    return str(val)


def _extract_kv_int(kv: dict, key: str) -> int:
    """Extract an integer value from GGUF key-value data.

    Args:
        kv: Key-value mapping from GGUF reader.
        key: The GGUF metadata key to extract.

    Returns:
        Integer value, or 0 if not found or not convertible.
    """
    val = kv.get(key)
    if val is None:
        return 0
    # Handle numpy scalars and arrays
    if hasattr(val, "item"):
        return int(val.item())
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if hasattr(first, "item"):
            return int(first.item())
        return int(first)
    try:
        return int(val)
    except (TypeError, ValueError):
        logger.warning("Could not parse GGUF value as integer for key %r — using 0 as fallback", key)
        return 0


def _detect_quantization(file_type: int, filename: str) -> str:
    """Detect the quantization type from GGUF file_type code or filename.

    Args:
        file_type: GGUF general.file_type integer code.
        filename: Model filename for pattern matching fallback.

    Returns:
        Quantization string like ``"q4_k_m"``, ``"q8_0"``, ``"f16"``.
    """
    # GGUF file_type codes (from llama.cpp ggml-common.h)
    _file_type_map: dict[int, str] = {
        0: "f32",
        1: "f16",
        2: "q4_0",
        3: "q4_1",
        7: "q8_0",
        8: "q5_0",
        9: "q5_1",
        10: "q2_k",
        11: "q3_k_s",
        12: "q3_k_m",
        13: "q3_k_l",
        14: "q4_k_s",
        15: "q4_k_m",
        16: "q5_k_s",
        17: "q5_k_m",
        18: "q6_k",
    }
    if file_type in _file_type_map:
        return _file_type_map[file_type]

    # Fallback: extract from filename
    lower = filename.lower()
    for quant in ("q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q5_0", "q5_1", "f16", "bf16"):
        if quant in lower:
            return quant
    return "unknown"


def _metadata_from_filename(model_path: Path) -> GGUFMetadata:
    """Infer approximate metadata from filename when GGUF reader is unavailable.

    Args:
        model_path: Path to the .gguf file.

    Returns:
        GGUFMetadata with best-effort values from filename patterns.
    """
    name = model_path.stem.lower()
    file_size_gb = model_path.stat().st_size / (1024**3) if model_path.exists() else 0.0

    arch = ""
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, name):
            arch = family
            break

    # Estimate block count from parameter size (rough heuristic: layers ≈ params_b * 4)
    param_match = re.search(r"(\d+)[bB]", name)
    params_b = int(param_match.group(1)) if param_match else 7
    block_count = max(16, params_b * 4)

    quantization = _detect_quantization(0, model_path.name)

    return GGUFMetadata(
        architecture=arch,
        block_count=block_count,
        context_length=0,  # Unknown from filename alone
        quantization=quantization,
        file_size_gb=round(file_size_gb, 2),
    )


# ── Core calculation functions ────────────────────────────────────────────────


def detect_family(architecture: str) -> str:
    """Map a GGUF general.architecture string to a canonical model family.

    Args:
        architecture: The architecture string from GGUF metadata
            (e.g. ``"llama"``, ``"qwen2"``, ``"gpt2"``).

    Returns:
        Canonical family name, or ``"unknown"`` if not recognized.
    """
    lower = architecture.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, lower):
            return family
    return "unknown"


def calculate_optimal_context(
    free_vram_gb: float,
    model_vram_gb: float,
    kv_per_token: float,
    trained_limit: int,
) -> int:
    """Calculate the optimal context length that fits in available VRAM.

    Formula: ``n_ctx = (free_vram - model_vram - overhead) / kv_per_token``,
    capped at the model's trained context limit.

    Args:
        free_vram_gb: Available VRAM in GB after other allocations.
        model_vram_gb: VRAM consumed by the model weights.
        kv_per_token: Bytes of KV cache per token (depends on quant type
            and number of KV heads).
        trained_limit: Maximum context length the model was trained on.

    Returns:
        Optimal context length in tokens, minimum 2048.
    """
    available_gb = free_vram_gb - model_vram_gb - _RUNTIME_OVERHEAD_GB
    if available_gb <= 0:
        return min(2048, trained_limit) if trained_limit > 0 else 2048

    available_bytes = available_gb * (1024**3)
    if kv_per_token <= 0:
        kv_per_token = 1.0  # Conservative fallback

    n_ctx = int(available_bytes / kv_per_token)

    # Cap at trained limit
    if trained_limit > 0:
        n_ctx = min(n_ctx, trained_limit)

    # Clamp to reasonable range (minimum 2048, aligned to 256 for efficiency)
    return max(2048, (n_ctx // 256) * 256)


def calculate_gpu_layers(
    free_vram_gb: float,
    bytes_per_layer: float,
    num_layers: int,
    kv_reserve_gb: float = 0.0,
    expert_count: int = 0,
    expert_used_count: int = 0,
) -> int:
    """Calculate how many layers can fit on GPU, with MoE adjustment.

    For MoE models, the effective VRAM cost per layer is adjusted by the
    ratio of used experts to total experts:
    ``effective_cost = cost * (0.4 + 0.6 * used / total)``.

    Args:
        free_vram_gb: Available VRAM in GB.
        bytes_per_layer: VRAM cost per layer in bytes.
        num_layers: Total number of layers in the model.
        kv_reserve_gb: VRAM reserved for KV cache.
        expert_count: Total number of experts (0 for non-MoE).
        expert_used_count: Number of experts activated per token.

    Returns:
        Number of layers to place on GPU. -1 means all layers fit.
    """
    available_gb = free_vram_gb * (1 - _GPU_SAFETY_MARGIN) - kv_reserve_gb - _RUNTIME_OVERHEAD_GB
    if available_gb <= 0:
        return 0

    available_bytes = available_gb * (1024**3)

    # MoE adjustment: only active experts consume full compute per layer
    effective_bytes_per_layer = bytes_per_layer
    if expert_count > 1 and expert_used_count > 0:
        moe_factor = 0.4 + 0.6 * (expert_used_count / expert_count)
        effective_bytes_per_layer = bytes_per_layer * moe_factor

    if effective_bytes_per_layer <= 0:
        return -1  # Cannot estimate; offload all

    layers_fit = int(available_bytes / effective_bytes_per_layer)
    if layers_fit >= num_layers:
        return -1  # All layers fit on GPU

    return max(0, layers_fit)


def get_temperature(family: str, task_type: str, quantization: str = "") -> float:
    """Look up the optimal temperature, preferring learned values over hardcoded.

    Checks Thompson-learned temperature overrides first (populated by
    :func:`update_learned_temperatures`). Falls back to the hardcoded
    ``_TEMPERATURE_MATRIX`` for families/task types without enough data.
    Applies a quantization offset for lower-precision quants.

    Args:
        family: Canonical model family string (e.g. ``"llama"``, ``"qwen2"``).
        task_type: Task type string (e.g. ``"coding"``, ``"reasoning"``).
        quantization: Quantization type string (e.g. ``"q4_k_m"``).

    Returns:
        Recommended temperature value (0.0 to 1.5).
    """
    # Check learned overrides first (populated by Thompson Sampling feedback)
    learned = _learned_temperature_overrides.get(family, {}).get(task_type)
    if learned is not None:
        quant_offset = _QUANT_TEMP_OFFSETS.get(quantization.lower(), 0.0) if quantization else 0.0
        return round(min(1.5, learned + quant_offset), 3)

    family_temps = _TEMPERATURE_MATRIX.get(family, _DEFAULT_TEMPERATURES)
    base_temp = family_temps.get(task_type, family_temps.get("general", 0.5))
    quant_offset = _QUANT_TEMP_OFFSETS.get(quantization.lower(), 0.0) if quantization else 0.0
    return round(min(1.5, base_temp + quant_offset), 3)


# Module-level learned temperature overrides — populated by
# update_learned_temperatures() when Thompson has enough data.
# Keys mirror _TEMPERATURE_MATRIX: {family: {task_type: temperature}}.
# Protected by _learned_temps_lock for thread-safe updates.
_learned_temperature_overrides: dict[str, dict[str, float]] = {}
_learned_temps_lock = __import__("threading").Lock()

# Minimum Thompson observations per temperature value before we trust the learned data
_MIN_LEARNED_TEMP_OBSERVATIONS = 50


def update_learned_temperatures() -> int:
    """Read Thompson strategy arms and update temperature overrides when data is mature.

    Scans all ``strategy:*:temperature:*`` arms from the Thompson selector.
    When a family+task_type combination has >= 50 total observations across
    all temperature values, the best-performing temperature replaces the
    hardcoded matrix entry.

    For unknown model families, finds the closest known family by name prefix
    and seeds its initial temperatures from that family's values.

    Returns:
        Number of temperature overrides updated.

    Side effects:
        - Mutates ``_learned_temperature_overrides`` (thread-safe via lock)
        - Logs INFO for each temperature correction applied
    """
    try:
        from vetinari.learning.model_selector import get_thompson_selector
        from vetinari.learning.thompson_selectors import STRATEGY_VALUE_SPACES

        ts = get_thompson_selector()
        temp_values = STRATEGY_VALUE_SPACES.get("temperature", [])
        if not temp_values:
            return 0

        # Collect per-agent/mode temperature arm stats
        # Arms are keyed: "strategy:{agent}:{mode}:temperature:{value}"
        family_task_scores: dict[tuple[str, str], dict[float, tuple[float, int]]] = {}

        for arm_key, arm in ts._arms.items():
            if ":temperature:" not in arm_key:
                continue
            parts = arm_key.split(":")
            if len(parts) < 5:
                continue
            # parts = ["strategy", agent, mode, "temperature", value]
            temp_val_str = parts[4]
            try:
                temp_val = float(temp_val_str)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed learned temperature value %r for arm %s: %s",
                    temp_val_str,
                    arm_key,
                    exc,
                )
                continue

            agent_type = parts[1]
            mode = parts[2]
            key = (agent_type, mode)

            if key not in family_task_scores:
                family_task_scores[key] = {}

            mean = arm.alpha / (arm.alpha + arm.beta) if (arm.alpha + arm.beta) > 0 else 0.5
            family_task_scores[key][temp_val] = (mean, arm.total_pulls)

        updated = 0
        new_overrides: dict[str, dict[str, float]] = {}

        for (_agent_type, mode), scores_by_temp in family_task_scores.items():
            total_obs = sum(pulls for _, pulls in scores_by_temp.values())
            if total_obs < _MIN_LEARNED_TEMP_OBSERVATIONS:
                continue

            # Find best temperature by mean quality score
            best_temp = max(scores_by_temp, key=lambda t: scores_by_temp[t][0])
            best_mean = scores_by_temp[best_temp][0]

            # Map agent mode to task_type for the temperature matrix.
            # Unknown modes are skipped entirely — we never fall back to "general"
            # because that would silently corrupt the general temperature entry.
            task_key = mode if mode in _DEFAULT_TEMPERATURES else None
            if task_key is None:
                continue

            # Update all known families with this learned temperature
            for family in _TEMPERATURE_MATRIX:
                old_temp = _TEMPERATURE_MATRIX[family].get(task_key)
                if old_temp is not None and abs(old_temp - best_temp) > 0.05:
                    if family not in new_overrides:
                        new_overrides[family] = {}
                    new_overrides[family][task_key] = best_temp
                    logger.info(
                        "Temperature learning: %s/%s corrected %.2f -> %.2f (Thompson mean=%.3f, observations=%d)",
                        family,
                        task_key,
                        old_temp,
                        best_temp,
                        best_mean,
                        total_obs,
                    )
                    updated += 1

        if new_overrides:
            with _learned_temps_lock:
                for family, temps in new_overrides.items():
                    if family not in _learned_temperature_overrides:
                        _learned_temperature_overrides[family] = {}
                    _learned_temperature_overrides[family].update(temps)

        return updated
    except Exception:
        logger.warning("Temperature learning update failed — keeping hardcoded values")
        return 0


def estimate_kv_per_token(
    head_count_kv: int,
    embedding_length: int,
    head_count: int,
    kv_quant: str = "q8_0",
) -> float:
    """Estimate KV cache bytes per token based on model architecture.

    Formula: ``2 * num_kv_heads * head_dim * bytes_per_element``
    (factor of 2 for K + V caches).

    Args:
        head_count_kv: Number of key-value attention heads.
        embedding_length: Model embedding dimension.
        head_count: Number of attention heads (used to derive head_dim).
        kv_quant: Quantization type for KV cache (default q8_0).

    Returns:
        Bytes per token for the KV cache.
    """
    if head_count <= 0 or head_count_kv <= 0:
        return 512.0  # Conservative fallback

    head_dim = embedding_length / head_count if head_count > 0 else 128
    bytes_per_element = _KV_BYTES_PER_TOKEN.get(kv_quant, 1.0)

    # K cache + V cache = 2 * kv_heads * head_dim * bytes
    return 2 * head_count_kv * head_dim * bytes_per_element


# ── Unknown-family learning protocol ─────────────────────────────────────────
# Keyed by model_id. Written by store_model_temperature_overrides(), read by
# seed_unknown_family() and create_family_entry(). Protected by
# _per_model_temps_lock for thread safety.
# NOTE: This is distinct from _learned_temperature_overrides (line 708), which
# is keyed by family name and used by get_recommended_temperature().
_per_model_temperature_overrides: dict[str, dict[str, float]] = {}
_per_model_temps_lock = threading.Lock()

# Keyed by model_id. Written by record_unknown_family_task(), read by the
# same function to check threshold. Protected by _UNKNOWN_FAMILY_LOCK.
_UNKNOWN_FAMILY_TASK_COUNTS: dict[str, int] = {}
_UNKNOWN_FAMILY_LOCK = threading.Lock()

# Number of tasks an unknown-family model must complete before a new family
# entry is written to model_families.yaml.
_FAMILY_ENTRY_THRESHOLD = 20


def store_model_temperature_overrides(model_id: str, temperatures: dict[str, float]) -> None:
    """Store learned temperature overrides for a specific model.

    Merges the provided temperature map into the per-model learned overrides
    dict. Existing task-type entries are overwritten; unmentioned task types
    are preserved. Thread-safe.

    Args:
        model_id: The model identifier to update overrides for.
        temperatures: Mapping of task type (e.g. ``"coding"``) to temperature
            value. Merged into any existing overrides for this model.
    """
    with _per_model_temps_lock:
        existing = _per_model_temperature_overrides.get(model_id, {})
        merged = {**existing, **temperatures}
        _per_model_temperature_overrides[model_id] = merged
    logger.debug(
        "Updated learned temperatures for model %s (%d task types)",
        model_id,
        len(temperatures),
    )


def find_closest_known_family(architecture: str) -> str:
    """Find the closest known model family using string similarity.

    Uses SequenceMatcher to compare the architecture string against known
    family slugs from ``_FAMILY_PATTERNS``. Returns the best-matching family
    slug, or ``"unknown"`` if no match scores above 0.4.

    Args:
        architecture: The model architecture string (e.g., from GGUF metadata).

    Returns:
        The closest known family slug, or ``"unknown"`` if no good match found.
    """
    best_score = 0.0
    best_family = "unknown"
    arch_lower = architecture.lower()
    for _pattern, family_slug in _FAMILY_PATTERNS:
        ratio = SequenceMatcher(None, arch_lower, family_slug.lower()).ratio()
        if ratio > best_score:
            best_score = ratio
            best_family = family_slug

    if best_score < 0.4:
        return "unknown"
    return best_family


def seed_unknown_family(model_id: str, architecture: str, closest_family: str) -> None:
    """Bootstrap an unknown model family with parameters from the closest known family.

    Copies the closest family's temperature settings into
    ``_per_model_temperature_overrides`` so the unknown model starts with
    reasonable defaults instead of generic ones.

    Args:
        model_id: The model identifier for the unknown model.
        architecture: The model's architecture string.
        closest_family: The family slug of the closest known family.
    """
    closest_temps = _TEMPERATURE_MATRIX.get(closest_family)
    if not closest_temps:
        logger.warning(
            "Cannot seed unknown family — closest family %s has no temperature data",
            closest_family,
        )
        return

    store_model_temperature_overrides(model_id, dict(closest_temps))
    logger.info(
        "Seeded unknown model %s (arch=%s) with temperatures from family %s",
        model_id,
        architecture,
        closest_family,
    )


def record_unknown_family_task(model_id: str, architecture: str, quality_score: float) -> None:
    """Record a task execution for an unknown-family model and check graduation threshold.

    Increments the per-model task counter. After ``_FAMILY_ENTRY_THRESHOLD``
    tasks, creates a new family entry in ``model_families.yaml`` with observed
    performance data and logs the discovery via ADR. Thread-safe.

    Args:
        model_id: The model identifier.
        architecture: The model's architecture string.
        quality_score: Quality score from this task execution (0.0-1.0).
            Reserved for future use in calibrating the new entry.
    """
    # Only count models that were seeded via seed_unknown_family() — known-family
    # models never appear in _per_model_temperature_overrides.
    with _per_model_temps_lock:
        is_unknown = model_id in _per_model_temperature_overrides
    if not is_unknown:
        return

    with _UNKNOWN_FAMILY_LOCK:
        count = _UNKNOWN_FAMILY_TASK_COUNTS.get(model_id, 0) + 1
        _UNKNOWN_FAMILY_TASK_COUNTS[model_id] = count

    if count >= _FAMILY_ENTRY_THRESHOLD:
        closest = find_closest_known_family(architecture)
        create_family_entry(model_id, architecture, closest)


def create_family_entry(model_id: str, architecture: str, closest_family: str) -> None:
    """Create a new model family entry in model_families.yaml from observed data.

    Reads the closest family's profile as a template, incorporates any learned
    temperature overrides, and appends a new family entry keyed by an
    architecture-derived slug. Also records the discovery as an ADR so the
    decision is preserved in the decision journal.

    Args:
        model_id: The model identifier that triggered family creation.
        architecture: The model architecture string (used as family slug base).
        closest_family: The closest known family used as template.
    """
    import yaml

    families_path = Path("config/knowledge/model_families.yaml")
    if not families_path.exists():
        logger.warning(
            "Cannot create family entry — model_families.yaml not found at %s",
            families_path,
        )
        return

    # Derive a stable slug from the architecture string
    family_slug = re.sub(r"[^a-z0-9_]", "_", architecture.lower())[:30].strip("_")

    try:
        with families_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error(
            "Could not read model_families.yaml — new family %s not created: %s",
            family_slug,
            exc,
        )
        return

    families = data.get("model_families", {})
    if family_slug in families:
        logger.info("Family %s already exists in model_families.yaml — skipping creation", family_slug)
        return

    template = families.get(closest_family, {})

    with _per_model_temps_lock:
        learned_temps = dict(_per_model_temperature_overrides.get(model_id, {}))

    new_entry: dict[str, Any] = {
        "name": f"Auto-discovered: {architecture}",
        "vendor": "unknown",
        "architecture_type": architecture,
        "capabilities": template.get(
            "capabilities",
            {
                "context_window": 4096,
                "supports_function_calling": False,
                "supports_vision": False,
            },
        ),
        "strengths": [f"Derived from {closest_family} family"],
        "weaknesses": ["Auto-discovered — limited performance data"],
        "discovered_from_model": model_id,
        "discovered_via": "unknown_family_protocol",
    }

    if learned_temps:
        new_entry["learned_temperatures"] = learned_temps

    families[family_slug] = new_entry
    data["model_families"] = families

    try:
        with families_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(
            "Created new model family entry '%s' from model %s (based on %s)",
            family_slug,
            model_id,
            closest_family,
        )
    except Exception as exc:
        logger.error(
            "Could not write model_families.yaml — family entry %s not saved: %s",
            family_slug,
            exc,
        )
        return

    # Record the discovery in the ADR decision journal (best-effort — never
    # block family creation if the ADR system is unavailable).
    try:
        from vetinari.adr import ADRCategory, ADRStatus, get_adr_system

        get_adr_system().create_adr(
            title=f"New model family discovered: {family_slug}",
            category=ADRCategory.AGENT_DESIGN.value,
            context=(
                f"Model {model_id} with architecture '{architecture}' completed "
                f"{_FAMILY_ENTRY_THRESHOLD}+ tasks without a matching entry in "
                f"model_families.yaml. Closest known family: {closest_family}."
            ),
            decision=(
                f"Created new family entry '{family_slug}' in model_families.yaml "
                f"using {closest_family} as template, with observed temperature "
                f"adjustments from {_FAMILY_ENTRY_THRESHOLD} task executions."
            ),
            consequences=(
                f"Future models with architecture '{architecture}' will use "
                f"{family_slug} parameters instead of generic defaults. "
                f"The entry will be refined as more tasks accumulate."
            ),
            status=ADRStatus.ACCEPTED.value,
        )
    except Exception as exc:
        logger.warning(
            "Could not create ADR for new family %s — discovery logged but not in decision journal: %s",
            family_slug,
            exc,
        )
