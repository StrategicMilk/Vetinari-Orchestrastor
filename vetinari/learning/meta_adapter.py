"""Meta-Learning Strategy Adapter — Department 9.1.

Extends Thompson Sampling to strategy parameters (temperature, context_window,
decomposition_granularity, prompt_template_variant) by matching new tasks to
prototypes built from past successful episodes.

When a new task arrives, the MetaAdapter:
1. Computes similarity to known TaskPrototypes via episode memory embeddings
2. If a close match exists, recommends the strategy that worked for that prototype
3. Falls back to Thompson Sampling bandit selection when no prototype matches
4. Records outcomes to refine prototypes over time

This is "meta" learning because it learns *how to configure the learning system
itself* — not what to do for a specific task, but which strategy parameters
lead to better outcomes for each task type.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import VETINARI_STATE_DIR
from vetinari.types import AgentType
from vetinari.utils.math_helpers import cosine_similarity
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)

# Minimum similarity score to consider a prototype match
_PROTOTYPE_MATCH_THRESHOLD = 0.6

# Minimum samples before a prototype's strategy is trusted over Thompson defaults
_MIN_PROTOTYPE_SAMPLES = 3

# Maximum prototypes to store (LRU eviction beyond this)
_MAX_PROTOTYPES = 200


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class StrategyBundle:
    """A complete set of strategy parameters for task execution.

    Attributes:
        temperature: LLM sampling temperature (0.0 = deterministic, 1.0 = creative).
        context_window: Maximum context window size in tokens.
        decomposition_granularity: How finely to decompose tasks (coarse/medium/fine).
        prompt_template_variant: Which prompt template style to use.
        source: How this bundle was selected (prototype_match, thompson, default).
    """

    temperature: float = 0.3
    context_window: int = 4096
    decomposition_granularity: str = "medium"
    prompt_template_variant: str = "standard"
    source: str = "default"

    def __repr__(self) -> str:
        return f"StrategyBundle(source={self.source!r}, temperature={self.temperature!r}, decomposition_granularity={self.decomposition_granularity!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


@dataclass
class TaskPrototype:
    """A learned task archetype with associated strategy performance data.

    Built from clusters of similar successful episodes. Each prototype
    represents a "type of task" and the strategy parameters that worked
    best for it.

    Attributes:
        prototype_id: Unique identifier for this prototype.
        task_type: General task category (coding, research, docs, etc.).
        domain: Specific domain (python, infrastructure, etc.).
        complexity: Estimated complexity 1-10.
        successful_strategies: Map of strategy_key -> {value -> avg_quality}.
        preferred_mode: Agent mode that works best for this prototype.
        avg_quality: Average quality score across all matching episodes.
        sample_count: Number of episodes that contributed to this prototype.
        representative_query: A representative task description for similarity matching.
        embedding: Cached embedding for similarity lookups.
        last_updated: ISO timestamp of last update.
    """

    prototype_id: str
    task_type: str = "general"
    domain: str = "general"
    complexity: int = 5
    successful_strategies: dict[str, dict[str, float]] = field(default_factory=dict)
    preferred_mode: str = ""
    avg_quality: float = 0.0
    sample_count: int = 0
    representative_query: str = ""
    embedding: list[float] = field(default_factory=list)  # noqa: VET220 — fixed-length embedding vector, not a growing buffer
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return f"TaskPrototype(prototype_id={self.prototype_id!r}, task_type={self.task_type!r}, avg_quality={self.avg_quality!r})"

    def get_best_strategy(self, strategy_key: str) -> str | int | float | None:
        """Return the best-performing value for a strategy key.

        Args:
            strategy_key: One of temperature, context_window, etc.

        Returns:
            The value with the highest average quality, or None if no data.
        """
        strategy_data = self.successful_strategies.get(strategy_key, {})
        if not strategy_data:
            return None
        return max(strategy_data, key=lambda v: strategy_data[v])


# ── MetaAdapter ──────────────────────────────────────────────────────


class MetaAdapter:
    """Selects strategy parameters for new tasks based on episode memory similarity.

    Combines prototype matching (experience-based) with Thompson Sampling
    (exploration-exploitation) for strategy parameter selection.

    Args:
        state_path: Path to persist prototype state. Defaults to
            ``$VETINARI_STATE_DIR/meta_adapter_state.json``.
    """

    # Weight for new observations in exponential moving average — matches
    # FeedbackLoop.EMA_ALPHA for consistency across the learning subsystem
    EMA_ALPHA = 0.2

    def __init__(self, state_path: Path | None = None) -> None:
        self._prototypes: dict[str, TaskPrototype] = {}
        self._lock = threading.Lock()
        self._state_path = state_path or self._default_state_path()
        self._embedder = self._get_embedder()
        # Monotonic counter for prototype ID allocation.  Using dict length
        # would repeat IDs when prototypes are evicted (len decreases after
        # _evict_least_used), causing silent state collisions on reload.
        self._next_proto_counter: int = 0
        self._load_state()

    def select_strategy(
        self,
        task_description: str,
        task_type: str = "general",
        agent_type: str = AgentType.WORKER.value,
        mode: str = "build",
    ) -> StrategyBundle:
        """Select the best strategy bundle for a new task.

        First attempts prototype matching via embedding similarity.
        Falls back to Thompson Sampling if no prototype match is found.

        Args:
            task_description: Natural language description of the task.
            task_type: General task category.
            agent_type: Agent that will execute the task.
            mode: Agent mode for execution.

        Returns:
            A StrategyBundle with recommended parameters and source attribution.
        """
        # Try prototype matching first
        query_embedding = self._embedder(task_description)
        best_match = self._find_best_prototype(query_embedding)

        if best_match is not None:
            prototype, similarity = best_match
            if prototype.sample_count >= _MIN_PROTOTYPE_SAMPLES:
                bundle = self._bundle_from_prototype(prototype)
                bundle.source = f"prototype_match:{prototype.prototype_id}:{similarity:.3f}"
                logger.info(
                    "MetaAdapter: prototype match %s (sim=%.3f, samples=%d) for '%s'",
                    prototype.prototype_id,
                    similarity,
                    prototype.sample_count,
                    task_description[:60],
                )
                return bundle

        # Fall back to Thompson Sampling
        return self._bundle_from_thompson(agent_type, mode)

    def record_outcome(
        self,
        task_description: str,
        task_type: str,
        strategy_used: StrategyBundle,
        quality_score: float,
        success: bool,
        mode: str = "",
    ) -> str:
        """Record the outcome of a task to refine prototypes.

        Updates existing prototype if a match is found, otherwise creates
        a new prototype from this observation.

        Args:
            task_description: The task that was executed.
            task_type: General task category.
            strategy_used: The StrategyBundle that was applied.
            quality_score: Observed quality 0.0-1.0.
            success: Whether the task completed successfully.
            mode: Agent mode that was used.

        Returns:
            The prototype_id that was updated or created.
        """
        if not success:
            # Only build prototypes from successes (failures are noise)
            return ""

        query_embedding = self._embedder(task_description)
        best_match = self._find_best_prototype(query_embedding)

        with self._lock:
            if best_match is not None:
                prototype, _similarity = best_match
                # Update existing prototype
                self._update_prototype(prototype, strategy_used, quality_score, mode)
                self._save_state()
                return prototype.prototype_id

            # Create new prototype
            prototype_id = self._next_prototype_id()
            prototype = TaskPrototype(
                prototype_id=prototype_id,
                task_type=task_type,
                representative_query=task_description[:300],
                embedding=query_embedding,
                avg_quality=quality_score,
                sample_count=1,
                preferred_mode=mode,
            )
            self._record_strategy_in_prototype(prototype, strategy_used, quality_score)
            self._prototypes[prototype_id] = prototype

            # Evict if over limit
            if len(self._prototypes) > _MAX_PROTOTYPES:
                self._evict_least_used()

            self._save_state()
            logger.info(
                "MetaAdapter: created prototype %s from '%s' (quality=%.3f)",
                prototype_id,
                task_description[:60],
                quality_score,
            )
            return prototype_id

    def get_stats(self) -> dict[str, Any]:
        """Return meta-adapter statistics.

        Returns:
            Dictionary with prototype counts, average quality, coverage.
        """
        with self._lock:
            if not self._prototypes:
                return {
                    "prototype_count": 0,
                    "avg_quality": 0.0,
                    "avg_samples": 0.0,
                    "task_types": [],
                }
            prototypes = list(self._prototypes.values())
            return {
                "prototype_count": len(prototypes),
                "avg_quality": sum(p.avg_quality for p in prototypes) / len(prototypes),
                "avg_samples": sum(p.sample_count for p in prototypes) / len(prototypes),
                "task_types": list({p.task_type for p in prototypes}),
            }

    def get_prototype(self, prototype_id: str) -> TaskPrototype | None:
        """Look up a prototype by ID.

        Args:
            prototype_id: The prototype identifier.

        Returns:
            The TaskPrototype, or None if not found.
        """
        with self._lock:
            return self._prototypes.get(prototype_id)

    # ── Internal helpers ─────────────────────────────────────────────

    def _find_best_prototype(
        self,
        query_embedding: list[float],
    ) -> tuple[TaskPrototype, float] | None:
        """Find the most similar prototype above threshold.

        Args:
            query_embedding: Embedding vector for the query.

        Returns:
            Tuple of (prototype, similarity) or None if no match above threshold.
        """
        with self._lock:
            best_score = 0.0
            best_proto: TaskPrototype | None = None

            for proto in self._prototypes.values():
                if not proto.embedding:
                    continue
                score = cosine_similarity(query_embedding, proto.embedding)
                if score > best_score:
                    best_score = score
                    best_proto = proto

            if best_proto is not None and best_score >= _PROTOTYPE_MATCH_THRESHOLD:
                return (best_proto, best_score)
            return None

    def _bundle_from_prototype(self, prototype: TaskPrototype) -> StrategyBundle:
        """Build a StrategyBundle from a prototype's learned strategies.

        Args:
            prototype: The matched TaskPrototype.

        Returns:
            StrategyBundle with best-performing values from the prototype.
        """
        temp = prototype.get_best_strategy("temperature")
        ctx = prototype.get_best_strategy("context_window")
        gran = prototype.get_best_strategy("decomposition_granularity")
        tmpl = prototype.get_best_strategy("prompt_template_variant")

        return StrategyBundle(
            temperature=float(temp) if temp is not None else 0.3,
            context_window=int(ctx) if ctx is not None else 4096,
            decomposition_granularity=str(gran) if gran is not None else "medium",
            prompt_template_variant=str(tmpl) if tmpl is not None else "standard",
        )

    def _bundle_from_thompson(self, agent_type: str, mode: str) -> StrategyBundle:
        """Build a StrategyBundle using Thompson Sampling selection.

        Args:
            agent_type: Agent type string.
            mode: Agent mode string.

        Returns:
            StrategyBundle with Thompson-selected values.
        """
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            selector = get_thompson_selector()
            temp = selector.select_strategy(agent_type, mode, "temperature")
            ctx = selector.select_strategy(agent_type, mode, "context_window_size")
            gran = selector.select_strategy(agent_type, mode, "decomposition_granularity")
            tmpl = selector.select_strategy(agent_type, mode, "prompt_template_variant")

            return StrategyBundle(
                temperature=float(temp),
                context_window=int(ctx),
                decomposition_granularity=str(gran),
                prompt_template_variant=str(tmpl),
                source="thompson",
            )
        except Exception:
            logger.warning("Thompson Sampling unavailable, using defaults")
            return StrategyBundle(source="default")

    def _update_prototype(
        self,
        prototype: TaskPrototype,
        strategy: StrategyBundle,
        quality: float,
        mode: str,
    ) -> None:
        """Update a prototype with a new observation.

        Args:
            prototype: The prototype to update.
            strategy: The strategy that was used.
            quality: Observed quality score.
            mode: Agent mode that was used.
        """
        # Running average for quality
        n = prototype.sample_count
        prototype.avg_quality = (prototype.avg_quality * n + quality) / (n + 1)
        prototype.sample_count = n + 1
        prototype.last_updated = datetime.now(timezone.utc).isoformat()

        if mode:
            prototype.preferred_mode = mode

        self._record_strategy_in_prototype(prototype, strategy, quality)

    @staticmethod
    def _record_strategy_in_prototype(
        prototype: TaskPrototype,
        strategy: StrategyBundle,
        quality: float,
    ) -> None:
        """Record strategy parameter effectiveness in a prototype.

        Args:
            prototype: The prototype to update.
            strategy: The strategy bundle used.
            quality: Quality score achieved.
        """
        mappings = {
            "temperature": str(strategy.temperature),
            "context_window": str(strategy.context_window),
            "decomposition_granularity": strategy.decomposition_granularity,
            "prompt_template_variant": strategy.prompt_template_variant,
        }
        for key, value in mappings.items():
            if key not in prototype.successful_strategies:
                prototype.successful_strategies[key] = {}
            strategy_data = prototype.successful_strategies[key]
            # Exponential moving average for strategy quality tracking
            if value in strategy_data:
                old_avg = strategy_data[value]
                strategy_data[value] = (1 - MetaAdapter.EMA_ALPHA) * old_avg + MetaAdapter.EMA_ALPHA * quality
            else:
                strategy_data[value] = quality

    def _evict_least_used(self) -> None:
        """Remove the prototype with the lowest sample count."""
        if not self._prototypes:
            return
        worst_id = min(
            self._prototypes,
            key=lambda pid: self._prototypes[pid].sample_count,
        )
        del self._prototypes[worst_id]
        logger.debug("MetaAdapter: evicted prototype %s", worst_id)

    def _next_prototype_id(self) -> str:
        """Generate the next sequential prototype ID.

        Uses a monotonically-incrementing counter so that evictions (which
        decrease ``len(self._prototypes)``) cannot cause ID reuse.

        Returns:
            String like 'proto_001', 'proto_002', etc.
        """
        self._next_proto_counter += 1
        return f"proto_{self._next_proto_counter:03d}"

    @staticmethod
    def _get_embedder():
        """Get an embedding function (reuses episode memory's embedder).

        Returns:
            A callable that takes a string and returns a list of floats.
        """
        try:
            from vetinari.learning.episode_memory import _simple_embedding

            return _simple_embedding
        except ImportError:
            # Inline fallback — should never happen in practice
            import hashlib

            def _fallback(text: str, dim: int = 256) -> list[float]:
                vec = [0.0] * dim
                for i in range(len(text) - 2):
                    gram = text[i : i + 3].lower()
                    h = int(hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest(), 16)
                    vec[h % dim] += 1.0
                norm = (sum(x * x for x in vec) ** 0.5) or 1.0
                return [x / norm for x in vec]

            logger.warning("vetinari.learning.episode_memory not importable — using inline trigram fallback embedder")
            return _fallback

    # ── Persistence ──────────────────────────────────────────────────

    @staticmethod
    def _default_state_path() -> Path:
        """Resolve default path for meta-adapter state.

        Returns:
            Path to meta_adapter_state.json.
        """
        state_dir_env = os.environ.get("VETINARI_STATE_DIR", "")
        if state_dir_env:
            state_dir = Path(state_dir_env)
        else:
            state_dir = VETINARI_STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "meta_adapter_state.json"

    def _load_state(self) -> None:
        """Load prototype state from JSON file."""
        try:
            if self._state_path.exists():
                with Path(self._state_path).open(encoding="utf-8") as f:
                    data = json.load(f)
                for pid, proto_data in data.items():
                    self._prototypes[pid] = TaskPrototype(**proto_data)
                # Seed the monotonic counter from the highest ID seen so that
                # new IDs never collide with existing ones after a reload.
                for pid in data:
                    if pid.startswith("proto_"):
                        try:
                            n = int(pid[len("proto_"):])
                            if n > self._next_proto_counter:
                                self._next_proto_counter = n
                        except ValueError:  # noqa: VET022 — non-numeric proto ID suffix is silently skipped by design
                            pass
                logger.debug(
                    "MetaAdapter: loaded %d prototypes from %s",
                    len(self._prototypes),
                    self._state_path,
                )
        except Exception:
            logger.warning(
                "MetaAdapter: could not load state from %s",
                self._state_path,
                exc_info=True,
            )

    def _save_state(self) -> None:
        """Persist prototype state to JSON file."""
        try:
            data = {pid: dataclass_to_dict(proto) for pid, proto in self._prototypes.items()}
            with Path(self._state_path).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            logger.warning(
                "MetaAdapter: could not save state to %s",
                self._state_path,
                exc_info=True,
            )


# ── Module-level singleton ───────────────────────────────────────────

_meta_adapter: MetaAdapter | None = None
_meta_adapter_lock = threading.Lock()


def get_meta_adapter() -> MetaAdapter:
    """Return the singleton MetaAdapter instance (thread-safe).

    Returns:
        The shared MetaAdapter instance.
    """
    global _meta_adapter
    if _meta_adapter is None:
        with _meta_adapter_lock:
            if _meta_adapter is None:
                _meta_adapter = MetaAdapter()
    return _meta_adapter
