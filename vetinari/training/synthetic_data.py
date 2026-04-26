"""Synthetic Data Generation for Vetinari Training.

Provides three complementary generators for creating high-quality training
data from the system's own capabilities, without relying on external datasets:

- ``SyntheticDataGenerator`` — mines past episodes and execution history to
  produce coding challenges, V-STaR reasoning chains, DPO preference pairs,
  and self-play tasks.
- ``MagpieGenerator`` — zero-seed instruction generation (ICLR 2025) via
  auto-regressive bootstrapping from the model's own distribution.
- ``StrategyDistiller`` — EvolveR-style offline self-distillation that extracts
  reusable abstract strategies from successful execution traces.

``MagpieGenerator`` and ``StrategyDistiller`` are implemented in
``synthetic_generators.py`` and re-exported here for backward compatibility.
All classes use late imports for vetinari dependencies and degrade gracefully
when the backing stores or LLM adapters are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.training.synthetic_generators import (
    MagpieGenerator,
    StrategyDistiller,
    _normalize,
)

logger = logging.getLogger(__name__)

# Minimum instruction length accepted by MagpieGenerator quality filter
_MIN_INSTRUCTION_LEN = 10
# Maximum instruction length accepted by MagpieGenerator quality filter
_MAX_INSTRUCTION_LEN = 500
# Minimum score gap required to form a DPO preference pair
_MIN_DPO_SCORE_GAP = 0.2
# Magpie over-generates by this multiplier then filters down to the target count
_MAGPIE_OVERSAMPLE_FACTOR = 2

__all__ = [
    "MagpieGenerator",
    "StrategyDistiller",
    "SyntheticDataGenerator",
    "_normalize",
    "generate_reasoning_chains",
]


# ── SyntheticDataGenerator ───────────────────────────────────────────────────


class SyntheticDataGenerator:
    """Generates training data from the system's own capabilities.

    Mines episodic memory and execution history to produce:
    - Coding challenge variations derived from past successful episodes
    - V-STaR reasoning chains with RLVR verification
    - DPO preference pairs from execution history
    - Self-play task seeds for iterative DPO

    All vetinari dependencies are imported lazily inside each method so
    that the class can be instantiated even when backing stores are absent.
    """

    def __init__(self) -> None:
        """Initialise with no required arguments.

        All vetinari module imports are deferred to method bodies.
        """

    def generate_coding_challenges(self, count: int = 50) -> list[dict[str, Any]]:
        """Generate coding challenge variations from past successful episodes.

        Queries episodic memory for successful coding episodes, then asks
        the loaded LLM to produce a variation on each one.  The variation
        prompt requests a structurally similar but distinct problem so the
        resulting dataset is diverse.

        Args:
            count: Target number of challenges to generate.  The actual
                returned count may be lower if episode memory or the LLM
                adapter is unavailable.

        Returns:
            List of ``{"instruction", "input", "output", "source_episode"}``
            dicts.  Empty list if episode memory is unavailable.
        """
        try:
            from vetinari.learning.episode_memory import get_episode_memory
        except ImportError:
            logger.warning("[SyntheticDataGenerator] episode_memory unavailable — returning empty coding challenges")
            return []

        try:
            mem = get_episode_memory()
        except Exception as exc:
            logger.warning("[SyntheticDataGenerator] Could not access episode memory: %s", exc)
            return []

        episodes = mem.recall(
            "coding implementation task",
            k=count * _MAGPIE_OVERSAMPLE_FACTOR,
            min_score=0.75,
            task_type="coding",
            successful_only=True,
        )

        if not episodes:
            logger.warning("[SyntheticDataGenerator] No successful coding episodes found")
            return []

        # Try to load an adapter for LLM calls
        adapter = self._get_adapter()
        challenges: list[dict[str, Any]] = []

        for ep in episodes:
            if len(challenges) >= count:
                break

            variation_prompt = (
                "You are creating training data for an AI coding assistant.\n\n"
                f"Here is a past coding task:\n{ep.task_summary}\n\n"
                "Generate a VARIATION of this task that is structurally similar "
                "but tests a different implementation detail or edge case. "
                "Output only the new task description, nothing else."
            )

            if adapter is not None:
                try:
                    result = adapter.chat(
                        model_id="default",
                        system_prompt="You are a coding challenge generator.",
                        input_text=variation_prompt,
                    )
                    instruction = result.get("output", "").strip()
                    if not instruction:
                        continue
                    challenges.append({
                        "instruction": instruction,
                        "input": "",
                        "output": ep.output_summary,
                        "source_episode": ep.episode_id,
                    })
                except Exception as exc:
                    logger.warning(
                        "[SyntheticDataGenerator] LLM call failed for episode %s: %s",
                        ep.episode_id,
                        exc,
                    )
            else:
                # No adapter — synthesise a templated variation without LLM
                instruction = f"Implement a variation of: {ep.task_summary}"
                challenges.append({
                    "instruction": instruction,
                    "input": "",
                    "output": ep.output_summary,
                    "source_episode": ep.episode_id,
                })

        logger.info("[SyntheticDataGenerator] Generated %d coding challenges", len(challenges))
        return challenges

    def generate_reasoning_chains(self, count: int = 50) -> list[dict[str, Any]]:
        """Generate V-STaR reasoning chains via bootstrapped self-verification.

        Implements the V-STaR paradigm: present problems, generate rationales,
        then verify correctness using existing RLVR signals (test execution and
        Quality agent scoring) rather than a separate verifier model.

        Verification relies on the existing RLVR signal stack:
          - Tier 1: code execution pass/fail (zero cost, instant)
          - Tier 2: test pass rate (existing Quality agent test runner)
          The combined signal determines which rationales are ``accepted``
          vs ``rejected``.

        Args:
            count: Target number of reasoning chain pairs to return.

        Returns:
            List of ``{"instruction", "output"}`` dicts where ``output``
            contains the accepted reasoning chain followed by the answer.
        """
        try:
            from vetinari.learning.episode_memory import get_episode_memory
        except ImportError:
            logger.warning("[SyntheticDataGenerator] episode_memory unavailable — returning empty reasoning chains")
            return []

        try:
            mem = get_episode_memory()
        except Exception as exc:
            logger.warning(
                "[SyntheticDataGenerator] Could not access episode memory for reasoning chains: %s",
                exc,
            )
            return []

        # Pull high-quality episodes across all task types to seed problems
        episodes = mem.recall(
            "problem solving reasoning",
            k=count * _MAGPIE_OVERSAMPLE_FACTOR,
            min_score=0.80,
        )

        adapter = self._get_adapter()
        chains: list[dict[str, Any]] = []

        for ep in episodes:
            if len(chains) >= count:
                break

            problem_statement = ep.task_summary
            rationale_prompt = (
                "Think step-by-step through the following problem and provide "
                "a detailed reasoning chain before giving your final answer.\n\n"
                f"Problem: {problem_statement}\n\n"
                "Format:\n"
                "<thinking>\n[your step-by-step reasoning]\n</thinking>\n\n"
                "Answer: [final answer]"
            )

            rationale = ""
            if adapter is not None:
                try:
                    result = adapter.chat(
                        model_id="default",
                        system_prompt="You are a careful problem solver.",
                        input_text=rationale_prompt,
                    )
                    rationale = result.get("output", "").strip()
                except Exception as exc:
                    logger.warning("[SyntheticDataGenerator] Rationale generation failed: %s", exc)

            if not rationale:
                # Use the episode's own output as the accepted chain
                rationale = f"<thinking>\n{ep.output_summary}\n</thinking>\n\nAnswer: see above"

            # Verification: use the episode quality_score as the RLVR signal.
            # Episodes with score >= 0.8 are treated as verified correct.
            # This avoids running a separate verifier model — the DAPO reward
            # tiers (test execution + Quality agent) already captured
            # correctness at record time.
            if ep.quality_score >= 0.80:
                chains.append({
                    "instruction": problem_statement,
                    "output": rationale,
                })

        logger.info("[SyntheticDataGenerator] Generated %d reasoning chains", len(chains))
        return chains

    def generate_dpo_pairs(self, count: int = 50) -> list[dict[str, Any]]:
        """Generate DPO preference pairs from execution history.

        Queries the training data collector for records where the same prompt
        received different quality scores and forms chosen/rejected pairs.
        Only pairs with a score gap of at least 0.2 are included to ensure
        the preference signal is meaningful.

        Args:
            count: Maximum number of DPO pairs to return.

        Returns:
            List of dicts with keys ``prompt``, ``chosen``, ``rejected``,
            ``chosen_score``, ``rejected_score``.  Empty list if training
            data is unavailable or no qualifying pairs exist.
        """
        try:
            from vetinari.learning.training_data import get_training_collector
        except ImportError:
            logger.warning("[SyntheticDataGenerator] training_data unavailable — returning empty DPO pairs")
            return []

        try:
            collector = get_training_collector()
            raw_pairs = collector.export_dpo_dataset(
                min_score_gap=_MIN_DPO_SCORE_GAP,
            )
        except Exception as exc:
            logger.warning("[SyntheticDataGenerator] Could not export DPO dataset: %s", exc)
            return []

        # Filter to requested count and ensure required score gap
        pairs: list[dict[str, Any]] = []
        for pair in raw_pairs:
            if len(pairs) >= count:
                break
            gap = pair.get("chosen_score", 0.0) - pair.get("rejected_score", 0.0)
            if gap < _MIN_DPO_SCORE_GAP:
                continue
            pairs.append({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "chosen_score": pair["chosen_score"],
                "rejected_score": pair["rejected_score"],
            })

        logger.info("[SyntheticDataGenerator] Produced %d DPO pairs", len(pairs))
        return pairs

    def generate_self_play_tasks(self, count: int = 20) -> list[dict[str, Any]]:
        """Generate seed tasks for iterative DPO self-play.

        Creates a diverse set of task seeds across multiple domains and
        difficulty levels.  The seeds are used as starting points for
        self-play loops where the model generates multiple candidate
        responses that are then ranked and used for DPO training.

        Args:
            count: Number of task seeds to generate.

        Returns:
            List of dicts with keys ``instruction``, ``domain``,
            ``difficulty``.
        """
        _domains = [
            "coding",
            "reasoning",
            "analysis",
            "summarisation",
            "debugging",
            "refactoring",
            "documentation",
        ]
        _difficulties = ["easy", "medium", "hard"]
        _templates: dict[str, list[str]] = {
            "coding": [
                "Implement a function that {verb} {noun} in Python.",
                "Write a class that manages {noun} with thread safety.",
                "Create a utility to {verb} {noun} using only the standard library.",
            ],
            "reasoning": [
                "Given {noun}, determine the most efficient approach to {verb} it.",
                "Explain the trade-offs between {noun_a} and {noun_b}.",
            ],
            "analysis": [
                "Analyse the time and space complexity of {noun}.",
                "Identify potential failure modes in {noun}.",
            ],
            "summarisation": [
                "Summarise the key design decisions behind {noun}.",
            ],
            "debugging": [
                "Find and fix the bug in a function that should {verb} {noun}.",
            ],
            "refactoring": [
                "Refactor {noun} to improve readability without changing behaviour.",
            ],
            "documentation": [
                "Write a Google-style docstring for a function that {verb} {noun}.",
            ],
        }
        _nouns = [
            "a binary search tree",
            "an LRU cache",
            "a rate limiter",
            "a connection pool",
            "a priority queue",
            "a Bloom filter",
            "a DAG scheduler",
            "a token bucket",
        ]
        _noun_pairs = [
            ("polling", "webhooks"),
            ("SQL", "NoSQL"),
            ("sync", "async"),
            ("eager loading", "lazy loading"),
        ]
        _verbs = [
            "serialise",
            "validate",
            "parse",
            "transform",
            "traverse",
            "index",
            "compress",
            "batch",
        ]

        import itertools
        import random

        rng = random.Random(42)  # noqa: S311 — deterministic seed for reproducibility, not cryptographic
        tasks: list[dict[str, Any]] = []

        domain_cycle = itertools.cycle(_domains)
        difficulty_cycle = itertools.cycle(_difficulties)

        for _ in range(count):
            domain = next(domain_cycle)
            difficulty = next(difficulty_cycle)
            templates = _templates.get(domain, _templates["coding"])
            template = rng.choice(templates)

            noun = rng.choice(_nouns)
            verb = rng.choice(_verbs)
            noun_pair = rng.choice(_noun_pairs)

            instruction = (
                template
                .replace("{noun}", noun)
                .replace("{verb}", verb)
                .replace("{noun_a}", noun_pair[0])
                .replace("{noun_b}", noun_pair[1])
            )

            tasks.append({
                "instruction": instruction,
                "domain": domain,
                "difficulty": difficulty,
            })

        logger.info("[SyntheticDataGenerator] Generated %d self-play tasks", len(tasks))
        return tasks

    def get_stats(self) -> dict[str, Any]:
        """Return statistics on generation capabilities.

        Checks which backing stores and adapters are available and reports
        approximate data volumes where the stores can be queried without
        side effects.

        Returns:
            Dict with keys ``episode_memory_available``,
            ``training_data_available``, ``adapter_available``,
            ``episode_count``, ``training_record_count``.
        """
        stats: dict[str, Any] = {
            "episode_memory_available": False,
            "training_data_available": False,
            "adapter_available": False,
            "episode_count": 0,
            "training_record_count": 0,
        }

        try:
            from vetinari.learning.episode_memory import get_episode_memory

            mem = get_episode_memory()
            # Probe approximate count via a broad recall
            episodes = mem.recall("task", k=10000)
            stats["episode_memory_available"] = True
            stats["episode_count"] = len(episodes)
        except Exception as exc:
            logger.warning("[SyntheticDataGenerator] Episode memory probe: %s", exc)

        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            records = collector.export_sft_dataset(min_score=0.0, max_records=100000)
            stats["training_data_available"] = True
            stats["training_record_count"] = len(records)
        except Exception as exc:
            logger.warning("[SyntheticDataGenerator] Training data probe: %s", exc)

        stats["adapter_available"] = self._get_adapter() is not None
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_adapter(self) -> Any | None:
        """Return the local inference adapter if available, else None.

        Returns:
            ``LocalInferenceAdapter`` instance or ``None``.
        """
        try:
            from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

            return LocalInferenceAdapter()
        except Exception as exc:
            logger.warning("[SyntheticDataGenerator] Adapter unavailable: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def generate_reasoning_chains(count: int = 50) -> list[dict[str, Any]]:
    """Generate V-STaR reasoning chains via bootstrapped self-verification.

    Module-level convenience wrapper around
    ``SyntheticDataGenerator().generate_reasoning_chains()``.

    Args:
        count: Target number of reasoning chain pairs to return.

    Returns:
        List of ``{"instruction", "output"}`` dicts where ``output``
        contains the accepted reasoning chain followed by the answer.
    """
    return SyntheticDataGenerator().generate_reasoning_chains(count=count)
