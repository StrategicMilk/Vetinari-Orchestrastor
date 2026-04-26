"""Test-Time Compute Scaling for Vetinari.

===========================================
Implements three levels of test-time compute scaling:

  Level 1: Best-of-N — generate N candidates and pick the best
  Level 2: Heuristic-guided — score intermediate reasoning steps and prune
           low-quality paths using an n-gram coherence + character-entropy
           heuristic (NGramHeuristicScorer). This is NOT a process reward
           model; see LATER-WAVE-REGISTRY for the reserved PRMScorer name.
  Level 3: MCTS — Monte Carlo Tree Search over decomposition candidates
            using UCB1 selection, expansion, simulation, and backpropagation

Usage::

    from vetinari.optimization.test_time_compute import TestTimeComputeScaler

    scaler = TestTimeComputeScaler()

    # Auto-select level based on task complexity (1-10 scale)
    level = scaler.auto_select_level(task_complexity=6)  # -> 2

    # Scale compute for a task
    result = scaler.scale(
        task="Implement a Redis cache with TTL support",
        level=2,
        evaluate_fn=lambda s: 0.9 if "ttl" in s.lower() else 0.5,
    )
    logger.debug(result.quality_estimate)
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vetinari.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# UCB1 exploration constant: sqrt(2) is the theoretical optimum
_UCB1_DEFAULT_C: float = 1.414

# Minimum visit count before a node is eligible for UCB1 selection
_MIN_VISITS_FOR_UCB: int = 1

# Entropy bounds for coherence scoring
_ENTROPY_LOW_THRESHOLD: float = 0.5  # below this → trivially short/empty
_ENTROPY_HIGH_THRESHOLD: float = 4.5  # above this → incoherent noise


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ComputeStepScore:
    """Quality score for a single reasoning step."""

    step_text: str
    score: float
    reasoning: str  # Human-readable explanation of why this score was assigned


StepScore = ComputeStepScore


@dataclass
class ComputeResult:
    """Result produced by TestTimeComputeScaler.scale()."""

    level_used: int
    result: str
    quality_estimate: float
    steps_evaluated: int
    computation_budget_used: float

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"ComputeResult(level_used={self.level_used!r},"
            f" quality_estimate={self.quality_estimate!r},"
            f" steps_evaluated={self.steps_evaluated!r})"
        )


@dataclass
class MCTSNode:
    """A single node in the MCTS search tree."""

    state: str  # Task description at this level
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    parent: MCTSNode | None = None
    action: str = ""  # Decomposition step that led here

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"MCTSNode(visits={self.visits!r}, total_value={self.total_value!r}, action={self.action!r})"

    @property
    def value(self) -> float:
        """Mean value over all visits.

        Returns:
            Mean reward, or 0.0 if never visited.
        """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits


# ---------------------------------------------------------------------------
# NGramHeuristicScorer
# ---------------------------------------------------------------------------


class NGramHeuristicScorer:
    """Scores intermediate reasoning steps using an n-gram coherence heuristic.

    This is NOT a process reward model. It computes a deterministic score
    from two signals over the candidate step and the preceding context:

      - Bigram overlap between the step and the accumulated context.
      - Shannon entropy over the step's character distribution.

    The final score is a weighted blend (0.6 * overlap + 0.4 * entropy_score)
    clamped to [0.0, 1.0]. There is no learned reward model loaded and no
    value head trained on preference data — the scorer is strictly heuristic.

    The name `PRMScorer` is reserved for a future implementation that loads
    an actual process-reward-model head (tracked in LATER-WAVE-REGISTRY).

    Args:
        coherence_threshold: Steps scoring below this are considered
            low-quality and eligible for pruning (0.0-1.0).
    """

    def __init__(self, coherence_threshold: float = 0.5) -> None:
        self._threshold = coherence_threshold

    def score_steps(self, steps: list[str]) -> list[ComputeStepScore]:
        """Score each step independently, passing the preceding steps as context.

        Args:
            steps: Ordered list of reasoning step strings.

        Returns:
            List of StepScore objects, one per input step.
        """
        context = ""
        scored: list[ComputeStepScore] = []
        for step in steps:
            s = self.score_step(step, context)
            reasoning = self._describe_score(s, step)
            scored.append(ComputeStepScore(step_text=step, score=s, reasoning=reasoning))
            context = f"{context} {step}".strip()
        return scored

    def score_step(self, step: str, context: str = "") -> float:
        """Score a single step given the preceding context.

        Args:
            step: The reasoning step text to evaluate.
            context: All prior steps concatenated (may be empty string).

        Returns:
            Quality score in [0.0, 1.0].
        """
        if not step.strip():
            return 0.0
        return self._compute_coherence(step, context)

    def prune_low_quality(
        self,
        steps: list[str],
        threshold: float | None = None,
    ) -> list[str]:
        """Remove steps that score below the coherence threshold.

        Args:
            steps: Ordered list of reasoning step strings.
            threshold: Override the instance-level threshold for this call.

        Returns:
            Filtered list containing only steps that pass the threshold.
        """
        cutoff = threshold if threshold is not None else self._threshold
        scored = self.score_steps(steps)
        kept = [ss.step_text for ss in scored if ss.score >= cutoff]
        pruned = len(steps) - len(kept)
        if pruned:
            logger.info("[NGramHeuristicScorer] Pruned %d/%d low-quality steps", pruned, len(steps))
        return kept

    def _compute_coherence(self, text: str, context: str) -> float:
        """Compute a coherence score using n-gram overlap and entropy.

        Strategy:
        - N-gram overlap: fraction of text bigrams that also appear in context.
          High overlap = the step continues existing threads (good).
        - Entropy: Shannon entropy over character frequencies. Very low = trivial;
          very high = random noise. We reward the middle ground.
        - Final score = 0.6 * overlap + 0.4 * entropy_score

        Args:
            text: The step text to evaluate.
            context: All prior text (may be empty).

        Returns:
            Coherence score in [0.0, 1.0].
        """
        # --- n-gram overlap ---
        overlap_score = self._ngram_overlap(text, context) if context else 0.5

        # --- character-level Shannon entropy ---
        entropy = self._char_entropy(text)
        if entropy < _ENTROPY_LOW_THRESHOLD:
            entropy_score = entropy / _ENTROPY_LOW_THRESHOLD * 0.3
        elif entropy > _ENTROPY_HIGH_THRESHOLD:
            excess = entropy - _ENTROPY_HIGH_THRESHOLD
            entropy_score = max(0.0, 1.0 - excess * 0.3)
        else:
            # Linear ramp from 0.3 at low threshold to 1.0 at mid, back to 0.7 at high
            mid = (_ENTROPY_LOW_THRESHOLD + _ENTROPY_HIGH_THRESHOLD) / 2
            if entropy <= mid:
                entropy_score = 0.3 + 0.7 * (entropy - _ENTROPY_LOW_THRESHOLD) / (mid - _ENTROPY_LOW_THRESHOLD)
            else:
                entropy_score = 1.0 - 0.3 * (entropy - mid) / (_ENTROPY_HIGH_THRESHOLD - mid)

        coherence = 0.6 * overlap_score + 0.4 * entropy_score
        return round(min(1.0, max(0.0, coherence)), 4)

    @staticmethod
    def _ngram_overlap(text: str, context: str, n: int = 2) -> float:
        """Fraction of text n-grams that appear in context.

        Args:
            text: The text to analyse.
            context: Reference text.
            n: N-gram size.

        Returns:
            Overlap ratio in [0.0, 1.0].
        """
        text_lower = text.lower()
        context_lower = context.lower()

        text_grams = {text_lower[i : i + n] for i in range(len(text_lower) - n + 1)}
        if not text_grams:
            return 0.5

        context_grams = {context_lower[i : i + n] for i in range(len(context_lower) - n + 1)}
        if not context_grams:
            return 0.5  # No context yet — neutral score

        overlap = len(text_grams & context_grams) / len(text_grams)
        # Clamp: very high overlap (copying) is also penalised slightly
        return min(1.0, overlap * 1.5) if overlap < 0.8 else 0.6 + overlap * 0.1

    @staticmethod
    def _char_entropy(text: str) -> float:
        """Shannon entropy over character frequency distribution.

        Args:
            text: Input text.

        Returns:
            Entropy in bits (unbounded above, typically 0-5 for natural text).
        """
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    @staticmethod
    def _describe_score(score: float, step: str) -> str:
        """Generate a human-readable explanation for a step score.

        Args:
            score: Computed coherence score.
            step: The step text.

        Returns:
            Short reasoning string.
        """
        length = len(step.strip())
        if score >= 0.8:
            return f"High coherence (score={score:.2f}): step is well-structured and {length} chars"
        if score >= 0.5:
            return f"Moderate coherence (score={score:.2f}): step is adequate at {length} chars"
        return f"Low coherence (score={score:.2f}): step may be too short, repetitive, or noisy ({length} chars)"


# ---------------------------------------------------------------------------
# MCTSPlanner
# ---------------------------------------------------------------------------


class MCTSPlanner:
    """Tree search over candidate task decompositions.

    Uses UCB1 for selection, random expansion, rollout simulation,
    and backpropagation to find the best decomposition path.

    Args:
        exploration_weight: UCB1 exploration constant C. Higher = more
            exploration vs exploitation. Default sqrt(2) ≈ 1.414.
        max_iterations: Number of MCTS simulation loops.
        max_depth: Maximum depth of the search tree.
    """

    def __init__(
        self,
        exploration_weight: float = _UCB1_DEFAULT_C,
        max_iterations: int = 100,
        max_depth: int = 5,
    ) -> None:
        self._c = exploration_weight
        self._max_iter = max_iterations
        self._max_depth = max_depth

    def search(
        self,
        root_task: str,
        evaluate_fn: Callable[[str], float],
    ) -> list[str]:
        """Run MCTS and return the best decomposition path found.

        Args:
            root_task: The top-level task description to decompose.
            evaluate_fn: Callable that scores a task description string,
                returning a quality estimate in [0.0, 1.0].

        Returns:
            Ordered list of decomposition steps along the best path.
        """
        root = MCTSNode(state=root_task)

        for _ in range(self._max_iter):
            node = self._select(root)
            if node.visits > 0 and self._depth(node) < self._max_depth:
                node = self._expand(node)
            value = self._simulate(node, evaluate_fn)
            self._backpropagate(node, value)

        return self._get_best_path(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend tree using UCB1 until an unvisited or leaf node.

        Args:
            node: Starting node for selection.

        Returns:
            Selected node for expansion/simulation.
        """
        while node.children:
            # If any child is unvisited, select it
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return unvisited[0]
            node = max(node.children, key=lambda c: self._ucb1(c))
        return node

    def _ucb1(self, node: MCTSNode) -> float:
        """Compute UCB1 score for a node.

        Args:
            node: The node to score.

        Returns:
            UCB1 value (exploitation + exploration bonus).
        """
        if node.visits == 0:
            return float("inf")
        parent_visits = node.parent.visits if node.parent else node.visits
        exploitation = node.value
        exploration = self._c * math.sqrt(math.log(max(1, parent_visits)) / node.visits)
        return exploitation + exploration

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Generate child nodes by decomposing the current task.

        Args:
            node: The node to expand.

        Returns:
            A newly created child node.
        """
        if not node.children:
            subtasks = self._decompose_step(node.state)
            for subtask in subtasks:
                child = MCTSNode(state=subtask, parent=node, action=subtask)
                node.children.append(child)

        # Return first unvisited child, or the first child if all visited
        for child in node.children:
            if child.visits == 0:
                return child
        return node.children[0]

    def _simulate(
        self,
        node: MCTSNode,
        evaluate_fn: Callable[[str], float],
    ) -> float:
        """Rollout: evaluate the current node's task quality.

        Args:
            node: Node to evaluate.
            evaluate_fn: Quality scoring function.

        Returns:
            Quality estimate in [0.0, 1.0].
        """
        try:
            score = evaluate_fn(node.state)
            return float(max(0.0, min(1.0, score)))
        except Exception:
            logger.warning("[MCTSPlanner] evaluate_fn raised; defaulting to 0.5")
            return 0.5

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the simulation result up to the root.

        Args:
            node: Leaf node where simulation completed.
            value: Reward to propagate.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def _decompose_step(self, task: str) -> list[str]:
        """Heuristic decomposition: split a task into 2-4 subtasks.

        This is MCTS's **action-proposal policy** — it generates the set of
        candidate child states when `_expand` is called on a node. It is NOT
        the search algorithm itself; the UCB1 selection in `_select`, the
        tree expansion in `_expand`, the rollout evaluation in `_simulate`,
        and the value backpropagation in `_backpropagate` together constitute
        the search. A domain-specific proposal policy (regex, heuristic
        grammar, or a neural policy head in stronger variants) is a standard
        MCTS component — see ADR-0100 for evidence.

        Uses pattern matching to identify common decomposition points
        (and, then, also, first/second/finally). Falls back to sentence
        splitting or generic subtasks if no patterns are found.

        Args:
            task: Task description to decompose.

        Returns:
            List of 2-4 subtask strings.
        """
        # Try splitting on conjunction keywords
        conjunctions = re.split(r"\b(?:and then|and also|, then|; then|; and)\b", task, flags=re.IGNORECASE)
        if len(conjunctions) >= 2:
            return [s.strip() for s in conjunctions[:4] if s.strip()]

        # Try splitting on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", task.strip())
        if len(sentences) >= 2:
            return [s.strip() for s in sentences[:4] if s.strip()]

        # Generic subtask generation based on task length
        words = task.split()
        if len(words) <= 5:
            return [f"Prepare for: {task}", f"Execute: {task}"]
        mid = len(words) // 2
        part1 = " ".join(words[:mid])
        part2 = " ".join(words[mid:])
        return [f"Phase 1 — {part1}", f"Phase 2 — {part2}"]

    def _get_best_path(self, root: MCTSNode) -> list[str]:
        """Trace the highest-value path from root to deepest best child.

        Args:
            root: Root node of the MCTS tree.

        Returns:
            List of action strings along the best path (excludes root action).
        """
        path: list[str] = []
        node = root
        while node.children:
            best = max(node.children, key=lambda c: c.value)
            if best.visits == 0:
                break
            if best.action:
                path.append(best.action)
            node = best
        return path or [root.state]

    @staticmethod
    def _depth(node: MCTSNode) -> int:
        """Compute depth of a node from the root.

        Args:
            node: Node to measure.

        Returns:
            Integer depth (root = 0).
        """
        depth = 0
        current: MCTSNode | None = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth


# ---------------------------------------------------------------------------
# TestTimeComputeScaler
# ---------------------------------------------------------------------------


class TestTimeComputeScaler:
    """Orchestrates test-time compute scaling across 3 levels.

    Level 1: Best-of-N (existing) — generate N responses, pick best
    Level 2: Heuristic-guided — score reasoning steps with
             NGramHeuristicScorer, prune low-quality paths
    Level 3: MCTS — tree search over decomposition candidates

    Args:
        ngram_heuristic_scorer: Optional NGramHeuristicScorer instance.
            Created on first use if None.
        mcts_planner: Optional MCTSPlanner instance. Created on first use if None.
    """

    # Complexity thresholds for auto_select_level
    _L2_THRESHOLD: int = 4  # complexity >= 4 → Level 2
    _L3_THRESHOLD: int = 8  # complexity >= 8 → Level 3

    def __init__(
        self,
        ngram_heuristic_scorer: NGramHeuristicScorer | None = None,
        mcts_planner: MCTSPlanner | None = None,
    ) -> None:
        self._scorer = ngram_heuristic_scorer
        self._mcts = mcts_planner

    @property
    def ngram_heuristic_scorer(self) -> NGramHeuristicScorer:
        """Lazy-initialised n-gram heuristic scorer.

        Returns:
            NGramHeuristicScorer instance.
        """
        if self._scorer is None:
            self._scorer = NGramHeuristicScorer()
        return self._scorer

    @property
    def mcts_planner(self) -> MCTSPlanner:
        """Lazy-initialised MCTSPlanner.

        Returns:
            MCTSPlanner instance.
        """
        if self._mcts is None:
            self._mcts = MCTSPlanner()
        return self._mcts

    def scale(
        self,
        task: str,
        level: int,
        evaluate_fn: Callable[[str], float] | None = None,
        n: int = 3,
    ) -> ComputeResult:
        """Run the specified compute level and return the result.

        Args:
            task: Task description to process.
            level: Compute level 1, 2, or 3.
            evaluate_fn: Optional quality-scoring callable (required for L3;
                used as heuristic in L2 if provided).
            n: Number of candidates for Level 1 Best-of-N.

        Returns:
            ComputeResult with result string, quality estimate, and budget info.

        Raises:
            ValueError: If an unsupported level is requested.
        """
        if level == 1:
            return self._level1_best_of_n(task, n, evaluate_fn)
        if level == 2:
            return self._level2_prm(task, evaluate_fn)
        if level == 3:
            return self._level3_mcts(task, evaluate_fn)
        raise ConfigurationError(f"Unsupported compute level: {level}. Must be 1, 2, or 3.")

    def auto_select_level(self, task_complexity: int) -> int:
        """Recommend a compute level based on task complexity.

        Args:
            task_complexity: Integer complexity rating from 1 (trivial) to
                10 (extremely complex).

        Returns:
            Recommended level: 1, 2, or 3.
        """
        if task_complexity >= self._L3_THRESHOLD:
            return 3
        if task_complexity >= self._L2_THRESHOLD:
            return 2
        return 1

    # ------------------------------------------------------------------
    # Level implementations
    # ------------------------------------------------------------------

    def _level1_best_of_n(
        self,
        task: str,
        n: int,
        evaluate_fn: Callable[[str], float] | None,
    ) -> ComputeResult:
        """Level 1: generate N candidate responses, return the best.

        Attempts to delegate to BestOfNSelector from best_of_n module if
        available; falls back to an inline heuristic.

        Args:
            task: Task description.
            n: Number of candidates to generate.
            evaluate_fn: Optional scoring function.

        Returns:
            ComputeResult at level 1.
        """
        try:
            from vetinari.optimization.best_of_n import BestOfNSelector  # type: ignore[import]

            selector = BestOfNSelector()
            candidates = selector.generate_candidates(task, n=n)
            best, score = selector.select_best(candidates, evaluate_fn=evaluate_fn)
            return ComputeResult(
                level_used=1,
                result=best,
                quality_estimate=score,
                steps_evaluated=n,
                computation_budget_used=float(n),
            )
        except ImportError:
            logger.debug("test_time_compute optional dependencies unavailable")

        # Inline fallback: score each candidate variant with PRM
        candidates = self._generate_simple_candidates(task, n)
        if evaluate_fn is not None:
            scored = [(c, evaluate_fn(c)) for c in candidates]
        else:
            scored = [(c, self.ngram_heuristic_scorer.score_step(c)) for c in candidates]

        best_candidate, best_score = max(scored, key=lambda x: x[1])
        logger.info("[L1] Best-of-%d selected candidate with score=%.3f", n, best_score)
        return ComputeResult(
            level_used=1,
            result=best_candidate,
            quality_estimate=best_score,
            steps_evaluated=n,
            computation_budget_used=float(n),
        )

    def _level2_prm(
        self,
        task: str,
        evaluate_fn: Callable[[str], float] | None,
    ) -> ComputeResult:
        """Level 2: decompose task, score steps with PRM, prune and combine.

        Args:
            task: Task description.
            evaluate_fn: Optional external scoring function.

        Returns:
            ComputeResult at level 2.
        """
        # Decompose task into steps using MCTS heuristic decomposer
        raw_steps = self.mcts_planner._decompose_step(task)
        scored = self.ngram_heuristic_scorer.score_steps(raw_steps)
        kept = self.ngram_heuristic_scorer.prune_low_quality(raw_steps)

        if not kept:
            kept = raw_steps  # Fall back to all steps if everything was pruned

        combined = " -> ".join(kept)
        if evaluate_fn is not None:
            quality = evaluate_fn(combined)
        else:
            quality = sum(ss.score for ss in scored) / max(len(scored), 1)

        logger.info("[L2] PRM kept %d/%d steps, quality=%.3f", len(kept), len(raw_steps), quality)
        return ComputeResult(
            level_used=2,
            result=combined,
            quality_estimate=float(quality),
            steps_evaluated=len(raw_steps),
            computation_budget_used=float(len(raw_steps)) * 2.0,
        )

    def _level3_mcts(
        self,
        task: str,
        evaluate_fn: Callable[[str], float] | None,
    ) -> ComputeResult:
        """Level 3: MCTS search over decomposition space.

        Args:
            task: Task description.
            evaluate_fn: Scoring function. Defaults to PRM coherence if None.

        Returns:
            ComputeResult at level 3.
        """
        eff_evaluate = evaluate_fn if evaluate_fn is not None else self.ngram_heuristic_scorer.score_step
        path = self.mcts_planner.search(task, eff_evaluate)
        result = " -> ".join(path) if path else task
        quality = eff_evaluate(path[-1] if path else task)
        iterations = self.mcts_planner._max_iter

        logger.info("[L3] MCTS found path of %d steps, quality=%.3f", len(path), quality)
        return ComputeResult(
            level_used=3,
            result=result,
            quality_estimate=float(quality),
            steps_evaluated=iterations,
            computation_budget_used=float(iterations),
        )

    @staticmethod
    def _generate_simple_candidates(task: str, n: int) -> list[str]:
        """Generate simple textual variants of the task as L1 candidates.

        Args:
            task: Original task description.
            n: Number of candidates.

        Returns:
            List of n candidate strings.
        """
        variants: list[str] = [task]
        prefixes = [
            "Step by step: ",
            "Carefully consider and then: ",
            "Breaking it down: ",
            "Systematically: ",
        ]
        for i in range(n - 1):
            prefix = prefixes[i % len(prefixes)]
            variants.append(f"{prefix}{task}")
        return variants[:n]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_test_time_scaler(**kwargs: Any) -> TestTimeComputeScaler:
    """Convenience factory returning a TestTimeComputeScaler.

    Args:
        **kwargs: Forwarded to TestTimeComputeScaler.__init__.

    Returns:
        A new TestTimeComputeScaler instance.
    """
    return TestTimeComputeScaler(**kwargs)
