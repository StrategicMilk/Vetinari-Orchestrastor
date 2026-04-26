"""Structured Prompt Mutation Operators — Department 9.6.

Implements deterministic, composable mutation operators for agent system
prompts.  Each operator transforms a prompt section in a specific way
(rephrase instructions, inject constraints, restructure format, etc.).
The operators are selected by Thompson Sampling via OperatorSelector
and validated by the existing PromptEvolver A/B testing pipeline.

Based on MASPOB (Multi-Agent System Prompt Optimization Benchmark, 2025)
findings: prompt section positioning has measurable effects on agent
output quality.  Identity/role sections perform best at prompt start;
constraints near the end; examples immediately before the task
description.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import threading
from enum import Enum
from itertools import islice, permutations
from pathlib import Path

from vetinari.constants import VETINARI_STATE_DIR
from vetinari.exceptions import AgentError

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def _compile_section_keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile and cache a section-header pattern for a given keyword.

    The lru_cache avoids re.compile overhead when the same keyword frozenset
    is used repeatedly across mutation calls.

    Args:
        keyword: The keyword to search for in a markdown section header.

    Returns:
        Compiled regex pattern matching a heading containing the keyword.
    """
    return re.compile(
        rf"^(#{{1,3}})\s+.*{re.escape(keyword)}.*$",
        re.MULTILINE | re.IGNORECASE,
    )


# ── Section detection ────────────────────────────────────────────────
# Matches markdown-style section headers: ## Title or # Title
_SECTION_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Common section role classifications for MASPOB ordering
_ROLE_KEYWORDS = frozenset({
    "identity",
    "role",
    "persona",
    "who you are",
    "you are",
})
_CONSTRAINT_KEYWORDS = frozenset({
    "constraints",
    "rules",
    "requirements",
    "limitations",
    "guardrails",
    "boundaries",
    "restrictions",
})
_EXAMPLE_KEYWORDS = frozenset({
    "examples",
    "example",
    "few-shot",
    "demonstrations",
    "samples",
})
_TASK_KEYWORDS = frozenset({
    "task",
    "objective",
    "goal",
    "instructions",
    "what to do",
})


class MutationOperator(Enum):
    """Deterministic mutation operators for prompt sections.

    Each operator transforms prompt text in a specific, reproducible way.
    Operators are composable — multiple operators can be applied in
    sequence to produce compound mutations.
    """

    INSTRUCTION_REPHRASE = "instruction_rephrase"
    CONSTRAINT_INJECTION = "constraint_injection"
    EXAMPLE_INJECTION = "example_injection"
    FORMAT_RESTRUCTURE = "format_restructure"
    REASONING_SCAFFOLD = "reasoning_scaffold"
    ROLE_REINFORCEMENT = "role_reinforcement"
    OUTPUT_SCHEMA_TIGHTEN = "output_schema_tighten"
    CONTEXT_PRUNE = "context_prune"


class PromptMutator:
    """Applies structured mutations to prompt sections.

    Each mutation operator is deterministic (no LLM calls required) and
    composable.  The mutator parses prompts into sections, applies the
    requested operator, and returns the modified prompt.

    Args:
        seed_constraints: Extra constraint phrases available for injection.
        seed_examples: Few-shot example strings available for injection.
    """

    # Default constraint bank — domain-agnostic constraints that can be
    # injected to tighten agent behavior
    DEFAULT_CONSTRAINTS = (
        "Think step by step before answering.",
        "If uncertain, state your confidence level explicitly.",
        "Cite specific evidence for every claim.",
        "Prefer concrete examples over abstract descriptions.",
        "Keep responses focused — do not add unrequested information.",
    )

    # Reasoning scaffold templates
    REASONING_SCAFFOLDS = (
        "\n\nBefore answering, reason through the problem step by step:\n"
        "1. Identify the core question or task.\n"
        "2. List relevant facts and constraints.\n"
        "3. Consider alternative approaches.\n"
        "4. Choose the best approach and explain why.\n"
        "5. Produce your answer.\n",
        "\n\nUse the following thinking framework:\n"
        "- What do I know? (given information)\n"
        "- What do I need? (desired output)\n"
        "- What could go wrong? (edge cases)\n"
        "- What is my answer? (conclusion)\n",
    )

    # Output schema tightening patterns
    OUTPUT_SCHEMAS = (
        "\n\nFormat your response as:\n"
        "## Analysis\n[Your analysis here]\n\n"
        "## Recommendation\n[Your recommendation here]\n\n"
        "## Confidence\n[HIGH/MEDIUM/LOW with justification]\n",
        "\n\nStructure your output with clear sections:\n"
        "1. **Summary** — one-sentence answer\n"
        "2. **Details** — supporting evidence\n"
        "3. **Caveats** — limitations or risks\n",
    )

    def __init__(
        self,
        seed_constraints: tuple[str, ...] | None = None,
        seed_examples: tuple[str, ...] | None = None,
    ) -> None:
        self._constraints = seed_constraints or self.DEFAULT_CONSTRAINTS
        self._examples = seed_examples or ()
        # Track mutation count for deterministic variant selection
        self._mutation_counter: int = 0

    def mutate(
        self,
        prompt: str,
        operator: MutationOperator,
        section: str | None = None,
    ) -> str:
        """Apply a specific mutation operator to a prompt or section.

        Args:
            prompt: The full prompt text to mutate.
            operator: Which mutation operator to apply.
            section: Optional section name to target.  If None, the
                operator is applied to the full prompt.

        Returns:
            The mutated prompt text.

        Raises:
            ValueError: If the operator is not recognised.
        """
        self._mutation_counter += 1

        match operator:
            case MutationOperator.INSTRUCTION_REPHRASE:
                return self._rephrase_instructions(prompt, section)
            case MutationOperator.CONSTRAINT_INJECTION:
                return self._inject_constraint(prompt, section)
            case MutationOperator.EXAMPLE_INJECTION:
                return self._inject_example(prompt, section)
            case MutationOperator.FORMAT_RESTRUCTURE:
                return self._restructure_format(prompt, section)
            case MutationOperator.REASONING_SCAFFOLD:
                return self._add_reasoning_scaffold(prompt, section)
            case MutationOperator.ROLE_REINFORCEMENT:
                return self._reinforce_role(prompt, section)
            case MutationOperator.OUTPUT_SCHEMA_TIGHTEN:
                return self._tighten_output_schema(prompt, section)
            case MutationOperator.CONTEXT_PRUNE:
                return self._prune_context(prompt, section)
            case _:
                msg = f"Unknown mutation operator: {operator}"
                raise AgentError(msg)

    # ── Operator implementations ─────────────────────────────────────

    def _rephrase_instructions(self, prompt: str, section: str | None) -> str:
        """Rephrase imperative instructions to declarative or vice versa.

        Alternates between imperative→declarative and positive→negative
        framing based on mutation counter.
        """
        target = self._extract_section(prompt, section) if section else prompt

        if self._mutation_counter % 2 == 0:
            # Imperative → declarative framing
            rephrased = self._imperative_to_declarative(target)
        else:
            # Positive → negative framing ("Do X" → "Never fail to X")
            rephrased = self._positive_to_negative(target)

        # Fallback: if no patterns matched, add an instruction-clarity prefix
        if rephrased == target:
            rephrased = "Important: follow these instructions precisely.\n\n" + target

        if section:
            return prompt.replace(target, rephrased)
        return rephrased

    def _inject_constraint(self, prompt: str, section: str | None) -> str:
        """Add a constraint from the constraint bank."""
        idx = self._mutation_counter % len(self._constraints)
        constraint = self._constraints[idx]

        # Find the best insertion point — after constraints section or at end
        constraint_line = f"\n- {constraint}\n"

        if section:
            target = self._extract_section(prompt, section)
            return prompt.replace(target, target + constraint_line)

        # Try to find an existing constraints section
        for kw in _CONSTRAINT_KEYWORDS:
            match = _compile_section_keyword_pattern(kw).search(prompt)
            if match:
                # Insert after the section header's content
                next_section = _SECTION_HEADER_RE.search(prompt, match.end() + 1)
                insert_pos = next_section.start() if next_section else len(prompt)
                return prompt[:insert_pos] + constraint_line + prompt[insert_pos:]

        # No constraints section — append before the last section
        return prompt + constraint_line

    def _inject_example(self, prompt: str, section: str | None) -> str:
        """Add a few-shot example from the example bank."""
        if not self._examples:
            # No examples available — add a generic example request
            example_block = "\n\n**Example:**\nInput: [representative input]\nOutput: [expected output format]\n"
        else:
            idx = self._mutation_counter % len(self._examples)
            example_block = f"\n\n**Example:**\n{self._examples[idx]}\n"

        if section:
            target = self._extract_section(prompt, section)
            return prompt.replace(target, target + example_block)

        # Insert before task section (MASPOB: examples before task description)
        for kw in _TASK_KEYWORDS:
            match = _compile_section_keyword_pattern(kw).search(prompt)
            if match:
                return prompt[: match.start()] + example_block + prompt[match.start() :]

        return prompt + example_block

    def _restructure_format(self, prompt: str, section: str | None) -> str:
        """Reorder prompt sections based on MASPOB optimal positioning.

        When sufficient quality data has been accumulated by
        :class:`MASPOBAnalyzer`, uses learned ordering instead of the static
        MASPOB heuristic.  Falls back to the heuristic when data is sparse.

        MASPOB heuristic:
        - Identity/role sections perform best at prompt start
        - Constraints perform best near the end
        - Examples perform best immediately before the task description
        """
        sections = self._parse_sections(prompt)
        if len(sections) < 2:
            # Not enough sections to restructure meaningfully
            return prompt

        # ── Learned ordering via MASPOBAnalyzer ──────────────────────
        section_names = [title for title, _ in sections]
        analyzer = get_maspob_analyzer()
        if analyzer.has_sufficient_data(section_names):
            optimal_order = analyzer.get_optimal_ordering(section_names)
            section_map = dict(sections)
            ordered = [(name, section_map[name]) for name in optimal_order if name in section_map]
            # Append any sections not covered by the learned ordering
            covered = set(optimal_order)
            ordered += [(t, c) for t, c in sections if t not in covered]
        else:
            # ── Static MASPOB heuristic ───────────────────────────────
            role_sections: list[tuple[str, str]] = []
            constraint_sections: list[tuple[str, str]] = []
            example_sections: list[tuple[str, str]] = []
            task_sections: list[tuple[str, str]] = []
            other_sections: list[tuple[str, str]] = []

            for title, content in sections:
                title_lower = title.lower()
                if any(kw in title_lower for kw in _ROLE_KEYWORDS):
                    role_sections.append((title, content))
                elif any(kw in title_lower for kw in _CONSTRAINT_KEYWORDS):
                    constraint_sections.append((title, content))
                elif any(kw in title_lower for kw in _EXAMPLE_KEYWORDS):
                    example_sections.append((title, content))
                elif any(kw in title_lower for kw in _TASK_KEYWORDS):
                    task_sections.append((title, content))
                else:
                    other_sections.append((title, content))

            # MASPOB optimal order: role → other → examples → task → constraints
            ordered = role_sections + other_sections + example_sections + task_sections + constraint_sections

        # Extract preamble (text before first section header)
        first_header = _SECTION_HEADER_RE.search(prompt)
        preamble = prompt[: first_header.start()].rstrip() if first_header else ""

        # Reconstruct prompt
        parts = [preamble] if preamble else []
        for title, content in ordered:
            parts.append(f"\n\n## {title}\n{content}")

        return "\n".join(parts).strip()

    def _add_reasoning_scaffold(self, prompt: str, section: str | None) -> str:
        """Add chain-of-thought or step-by-step reasoning instructions."""
        idx = self._mutation_counter % len(self.REASONING_SCAFFOLDS)
        scaffold = self.REASONING_SCAFFOLDS[idx]

        if section:
            target = self._extract_section(prompt, section)
            return prompt.replace(target, target + scaffold)

        # Add before the last section or at the end
        return prompt + scaffold

    def _reinforce_role(self, prompt: str, section: str | None) -> str:
        """Strengthen or refine the agent's identity/role description."""
        # Find existing role section
        for kw in _ROLE_KEYWORDS:
            match = _compile_section_keyword_pattern(kw).search(prompt)
            if match:
                # Add emphasis to role section
                next_section = _SECTION_HEADER_RE.search(prompt, match.end() + 1)
                end_pos = next_section.start() if next_section else len(prompt)

                reinforcement = (
                    "\n\nThis is your primary function.  Every response must "
                    "reflect this role.  If a request falls outside your role, "
                    "explicitly acknowledge the boundary before proceeding.\n"
                )

                return prompt[:end_pos].rstrip() + reinforcement + prompt[end_pos:]

        # No role section found — prepend role reinforcement
        reinforcement = "You are a specialist agent.  Stay focused on your designated role and expertise area.\n\n"
        return reinforcement + prompt

    def _tighten_output_schema(self, prompt: str, section: str | None) -> str:
        """Add or tighten output format specifications."""
        idx = self._mutation_counter % len(self.OUTPUT_SCHEMAS)
        schema = self.OUTPUT_SCHEMAS[idx]

        if section:
            target = self._extract_section(prompt, section)
            return prompt.replace(target, target + schema)

        return prompt + schema

    def _prune_context(self, prompt: str, section: str | None) -> str:
        """Remove low-signal sections identified by length heuristic.

        Removes sections that are:
        - Very short (< 20 chars, likely empty or trivial)
        - Duplicative (content appears elsewhere in the prompt)
        """
        sections = self._parse_sections(prompt)
        if len(sections) < 3:
            return prompt

        # Find sections to prune (short or near-duplicate)
        pruned_titles: set[str] = set()
        seen_content: set[str] = set()

        for title, content in sections:
            normalised = content.strip().lower()
            if len(normalised) < 20:
                pruned_titles.add(title)
                logger.debug("Pruning short section: %s (%d chars)", title, len(normalised))
            elif normalised in seen_content:
                pruned_titles.add(title)
                logger.debug("Pruning duplicate section: %s", title)
            else:
                seen_content.add(normalised)

        if not pruned_titles:
            return prompt

        # Reconstruct without pruned sections
        first_header = _SECTION_HEADER_RE.search(prompt)
        preamble = prompt[: first_header.start()].rstrip() if first_header else ""

        parts = [preamble] if preamble else []
        for title, content in sections:
            if title not in pruned_titles:
                parts.append(f"\n\n## {title}\n{content}")

        return "\n".join(parts).strip()

    # ── Helper methods ───────────────────────────────────────────────

    def _parse_sections(self, prompt: str) -> list[tuple[str, str]]:
        """Parse a prompt into (title, content) sections.

        Args:
            prompt: The prompt text with markdown-style section headers.

        Returns:
            List of (section_title, section_content) tuples.
        """
        matches = list(_SECTION_HEADER_RE.finditer(prompt))
        if not matches:
            return [("", prompt)]

        sections: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
            content = prompt[start:end].strip()
            sections.append((title, content))

        return sections

    def _extract_section(self, prompt: str, section_name: str) -> str:
        """Extract the content of a named section from the prompt.

        Args:
            prompt: The full prompt text.
            section_name: The section title to find.

        Returns:
            The section content, or the full prompt if section not found.
        """
        for title, content in self._parse_sections(prompt):
            if title.lower() == section_name.lower():
                return content
        return prompt

    @staticmethod
    def _imperative_to_declarative(text: str) -> str:
        """Convert imperative instructions to declarative framing.

        Transforms patterns like "Do X" → "The agent does X",
        "Generate Y" → "The output includes Y".
        """
        replacements = [
            (r"(?i)^(\s*)Do\b", r"\1The agent should"),
            (r"(?i)^(\s*)Generate\b", r"\1The output includes"),
            (r"(?i)^(\s*)Analyze\b", r"\1The analysis covers"),
            (r"(?i)^(\s*)Review\b", r"\1The review examines"),
            (r"(?i)^(\s*)Create\b", r"\1The deliverable is"),
            (r"(?i)^(\s*)Write\b", r"\1The output provides"),
            (r"(?i)^(\s*)Check\b", r"\1Verification covers"),
            (r"(?i)^(\s*)Ensure\b", r"\1It is required that"),
        ]
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        return result

    @staticmethod
    def _positive_to_negative(text: str) -> str:
        """Convert positive framing to negative emphasis.

        Transforms patterns like "Be concise" → "Never be verbose",
        adding emphasis through negative framing.
        """
        replacements = [
            (r"(?i)\bBe concise\b", "Never be verbose or add unnecessary detail"),
            (r"(?i)\bBe specific\b", "Never be vague or abstract"),
            (r"(?i)\bBe accurate\b", "Never include unverified information"),
            (r"(?i)\bBe clear\b", "Never use ambiguous language"),
            (r"(?i)\bBe thorough\b", "Never skip important details"),
            (r"(?i)\bBe helpful\b", "Never provide unhelpful or irrelevant responses"),
        ]
        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        return result


# ---------------------------------------------------------------------------
# MASPOB position-sensitivity analyzer
# ---------------------------------------------------------------------------

_DEFAULT_STATE_PATH = VETINARI_STATE_DIR / "maspob_state.json"
_MIN_SAMPLES: int = 10
_MAX_PERMUTATIONS: int = 24  # cap permutation generation for efficiency


class MASPOBAnalyzer:
    """Learns optimal prompt section ordering via position-sensitivity analysis.

    Records quality scores for different section orderings and converges
    on the best arrangement.  When sufficient data exists (>= 10 samples),
    provides learned ordering to ``_restructure_format()`` instead of the
    static MASPOB heuristic.

    Args:
        state_path: Path to the JSON file used for persistent state.
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self._state_path = state_path or _DEFAULT_STATE_PATH
        # Maps section_name → {position_str → [quality_scores]}
        self._position_stats: dict[str, dict[str, list[float]]] = {}
        self._lock = threading.Lock()
        self._load_state()

    # ── Public methods ────────────────────────────────────────────────

    def analyze_section_ordering(self, prompt: str) -> list[list[str]]:
        """Generate candidate section orderings (permutations) from a prompt.

        Parses section headers from *prompt* and returns up to
        ``_MAX_PERMUTATIONS`` distinct orderings.

        Args:
            prompt: The prompt text containing markdown-style section headers.

        Returns:
            List of section-name lists representing candidate orderings.
        """
        # Reuse the module-level _SECTION_HEADER_RE (group 2 is the title text)
        section_names = [m.group(2).strip() for m in _SECTION_HEADER_RE.finditer(prompt)]
        if not section_names:
            return []
        # Cap permutations to avoid combinatorial explosion
        all_perms = list(islice(permutations(section_names), _MAX_PERMUTATIONS))
        return [list(p) for p in all_perms]

    def record_quality(self, section_name: str, position: int, quality_score: float) -> None:
        """Record an observed quality score for a section at a given position.

        Args:
            section_name: The name of the prompt section.
            position: Zero-based position index in the ordering.
            quality_score: Quality metric (higher is better, e.g. 0.0-1.0).
        """
        pos_key = str(position)
        with self._lock:
            if section_name not in self._position_stats:
                self._position_stats[section_name] = {}
            section_stats = self._position_stats[section_name]
            if pos_key not in section_stats:
                section_stats[pos_key] = []
            section_stats[pos_key].append(quality_score)
        self._save_state()

    def get_optimal_ordering(self, section_names: list[str]) -> list[str]:
        """Return the best section ordering based on accumulated quality data.

        Uses a greedy Hungarian-style algorithm: iteratively assigns each
        section to the available position with its highest average quality
        score.

        Args:
            section_names: Sections to order.

        Returns:
            Ordered list of section names.  Falls back to input order if
            insufficient data for any section.
        """
        if not self.has_sufficient_data(section_names):
            return list(section_names)

        n = len(section_names)
        available_positions = set(range(n))
        ordered: list[tuple[int, str]] = []  # (position, section_name)

        with self._lock:
            remaining = list(section_names)
            while remaining and available_positions:
                best_section: str | None = None
                best_pos: int | None = None
                best_avg: float = -1.0

                for sec in remaining:
                    stats = self._position_stats.get(sec, {})
                    for pos in available_positions:
                        scores = stats.get(str(pos), [])
                        if scores:
                            avg = sum(scores) / len(scores)
                            if avg > best_avg:
                                best_avg = avg
                                best_section = sec
                                best_pos = pos

                if best_section is None or best_pos is None:
                    # No data for any remaining pair — append in input order
                    for sec in remaining:
                        pos = min(available_positions)
                        ordered.append((pos, sec))
                        available_positions.discard(pos)
                    break

                ordered.append((best_pos, best_section))
                remaining.remove(best_section)
                available_positions.discard(best_pos)

        ordered.sort(key=lambda x: x[0])
        return [sec for _, sec in ordered]

    def has_sufficient_data(self, section_names: list[str]) -> bool:
        """Return True when every (section, position) slot has sufficient evidence.

        The learned ordering is only used when *each* (section, position) pair has
        at least ``_MIN_SAMPLES`` observations.  A section with many samples
        concentrated at one position still lacks evidence for the others — the
        greedy assignment in ``get_optimal_ordering()`` needs per-slot confidence
        to produce a trustworthy result.

        Args:
            section_names: Section names to check. Position slots are
                ``range(len(section_names))``, mirroring ``get_optimal_ordering()``.

        Returns:
            True when every (section, position) pair has ``_MIN_SAMPLES``
            observations, False otherwise (falls back to heuristic ordering).
        """
        n = len(section_names)
        with self._lock:
            for name in section_names:
                stats = self._position_stats.get(name, {})
                for pos in range(n):
                    if len(stats.get(str(pos), [])) < _MIN_SAMPLES:
                        return False
        return True

    # ── Persistence ───────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load persisted position stats from disk, if available."""
        try:
            if self._state_path.exists():
                with self._state_path.open("r", encoding="utf-8") as fh:
                    self._position_stats = json.load(fh)
        except Exception as exc:
            logger.warning("MASPOBAnalyzer: failed to load state from %s: %s", self._state_path, exc)
            self._position_stats = {}

    def _save_state(self) -> None:
        """Persist position stats to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = dict(self._position_stats)
            with self._state_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.warning("MASPOBAnalyzer: failed to save state to %s: %s", self._state_path, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_maspob_analyzer: MASPOBAnalyzer | None = None
_maspob_lock: threading.Lock = threading.Lock()


def get_maspob_analyzer(state_path: Path | None = None) -> MASPOBAnalyzer:
    """Return the module-level singleton :class:`MASPOBAnalyzer`.

    When ``state_path`` is provided the singleton is (re-)created with that
    path.  This allows tests to inject a temporary directory without relying
    on the process-wide default.

    Args:
        state_path: Optional override for the state file path.  Passing a
            non-None value always creates a fresh instance, even if one
            already exists.

    Returns:
        The singleton :class:`MASPOBAnalyzer` instance.
    """
    global _maspob_analyzer
    if _maspob_analyzer is None or state_path is not None:
        with _maspob_lock:
            if _maspob_analyzer is None or state_path is not None:
                _maspob_analyzer = MASPOBAnalyzer(state_path=state_path)
    return _maspob_analyzer
