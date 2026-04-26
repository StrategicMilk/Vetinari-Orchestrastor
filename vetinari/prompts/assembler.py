"""Vetinari Dynamic Prompt Assembler.

===================================
Builds optimised prompts from modular components selected by agent type
and task type.  Replaces monolithic system prompts with composable pieces:

  Role definition  (short, per-agent)
  Task instructions (selected by task type)
  Output format     (JSON schema)
  Few-shot examples (dynamically selected by embedding similarity)
  Learned rules     (failure-pattern warnings from WorkflowLearner)

The assembler is context-window-aware: it respects the active model's
context limit and the target budget for each section.

Usage::

    from vetinari.prompts.assembler import get_prompt_assembler

    asm = get_prompt_assembler()
    prompt = asm.build(
        agent_type=AgentType.WORKER,
        task_type="coding",
        task_description="Implement a Redis cache wrapper",
        context_budget=28000,   # chars available after system prompt overhead
    )
    # -> {"system": "...", "user": "...", "total_chars": 4200}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vetinari.types import AgentType

logger = logging.getLogger(__name__)

try:
    from vetinari.adapters.llama_cpp_adapter import SYSTEM_PROMPT_BOUNDARY as _KV_CACHE_BOUNDARY
except Exception:
    _KV_CACHE_BOUNDARY = "<<<CONTEXT_BOUNDARY>>>"


# ---------------------------------------------------------------------------
# Role definitions (short, per agent — loaded lazily from existing SKILL.md)
# ---------------------------------------------------------------------------

_ROLE_DEFS: dict[str, str] = {
    AgentType.FOREMAN.value: (
        "You are Vetinari's Foreman — the planning and orchestration agent.\n\n"
        "Your responsibilities:\n"
        "- Decompose user goals into precise, dependency-aware task DAGs\n"
        "- Assign tasks to Worker agents with clear inputs, outputs, and success criteria\n"
        "- Monitor execution progress and handle task failures with re-planning\n"
        "- Never execute tasks directly — only plan, delegate, and coordinate\n\n"
        "You operate in modes: plan, decompose, monitor, delegate, prune, extract."
    ),
    AgentType.WORKER.value: (
        "You are Vetinari's Worker — the execution agent.\n\n"
        "Your responsibilities:\n"
        "- Execute implementation tasks: coding, research, documentation, data engineering, devops, operations\n"
        "- Produce production-quality output with proper error handling, type hints, and documentation\n"
        "- Self-reflect on output quality before submitting (draft → evaluate → refine → submit)\n"
        "- Report accurate confidence levels and flag uncertainty honestly\n\n"
        "You operate in modes: build, research, oracle, operations, and their sub-modes.\n"
        "Output must be structured JSON matching the task's output schema."
    ),
    AgentType.INSPECTOR.value: (
        "You are Vetinari's Inspector — the quality assurance agent.\n\n"
        "Your responsibilities:\n"
        "- Verify Worker output meets acceptance criteria with quantified assessments\n"
        "- Perform code review, security audits, test generation, and simplification\n"
        "- Extract per-claim verification with deterministic checks before model-based evaluation\n"
        "- Never generate new content — only evaluate, score, and provide remediation guidance\n\n"
        "You operate in modes: code_review, security_audit, test_generation, simplification.\n"
        "Every assessment must include: score (0.0-1.0), specific issues found, and remediation steps."
    ),
}

# Task-type-specific instructions (injected after role def)
_TASK_INSTRUCTIONS: dict[str, str] = {
    "coding": (
        "Generate clean, well-documented Python code. "
        "Include type hints, docstrings, error handling, and logging. "
        "Ensure all generated code is syntactically valid."
    ),
    "research": (
        "Cite sources where possible. Distinguish between established facts and your inference. "
        "Structure findings in sections with clear headings."
    ),
    "analysis": ("Be specific and evidence-based. Quantify where possible. Separate facts from recommendations."),
    "planning": (
        "Produce a concrete, actionable plan. Each step must have clear inputs, outputs, "
        "and success criteria. Identify dependencies and risks."
    ),
    "review": (
        "Be constructive and specific. For each issue, explain WHY it is a problem and provide a corrected example."
    ),
    "documentation": (
        "Write clear, accurate documentation for the target audience. Include examples for every "
        "non-trivial concept. Use consistent terminology and keep it DRY — don't repeat code verbatim."
    ),
    "general": ("Be precise and complete. If you are unsure about anything, say so explicitly rather than guessing."),
    "security": (
        "Identify all vulnerabilities with CWE IDs, severity levels (critical/high/medium/low), "
        "and provide concrete remediation code. Never approve code with critical issues."
    ),
    "devops": (
        "Produce complete, runnable configuration files. Include comments explaining non-obvious "
        "choices. Follow security best practices for all infrastructure code."
    ),
    "data": (
        "Use appropriate data types and constraints. Include both up and down migration scripts. "
        "Consider indexing, partitioning, and performance implications."
    ),
    "image": (
        "Produce optimized Stable Diffusion prompts. Be specific about style, composition, "
        "and technical quality. Provide SVG fallbacks for simple geometric designs."
    ),
}

# Output format templates by task type
_OUTPUT_FORMATS: dict[str, str] = {
    "coding": 'Return ONLY valid JSON:\n{"scaffold_code": "...", "tests": [...], "artifacts": [...], "implementation_notes": [...]}',
    "planning": 'Return ONLY valid JSON:\n{"tasks": [...], "dependencies": {...}, "risks": [...], "notes": "..."}',
    "research": 'Return ONLY valid JSON:\n{"summary": "...", "findings": [...], "sources": [...], "recommendations": [...]}',
    "analysis": 'Return ONLY valid JSON:\n{"summary": "...", "findings": [...], "recommendations": [...]}',
    "review": 'Return ONLY valid JSON:\n{"score": 0.0, "issues": [...], "suggestions": [...], "summary": "..."}',
    "general": "Return clear, structured output. Use JSON when the output is data-oriented.",
}

# Minimum assembled system prompt length — shorter indicates a loading failure
_MIN_SYSTEM_PROMPT_CHARS = 100


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class PromptAssembler:
    """Builds modular, context-window-aware prompts for any agent/task combo.

    Prompt structure is optimized for prefix caching (KV cache reuse):

    **Static prefix** (cacheable across requests for same agent/task type):
        1. Role definition
        2. Standards/rules prefix
        3. Task-type instructions
        4. Output format spec

    **Dynamic suffix** (varies per request):
        5. Learned failure-pattern rules
        6. Few-shot examples (from episode memory)
        7. Task description (user message)

    The static prefix is hashed and cached via ``prompt_cache.py`` to track
    cache hit rates.  LiteLLM-compatible ``cache_control`` metadata is
    included in the output for providers that support prefix caching
    (Anthropic, OpenAI with prompt caching enabled).

    **IMPORTANT: The assembler REPLACES the system prompt, not composes with it.**
    The output of ``build()["system"]`` is used as the complete system prompt
    at ``inference.py:407-408``, overriding any prior prompt from
    ``base_agent_prompts.build_system_prompt()``.  This is intentional: the
    assembler consolidates role identity, rules, task instructions, and
    dynamic content (few-shot examples, learned rules) into a single
    KV-cache-optimized prompt.
    """

    # Section budgets as fractions of total context_budget
    BUDGET_ROLE = 0.05
    BUDGET_INSTRUCTIONS = 0.08
    BUDGET_FORMAT = 0.05
    BUDGET_RULES = 0.05
    BUDGET_EXAMPLES = 0.15
    BUDGET_TASK = 1.0 - 0.05 - 0.08 - 0.05 - 0.05 - 0.15  # ~62%

    def __init__(self):
        self._examples_cache: dict[str, list[dict]] = {}
        self._rules_cache: dict[str, list[str]] = {}
        self._prompt_cache = None

    def _get_prompt_cache(self):
        """Lazy-load the prompt cache singleton.

        Returns:
            PromptCache instance, or None if unavailable.
        """
        if self._prompt_cache is None:
            try:
                from vetinari.optimization.prompt_cache import get_prompt_cache

                self._prompt_cache = get_prompt_cache()
            except Exception:
                logger.warning("Prompt cache unavailable", exc_info=True)
        return self._prompt_cache

    def build(
        self,
        agent_type: str,
        task_type: str,
        task_description: str,
        mode: str | None = None,
        context_budget: int = 28000,
        include_examples: bool = True,
        include_rules: bool = True,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a complete prompt dict for the given agent and task.

        The system prompt is structured for prefix caching: static content
        (role, standards, instructions, format) comes first as a cacheable
        prefix, followed by dynamic content (rules, examples) as a suffix.

        Args:
            agent_type:       AgentType enum value string (e.g. "WORKER").
            task_type:        Task category (coding / research / planning / ...).
            task_description: The actual task request.
            mode:             Optional agent mode (e.g. "build", "code_review").
                              Passed to ``_get_role()`` to load mode-specific
                              prompt content from ``config/agents/``.
            context_budget:   Total character budget for the entire prompt.
            include_examples: Whether to inject few-shot examples.
            include_rules:    Whether to inject learned failure-pattern rules.
            project_id:       Optional project ID for project-scoped rules.
            model_id:         Optional model ID for model-scoped rules.

        Returns:
            Dict with keys: ``system``, ``user``, ``total_chars``,
            ``agent_type``, ``task_type``, ``cache_control``, ``cache_hit``.
        """
        budget = max(context_budget, 4000)

        # ── Static Prefix (cacheable) ─────────────────────────────────

        # 1. Role definition (most stable — changes only with agent type)
        role = self._get_role(agent_type, mode=mode)
        role = self._truncate(role, int(budget * self.BUDGET_ROLE))

        # 2. Standards/rules prefix (stable per project/model config)
        rules_prefix = ""
        try:
            from vetinari.rules_manager import get_rules_manager

            rm = get_rules_manager()
            rules_prefix = rm.build_system_prompt_prefix(project_id=project_id, model_id=model_id)
        except Exception:
            logger.warning("Failed to load rules prefix for prompt assembly", exc_info=True)

        # 3. Task-type instructions (stable per task type)
        instructions = self._get_instructions(task_type)
        instructions = self._truncate(instructions, int(budget * self.BUDGET_INSTRUCTIONS))

        # 4. Output format (stable per task type)
        fmt = self._get_format(task_type)
        fmt = self._truncate(fmt, int(budget * self.BUDGET_FORMAT))

        # Assemble static prefix — this is the cacheable portion
        static_parts = [p for p in [role, rules_prefix, instructions, fmt] if p]
        static_prefix = "\n\n".join(static_parts)

        # Track cache hit on static prefix
        cache_hit = False
        cache = self._get_prompt_cache()
        if cache is not None:
            try:
                from vetinari.optimization.prompt_cache import hash_prompt

                prefix_hash = hash_prompt(static_prefix)
                cache_result = cache.get_or_cache(prefix_hash, static_prefix)
                cache_hit = cache_result.hit
                if cache_hit:
                    logger.debug(
                        "Prompt prefix cache HIT for %s/%s (saved ~%d tokens)",
                        agent_type,
                        task_type,
                        cache_result.savings_tokens,
                    )
            except Exception:
                logger.warning("Prompt cache lookup failed", exc_info=True)

        # ── Dynamic Suffix (varies per request) ──────────────────────

        # 5. Learned failure rules (changes as learner accumulates data)
        rules_text = ""
        if include_rules:
            rules = self._get_rules(agent_type, task_type)
            if rules:
                rules_text = "IMPORTANT — avoid these known failure patterns:\n" + "\n".join(
                    f"- {r}" for r in rules[:5]
                )
            rules_text = self._truncate(rules_text, int(budget * self.BUDGET_RULES))

        # 6. Few-shot examples (dynamically selected per task)
        examples_text = ""
        if include_examples:
            examples = self._get_examples(agent_type, task_type, task_description)
            if examples:
                examples_text = "EXAMPLES:\n" + "\n---\n".join(
                    f"Task: {e.get('task', '')[:200]}\nOutput: {e.get('output', '')[:400]}" for e in examples
                )
            examples_text = self._truncate(examples_text, int(budget * self.BUDGET_EXAMPLES))

        dynamic_parts = [p for p in [rules_text, examples_text] if p]
        dynamic_suffix = "\n\n".join(dynamic_parts)

        # ── Assemble Final System Prompt ──────────────────────────────
        # Static prefix first (cacheable), then a stable boundary marker that
        # the local llama.cpp adapter can split on for KV-prefix reuse.
        if static_prefix and dynamic_suffix:
            system_prompt = static_prefix + _KV_CACHE_BOUNDARY + dynamic_suffix
        else:
            system_parts = [p for p in [static_prefix, dynamic_suffix] if p]
            system_prompt = "\n\n".join(system_parts)

        # Phase 8.43: Compress dynamic suffix when prompt exceeds budget
        if len(system_prompt) > budget:
            try:
                from vetinari.optimization.prompt_compressor import PerplexityCompressor

                compressor = PerplexityCompressor()
                target_ratio = budget / len(system_prompt)
                # Only compress the dynamic suffix — keep static prefix intact
                if dynamic_suffix and len(dynamic_suffix) > 200:
                    compressed_suffix = compressor.compress(dynamic_suffix, target_ratio)
                    if static_prefix and compressed_suffix:
                        system_prompt = static_prefix + _KV_CACHE_BOUNDARY + compressed_suffix
                    else:
                        system_prompt = "\n\n".join(
                            [p for p in [static_prefix, compressed_suffix] if p],
                        )
                    logger.debug(
                        "Compressed prompt from %d to %d chars (ratio=%.2f)",
                        len(static_prefix) + len(dynamic_suffix),
                        len(system_prompt),
                        target_ratio,
                    )
            except Exception:
                logger.warning("Prompt compressor unavailable for budget enforcement")

        # Validate minimum prompt structure — very short prompts indicate a loading failure
        if len(system_prompt) < _MIN_SYSTEM_PROMPT_CHARS:
            logger.warning(
                "Assembled prompt for %s/%s is suspiciously short (%d chars < %d minimum) — "
                "check that agent identity and task instructions loaded correctly",
                agent_type,
                task_type,
                len(system_prompt),
                _MIN_SYSTEM_PROMPT_CHARS,
            )

        # 7. User message (task description)
        remaining = budget - len(system_prompt)
        user_prompt = self._truncate(task_description, max(remaining, 500))

        total_chars = len(system_prompt) + len(user_prompt)

        return {
            "system": system_prompt,
            "user": user_prompt,
            "total_chars": total_chars,
            "agent_type": agent_type,
            "task_type": task_type,
            "cache_control": {
                "type": "ephemeral",
                "prefix_chars": len(static_prefix),
            },
            "cache_hit": cache_hit,
        }

    # ------------------------------------------------------------------
    # Component getters
    # ------------------------------------------------------------------

    def _get_role(self, agent_type: str, mode: str | None = None) -> str:
        """Load agent identity, preferring config/agents/ markdown files.

        Tries ``prompt_loader.load_agent_prompt()`` first so that rich markdown
        prompts in ``config/agents/`` are used when available.  Falls back to
        the hardcoded ``_ROLE_DEFS`` entries when no markdown file exists,
        logging a warning so missing prompts are visible and don't silently
        degrade inference quality.

        Args:
            agent_type: AgentType enum value string (e.g. "WORKER").
            mode: Optional agent mode to load mode-specific prompt content.

        Returns:
            Agent identity string suitable for use as the role section of the
            system prompt.
        """
        try:
            from vetinari.agents.prompt_loader import load_agent_prompt

            loaded = load_agent_prompt(agent_type, mode=mode)
            if loaded:
                return loaded
        except Exception:
            logger.warning(
                "Failed to load prompt from markdown for %s — falling back to hardcoded role",
                agent_type,
                exc_info=True,
            )

        # Fallback to hardcoded role defs
        role = _ROLE_DEFS.get(agent_type.upper())
        if role:
            logger.info(
                "Using fallback role definition for %s — no config/agents/ markdown found",
                agent_type,
            )
            return role
        return f"You are a Vetinari {agent_type} specialist."

    def _get_instructions(self, task_type: str) -> str:
        return _TASK_INSTRUCTIONS.get(task_type.lower(), _TASK_INSTRUCTIONS["general"])

    def _get_format(self, task_type: str) -> str:
        return _OUTPUT_FORMATS.get(task_type.lower(), _OUTPUT_FORMATS["general"])

    def _get_rules(self, agent_type: str, task_type: str) -> list[str]:
        """Load learned failure-pattern rules from WorkflowLearner."""
        cache_key = f"{agent_type}:{task_type}"
        if cache_key in self._rules_cache:
            return self._rules_cache[cache_key]
        rules: list[str] = []
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner

            learner = get_workflow_learner()
            domain = self._task_to_domain(task_type)
            rec = learner.get_recommendations(domain)
            if rec and rec.get("warnings"):
                rules = rec["warnings"][:5]
        except Exception:
            logger.warning("Failed to load learned failure rules for %s:%s", agent_type, task_type, exc_info=True)
        self._rules_cache[cache_key] = rules
        return rules

    def _get_examples(
        self,
        agent_type: str,
        task_type: str,
        task_description: str,
    ) -> list[dict]:
        """Return 1-3 few-shot examples relevant to this task.

        Falls back to static template examples when the training corpus
        is not yet populated.
        """
        try:
            from vetinari.memory import get_unified_memory_store

            memory = get_unified_memory_store()
            episodes = memory.recall_episodes(task_description, k=3, min_score=0.7)
            if episodes:
                return [{"task": e.task_summary, "output": e.output_summary} for e in episodes]
        except Exception:
            logger.warning("Failed to recall few-shot examples from episode memory", exc_info=True)

        # Static examples from template files
        tmpl_path = Path(__file__).parent.parent.parent / "templates" / "v1" / f"{agent_type.lower()}.json"
        if tmpl_path.exists():
            try:
                with Path(tmpl_path).open(encoding="utf-8") as f:
                    templates = json.load(f)
                examples = templates.get("templates", [])[:3]
                return [
                    {
                        "task": ex.get("description", ""),
                        "output": json.dumps(ex.get("expected_output", {}))[:300],
                    }
                    for ex in examples
                    if ex.get("description")
                ]
            except Exception:
                logger.warning("Failed to load static template examples from %s", tmpl_path, exc_info=True)

        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate text to max_chars, appending a truncation indicator when shortened.

        Args:
            text: Text to truncate.
            max_chars: Maximum allowed character length including the indicator.

        Returns:
            Original text if within the limit, otherwise the text shortened to
            fit max_chars with "[...truncated...]" appended.
        """
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        _indicator = "[...truncated...]"
        return text[: max(max_chars - len(_indicator), 0)] + _indicator

    @staticmethod
    def _task_to_domain(task_type: str) -> str:
        mapping = {
            "coding": "coding",
            "code_gen": "coding",
            "builder": "coding",
            "research": "research",
            "researcher": "research",
            "data": "data",
            "data_engineering": "data",
            "documentation": "docs",
            "docs": "docs",
        }
        return mapping.get(task_type.lower(), "general")


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_assembler: PromptAssembler | None = None
_assembler_lock = __import__("threading").Lock()


def get_prompt_assembler() -> PromptAssembler:
    """Return the global PromptAssembler singleton.

    Returns:
        The PromptAssembler result.
    """
    global _assembler
    if _assembler is None:
        with _assembler_lock:
            if _assembler is None:
                _assembler = PromptAssembler()
    return _assembler
