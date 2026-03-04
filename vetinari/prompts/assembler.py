"""
Vetinari Dynamic Prompt Assembler
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
        agent_type=AgentType.BUILDER,
        task_type="coding",
        task_description="Implement a Redis cache wrapper",
        context_budget=28000,   # chars available after system prompt overhead
    )
    # -> {"system": "...", "user": "...", "total_chars": 4200}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role definitions (short, per agent — loaded lazily from existing SKILL.md)
# ---------------------------------------------------------------------------

_ROLE_DEFS: Dict[str, str] = {
    "PLANNER": "You are Vetinari's Planner. Decompose goals into precise, dependency-aware task DAGs.",
    "EXPLORER": "You are Vetinari's Explorer. Search codebases and documentation to find exactly what is asked.",
    "ORACLE": "You are Vetinari's Oracle. Give authoritative architectural guidance backed by evidence.",
    "LIBRARIAN": "You are Vetinari's Librarian. Research libraries, APIs, and documentation with precision.",
    "RESEARCHER": "You are Vetinari's Researcher. Synthesise information from multiple sources into actionable insights.",
    "EVALUATOR": "You are Vetinari's Evaluator. Review code and outputs with rigorous, objective quality criteria.",
    "SYNTHESIZER": "You are Vetinari's Synthesizer. Merge outputs from multiple agents into a coherent, conflict-free whole.",
    "BUILDER": "You are Vetinari's Builder. Generate production-quality code scaffolds and implementation.",
    "UI_PLANNER": "You are Vetinari's UI Planner. Design accessible, modern UI/UX with full component specifications.",
    "SECURITY_AUDITOR": "You are Vetinari's Security Auditor. Identify vulnerabilities and provide remediation code.",
    "DATA_ENGINEER": "You are Vetinari's Data Engineer. Design correct, performant schemas, migrations, and pipelines.",
    "DOCUMENTATION_AGENT": "You are Vetinari's Documentation Agent. Produce clear, complete technical documentation.",
    "COST_PLANNER": "You are Vetinari's Cost Planner. Analyse and minimise inference costs while maintaining quality.",
    "TEST_AUTOMATION": "You are Vetinari's Test Automation Engineer. Generate real, meaningful tests with full coverage.",
    "EXPERIMENTATION_MANAGER": "You are Vetinari's Experimentation Manager. Design and analyse rigorous A/B experiments.",
    "IMPROVEMENT": "You are Vetinari's Improvement Agent. Analyse system performance and recommend optimisations.",
    "USER_INTERACTION": "You are Vetinari's User Interaction Agent. Clarify ambiguous requests with minimal questions.",
    "DEVOPS": "You are Vetinari's DevOps Agent. Design robust CI/CD pipelines and infrastructure configurations.",
    "VERSION_CONTROL": "You are Vetinari's Version Control Agent. Provide git strategy, commit messages, and PR templates.",
    "ERROR_RECOVERY": "You are Vetinari's Error Recovery Agent. Diagnose failures and generate recovery strategies.",
    "CONTEXT_MANAGER": "You are Vetinari's Context Manager. Consolidate and optimise long-term memory.",
    "IMAGE_GENERATOR": "You are Vetinari's Image Generator. Create high-quality visual assets from descriptions.",
}

# Task-type-specific instructions (injected after role def)
_TASK_INSTRUCTIONS: Dict[str, str] = {
    "coding": (
        "Generate clean, well-documented Python code. "
        "Include type hints, docstrings, error handling, and logging. "
        "Ensure all generated code is syntactically valid."
    ),
    "research": (
        "Cite sources where possible. Distinguish between established facts and your inference. "
        "Structure findings in sections with clear headings."
    ),
    "analysis": (
        "Be specific and evidence-based. Quantify where possible. "
        "Separate facts from recommendations."
    ),
    "planning": (
        "Produce a concrete, actionable plan. Each step must have clear inputs, outputs, "
        "and success criteria. Identify dependencies and risks."
    ),
    "review": (
        "Be constructive and specific. For each issue, explain WHY it is a problem "
        "and provide a corrected example."
    ),
    "documentation": (
        "Write for the target audience. Include examples for every non-trivial concept. "
        "Use consistent terminology throughout."
    ),
    "general": (
        "Be precise and complete. If you are unsure about anything, say so explicitly "
        "rather than guessing."
    ),
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
    "documentation": (
        "Write clear, accurate documentation with examples. Target the appropriate audience. "
        "Keep it DRY — don't repeat code from the codebase verbatim."
    ),
    "image": (
        "Produce optimized Stable Diffusion prompts. Be specific about style, composition, "
        "and technical quality. Provide SVG fallbacks for simple geometric designs."
    ),
}

# Output format templates by task type
_OUTPUT_FORMATS: Dict[str, str] = {
    "coding": 'Return ONLY valid JSON:\n{"scaffold_code": "...", "tests": [...], "artifacts": [...], "implementation_notes": [...]}',
    "planning": 'Return ONLY valid JSON:\n{"tasks": [...], "dependencies": {...}, "risks": [...], "notes": "..."}',
    "research": 'Return ONLY valid JSON:\n{"summary": "...", "findings": [...], "sources": [...], "recommendations": [...]}',
    "analysis": 'Return ONLY valid JSON:\n{"summary": "...", "findings": [...], "recommendations": [...]}',
    "review": 'Return ONLY valid JSON:\n{"score": 0.0, "issues": [...], "suggestions": [...], "summary": "..."}',
    "general": "Return clear, structured output. Use JSON when the output is data-oriented.",
}


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class PromptAssembler:
    """Builds modular, context-window-aware prompts for any agent/task combo."""

    # Section budgets as fractions of total context_budget
    BUDGET_ROLE = 0.05
    BUDGET_INSTRUCTIONS = 0.08
    BUDGET_FORMAT = 0.05
    BUDGET_RULES = 0.05
    BUDGET_EXAMPLES = 0.15
    BUDGET_TASK = 1.0 - 0.05 - 0.08 - 0.05 - 0.05 - 0.15  # ~62%

    def __init__(self):
        self._examples_cache: Dict[str, List[Dict]] = {}
        self._rules_cache: Dict[str, List[str]] = {}

    def build(
        self,
        agent_type: str,
        task_type: str,
        task_description: str,
        context_budget: int = 28000,
        include_examples: bool = True,
        include_rules: bool = True,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a complete prompt dict for the given agent and task.

        Args:
            agent_type:       AgentType enum value string (e.g. "BUILDER")
            task_type:        Task category (coding / research / planning / …)
            task_description: The actual task request
            context_budget:   Total character budget for the entire prompt
            include_examples: Whether to inject few-shot examples
            include_rules:    Whether to inject learned failure-pattern rules
            project_id:       Optional project ID for project-scoped rules
            model_id:         Optional model ID for model-scoped rules

        Returns:
            Dict with "system", "user", and "total_chars" keys.
        """
        budget = max(context_budget, 4000)

        # 0. Inject global system prompt prefix + hierarchical user rules
        rules_prefix = ""
        try:
            from vetinari.rules_manager import get_rules_manager
            rm = get_rules_manager()
            rules_prefix = rm.build_system_prompt_prefix(
                project_id=project_id, model_id=model_id
            )
        except Exception:
            pass

        # 1. Role definition
        role = self._get_role(agent_type)
        role = self._truncate(role, int(budget * self.BUDGET_ROLE))

        # 2. Task-type instructions
        instructions = self._get_instructions(task_type)
        instructions = self._truncate(instructions, int(budget * self.BUDGET_INSTRUCTIONS))

        # 3. Output format
        fmt = self._get_format(task_type)
        fmt = self._truncate(fmt, int(budget * self.BUDGET_FORMAT))

        # 4. Learned failure rules
        rules_text = ""
        if include_rules:
            rules = self._get_rules(agent_type, task_type)
            if rules:
                rules_text = "IMPORTANT — avoid these known failure patterns:\n" + "\n".join(
                    f"- {r}" for r in rules[:5]
                )
            rules_text = self._truncate(rules_text, int(budget * self.BUDGET_RULES))

        # 5. Few-shot examples (dynamically selected)
        examples_text = ""
        if include_examples:
            examples = self._get_examples(agent_type, task_type, task_description)
            if examples:
                examples_text = "EXAMPLES:\n" + "\n---\n".join(
                    f"Task: {e.get('task', '')[:200]}\nOutput: {e.get('output', '')[:400]}"
                    for e in examples
                )
            examples_text = self._truncate(examples_text, int(budget * self.BUDGET_EXAMPLES))

        # Assemble system prompt (rules_prefix first so it's always visible)
        system_parts = [p for p in [rules_prefix, role, instructions, rules_text, examples_text, fmt] if p]
        system_prompt = "\n\n".join(system_parts)

        # 6. User message (task description)
        remaining = budget - len(system_prompt)
        user_prompt = self._truncate(task_description, max(remaining, 500))

        total_chars = len(system_prompt) + len(user_prompt)

        return {
            "system": system_prompt,
            "user": user_prompt,
            "total_chars": total_chars,
            "agent_type": agent_type,
            "task_type": task_type,
        }

    # ------------------------------------------------------------------
    # Component getters
    # ------------------------------------------------------------------

    def _get_role(self, agent_type: str) -> str:
        return _ROLE_DEFS.get(agent_type.upper(), f"You are a Vetinari {agent_type} specialist.")

    def _get_instructions(self, task_type: str) -> str:
        return _TASK_INSTRUCTIONS.get(task_type.lower(), _TASK_INSTRUCTIONS["general"])

    def _get_format(self, task_type: str) -> str:
        return _OUTPUT_FORMATS.get(task_type.lower(), _OUTPUT_FORMATS["general"])

    def _get_rules(self, agent_type: str, task_type: str) -> List[str]:
        """Load learned failure-pattern rules from WorkflowLearner."""
        cache_key = f"{agent_type}:{task_type}"
        if cache_key in self._rules_cache:
            return self._rules_cache[cache_key]
        rules: List[str] = []
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner
            learner = get_workflow_learner()
            domain = self._task_to_domain(task_type)
            rec = learner.get_recommendations(domain)
            if rec and rec.get("warnings"):
                rules = rec["warnings"][:5]
        except Exception:
            pass
        self._rules_cache[cache_key] = rules
        return rules

    def _get_examples(
        self,
        agent_type: str,
        task_type: str,
        task_description: str,
    ) -> List[Dict]:
        """Return 1-3 few-shot examples relevant to this task.

        Falls back to static template examples when the training corpus
        is not yet populated.
        """
        try:
            from vetinari.learning.episode_memory import get_episode_memory
            memory = get_episode_memory()
            episodes = memory.recall(task_description, k=3, min_score=0.7)
            if episodes:
                return [{"task": e.task_summary, "output": e.output_summary} for e in episodes]
        except Exception:
            pass

        # Static examples from template files
        tmpl_path = (
            Path(__file__).parent.parent.parent
            / "templates"
            / "v1"
            / f"{agent_type.lower()}.json"
        )
        if tmpl_path.exists():
            try:
                with open(tmpl_path, encoding="utf-8") as f:
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
                pass

        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max(max_chars - 3, 0)] + "..."

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

_assembler: Optional[PromptAssembler] = None
_assembler_lock = __import__("threading").Lock()


def get_prompt_assembler() -> PromptAssembler:
    """Return the global PromptAssembler singleton."""
    global _assembler
    if _assembler is None:
        with _assembler_lock:
            if _assembler is None:
                _assembler = PromptAssembler()
    return _assembler
