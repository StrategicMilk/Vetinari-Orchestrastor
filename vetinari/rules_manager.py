"""Vetinari Rules Manager.

=======================
Hierarchical rules system that injects user-defined rules into agent
system prompts at different scopes:

  GLOBAL → PROJECT → MODEL → PROJECT+MODEL

Rules are stored in vetinari/config/rules.yaml and loaded at startup.
They are injected into every agent system prompt by the prompt assembler.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Rules file location
_DEFAULT_RULES_FILE = _PROJECT_ROOT / "vetinari" / "config" / "rules.yaml"


def _load_yaml_safe(path: Path) -> dict[str, Any]:
    """Load a YAML file safely, returning empty dict on error."""
    try:
        with Path(path).open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save data to a YAML file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error("Failed to save rules to %s: %s", path, e)


class RulesManager:
    """Manages the hierarchical rules configuration for Vetinari.

    Scope hierarchy (later scopes override earlier):
        global → project → model → project+model

    Usage:
        rules = get_rules_manager()

        # Get all rules for a specific context
        combined = rules.get_rules(project_id="my_app", model_id="qwen2.5-7b")

        # Returns a formatted string ready for injection into system prompts
        rules_text = rules.format_rules(project_id="my_app", model_id="qwen2.5-7b")
    """

    def __init__(self, rules_file: Path | None = None):
        self._path = rules_file or _DEFAULT_RULES_FILE
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load rules from disk."""
        with self._lock:
            self._data = _load_yaml_safe(self._path)
            if not self._data:
                self._data = {
                    "global": [],
                    "projects": {},
                    "models": {},
                    "global_system_prompt": "",
                }

    def _save(self) -> None:
        """Persist rules to disk."""
        with self._lock:
            _save_yaml(self._path, self._data)

    # ─── Global rules ────────────────────────────────────────────────────────

    def get_global_rules(self) -> list[str]:
        """Return the global rules list.

        Returns:
            List of global rule strings.
        """
        with self._lock:
            return list(self._data.get("global", []))

    def set_global_rules(self, rules: list[str]) -> None:
        """Replace the global rules list."""
        with self._lock:
            self._data["global"] = [r.strip() for r in rules if r.strip()]
            self._save()

    def get_global_system_prompt(self) -> str:
        """Return the global system prompt override.

        Returns:
            The global system prompt string, or empty string if not set.
        """
        with self._lock:
            return self._data.get("global_system_prompt", "")

    def set_global_system_prompt(self, prompt: str) -> None:
        """Replace the global system prompt override and persist it to disk immediately.

        Args:
            prompt: The new system prompt; leading/trailing whitespace is stripped before saving.
        """
        with self._lock:
            self._data["global_system_prompt"] = prompt.strip()
            self._save()

    # ─── Project rules ───────────────────────────────────────────────────────

    def get_project_rules(self, project_id: str) -> list[str]:
        """Return rules for a specific project.

        Returns:
            List of rule strings for the given project.
        """
        with self._lock:
            return list(self._data.get("projects", {}).get(project_id, []))

    def set_project_rules(self, project_id: str, rules: list[str]) -> None:
        """Set rules for a specific project.

        Args:
            project_id: The project id.
            rules: The rules.
        """
        with self._lock:
            if "projects" not in self._data:
                self._data["projects"] = {}
            self._data["projects"][project_id] = [r.strip() for r in rules if r.strip()]
            self._save()

    # ─── Model rules ─────────────────────────────────────────────────────────

    def get_model_rules(self, model_id: str) -> list[str]:
        """Return rules for a specific model.

        Returns:
            List of rule strings for the given model.
        """
        with self._lock:
            return list(self._data.get("models", {}).get(model_id, []))

    def set_model_rules(self, model_id: str, rules: list[str]) -> None:
        """Set rules for a specific model.

        Args:
            model_id: The model id.
            rules: The rules.
        """
        with self._lock:
            if "models" not in self._data:
                self._data["models"] = {}
            self._data["models"][model_id] = [r.strip() for r in rules if r.strip()]
            self._save()

    # ─── Combined rules ──────────────────────────────────────────────────────

    def get_rules(
        self,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> list[str]:
        """Get all applicable rules for the given context, in injection order:.

        global → project → model → project+model (combined).

        Returns deduplicated list preserving order.

        Args:
            project_id: The project id.
            model_id: The model id.

        Returns:
            Deduplicated list of rule strings in injection order.
        """
        rules: list[str] = []
        seen: set = set()

        def add(r_list: list[str]) -> None:
            """Add for the current context."""
            for r in r_list:
                if r and r not in seen:
                    rules.append(r)
                    seen.add(r)

        add(self.get_global_rules())
        if project_id:
            add(self.get_project_rules(project_id))
        if model_id:
            add(self.get_model_rules(model_id))
        if project_id and model_id:
            # Project+model specific rules
            combo_key = f"{project_id}::{model_id}"
            add(self._data.get("combo", {}).get(combo_key, []))

        return rules

    def format_rules(
        self,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> str:
        """Format applicable rules as a string for injection into system prompts.

        Returns empty string if no rules are defined.

        Args:
            project_id: The project id.
            model_id: The model id.

        Returns:
            Formatted rules string with bullet points, or empty string if none.
        """
        rules = self.get_rules(project_id=project_id, model_id=model_id)
        if not rules:
            return ""
        lines = "\n".join(f"- {r}" for r in rules)
        return f"\n## Project Rules\nAlways follow these rules:\n{lines}\n"

    def build_system_prompt_prefix(
        self,
        project_id: str | None = None,
        model_id: str | None = None,
    ) -> str:
        """Build the full system prompt prefix to prepend to every agent's system prompt.

        Includes: global system prompt + formatted rules.

        Args:
            project_id: The project id.
            model_id: The model id.

        Returns:
            Combined system prompt prefix string, or empty string if none.
        """
        parts = []
        gsp = self.get_global_system_prompt()
        if gsp:
            parts.append(gsp)
        rules_text = self.format_rules(project_id=project_id, model_id=model_id)
        if rules_text:
            parts.append(rules_text)
        return "\n\n".join(parts)

    def get_agent_rules(self, agent_type: str) -> list[str]:
        """Return rules for a specific agent type.

        Args:
            agent_type: The agent type value (e.g., "WORKER", "FOREMAN").

        Returns:
            List of rule strings for the agent, or empty list.
        """
        with self._lock:
            agents = self._data.get("agents", {})
            return list(agents.get(agent_type, []))

    def get_rules_for_context(
        self,
        agent_type: str,
        model_name: str | None = None,
        project_name: str | None = None,
    ) -> list[str]:
        """Get merged rules for a full agent execution context.

        Merges rules from all applicable scopes:
        global → agent-specific → project → model → project+model

        Most specific rules take precedence (appended last).

        Args:
            agent_type: Agent type value (e.g., "WORKER").
            model_name: Optional model identifier for model-specific rules.
            project_name: Optional project name for project-specific rules.

        Returns:
            Deduplicated list of rule strings in injection order.
        """
        rules: list[str] = []
        seen: set[str] = set()

        def _add(r_list: list[str]) -> None:
            for r in r_list:
                if r and r not in seen:
                    rules.append(r)
                    seen.add(r)

        # 1. Global rules
        _add(self.get_global_rules())

        # 2. Agent-specific rules
        _add(self.get_agent_rules(agent_type))

        # 3. Project rules
        if project_name:
            _add(self.get_project_rules(project_name))

        # 4. Model rules
        if model_name:
            _add(self.get_model_rules(model_name))

        # 5. Project+model combo
        if project_name and model_name:
            combo_key = f"{project_name}::{model_name}"
            with self._lock:
                _add(self._data.get("combo", {}).get(combo_key, []))

        return rules

    def format_rules_for_context(
        self,
        agent_type: str,
        model_name: str | None = None,
        project_name: str | None = None,
    ) -> str:
        """Format rules for a full agent context as numbered list with scope.

        Args:
            agent_type: Agent type value.
            model_name: Optional model identifier.
            project_name: Optional project name.

        Returns:
            Formatted rules string, or empty string if no rules.
        """
        rules = self.get_rules_for_context(agent_type, model_name, project_name)
        if not rules:
            return ""
        lines = [f"{i}. {r}" for i, r in enumerate(rules, 1)]
        return "## Active Rules\n\n" + "\n".join(lines)

    # ─── Rule learning from Quality feedback ────────────────────────────────

    def propose_rule_from_feedback(
        self,
        agent_type: str,
        mode: str,
        violation_description: str,
        model_name: str | None = None,
    ) -> bool:
        """Propose a new rule based on Quality rejection feedback.

        When Quality rejects output for a pattern not covered by existing
        rules, this method creates a proposed rule. After 3 consistent
        observations (tracked via ``_proposed_observations``), the rule is
        auto-accepted into the appropriate scope.

        Args:
            agent_type: Agent type that produced the violation.
            mode: Agent mode during the violation.
            violation_description: Description of the violation pattern.
            model_name: Optional model name for model-specific rules.

        Returns:
            True if a new rule was proposed or an existing one promoted.
        """
        with self._lock:
            proposed = self._data.setdefault("proposed", {})
            rule_key = f"{agent_type}:{mode}:{violation_description}"

            if rule_key in proposed:
                proposed[rule_key]["observations"] += 1
                obs_count = proposed[rule_key]["observations"]
                if obs_count >= 3:
                    self._accept_proposed_rule(
                        proposed[rule_key],
                        agent_type,
                        model_name,
                    )
                    del proposed[rule_key]
                    logger.info(
                        "Rule auto-accepted after %d observations: %s",
                        obs_count,
                        violation_description,
                    )
                    self._save()
                    return True
                self._save()
                return False
            proposed[rule_key] = {
                "description": violation_description,
                "agent_type": agent_type,
                "mode": mode,
                "model_name": model_name,
                "observations": 1,
                "status": "proposed",
            }
            logger.info(
                "New rule proposed (1/3): %s for %s:%s",
                violation_description,
                agent_type,
                mode,
            )
            self._save()
            return False

    def _accept_proposed_rule(
        self,
        proposed: dict[str, Any],
        agent_type: str,
        model_name: str | None,
    ) -> None:
        """Promote a proposed rule to the appropriate scope.

        Args:
            proposed: The proposed rule data.
            agent_type: Agent type for agent-scoped rules.
            model_name: If set, add as model-specific rule instead.
        """
        desc = proposed["description"]

        if model_name:
            models = self._data.setdefault("models", {})
            model_rules = models.setdefault(model_name, [])
            if desc not in model_rules:
                model_rules.append(desc)
        else:
            agents = self._data.setdefault("agents", {})
            agent_rules = agents.setdefault(agent_type, [])
            if desc not in agent_rules:
                agent_rules.append(desc)

    # ─── Rule learning from human corrections ────────────────────────────────

    def extract_correction(
        self,
        original_output: str,
        corrected_output: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Analyse a human correction and characterise the type of change made.

        Compares word counts to detect verbosity or incompleteness, and falls
        back to a generic content-correction classification when lengths are
        similar but the text differs. Builds an evidence string combining the
        specific change description with up to 200 characters of context so
        that downstream consumers can log exactly what was corrected and why.

        Args:
            original_output: The agent's original output before the correction.
            corrected_output: The human-corrected version of the output.
            context: Optional free-text description of why the correction was
                made, injected into the evidence and used as the generalised
                rule for content corrections.

        Returns:
            A dict with the following keys:

            - ``correction_type``: One of ``"verbose"``, ``"incomplete"``,
              ``"content"``, or ``"unknown"``.
            - ``specific_change``: A short sentence describing what changed.
            - ``generalized_rule``: A rule that can be stored and injected into
              future system prompts.
            - ``evidence``: A combined string with the specific change and the
              first 200 characters of context.
        """
        original_words = len(original_output.split())
        corrected_words = len(corrected_output.split())

        if original_output == corrected_output:
            return {
                "correction_type": "unknown",
                "specific_change": "No change detected between original and corrected output.",
                "generalized_rule": "",
                "evidence": "Correction applied: no change detected.",
            }

        if corrected_words > original_words * 2:
            correction_type = "incomplete"
            specific_change = "Output was expanded — the original response was too brief."
            generalized_rule = "Provide complete responses that fully address the task requirements."
        elif original_words > corrected_words * 2:
            correction_type = "verbose"
            specific_change = "Output was shortened — the original response was too verbose."
            generalized_rule = "Keep responses concise and avoid unnecessary repetition."
        else:
            correction_type = "content"
            specific_change = "Content was modified — factual or stylistic correction applied."
            # Use the caller-supplied context as the rule when available so
            # the stored rule is as specific as possible.
            generalized_rule = context or "Verify factual accuracy before responding."

        truncated_context = context[:200]
        evidence_parts = [f"Correction applied: {specific_change}"]
        if truncated_context:
            evidence_parts.append(truncated_context)
        evidence = " | ".join(evidence_parts)

        return {
            "correction_type": correction_type,
            "specific_change": specific_change,
            "generalized_rule": generalized_rule,
            "evidence": evidence,
        }

    def propose_rule_from_correction(
        self,
        agent_type: str,
        mode: str,
        original_output: str,
        corrected_output: str,
        context: str = "",
        model_name: str | None = None,
    ) -> bool:
        """Propose a rule derived from a human correction of agent output.

        Extracts the correction type and a generalised rule from the diff
        between ``original_output`` and ``corrected_output``, then feeds that
        rule through the same three-observation promotion pipeline used by
        :meth:`propose_rule_from_feedback`. The rule is accepted and persisted
        after three consistent observations of the same pattern.

        When the outputs are identical no rule can be derived, so the method
        returns ``False`` immediately without creating a proposed entry.

        Args:
            agent_type: The type of the agent that produced the original output
                (e.g. ``AgentType.WORKER.value``).
            mode: The agent mode in effect when the output was produced.
            original_output: The agent's original output before correction.
            corrected_output: The human-corrected version.
            context: Optional explanation of why the correction was made.
                Passed to :meth:`extract_correction` and used as the rule text
                for content-type corrections when provided.
            model_name: If supplied, the accepted rule is stored under
                model-specific rules instead of agent-scoped rules.

        Returns:
            ``True`` when the third observation of the same pattern causes the
            rule to be promoted and accepted; ``False`` for the first two
            observations and for identical outputs.
        """
        correction = self.extract_correction(original_output, corrected_output, context)

        if correction["correction_type"] == "unknown":
            # Identical outputs — no correctable pattern to learn from.
            return False

        generalized_rule = correction["generalized_rule"]

        return self.propose_rule_from_feedback(
            agent_type=agent_type,
            mode=mode,
            violation_description=generalized_rule,
            model_name=model_name,
        )

    def get_proposed_rules(self) -> dict[str, Any]:
        """Return all currently proposed (not yet accepted) rules.

        Returns:
            Dict of proposed rules keyed by rule key.
        """
        with self._lock:
            return dict(self._data.get("proposed", {}))

    def to_dict(self) -> dict[str, Any]:
        """Serialize rules to dict for API responses.

        Returns:
            Complete rules data as a dictionary.
        """
        with self._lock:
            return dict(self._data)


# ─── Module-level singleton ───────────────────────────────────────────────────

_rules_manager: RulesManager | None = None
_rules_lock = threading.Lock()


def get_rules_manager() -> RulesManager:
    """Get the singleton RulesManager instance.

    Returns:
        The shared RulesManager singleton instance.
    """
    global _rules_manager
    if _rules_manager is None:
        with _rules_lock:
            if _rules_manager is None:
                _rules_manager = RulesManager()
    return _rules_manager
