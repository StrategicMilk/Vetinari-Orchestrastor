"""
Vetinari Rules Manager
=======================
Hierarchical rules system that injects user-defined rules into agent
system prompts at different scopes:

  GLOBAL → PROJECT → MODEL → PROJECT+MODEL

Rules are stored in rules.yaml in the project root and loaded at startup.
They are injected into every agent system prompt by the prompt assembler.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_RULES_FILE = Path(__file__).resolve().parents[1] / "rules.yaml"


def _load_yaml_safe(path: Path) -> Dict[str, Any]:
    """Load a YAML file safely, returning empty dict on error."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.debug(f"Could not load {path}: {e}")
        return {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Save data to a YAML file."""
    try:
        import yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error(f"Failed to save rules to {path}: {e}")


class RulesManager:
    """
    Manages the hierarchical rules configuration for Vetinari.

    Scope hierarchy (later scopes override earlier):
        global → project → model → project+model

    Usage:
        rules = get_rules_manager()

        # Get all rules for a specific context
        combined = rules.get_rules(project_id="my_app", model_id="qwen2.5-7b")

        # Returns a formatted string ready for injection into system prompts
        rules_text = rules.format_rules(project_id="my_app", model_id="qwen2.5-7b")
    """

    def __init__(self, rules_file: Optional[Path] = None):
        self._path = rules_file or _DEFAULT_RULES_FILE
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
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

    def get_global_rules(self) -> List[str]:
        """Return the global rules list."""
        with self._lock:
            return list(self._data.get("global", []))

    def set_global_rules(self, rules: List[str]) -> None:
        """Replace the global rules list."""
        with self._lock:
            self._data["global"] = [r.strip() for r in rules if r.strip()]
            self._save()

    def get_global_system_prompt(self) -> str:
        """Return the global system prompt override."""
        with self._lock:
            return self._data.get("global_system_prompt", "")

    def set_global_system_prompt(self, prompt: str) -> None:
        """Set the global system prompt override."""
        with self._lock:
            self._data["global_system_prompt"] = prompt.strip()
            self._save()

    # ─── Project rules ───────────────────────────────────────────────────────

    def get_project_rules(self, project_id: str) -> List[str]:
        """Return rules for a specific project."""
        with self._lock:
            return list(self._data.get("projects", {}).get(project_id, []))

    def set_project_rules(self, project_id: str, rules: List[str]) -> None:
        """Set rules for a specific project."""
        with self._lock:
            if "projects" not in self._data:
                self._data["projects"] = {}
            self._data["projects"][project_id] = [r.strip() for r in rules if r.strip()]
            self._save()

    # ─── Model rules ─────────────────────────────────────────────────────────

    def get_model_rules(self, model_id: str) -> List[str]:
        """Return rules for a specific model."""
        with self._lock:
            return list(self._data.get("models", {}).get(model_id, []))

    def set_model_rules(self, model_id: str, rules: List[str]) -> None:
        """Set rules for a specific model."""
        with self._lock:
            if "models" not in self._data:
                self._data["models"] = {}
            self._data["models"][model_id] = [r.strip() for r in rules if r.strip()]
            self._save()

    # ─── Combined rules ──────────────────────────────────────────────────────

    def get_rules(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[str]:
        """
        Get all applicable rules for the given context, in injection order:
        global → project → model → project+model (combined).

        Returns deduplicated list preserving order.
        """
        rules: List[str] = []
        seen: set = set()

        def add(r_list: List[str]) -> None:
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
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Format applicable rules as a string for injection into system prompts.
        Returns empty string if no rules are defined.
        """
        rules = self.get_rules(project_id=project_id, model_id=model_id)
        if not rules:
            return ""
        lines = "\n".join(f"- {r}" for r in rules)
        return f"\n## Project Rules\nAlways follow these rules:\n{lines}\n"

    def build_system_prompt_prefix(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Build the full system prompt prefix to prepend to every agent's system prompt.
        Includes: global system prompt + formatted rules.
        """
        parts = []
        gsp = self.get_global_system_prompt()
        if gsp:
            parts.append(gsp)
        rules_text = self.format_rules(project_id=project_id, model_id=model_id)
        if rules_text:
            parts.append(rules_text)
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rules to dict for API responses."""
        with self._lock:
            return dict(self._data)


# ─── Module-level singleton ───────────────────────────────────────────────────

_rules_manager: Optional[RulesManager] = None
_rules_lock = threading.Lock()


def get_rules_manager() -> RulesManager:
    """Get the singleton RulesManager instance."""
    global _rules_manager
    if _rules_manager is None:
        with _rules_lock:
            if _rules_manager is None:
                _rules_manager = RulesManager()
    return _rules_manager
