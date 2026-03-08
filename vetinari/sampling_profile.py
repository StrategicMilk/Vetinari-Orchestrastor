"""Central sampling parameter manager with task-type and model-family defaults."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
import re

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    CODING = "coding"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    CREATIVE = "creative"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    SECURITY = "security"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


@dataclass
class SamplingProfile:
    temperature: float = 0.5
    min_p: Optional[float] = 0.08
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.05
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    source: str = "default"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# Task-type profiles (researched defaults)
TASK_PROFILES: Dict[TaskType, SamplingProfile] = {
    TaskType.CODING: SamplingProfile(temperature=0.2, min_p=0.10, top_p=0.95, top_k=40, repeat_penalty=1.05, source="task_profile"),
    TaskType.ANALYSIS: SamplingProfile(temperature=0.3, min_p=0.10, top_p=0.95, top_k=40, repeat_penalty=1.05, source="task_profile"),
    TaskType.PLANNING: SamplingProfile(temperature=0.4, min_p=0.08, top_p=0.95, top_k=40, repeat_penalty=1.10, source="task_profile"),
    TaskType.TESTING: SamplingProfile(temperature=0.2, min_p=0.10, top_p=0.95, top_k=40, repeat_penalty=1.05, source="task_profile"),
    TaskType.DOCUMENTATION: SamplingProfile(temperature=0.3, min_p=0.08, top_p=0.95, top_k=50, repeat_penalty=1.15, source="task_profile"),
    TaskType.CREATIVE: SamplingProfile(temperature=0.8, min_p=0.05, top_p=0.95, top_k=60, repeat_penalty=1.00, source="task_profile"),
    TaskType.CODE_REVIEW: SamplingProfile(temperature=0.2, min_p=0.10, top_p=0.95, top_k=40, repeat_penalty=1.05, source="task_profile"),
    TaskType.RESEARCH: SamplingProfile(temperature=0.4, min_p=0.08, top_p=0.95, top_k=50, repeat_penalty=1.10, source="task_profile"),
    TaskType.SECURITY: SamplingProfile(temperature=0.1, min_p=0.10, top_p=0.90, top_k=30, repeat_penalty=1.05, source="task_profile"),
    TaskType.SUMMARIZATION: SamplingProfile(temperature=0.3, min_p=0.08, top_p=0.95, top_k=40, repeat_penalty=1.20, source="task_profile"),
    TaskType.GENERAL: SamplingProfile(temperature=0.5, min_p=0.08, top_p=0.95, top_k=40, repeat_penalty=1.05, source="task_profile"),
}

# Model-family overrides (vendor-recommended)
MODEL_FAMILY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "qwen3_thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},
    "qwen3_nothinking": {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0},
    "deepseek_r1": {"temperature": 0.6, "top_p": 0.95},
    "llama": {"temperature": 0.6, "top_p": 0.90},
    "mistral_instruct": {"temperature": 0.15},
    "magistral": {"temperature": 0.7, "top_p": 0.95},
    "gemma": {"temperature": 1.0, "top_p": 0.96, "top_k": 64},
    "phi_reasoning": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
}

# Patterns to detect model family from model ID
MODEL_FAMILY_PATTERNS = [
    (r"qwen.*3.*thinking", "qwen3_thinking"),
    (r"qwen.*3", "qwen3_nothinking"),
    (r"deepseek.*r1", "deepseek_r1"),
    (r"llama", "llama"),
    (r"mistral.*instruct", "mistral_instruct"),
    (r"magistral", "magistral"),
    (r"gemma", "gemma"),
    (r"phi.*reasoning|phi-4", "phi_reasoning"),
]


def _detect_model_family(model_id: str) -> Optional[str]:
    model_lower = model_id.lower()
    for pattern, family in MODEL_FAMILY_PATTERNS:
        if re.search(pattern, model_lower):
            return family
    return None


class SamplingProfileManager:
    """Resolves sampling parameters with priority: user > learned > model-family > task-type > global."""

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = Path(persist_path or ".vetinari/sampling_profiles.json")
        self._learned: Dict[str, Dict[str, float]] = {}
        self._load()

    def resolve(self, task_type: str, model_id: str, user_overrides: Optional[Dict] = None) -> SamplingProfile:
        # 1. Start with task-type profile
        try:
            tt = TaskType(task_type.lower())
        except ValueError:
            tt = TaskType.GENERAL
        profile = SamplingProfile(**{k: v for k, v in asdict(TASK_PROFILES[tt]).items() if k != "source"})
        profile.source = f"task:{tt.value}"

        # 2. Apply model-family vendor override
        family = _detect_model_family(model_id)
        if family and family in MODEL_FAMILY_OVERRIDES:
            for k, v in MODEL_FAMILY_OVERRIDES[family].items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            profile.source = f"model_family:{family}"

        # 3. Apply learned adjustments
        learn_key = f"{tt.value}:{model_id}"
        if learn_key in self._learned:
            for k, v in self._learned[learn_key].items():
                if hasattr(profile, k):
                    setattr(profile, k, v)
            profile.source = f"learned:{learn_key}"

        # 4. Apply user overrides (highest priority)
        if user_overrides:
            for k, v in user_overrides.items():
                if hasattr(profile, k) and k != "source":
                    setattr(profile, k, v)
            profile.source = "user_override"

        return profile

    def record_outcome(self, task_type: str, model_id: str, profile: SamplingProfile, quality_score: float):
        """Learn from execution outcomes -- nudge parameters toward better quality."""
        key = f"{task_type}:{model_id}"
        if key not in self._learned:
            self._learned[key] = {}

        # Simple learning: if quality is high, reinforce current temperature
        if quality_score >= 0.8:
            self._learned[key]["temperature"] = profile.temperature
        elif quality_score < 0.4:
            # Poor quality -- nudge temperature toward task default
            try:
                tt = TaskType(task_type.lower())
                default_temp = TASK_PROFILES[tt].temperature
                current = self._learned[key].get("temperature", profile.temperature)
                self._learned[key]["temperature"] = round((current + default_temp) / 2, 2)
            except ValueError:
                pass

        self._persist()

    def _persist(self):
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "w") as f:
                json.dump(self._learned, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist sampling profiles: {e}")

    def _load(self):
        try:
            if self._persist_path.exists():
                with open(self._persist_path) as f:
                    self._learned = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load sampling profiles: {e}")
            self._learned = {}


# Module-level singleton
_manager: Optional[SamplingProfileManager] = None

def get_sampling_profile_manager() -> SamplingProfileManager:
    global _manager
    if _manager is None:
        _manager = SamplingProfileManager()
    return _manager
