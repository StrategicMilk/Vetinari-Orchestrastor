"""
Prompt Evolver - Vetinari Self-Improvement Subsystem

A/B tests prompt variations for each agent and promotes variants that
achieve statistically better quality scores.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A candidate prompt variant with performance tracking."""
    variant_id: str
    agent_type: str
    prompt_text: str
    is_baseline: bool = False
    trials: int = 0
    total_quality: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    promoted_at: Optional[str] = None
    status: str = "testing"   # "testing" | "promoted" | "deprecated"

    @property
    def avg_quality(self) -> float:
        return self.total_quality / max(self.trials, 1)

    def record(self, quality: float) -> None:
        self.trials += 1
        self.total_quality += quality


class PromptEvolver:
    """
    Automatically evolves agent system prompts through A/B testing.

    Strategy:
    1. Baseline prompt is the current system prompt for each agent.
    2. When quality degrades (below rolling average), generate new variants.
    3. Route a fraction of tasks to variant prompts.
    4. After MIN_TRIALS trials, promote variant if significantly better.
    5. Store all variants and their performance in memory.
    """

    MIN_TRIALS = 20          # Minimum trials before promotion decision
    VARIANT_FRACTION = 0.3   # Fraction of tasks routed to variant
    MIN_IMPROVEMENT = 0.05   # Minimum quality improvement to promote

    def __init__(self, adapter_manager=None):
        self._adapter_manager = adapter_manager
        # agent_type -> list of variants
        self._variants: Dict[str, List[PromptVariant]] = {}
        self._load_variants()

    def register_baseline(self, agent_type: str, prompt_text: str) -> None:
        """Register the current system prompt as the baseline."""
        if agent_type not in self._variants:
            self._variants[agent_type] = []

        # Check if baseline already exists
        for v in self._variants[agent_type]:
            if v.is_baseline:
                if v.prompt_text != prompt_text:
                    # Prompt changed externally -- update
                    v.prompt_text = prompt_text
                return

        variant = PromptVariant(
            variant_id=f"{agent_type}_baseline",
            agent_type=agent_type,
            prompt_text=prompt_text,
            is_baseline=True,
            status="promoted",
        )
        self._variants[agent_type].append(variant)

    def select_prompt(self, agent_type: str) -> Tuple[str, str]:
        """
        Select which prompt to use for this invocation.

        Returns:
            Tuple of (prompt_text, variant_id).
        """
        variants = self._variants.get(agent_type, [])
        if not variants:
            return "", "none"

        promoted = [v for v in variants if v.status == "promoted"]
        testing = [v for v in variants if v.status == "testing"]

        if testing and random.random() < self.VARIANT_FRACTION:
            # Route to a testing variant
            v = random.choice(testing)
            return v.prompt_text, v.variant_id

        if promoted:
            # Use the best promoted variant
            best = max(promoted, key=lambda v: v.avg_quality if v.trials > 0 else 0.5)
            return best.prompt_text, best.variant_id

        return "", "default"

    def record_result(self, agent_type: str, variant_id: str, quality: float) -> None:
        """Record a quality result for a variant and trigger promotion check."""
        variants = self._variants.get(agent_type, [])
        for v in variants:
            if v.variant_id == variant_id:
                v.record(quality)
                break

        self._check_promotion(agent_type)

    def _check_promotion(self, agent_type: str) -> None:
        """Decide whether to promote, deprecate, or keep testing variants."""
        variants = self._variants.get(agent_type, [])
        baseline = next((v for v in variants if v.is_baseline and v.status == "promoted"), None)
        baseline_quality = baseline.avg_quality if baseline and baseline.trials > 0 else 0.65

        for v in variants:
            if v.status != "testing" or v.trials < self.MIN_TRIALS:
                continue

            if v.avg_quality >= baseline_quality + self.MIN_IMPROVEMENT:
                logger.info(
                    f"[PromptEvolver] Promoting variant {v.variant_id} for {agent_type}: "
                    f"quality={v.avg_quality:.3f} vs baseline={baseline_quality:.3f}"
                )
                v.status = "promoted"
                v.promoted_at = datetime.now().isoformat()
                # Demote old baseline
                if baseline and v.variant_id != baseline.variant_id:
                    baseline.status = "deprecated"
            elif v.avg_quality < baseline_quality - 0.1:
                # Variant is worse -- deprecate it
                v.status = "deprecated"
                logger.info(f"[PromptEvolver] Deprecated variant {v.variant_id} for {agent_type}")

        self._save_variants()

    def generate_variant(self, agent_type: str, baseline_prompt: str) -> Optional[str]:
        """Use LLM to generate an improved prompt variant."""
        if not self._adapter_manager:
            return None

        try:
            from vetinari.adapters.base import InferenceRequest

            req = InferenceRequest(
                model_id="default",
                prompt=f"""You are a prompt engineer. Improve this agent system prompt to be more effective.

AGENT TYPE: {agent_type}

CURRENT PROMPT:
{baseline_prompt}

Generate an improved version that:
1. Is more specific and actionable
2. Gives clearer output format instructions
3. Reduces ambiguity
4. Maintains the same core role

Respond with ONLY the improved prompt text, no explanations.""",
                system_prompt="You are an expert AI prompt engineer.",
                max_tokens=1024,
                temperature=0.7,
            )
            resp = self._adapter_manager.infer(req)
            if resp.status == "ok" and resp.output.strip():
                variant_id = f"{agent_type}_v{len(self._variants.get(agent_type, []))+1}"
                variant = PromptVariant(
                    variant_id=variant_id,
                    agent_type=agent_type,
                    prompt_text=resp.output.strip(),
                )
                if agent_type not in self._variants:
                    self._variants[agent_type] = []
                self._variants[agent_type].append(variant)
                self._save_variants()
                logger.info(f"[PromptEvolver] Generated variant {variant_id} for {agent_type}")
                return resp.output.strip()
        except Exception as e:
            logger.debug(f"Variant generation failed: {e}")
        return None

    def get_stats(self, agent_type: str) -> Dict[str, Any]:
        """Get evolution statistics for an agent type."""
        variants = self._variants.get(agent_type, [])
        return {
            "agent_type": agent_type,
            "total_variants": len(variants),
            "promoted": [{"id": v.variant_id, "quality": round(v.avg_quality, 3), "trials": v.trials}
                         for v in variants if v.status == "promoted"],
            "testing": len([v for v in variants if v.status == "testing"]),
            "deprecated": len([v for v in variants if v.status == "deprecated"]),
        }

    def _load_variants(self) -> None:
        try:
            import os
            path = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari",
                ".vetinari", "prompt_variants.json"
            )
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for agent_type, variants in data.items():
                    self._variants[agent_type] = [PromptVariant(**v) for v in variants]
        except Exception as e:
            logger.debug(f"Could not load prompt variants: {e}")

    def _save_variants(self) -> None:
        try:
            import os
            from dataclasses import asdict
            state_dir = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari", ".vetinari"
            )
            os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, "prompt_variants.json")
            data = {k: [asdict(v) for v in vs] for k, vs in self._variants.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save prompt variants: {e}")


_prompt_evolver: Optional[PromptEvolver] = None


def get_prompt_evolver() -> PromptEvolver:
    global _prompt_evolver
    if _prompt_evolver is None:
        _prompt_evolver = PromptEvolver()
    return _prompt_evolver
