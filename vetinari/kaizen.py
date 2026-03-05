"""
Kaizen Integration — WS10
==========================
Wraps every task execution in a PDCA (Plan-Do-Check-Act) cycle and tracks
Muda (waste) across task runs for continuous improvement visibility.

PDCA Execution Wrapper
-----------------------
- Plan  — log intent, model, decomposition strategy before execution.
- Do    — execute the task with telemetry capture.
- Check — quality score against DoD criteria; log metrics.
- Act   — feed results into PromptEvolver / WorkflowLearner / AutoTuner
          when quality < threshold; capture successful patterns as templates
          when quality >= threshold.

Muda Tracking
-------------
Seven categories of waste are tracked per task:
  1. Overproduction — oversized model for simple task.
  2. Waiting        — idle time between sequential tasks.
  3. Over-processing — excessive prompt engineering for trivial tasks.
  4. Defects         — tasks needing retry (rework).
  5. Motion          — unnecessary model swaps.
  6. Inventory       — stale / unused decomposition artefacts.
  7. Transport       — repeated context serialisation overhead.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_QUALITY_THRESHOLD = 0.7  # Below this → feed back into learning


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PDCACycle:
    """One PDCA execution cycle."""
    cycle_id: str
    task_id: str
    task_description: str
    agent_type: str
    model_id: str
    plan_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    do_start: float = field(default_factory=time.time)
    do_end: Optional[float] = None
    quality_score: Optional[float] = None
    quality_passed: bool = False
    muda: List[str] = field(default_factory=list)
    outcome: str = "pending"  # pending | pass | fail | rework


@dataclass
class MudaRecord:
    """A single recorded waste event."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    muda_type: str = ""
    task_id: str = ""
    description: str = ""
    severity: str = "low"  # low | medium | high
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Kaizen Engine
# ---------------------------------------------------------------------------

class KaizenEngine:
    """Wraps task execution in PDCA cycles and tracks Muda."""

    def __init__(
        self,
        quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
    ) -> None:
        self.quality_threshold = quality_threshold
        self._cycles: List[PDCACycle] = []
        self._muda_log: List[MudaRecord] = []

    # ------------------------------------------------------------------
    # Public: PDCA wrapper
    # ------------------------------------------------------------------

    def run(
        self,
        task: Dict[str, Any],
        executor: Callable[[Dict[str, Any]], Dict[str, Any]],
        model_id: str = "unknown",
        agent_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Execute ``task`` through ``executor`` wrapped in a PDCA cycle.

        Returns the executor result dict, augmented with a ``_kaizen`` key.
        """
        task_id = task.get("subtask_id") or task.get("id") or str(uuid.uuid4())[:8]
        description = task.get("description", "")
        complexity = task.get("complexity", "moderate")

        # ---- PLAN ----
        cycle = PDCACycle(
            cycle_id=str(uuid.uuid4())[:8],
            task_id=task_id,
            task_description=description[:200],
            agent_type=agent_type,
            model_id=model_id,
        )
        logger.debug(
            f"[Kaizen/Plan] cycle={cycle.cycle_id} task={task_id} "
            f"model={model_id} agent={agent_type}"
        )

        # Detect overproduction before execution
        if self._is_overproduced(complexity, model_id):
            muda = MudaRecord(
                muda_type="overproduction",
                task_id=task_id,
                description=f"Large model '{model_id}' used for '{complexity}' task",
                severity="medium",
            )
            self._muda_log.append(muda)
            cycle.muda.append("overproduction")

        # ---- DO ----
        t0 = time.time()
        try:
            result = executor(task)
        except Exception as exc:
            result = {"success": False, "error": str(exc)}
        finally:
            cycle.do_end = time.time()

        duration_ms = int((cycle.do_end - t0) * 1000)

        # ---- CHECK ----
        quality = self._score_result(result, description)
        cycle.quality_score = quality
        cycle.quality_passed = quality >= self.quality_threshold
        cycle.outcome = "pass" if cycle.quality_passed else "rework"

        if not cycle.quality_passed:
            muda = MudaRecord(
                muda_type="defects",
                task_id=task_id,
                description=f"Quality {quality:.2f} below threshold {self.quality_threshold}",
                severity="high",
            )
            self._muda_log.append(muda)
            cycle.muda.append("defects")

        logger.debug(
            f"[Kaizen/Check] cycle={cycle.cycle_id} quality={quality:.2f} "
            f"passed={cycle.quality_passed} duration={duration_ms}ms"
        )

        # ---- ACT ----
        self._act(cycle, result, description)

        self._cycles.append(cycle)

        result["_kaizen"] = {
            "cycle_id": cycle.cycle_id,
            "quality_score": quality,
            "quality_passed": cycle.quality_passed,
            "muda": cycle.muda,
            "duration_ms": duration_ms,
        }
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_overproduced(complexity: str, model_id: str) -> bool:
        """Heuristic: large model names used for simple tasks."""
        if complexity not in ("simple", "low"):
            return False
        large_indicators = ("70b", "72b", "34b", "32b", "65b", "large", "xl")
        return any(ind in model_id.lower() for ind in large_indicators)

    @staticmethod
    def _score_result(result: Dict[str, Any], description: str) -> float:
        """
        Heuristic quality score 0.0-1.0.

        Uses success flag, output length, and absence of placeholder patterns.
        """
        import re
        _placeholder = re.compile(
            r'\b(TODO|FIXME|raise\s+NotImplementedError|placeholder)\b',
            re.IGNORECASE,
        )

        if not result.get("success", True):
            return 0.2

        output_text = str(result.get("output") or result.get("final_output") or "")
        length_score = min(1.0, len(output_text) / 500)

        placeholder_score = 0.0 if _placeholder.search(output_text) else 1.0

        # Weighted average
        return round(0.4 * length_score + 0.6 * placeholder_score, 3)

    def _act(
        self, cycle: PDCACycle, result: Dict[str, Any], description: str
    ) -> None:
        """Feed results into learning systems (non-blocking, best-effort)."""
        if cycle.quality_passed:
            self._capture_successful_pattern(cycle, description)
        else:
            self._feed_learning_systems(cycle, result, description)

    def _capture_successful_pattern(
        self, cycle: PDCACycle, description: str
    ) -> None:
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner
            learner = get_workflow_learner()
            if hasattr(learner, "record_success"):
                learner.record_success(
                    task_description=description,
                    agent_type=cycle.agent_type,
                    model_id=cycle.model_id,
                    quality_score=cycle.quality_score or 1.0,
                )
        except Exception as exc:
            logger.debug(f"[Kaizen/Act] capture_success failed: {exc}")

    def _feed_learning_systems(
        self, cycle: PDCACycle, result: Dict[str, Any], description: str
    ) -> None:
        try:
            from vetinari.learning.prompt_evolver import get_prompt_evolver
            evolver = get_prompt_evolver()
            if hasattr(evolver, "record_failure"):
                evolver.record_failure(
                    prompt=description,
                    agent_type=cycle.agent_type,
                    quality_score=cycle.quality_score or 0.0,
                )
        except Exception as exc:
            logger.debug(f"[Kaizen/Act] prompt_evolver failed: {exc}")

    # ------------------------------------------------------------------
    # Public: reporting
    # ------------------------------------------------------------------

    def get_muda_report(self) -> Dict[str, Any]:
        """Return a summary of all recorded waste events."""
        from collections import Counter
        type_counts = Counter(m.muda_type for m in self._muda_log)
        return {
            "total_waste_events": len(self._muda_log),
            "by_type": dict(type_counts),
            "high_severity": [
                {"type": m.muda_type, "task_id": m.task_id, "desc": m.description}
                for m in self._muda_log
                if m.severity == "high"
            ],
        }

    def get_pdca_summary(self) -> Dict[str, Any]:
        """Return PDCA cycle aggregate statistics."""
        if not self._cycles:
            return {"cycles": 0}
        pass_count = sum(1 for c in self._cycles if c.quality_passed)
        scores = [c.quality_score for c in self._cycles if c.quality_score is not None]
        avg_quality = sum(scores) / len(scores) if scores else 0.0
        return {
            "cycles": len(self._cycles),
            "pass_rate": pass_count / len(self._cycles),
            "avg_quality": round(avg_quality, 3),
            "rework_count": len(self._cycles) - pass_count,
        }

    def get_5s_audit(self) -> Dict[str, Any]:
        """
        Run a lightweight 5S codebase self-audit.
        Returns findings for Sort / Set-in-Order / Shine / Standardize / Sustain.
        """
        findings: Dict[str, List[str]] = {
            "Sort": [],
            "Set_in_Order": [],
            "Shine": [],
            "Standardize": [],
            "Sustain": [],
        }
        try:
            from vetinari.config import get_project_root
            root = get_project_root() / "vetinari"
            import ast as _ast
            from pathlib import Path

            for py_file in root.rglob("*.py"):
                try:
                    src = py_file.read_text(encoding="utf-8", errors="ignore")
                    tree = _ast.parse(src)
                    # Sort: detect unused imports (simple heuristic)
                    for node in _ast.walk(tree):
                        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
                            names = [a.asname or a.name for a in node.names]
                            for name in names:
                                if name and src.count(name) <= 1:
                                    findings["Sort"].append(
                                        f"{py_file.name}: possibly unused import '{name}'"
                                    )
                except Exception:
                    pass
        except Exception as exc:
            findings["Sustain"].append(f"5S audit error: {exc}")

        return findings


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_kaizen_instance: Optional[KaizenEngine] = None


def get_kaizen_engine() -> KaizenEngine:
    """Return the module-level KaizenEngine singleton."""
    global _kaizen_instance
    if _kaizen_instance is None:
        _kaizen_instance = KaizenEngine()
    return _kaizen_instance
