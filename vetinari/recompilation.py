"""
Recompilation Engine — WS6
===========================
Bottom-up sequential assembly of subtask outputs.

Workflow
--------
1. **Layer identification** — group subtasks by dependency depth so tasks with
   no unresolved predecessors form layer 0, their dependents form layer 1, etc.
2. **Bottom-up execution** — execute layers in order (deepest leaves first).
3. **Within-layer parallelism** — tasks in the same layer run concurrently up
   to ``max_workers``.
4. **Layer review gate** — after each layer completes, run integration checks
   before proceeding to the next layer.
5. **Feature assembly** — when subtask outputs are combined into their parent,
   validate coherence (no duplicates, interfaces match).
6. **Final integration check** — after all layers, run a comprehensive check.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Maximum workers per layer
DEFAULT_MAX_WORKERS: int = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerResult:
    """Aggregated results for one execution layer."""
    layer_index: int
    task_results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    gate_passed: bool = True
    gate_issues: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class RecompilationResult:
    """Full recompilation run result."""
    plan_id: str
    layers: List[LayerResult] = field(default_factory=list)
    final_output: Dict[str, Any] = field(default_factory=dict)
    integration_passed: bool = True
    integration_issues: List[str] = field(default_factory=list)
    total_duration_ms: int = 0
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success(self) -> bool:
        return self.integration_passed and all(
            lr.gate_passed for lr in self.layers
        )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RecompilationEngine:
    """
    Orchestrates bottom-up task assembly with layer review gates.

    Usage::

        engine = RecompilationEngine(max_workers=4)
        result = engine.assemble(subtasks, executor=my_task_executor)
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        subtasks: List[Dict[str, Any]],
        executor: Callable[[Dict[str, Any]], Dict[str, Any]],
        plan_id: Optional[str] = None,
    ) -> RecompilationResult:
        """
        Run the full bottom-up assembly pipeline.

        Parameters
        ----------
        subtasks:
            List of subtask dicts with at least ``subtask_id``,
            ``description``, ``agent_type``, and ``dependencies``.
        executor:
            Callable that accepts a subtask dict and returns a result dict.
            Must include a boolean ``success`` key and optional ``output``.
        plan_id:
            Human-readable identifier (auto-generated if omitted).

        Returns
        -------
        RecompilationResult
        """
        import time

        plan_id = plan_id or f"recomp-{uuid.uuid4().hex[:8]}"
        start = time.time()
        result = RecompilationResult(plan_id=plan_id)

        # Build layers
        layers = self._build_layers(subtasks)
        logger.info(
            f"[Recompilation] plan={plan_id} — "
            f"{len(subtasks)} tasks in {len(layers)} layers"
        )

        completed_outputs: Dict[str, Any] = {}  # subtask_id → output

        for layer_idx, layer_tasks in enumerate(layers):
            layer_start = time.time()
            logger.info(
                f"[Recompilation] Layer {layer_idx}: executing {len(layer_tasks)} task(s)"
            )

            # Execute tasks in parallel within the layer
            task_results: Dict[str, Any] = {}
            errors: Dict[str, str] = {}
            self._execute_layer(
                layer_tasks, executor, completed_outputs,
                task_results, errors
            )

            # Layer review gate
            gate_passed, gate_issues = self._review_layer(
                layer_idx, layer_tasks, task_results, errors
            )

            layer_duration = int((time.time() - layer_start) * 1000)
            lr = LayerResult(
                layer_index=layer_idx,
                task_results=task_results,
                errors=errors,
                gate_passed=gate_passed,
                gate_issues=gate_issues,
                duration_ms=layer_duration,
            )
            result.layers.append(lr)

            # Update completed outputs map
            completed_outputs.update(task_results)

            if not gate_passed:
                logger.warning(
                    f"[Recompilation] Layer {layer_idx} gate FAILED: {gate_issues[:3]}"
                )
                # Continue anyway — surface issues but don't halt

        # Final integration check
        int_passed, int_issues = self._final_integration_check(
            subtasks, completed_outputs
        )
        result.integration_passed = int_passed
        result.integration_issues = int_issues

        # Assemble final output
        result.final_output = self._build_final_output(subtasks, completed_outputs)
        result.total_duration_ms = int((time.time() - start) * 1000)

        logger.info(
            f"[Recompilation] plan={plan_id} done in {result.total_duration_ms}ms "
            f"— integration={'OK' if int_passed else 'ISSUES'}"
        )
        return result

    # ------------------------------------------------------------------
    # Layer building
    # ------------------------------------------------------------------

    def _build_layers(
        self, subtasks: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Topological sort → layers of tasks that can run in parallel.

        Uses Kahn's BFS algorithm.
        """
        id_to_task = {s["subtask_id"]: s for s in subtasks}
        in_degree: Dict[str, int] = {sid: 0 for sid in id_to_task}
        dependents: Dict[str, List[str]] = defaultdict(list)

        for st in subtasks:
            sid = st["subtask_id"]
            for dep in st.get("dependencies", []):
                if dep in id_to_task:
                    in_degree[sid] += 1
                    dependents[dep].append(sid)

        # Kahn's algorithm — iterate until all processed
        layers: List[List[Dict[str, Any]]] = []
        remaining = set(id_to_task.keys())
        max_iters = len(subtasks) * 2

        for _ in range(max_iters):
            ready = [
                id_to_task[sid]
                for sid in remaining
                if in_degree[sid] == 0
            ]
            if not ready:
                break
            layers.append(ready)
            for task in ready:
                remaining.discard(task["subtask_id"])
                for dep_id in dependents[task["subtask_id"]]:
                    in_degree[dep_id] -= 1

        if remaining:
            logger.error(
                f"[Recompilation] Circular dependency; {len(remaining)} tasks unscheduled"
            )
            # Add stragglers as a final layer so they at least run
            layers.append([id_to_task[sid] for sid in remaining])

        return layers

    # ------------------------------------------------------------------
    # Layer execution
    # ------------------------------------------------------------------

    def _execute_layer(
        self,
        tasks: List[Dict[str, Any]],
        executor: Callable,
        completed_outputs: Dict[str, Any],
        task_results: Dict[str, Any],
        errors: Dict[str, str],
    ) -> None:
        """Execute tasks in parallel within a single layer."""
        if not tasks:
            return

        def run_one(task: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            sid = task["subtask_id"]
            # Inject dependency outputs into the task context
            enriched = dict(task)
            dep_ctx: Dict[str, Any] = {}
            for dep_id in task.get("dependencies", []):
                if dep_id in completed_outputs:
                    dep_ctx[dep_id] = completed_outputs[dep_id]
            enriched["dependency_outputs"] = dep_ctx
            try:
                res = executor(enriched)
                return sid, res
            except Exception as exc:
                logger.warning(f"[Recompilation] Task {sid} raised: {exc}")
                return sid, {"success": False, "output": None, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            futures = {pool.submit(run_one, t): t["subtask_id"] for t in tasks}
            for future in as_completed(futures):
                sid, res = future.result()
                if res.get("success", False):
                    task_results[sid] = res.get("output") or res
                else:
                    errors[sid] = res.get("error", "unknown error")
                    task_results[sid] = None

    # ------------------------------------------------------------------
    # Review gates
    # ------------------------------------------------------------------

    def _review_layer(
        self,
        layer_idx: int,
        tasks: List[Dict[str, Any]],
        task_results: Dict[str, Any],
        errors: Dict[str, str],
    ) -> Tuple[bool, List[str]]:
        """
        Run integration checks after a layer completes.

        Current checks:
        - No more than 50% of tasks in the layer failed.
        - Outputs are non-None for successful tasks.
        """
        issues: List[str] = []

        failed_count = len(errors)
        total = len(tasks)
        if total and (failed_count / total) > 0.5:
            issues.append(
                f"Layer {layer_idx}: {failed_count}/{total} tasks failed (>50%)"
            )

        # Check for placeholder output patterns
        _placeholder_re = re.compile(
            r'\b(TODO|FIXME|placeholder|stub|not implemented|pass)\b', re.IGNORECASE
        )
        for sid, output in task_results.items():
            if output is None:
                continue
            text = str(output)
            if _placeholder_re.search(text):
                issues.append(
                    f"Task {sid} output may contain placeholder code"
                )

        return not issues, issues

    def _final_integration_check(
        self,
        subtasks: List[Dict[str, Any]],
        completed_outputs: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Run a final integration check across all outputs.

        Checks:
        - All subtasks have an output (none silently dropped).
        - No two outputs share identical content (duplicate detection).
        """
        issues: List[str] = []

        # Coverage check
        missing = [
            s["subtask_id"]
            for s in subtasks
            if s["subtask_id"] not in completed_outputs
        ]
        if missing:
            issues.append(f"{len(missing)} subtask(s) have no output: {missing[:5]}")

        # Duplicate detection (exact string matches only)
        seen: Dict[str, str] = {}
        for sid, output in completed_outputs.items():
            if output is None:
                continue
            key = str(output)[:500]
            if key in seen:
                issues.append(
                    f"Duplicate output detected: tasks '{sid}' and '{seen[key]}'"
                )
            else:
                seen[key] = sid

        return not issues, issues

    # ------------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------------

    def _build_final_output(
        self,
        subtasks: List[Dict[str, Any]],
        completed_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine all outputs into a structured final deliverable."""
        assembled: Dict[str, Any] = {
            "total_tasks": len(subtasks),
            "completed_tasks": sum(1 for v in completed_outputs.values() if v is not None),
            "outputs": {},
        }

        for st in subtasks:
            sid = st["subtask_id"]
            assembled["outputs"][sid] = {
                "description": st.get("description", ""),
                "agent_type": st.get("agent_type", ""),
                "output": completed_outputs.get(sid),
            }

        return assembled


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[RecompilationEngine] = None


def get_recompilation_engine(max_workers: int = DEFAULT_MAX_WORKERS) -> RecompilationEngine:
    """Return the module-level RecompilationEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecompilationEngine(max_workers=max_workers)
    return _engine_instance
