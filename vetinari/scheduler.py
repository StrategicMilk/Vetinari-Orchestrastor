"""Scheduler module."""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class Scheduler:
    """Schedules and dispatches waves of tasks to agents."""
    def __init__(self, config: dict, max_concurrent: int = 4):
        self.config = config
        self.max_concurrent = max_concurrent

    def build_schedule(self, config: dict) -> list[dict]:
        """Build a linear schedule (legacy method, use build_schedule_layers for parallelism).

        Returns:
            List of results.
        """
        tasks = config.get("tasks", [])
        # Simple topological sort based on dependencies
        order = []
        dep_map = {t["id"]: set(t.get("dependencies", [])) for t in tasks}
        remaining = {t["id"]: t for t in tasks}

        while remaining:
            progressed = False
            for tid, t in list(remaining.items()):
                deps = dep_map[tid]
                if all(d in [x["id"] for x in order] for d in deps):
                    order.append(t)
                    del remaining[tid]
                    progressed = True
            if not progressed:
                # Circular or unresolved; break to avoid infinite loop
                break
        return order

    def build_schedule_layers(self, config: dict) -> list[list[dict]]:
        """Build execution layers for parallel execution.

        Returns a list of layers, where each layer contains tasks that can run in parallel.
        Tasks in layer N only depend on tasks in layers 0 to N-1.

        Returns:
            List of results.
        """
        tasks = config.get("tasks", [])
        if not tasks:
            return []

        # Validate dependencies first
        task_ids = {t["id"] for t in tasks}
        for task in tasks:
            for dep in task.get("dependencies", []):
                if dep not in task_ids:
                    logger.warning("Task %s has unknown dependency: %s", task["id"], dep)

        # Build dependency graph
        task_map = {t["id"]: t for t in tasks}
        in_degree = {t["id"]: 0 for t in tasks}
        dependents = defaultdict(list)  # task_id -> list of tasks that depend on it
        unresolvable = set()  # Track tasks with missing dependencies

        for task in tasks:
            task_id = task["id"]
            deps = task.get("dependencies", [])
            # Check if all dependencies exist
            valid_deps = [d for d in deps if d in task_map]
            # If any dependency is missing, mark this task as unresolvable
            if len(valid_deps) != len(deps):
                unresolvable.add(task_id)
                logger.warning("Task %s has unknown dependency and will not be scheduled", task_id)
            in_degree[task_id] = len(valid_deps)
            for dep in valid_deps:
                # Always initialize dependent list and add task_id
                if dep not in dependents:
                    dependents[dep] = []
                dependents[dep].append(task_id)

        # Kahn's algorithm to build layers
        layers = []
        processed = set()
        max_iterations = len(tasks) * 2  # Prevent infinite loops
        iteration = 0

        while len(processed) < len(tasks) and iteration < max_iterations:
            iteration += 1
            # Find all tasks with in-degree 0 (no pending dependencies) and not unresolvable
            ready = []
            for task_id, degree in in_degree.items():
                if task_id not in processed and task_id not in unresolvable and degree == 0:
                    task = task_map[task_id]
                    ready.append(task)

            if not ready:
                # Circular dependency or missing task - log and break
                remaining = [tid for tid in task_ids if tid not in processed]
                logger.error("Possible circular dependency or missing tasks. Remaining: %s", remaining)
                break

            # Emit layers of at most max_concurrent tasks, but keep iterating
            # so that overflow tasks are scheduled in subsequent layers rather
            # than being silently dropped.
            current_layer = ready[: self.max_concurrent]

            layers.append(current_layer)

            # Mark these tasks as processed and update in-degrees
            for task in current_layer:
                processed.add(task["id"])
                # Reduce in-degree for dependents
                for dependent_id in dependents[task["id"]]:
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1

        if iteration >= max_iterations:
            logger.error("Scheduler exceeded maximum iterations - possible circular dependency")

        return layers

    def get_ready_tasks(self, config: dict, completed: set[str]) -> list[dict]:
        """Get tasks that are ready to run (all dependencies completed).

        Args:
            config: The config.
            completed: The completed.

        Returns:
            List of results.
        """
        tasks = config.get("tasks", [])
        ready = []

        for task in tasks:
            if task["id"] in completed:
                continue
            deps = task.get("dependencies", [])
            if all(dep in completed for dep in deps):
                ready.append(task)

        return ready[: self.max_concurrent]
