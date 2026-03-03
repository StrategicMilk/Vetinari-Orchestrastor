import logging
from typing import List, Dict, Set
from collections import defaultdict, deque


class Scheduler:
    def __init__(self, config: dict, max_concurrent: int = 4):
        self.config = config
        self.max_concurrent = max_concurrent

    def build_schedule(self, config: dict) -> List[Dict]:
        """Build a linear schedule (legacy method, use build_schedule_layers for parallelism)."""
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

    def build_schedule_layers(self, config: dict) -> List[List[Dict]]:
        """
        Build execution layers for parallel execution.
        Returns a list of layers, where each layer contains tasks that can run in parallel.
        Tasks in layer N only depend on tasks in layers 0 to N-1.
        """
        tasks = config.get("tasks", [])
        if not tasks:
            return []
        
        # Validate dependencies first
        task_ids = {t["id"] for t in tasks}
        for task in tasks:
            for dep in task.get("dependencies", []):
                if dep not in task_ids:
                    logging.warning(f"Task {task['id']} has unknown dependency: {dep}")
        
        # Build dependency graph
        task_map = {t["id"]: t for t in tasks}
        in_degree = {t["id"]: 0 for t in tasks}
        dependents = defaultdict(list)  # task_id -> list of tasks that depend on it
        
        for task in tasks:
            task_id = task["id"]
            deps = task.get("dependencies", [])
            in_degree[task_id] = len(deps)
            for dep in deps:
                if dep in dependents:  # Only add if dependency exists
                    dependents[dep].append(task_id)
        
        # Kahn's algorithm to build layers
        layers = []
        processed = set()
        max_iterations = len(tasks) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(processed) < len(tasks) and iteration < max_iterations:
            iteration += 1
            # Find all tasks with in-degree 0 (no pending dependencies)
            current_layer = []
            for task_id, degree in in_degree.items():
                if task_id not in processed and degree == 0:
                    task = task_map[task_id]
                    current_layer.append(task)
            
            if not current_layer:
                # Circular dependency or missing task - log and break
                remaining = [tid for tid in task_ids if tid not in processed]
                logging.error(f"Possible circular dependency or missing tasks. Remaining: {remaining}")
                break
            
            # Limit layer size to max_concurrent
            if len(current_layer) > self.max_concurrent:
                current_layer = current_layer[:self.max_concurrent]
            
            layers.append(current_layer)
            
            # Mark these tasks as processed and update in-degrees
            for task in current_layer:
                processed.add(task["id"])
                # Reduce in-degree for dependents
                for dependent_id in dependents[task["id"]]:
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1
        
        if iteration >= max_iterations:
            logging.error("Scheduler exceeded maximum iterations - possible circular dependency")
        
        return layers

    def get_ready_tasks(self, config: dict, completed: Set[str]) -> List[Dict]:
        """Get tasks that are ready to run (all dependencies completed)."""
        tasks = config.get("tasks", [])
        ready = []
        
        for task in tasks:
            if task["id"] in completed:
                continue
            deps = task.get("dependencies", [])
            if all(dep in completed for dep in deps):
                ready.append(task)
        
        return ready[:self.max_concurrent]
