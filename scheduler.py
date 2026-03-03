from typing import List, Dict

class Scheduler:
    def __init__(self, config: dict):
        self.config = config

    def build_schedule(self, config: dict) -> List[Dict]:
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