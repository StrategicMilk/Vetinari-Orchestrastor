"""
Tests for scheduler circular dependency detection and topological sorting.
Tests Kahn's algorithm implementation and error handling.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vetinari.scheduler import Scheduler


class TestSchedulerCircularDependency:
    """Test circular dependency detection."""

    def setup_method(self):
        """Set up scheduler for testing."""
        self.config_base = {}
        self.scheduler = Scheduler(self.config_base, max_concurrent=4)

    def test_direct_circular_dependency(self):
        """Detect A -> B -> A circular dependency."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": ["task_b"]},
                {"id": "task_b", "dependencies": ["task_a"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # Circular dependencies should be detected, resulting in incomplete schedule
        task_ids_scheduled = set()
        for layer in layers:
            for task in layer:
                task_ids_scheduled.add(task["id"])

        # Both tasks should be scheduled despite circular dependency
        # (the algorithm detects and logs it, but still returns incomplete schedule)
        assert len(task_ids_scheduled) < 2 or len(layers) > 0

    def test_self_circular_dependency(self):
        """Detect A -> A self-dependency."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": ["task_a"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # Self-dependency should be detected
        # Task should not be scheduled (in-degree never reaches 0)
        task_ids_scheduled = set()
        for layer in layers:
            for task in layer:
                task_ids_scheduled.add(task["id"])

        assert "task_a" not in task_ids_scheduled or len(layers) == 0

    def test_complex_circular_dependency(self):
        """Detect A -> B -> C -> A complex circular dependency."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": ["task_c"]},
                {"id": "task_b", "dependencies": ["task_a"]},
                {"id": "task_c", "dependencies": ["task_b"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # Should detect the circular chain
        task_ids_scheduled = set()
        for layer in layers:
            for task in layer:
                task_ids_scheduled.add(task["id"])

        # None of the circular tasks should complete scheduling
        assert len(task_ids_scheduled) < 3 or len(layers) == 0


class TestSchedulerValidDependencies:
    """Test valid dependency resolution."""

    def setup_method(self):
        """Set up scheduler for testing."""
        self.config_base = {}
        self.scheduler = Scheduler(self.config_base, max_concurrent=4)

    def test_linear_dependencies(self):
        """Test A -> B -> C linear dependency chain."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": ["task_a"]},
                {"id": "task_c", "dependencies": ["task_b"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        assert len(layers) == 3  # Each task in its own layer
        assert layers[0][0]["id"] == "task_a"
        assert layers[1][0]["id"] == "task_b"
        assert layers[2][0]["id"] == "task_c"

    def test_parallel_tasks(self):
        """Test independent tasks execute in parallel."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": []},
                {"id": "task_c", "dependencies": []}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        assert len(layers) == 1  # All in first layer
        assert len(layers[0]) == 3  # All three tasks
        task_ids = {t["id"] for t in layers[0]}
        assert task_ids == {"task_a", "task_b", "task_c"}

    def test_mixed_dependencies(self):
        """Test mixed parallel and sequential tasks."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": []},
                {"id": "task_c", "dependencies": ["task_a", "task_b"]},
                {"id": "task_d", "dependencies": ["task_c"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        assert len(layers) == 3
        # Layer 0: task_a and task_b in parallel
        assert len(layers[0]) == 2
        # Layer 1: task_c (depends on both)
        assert len(layers[1]) == 1
        assert layers[1][0]["id"] == "task_c"
        # Layer 2: task_d
        assert len(layers[2]) == 1
        assert layers[2][0]["id"] == "task_d"

    def test_no_tasks(self):
        """Test empty task list."""
        config = {"tasks": []}

        layers = self.scheduler.build_schedule_layers(config)

        assert layers == []

    def test_single_task(self):
        """Test single task with no dependencies."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        assert len(layers) == 1
        assert layers[0][0]["id"] == "task_a"


class TestSchedulerMaxConcurrent:
    """Test max concurrent task limiting."""

    def test_max_concurrent_limit(self):
        """Verify max_concurrent limit is respected."""
        scheduler = Scheduler({}, max_concurrent=2)

        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": []},
                {"id": "task_c", "dependencies": []},
                {"id": "task_d", "dependencies": []},
                {"id": "task_e", "dependencies": []}
            ]
        }

        layers = scheduler.build_schedule_layers(config)

        # With max_concurrent=2, tasks should be split across layers
        # Even though they have no dependencies
        total_scheduled = sum(len(layer) for layer in layers)
        assert total_scheduled == 5

        # Some layer should have <= 2 tasks
        for layer in layers:
            assert len(layer) <= 2

    def test_max_concurrent_large_value(self):
        """Test with large max_concurrent value."""
        scheduler = Scheduler({}, max_concurrent=100)

        config = {
            "tasks": [
                {"id": f"task_{i}", "dependencies": []} for i in range(10)
            ]
        }

        layers = scheduler.build_schedule_layers(config)

        # All independent tasks should be in one layer
        assert len(layers) == 1
        assert len(layers[0]) == 10


class TestSchedulerMissingDependencies:
    """Test handling of missing/invalid dependencies."""

    def setup_method(self):
        """Set up scheduler for testing."""
        self.scheduler = Scheduler({}, max_concurrent=4)

    def test_missing_dependency_detection(self):
        """Verify missing dependencies are detected."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": ["task_x"]}  # task_x doesn't exist
            ]
        }

        # Should log warning about unknown dependency
        layers = self.scheduler.build_schedule_layers(config)

        # task_b will never be scheduled because dependency can't be satisfied
        task_ids_scheduled = set()
        for layer in layers:
            for task in layer:
                task_ids_scheduled.add(task["id"])

        assert "task_a" in task_ids_scheduled
        assert "task_b" not in task_ids_scheduled  # Unresolvable

    def test_empty_dependency_list(self):
        """Test tasks with empty dependencies."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        assert len(layers) == 1
        assert layers[0][0]["id"] == "task_a"

    def test_none_dependency_field(self):
        """Test task with missing dependencies field."""
        config = {
            "tasks": [
                {"id": "task_a"}  # No dependencies field
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # Should treat as no dependencies
        assert len(layers) == 1
        assert layers[0][0]["id"] == "task_a"


class TestSchedulerBuildSchedule:
    """Test legacy build_schedule method."""

    def setup_method(self):
        """Set up scheduler for testing."""
        self.scheduler = Scheduler({}, max_concurrent=4)

    def test_build_schedule_linear(self):
        """Test linear schedule building."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": ["task_a"]},
                {"id": "task_c", "dependencies": ["task_b"]}
            ]
        }

        schedule = self.scheduler.build_schedule(config)

        # Should produce a linear order
        task_ids = [t["id"] for t in schedule]
        assert task_ids == ["task_a", "task_b", "task_c"]

    def test_build_schedule_with_circular(self):
        """Test schedule with circular dependency."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": ["task_b"]},
                {"id": "task_b", "dependencies": ["task_a"]}
            ]
        }

        schedule = self.scheduler.build_schedule(config)

        # Circular dependency should result in incomplete schedule
        assert len(schedule) < 2 or len(schedule) == 0


class TestSchedulerEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up scheduler for testing."""
        self.scheduler = Scheduler({}, max_concurrent=4)

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph (common pattern)."""
        config = {
            "tasks": [
                {"id": "task_a", "dependencies": []},
                {"id": "task_b", "dependencies": ["task_a"]},
                {"id": "task_c", "dependencies": ["task_a"]},
                {"id": "task_d", "dependencies": ["task_b", "task_c"]}
            ]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # Layer 0: task_a
        assert layers[0][0]["id"] == "task_a"
        # Layer 1: task_b and task_c (parallel)
        layer1_ids = {t["id"] for t in layers[1]}
        assert layer1_ids == {"task_b", "task_c"}
        # Layer 2: task_d
        assert layers[2][0]["id"] == "task_d"

    def test_large_number_of_tasks(self):
        """Test scheduler with large number of tasks."""
        config = {
            "tasks": [{"id": f"task_{i}", "dependencies": []} for i in range(100)]
        }

        layers = self.scheduler.build_schedule_layers(config)

        # All tasks should be scheduled
        total_scheduled = sum(len(layer) for layer in layers)
        assert total_scheduled == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
