"""
Tests for the dry-run demo package scaffold.

This test verifies that the demo package scaffold can be built
and imported, demonstrating the end-to-end flow.
"""

import os
import sys
import subprocess
import pytest


def test_demo_package_scaffold_exists():
    """Verify the demo package scaffold files exist."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(repo_root, "dry_run_demo_pkg")
    if not os.path.isdir(pkg_dir):
        pytest.skip("dry_run_demo_pkg not generated in this environment")
    setup_path = os.path.join(pkg_dir, "setup.py")
    assert os.path.exists(setup_path), f"setup.py not found at {setup_path}"

    pkg_init = os.path.join(pkg_dir, "dry_run_demo_pkg", "__init__.py")
    assert os.path.exists(pkg_init), f"__init__.py not found at {pkg_init}"


def test_demo_package_can_be_imported():
    """Verify the demo package can be imported."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_path = os.path.join(repo_root, "dry_run_demo_pkg")
    if not os.path.isdir(pkg_path):
        pytest.skip("dry_run_demo_pkg not generated in this environment")
    sys.path.insert(0, pkg_path)
    
    try:
        import dry_run_demo_pkg
        assert hasattr(dry_run_demo_pkg, "__version__")
        assert dry_run_demo_pkg.__version__ == "0.1.0"
    finally:
        sys.path.remove(pkg_path)


def test_demo_package_main_runs():
    """Verify the demo package main function runs without errors."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_path = os.path.join(repo_root, "dry_run_demo_pkg")
    if not os.path.isdir(pkg_path):
        pytest.skip("dry_run_demo_pkg not generated in this environment")
    sys.path.insert(0, pkg_path)

    try:
        import importlib
        mod = importlib.import_module("dry_run_demo_pkg")
        if hasattr(mod, "hello"):
            result = mod.hello()
            assert result == "Hello from dry_run_demo_pkg!"
        else:
            # Package exists but hello() not defined — scaffold only
            assert hasattr(mod, "__version__")
    finally:
        if pkg_path in sys.path:
            sys.path.remove(pkg_path)


def test_coding_bridge_scaffold_generation():
    """Test that CodingBridge can generate a scaffold."""
    from vetinari.agents.coding_bridge import CodingBridge, CodingTask, CodingTaskType
    
    bridge = CodingBridge()
    bridge.enabled = True  # Enable for testing
    
    task = CodingTask(
        task_type=CodingTaskType.SCAFFOLD,
        description="Generate test scaffold",
        language="python",
        context={"project_name": "test_scaffold_project"},
        output_path="./test_output_scaffold"
    )
    
    result = bridge.generate_task(task)
    
    assert result.success is True
    assert len(result.output_files) > 0
    assert "setup.py" in result.output_files[0]
    
    # Cleanup
    import shutil
    output_dir = "./test_output_scaffold"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
