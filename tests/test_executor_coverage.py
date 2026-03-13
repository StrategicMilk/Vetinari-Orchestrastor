"""Coverage tests for vetinari/executor.py — Phase 7D"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock


class TestTaskExecutor(unittest.TestCase):

    def _make_executor(self, tmpdir):
        from vetinari.executor import TaskExecutor
        adapter   = MagicMock()
        validator = MagicMock()
        validator.validate.return_value = (True, None)
        config = {
            "project_root": tmpdir,
            "prompts_dir":  os.path.join(tmpdir, "prompts"),
            "tasks": [
                {"id": "task_1",
                 "prompts": {"plan": "Plan something", "run": "Build something"}}
            ]
        }
        return TaskExecutor(adapter, validator, config), adapter, validator

    def test_load_prompt_from_config_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, _, _ = self._make_executor(d)
            prompt = exec_._load_prompt("task_1", "plan")
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_load_prompt_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            prompts_dir = os.path.join(d, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            prompt_path = os.path.join(prompts_dir, "task_1_plan.txt")
            with open(prompt_path, "w") as f:
                f.write("  File-based prompt  ")
            exec_, _, _ = self._make_executor(d)
            prompt = exec_._load_prompt("task_1", "plan")
            assert prompt == "File-based prompt"

    def test_parse_code_blocks_python(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, _, _ = self._make_executor(d)
            text   = "```python\ndef foo():\n    pass\n```"
            blocks = exec_._parse_code_blocks(text)
            assert isinstance(blocks, dict)
            assert len(blocks) > 0

    def test_parse_code_blocks_empty(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, _, _ = self._make_executor(d)
            blocks = exec_._parse_code_blocks("no code here")
            assert blocks == {}

    def test_parse_code_blocks_readme(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, _, _ = self._make_executor(d)
            text   = "```markdown\n# README content here\n```"
            blocks = exec_._parse_code_blocks(text)
            assert isinstance(blocks, dict)

    def test_write_files(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, _, _ = self._make_executor(d)
            blocks = {"output_0.py": "print('hello')", "README.md": "# Readme"}
            exec_._write_files(blocks, "task_1")
            output_dir = os.path.join(d, "outputs", "task_1", "generated")
            assert os.path.isdir(output_dir)
            assert os.path.exists(os.path.join(output_dir, "output_0.py"))

    def test_execute_task_calls_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            exec_, adapter, _ = self._make_executor(d)
            # adapter.chat must return a dict with an "output" key
            adapter.chat.return_value = {"output": "```python\nprint('result')\n```",
                                          "model": "test", "tokens": 10}
            result = exec_.execute_task("task_1")
            assert isinstance(result, dict)


if __name__ == "__main__":
    unittest.main()
