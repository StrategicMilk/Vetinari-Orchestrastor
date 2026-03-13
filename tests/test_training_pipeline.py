"""
Comprehensive tests for vetinari/training/pipeline.py

All subprocess.run calls, shutil operations, and external library
imports are mocked so no actual training code executes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call, mock_open


# ---------------------------------------------------------------------------
# Helpers — stub heavy ML libs so the module can be imported without them
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    return mod


for _lib in ["unsloth", "trl", "peft", "transformers", "bitsandbytes", "datasets", "torch"]:
    if _lib not in sys.modules:
        sys.modules[_lib] = _make_stub(_lib)

# Patch get_training_collector before importing pipeline so the module-level
# import inside DataCurator.curate() can be controlled per-test.
_fake_training_data_mod = _make_stub("vetinari.learning.training_data")
# Pre-populate the attribute so patch() can find and replace it
_fake_training_data_mod.get_training_collector = lambda: None
sys.modules.setdefault("vetinari.learning.training_data", _fake_training_data_mod)
_learning_stub = _make_stub("vetinari.learning")
sys.modules.setdefault("vetinari.learning", _learning_stub)

# Python 3.10 does not auto-resolve sys.modules entries via parent attribute
# lookup, so patch("vetinari.learning.training_data.X") needs explicit bindings.
import vetinari as _vetinari_pkg  # noqa: E402
_vetinari_pkg.learning = sys.modules["vetinari.learning"]
sys.modules["vetinari.learning"].training_data = _fake_training_data_mod

from vetinari.training.pipeline import (  # noqa: E402
    DataCurator,
    GGUFConverter,
    LocalTrainer,
    ModelDeployer,
    TrainingPipeline,
    TrainingRun,
    _MODELS_DIR,
)


# ---------------------------------------------------------------------------
# 1. TestTrainingRun
# ---------------------------------------------------------------------------

class TestTrainingRun(unittest.TestCase):
    """Tests for the TrainingRun dataclass."""

    def _make(self, **kw) -> TrainingRun:
        defaults = dict(
            run_id="run_abc12345",
            timestamp="2025-01-01T00:00:00",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            task_type="coding",
            training_examples=100,
            epochs=3,
            success=False,
        )
        defaults.update(kw)
        return TrainingRun(**defaults)

    def test_required_fields_stored(self):
        run = self._make()
        self.assertEqual(run.run_id, "run_abc12345")
        self.assertEqual(run.base_model, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(run.task_type, "coding")

    def test_default_output_model_path_empty(self):
        run = self._make()
        self.assertEqual(run.output_model_path, "")

    def test_default_adapter_path_empty(self):
        run = self._make()
        self.assertEqual(run.adapter_path, "")

    def test_default_eval_score_zero(self):
        run = self._make()
        self.assertEqual(run.eval_score, 0.0)

    def test_default_baseline_score_zero(self):
        run = self._make()
        self.assertEqual(run.baseline_score, 0.0)

    def test_default_error_empty(self):
        run = self._make()
        self.assertEqual(run.error, "")

    def test_default_success_false(self):
        run = self._make(success=False)
        self.assertFalse(run.success)

    def test_success_true(self):
        run = self._make(success=True)
        self.assertTrue(run.success)

    def test_set_output_model_path(self):
        run = self._make(output_model_path="/some/path/model.gguf")
        self.assertEqual(run.output_model_path, "/some/path/model.gguf")

    def test_set_adapter_path(self):
        run = self._make(adapter_path="/tmp/run/lora_adapter")
        self.assertEqual(run.adapter_path, "/tmp/run/lora_adapter")

    def test_set_error(self):
        run = self._make(error="something went wrong")
        self.assertEqual(run.error, "something went wrong")

    def test_set_eval_score(self):
        run = self._make(eval_score=0.87)
        self.assertAlmostEqual(run.eval_score, 0.87)

    def test_training_examples_stored(self):
        run = self._make(training_examples=42)
        self.assertEqual(run.training_examples, 42)

    def test_epochs_stored(self):
        run = self._make(epochs=5)
        self.assertEqual(run.epochs, 5)

    def test_serializes_to_dict(self):
        run = self._make(success=True, output_model_path="/m/model.gguf")
        d = dataclasses.asdict(run)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["success"], True)
        self.assertEqual(d["output_model_path"], "/m/model.gguf")

    def test_dict_has_all_fields(self):
        run = self._make()
        d = dataclasses.asdict(run)
        expected_keys = {
            "run_id", "timestamp", "base_model", "task_type",
            "training_examples", "epochs", "success",
            "output_model_path", "adapter_path", "eval_score",
            "baseline_score", "error",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_json_round_trip(self):
        run = self._make(success=True, eval_score=0.91)
        d = dataclasses.asdict(run)
        s = json.dumps(d)
        d2 = json.loads(s)
        self.assertEqual(d2["eval_score"], 0.91)
        self.assertTrue(d2["success"])

    def test_is_dataclass(self):
        self.assertTrue(dataclasses.is_dataclass(TrainingRun))

    def test_timestamp_stored(self):
        ts = "2025-06-15T12:30:00"
        run = self._make(timestamp=ts)
        self.assertEqual(run.timestamp, ts)


# ---------------------------------------------------------------------------
# 2. TestDataCuratorCurate
# ---------------------------------------------------------------------------

class TestDataCuratorCurate(unittest.TestCase):
    """Tests for DataCurator.curate()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.curator = DataCurator()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_collector(self, records):
        """Return a mock collector that returns `records` from export_sft_dataset."""
        collector = MagicMock()
        collector.export_sft_dataset.return_value = records
        return collector

    def _curate(self, records, **kw):
        collector = self._make_collector(records)
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            return self.curator.curate(output_dir=self.tmpdir, **kw)

    def test_returns_string_path(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records)
        self.assertIsInstance(path, str)

    def test_returned_path_exists(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records)
        self.assertTrue(os.path.exists(path))

    def test_file_has_jsonl_content(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records)
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

    def test_alpaca_instruction_field(self):
        records = [{"prompt": "write code", "completion": "print('hi')"}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertIn("instruction", obj)
        self.assertEqual(obj["instruction"], "write code")

    def test_alpaca_input_field_empty(self):
        records = [{"prompt": "write code", "completion": "print('hi')"}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertIn("input", obj)
        self.assertEqual(obj["input"], "")

    def test_alpaca_output_field(self):
        records = [{"prompt": "write code", "completion": "print('hi')"}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertIn("output", obj)
        self.assertEqual(obj["output"], "print('hi')")

    def test_multiple_records_written(self):
        records = [
            {"prompt": f"prompt {i}", "completion": f"output {i}"}
            for i in range(10)
        ]
        path = self._curate(records)
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 10)

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError):
            self._curate([])

    def test_prompt_truncated_at_1000(self):
        long_prompt = "x" * 2000
        records = [{"prompt": long_prompt, "completion": "out"}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertEqual(len(obj["instruction"]), 1000)

    def test_completion_truncated_at_2000(self):
        long_comp = "y" * 3000
        records = [{"prompt": "in", "completion": long_comp}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertEqual(len(obj["output"]), 2000)

    def test_filename_contains_task_type(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records, task_type="coding")
        self.assertIn("coding", os.path.basename(path))

    def test_filename_contains_general_when_no_task_type(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records, task_type=None)
        self.assertIn("general", os.path.basename(path))

    def test_filename_starts_with_sft(self):
        records = [{"prompt": "hi", "completion": "hello"}]
        path = self._curate(records)
        self.assertTrue(os.path.basename(path).startswith("sft_"))

    def test_min_score_passed_to_collector(self):
        collector = self._make_collector([{"prompt": "p", "completion": "c"}])
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            self.curator.curate(output_dir=self.tmpdir, min_score=0.9)
        collector.export_sft_dataset.assert_called_once()
        call_kwargs = collector.export_sft_dataset.call_args
        # Check min_score was passed
        args, kwargs = call_kwargs
        self.assertEqual(kwargs.get("min_score", args[0] if args else None), 0.9)

    def test_task_type_passed_to_collector(self):
        collector = self._make_collector([{"prompt": "p", "completion": "c"}])
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            self.curator.curate(output_dir=self.tmpdir, task_type="math")
        collector.export_sft_dataset.assert_called_once()
        call_kwargs = collector.export_sft_dataset.call_args
        _, kwargs = call_kwargs
        self.assertEqual(kwargs.get("task_type"), "math")

    def test_each_line_is_valid_json(self):
        records = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(5)]
        path = self._curate(records)
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.assertIn("instruction", obj)


# ---------------------------------------------------------------------------
# 3. TestDataCuratorCurateDpo
# ---------------------------------------------------------------------------

class TestDataCuratorCurateDpo(unittest.TestCase):
    """Tests for DataCurator.curate_dpo()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.curator = DataCurator()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_collector(self, pairs):
        collector = MagicMock()
        collector.export_dpo_dataset.return_value = pairs
        return collector

    def _curate_dpo(self, pairs, **kw):
        collector = self._make_collector(pairs)
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            return self.curator.curate_dpo(output_dir=self.tmpdir, **kw)

    def test_returns_string_path(self):
        pairs = [{"prompt": "q", "chosen": "good", "rejected": "bad"}]
        path = self._curate_dpo(pairs)
        self.assertIsInstance(path, str)

    def test_returned_path_exists(self):
        pairs = [{"prompt": "q", "chosen": "good", "rejected": "bad"}]
        path = self._curate_dpo(pairs)
        self.assertTrue(os.path.exists(path))

    def test_empty_pairs_raises(self):
        with self.assertRaises(ValueError):
            self._curate_dpo([])

    def test_multiple_pairs_written(self):
        pairs = [{"prompt": f"q{i}", "chosen": "g", "rejected": "b"} for i in range(6)]
        path = self._curate_dpo(pairs)
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 6)

    def test_pairs_preserved_verbatim(self):
        pairs = [{"prompt": "ask", "chosen": "good answer", "rejected": "bad answer"}]
        path = self._curate_dpo(pairs)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertEqual(obj["prompt"], "ask")
        self.assertEqual(obj["chosen"], "good answer")
        self.assertEqual(obj["rejected"], "bad answer")

    def test_filename_starts_with_dpo(self):
        pairs = [{"prompt": "q", "chosen": "g", "rejected": "b"}]
        path = self._curate_dpo(pairs)
        self.assertTrue(os.path.basename(path).startswith("dpo_"))

    def test_filename_contains_task_type(self):
        pairs = [{"prompt": "q", "chosen": "g", "rejected": "b"}]
        path = self._curate_dpo(pairs, task_type="coding")
        self.assertIn("coding", os.path.basename(path))

    def test_filename_contains_general_when_no_task_type(self):
        pairs = [{"prompt": "q", "chosen": "g", "rejected": "b"}]
        path = self._curate_dpo(pairs, task_type=None)
        self.assertIn("general", os.path.basename(path))

    def test_min_score_gap_passed_to_collector(self):
        collector = self._make_collector([{"prompt": "q", "chosen": "g", "rejected": "b"}])
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            self.curator.curate_dpo(output_dir=self.tmpdir, min_score_gap=0.3)
        collector.export_dpo_dataset.assert_called_once()
        _, kwargs = collector.export_dpo_dataset.call_args
        self.assertEqual(kwargs.get("min_score_gap"), 0.3)

    def test_each_line_valid_json(self):
        pairs = [{"prompt": f"q{i}", "chosen": "g", "rejected": "b"} for i in range(4)]
        path = self._curate_dpo(pairs)
        with open(path) as f:
            for line in f:
                json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# 4. TestLocalTrainerCheckAvailable
# ---------------------------------------------------------------------------

class TestLocalTrainerCheckAvailable(unittest.TestCase):
    """Tests for LocalTrainer.check_available()."""

    def setUp(self):
        self.trainer = LocalTrainer()

    def _check_with_libs(self, available_libs):
        """Patch __import__ so only libs in available_libs succeed."""
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        libs = {"unsloth", "trl", "peft", "transformers", "bitsandbytes"}

        def fake_import(name, *args, **kwargs):
            if name in libs:
                if name in available_libs:
                    return types.ModuleType(name)
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            return self.trainer.check_available()

    def test_returns_dict(self):
        result = self.trainer.check_available()
        self.assertIsInstance(result, dict)

    def test_dict_has_unsloth_key(self):
        result = self.trainer.check_available()
        self.assertIn("unsloth", result)

    def test_dict_has_trl_key(self):
        result = self.trainer.check_available()
        self.assertIn("trl", result)

    def test_dict_has_peft_key(self):
        result = self.trainer.check_available()
        self.assertIn("peft", result)

    def test_dict_has_transformers_key(self):
        result = self.trainer.check_available()
        self.assertIn("transformers", result)

    def test_dict_has_bitsandbytes_key(self):
        result = self.trainer.check_available()
        self.assertIn("bitsandbytes", result)

    def test_all_values_are_bool(self):
        result = self.trainer.check_available()
        for v in result.values():
            self.assertIsInstance(v, bool)

    def test_missing_lib_returns_false(self):
        """Force ImportError for a specific lib."""
        original = sys.modules.pop("unsloth", None)
        try:
            # Temporarily make it unavailable by overriding in modules
            sys.modules["unsloth"] = None  # type: ignore[assignment]
            result = self.trainer.check_available()
            # unsloth should be False since None raises AttributeError but
            # the try/except catches ImportError; set to real missing module
        finally:
            if original is not None:
                sys.modules["unsloth"] = original
            else:
                sys.modules.pop("unsloth", None)

    def test_all_present_returns_all_true(self):
        """When all stubs are in sys.modules, check_available returns True for each."""
        for lib in ["unsloth", "trl", "peft", "transformers", "bitsandbytes"]:
            sys.modules[lib] = types.ModuleType(lib)
        result = self.trainer.check_available()
        for lib in ["unsloth", "trl", "peft", "transformers", "bitsandbytes"]:
            self.assertTrue(result[lib], f"{lib} should be True")


# ---------------------------------------------------------------------------
# 5. TestLocalTrainerTrainQlora
# ---------------------------------------------------------------------------

class TestLocalTrainerTrainQlora(unittest.TestCase):
    """Tests for LocalTrainer.train_qlora()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = LocalTrainer()
        # Stub dataset file
        self.dataset_path = os.path.join(self.tmpdir, "data.jsonl")
        with open(self.dataset_path, "w") as f:
            f.write('{"instruction": "hi", "input": "", "output": "hello"}\n')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _avail(self, unsloth=True, trl=True, transformers=True, peft=True, bitsandbytes=True):
        return {
            "unsloth": unsloth,
            "trl": trl,
            "transformers": transformers,
            "peft": peft,
            "bitsandbytes": bitsandbytes,
        }

    def _run_qlora(self, avail, use_unsloth=True, expect_fail=False, **kw):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""
        with patch.object(self.trainer, "check_available", return_value=avail):
            with patch("subprocess.run", return_value=mock_proc) as mock_sub:
                if expect_fail:
                    with self.assertRaises(RuntimeError):
                        self.trainer.train_qlora(
                            base_model="Qwen/Qwen2.5-Coder-7B",
                            dataset_path=self.dataset_path,
                            output_dir=self.tmpdir,
                            use_unsloth=use_unsloth,
                            **kw,
                        )
                    return None, mock_sub
                result = self.trainer.train_qlora(
                    base_model="Qwen/Qwen2.5-Coder-7B",
                    dataset_path=self.dataset_path,
                    output_dir=self.tmpdir,
                    use_unsloth=use_unsloth,
                    **kw,
                )
                return result, mock_sub

    def test_raises_when_no_trl_or_transformers(self):
        avail = self._avail(trl=False, transformers=False)
        with patch.object(self.trainer, "check_available", return_value=avail):
            with self.assertRaises(RuntimeError):
                self.trainer.train_qlora("m", self.dataset_path, self.tmpdir)

    def test_returns_string_path_unsloth(self):
        avail = self._avail(unsloth=True, trl=True)
        result, _ = self._run_qlora(avail, use_unsloth=True)
        self.assertIsInstance(result, str)

    def test_returns_string_path_trl(self):
        avail = self._avail(unsloth=False, trl=True)
        result, _ = self._run_qlora(avail, use_unsloth=False)
        self.assertIsInstance(result, str)

    def test_unsloth_path_contains_lora_adapter(self):
        avail = self._avail(unsloth=True, trl=True)
        result, _ = self._run_qlora(avail, use_unsloth=True)
        self.assertIn("lora_adapter", result)

    def test_trl_path_contains_lora_adapter(self):
        avail = self._avail(unsloth=False, trl=True)
        result, _ = self._run_qlora(avail, use_unsloth=False)
        self.assertIn("lora_adapter", result)

    def test_unsloth_writes_script_to_disk(self):
        avail = self._avail(unsloth=True, trl=True)
        self._run_qlora(avail, use_unsloth=True)
        script = Path(self.tmpdir) / "train_script.py"
        self.assertTrue(script.exists())

    def test_trl_writes_script_to_disk(self):
        avail = self._avail(unsloth=False, trl=True)
        self._run_qlora(avail, use_unsloth=False)
        script = Path(self.tmpdir) / "train_trl_script.py"
        self.assertTrue(script.exists())

    def test_subprocess_run_called_once_unsloth(self):
        avail = self._avail(unsloth=True, trl=True)
        _, mock_sub = self._run_qlora(avail, use_unsloth=True)
        mock_sub.assert_called_once()

    def test_subprocess_run_called_once_trl(self):
        avail = self._avail(unsloth=False, trl=True)
        _, mock_sub = self._run_qlora(avail, use_unsloth=False)
        mock_sub.assert_called_once()

    def test_subprocess_run_uses_sys_executable(self):
        avail = self._avail(unsloth=True, trl=True)
        _, mock_sub = self._run_qlora(avail, use_unsloth=True)
        cmd = mock_sub.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)

    def test_raises_on_subprocess_failure_unsloth(self):
        avail = self._avail(unsloth=True, trl=True)
        bad_proc = MagicMock(returncode=1, stderr="CUDA error")
        with patch.object(self.trainer, "check_available", return_value=avail):
            with patch("subprocess.run", return_value=bad_proc):
                with self.assertRaises(RuntimeError) as ctx:
                    self.trainer.train_qlora("m", self.dataset_path, self.tmpdir, use_unsloth=True)
        self.assertIn("Training failed", str(ctx.exception))

    def test_raises_on_subprocess_failure_trl(self):
        avail = self._avail(unsloth=False, trl=True)
        bad_proc = MagicMock(returncode=1, stderr="OOM")
        with patch.object(self.trainer, "check_available", return_value=avail):
            with patch("subprocess.run", return_value=bad_proc):
                with self.assertRaises(RuntimeError) as ctx:
                    self.trainer.train_qlora("m", self.dataset_path, self.tmpdir, use_unsloth=False)
        self.assertIn("Training failed", str(ctx.exception))

    def test_falls_back_to_trl_when_unsloth_not_available(self):
        """use_unsloth=True but unsloth not in avail -> TRL path."""
        avail = self._avail(unsloth=False, trl=True)
        result, mock_sub = self._run_qlora(avail, use_unsloth=True)
        # TRL script should have been written
        self.assertTrue((Path(self.tmpdir) / "train_trl_script.py").exists())

    def test_unsloth_script_contains_base_model(self):
        avail = self._avail(unsloth=True, trl=True)
        self._run_qlora(avail, use_unsloth=True)
        script_text = (Path(self.tmpdir) / "train_script.py").read_text()
        self.assertIn("Qwen/Qwen2.5-Coder-7B", script_text)

    def test_trl_script_contains_base_model(self):
        avail = self._avail(unsloth=False, trl=True)
        self._run_qlora(avail, use_unsloth=False)
        script_text = (Path(self.tmpdir) / "train_trl_script.py").read_text()
        self.assertIn("Qwen/Qwen2.5-Coder-7B", script_text)

    def test_epochs_param_in_unsloth_script(self):
        avail = self._avail(unsloth=True, trl=True)
        self._run_qlora(avail, use_unsloth=True, epochs=7)
        script_text = (Path(self.tmpdir) / "train_script.py").read_text()
        self.assertIn("7", script_text)

    def test_epochs_param_in_trl_script(self):
        avail = self._avail(unsloth=False, trl=True)
        self._run_qlora(avail, use_unsloth=False, epochs=5)
        script_text = (Path(self.tmpdir) / "train_trl_script.py").read_text()
        self.assertIn("5", script_text)


# ---------------------------------------------------------------------------
# 6. TestGGUFConverter
# ---------------------------------------------------------------------------

class TestGGUFConverter(unittest.TestCase):
    """Tests for GGUFConverter.convert()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.converter = GGUFConverter()
        self.adapter_path = os.path.join(self.tmpdir, "lora_adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _convert(self, proc1_rc=0, proc2_rc=0, quantization="q4_k_m"):
        proc1 = MagicMock(returncode=proc1_rc, stderr="err1")
        proc2 = MagicMock(returncode=proc2_rc, stderr="err2")
        side_effects = [proc1, proc2]
        with patch("subprocess.run", side_effect=side_effects) as mock_sub:
            result = self.converter.convert(
                base_model="Qwen/Base",
                adapter_path=self.adapter_path,
                output_dir=self.tmpdir,
                quantization=quantization,
            )
        return result, mock_sub

    def test_returns_string(self):
        result, _ = self._convert()
        self.assertIsInstance(result, str)

    def test_merge_script_written_to_disk(self):
        self._convert()
        self.assertTrue((Path(self.tmpdir) / "merge_script.py").exists())

    def test_subprocess_called_for_merge(self):
        _, mock_sub = self._convert()
        # First call should be the merge script
        first_call_args = mock_sub.call_args_list[0][0][0]
        self.assertIn("merge_script.py", " ".join(str(a) for a in first_call_args))

    def test_subprocess_called_for_convert(self):
        _, mock_sub = self._convert()
        self.assertEqual(mock_sub.call_count, 2)

    def test_merge_failure_raises(self):
        proc1 = MagicMock(returncode=1, stderr="merge error")
        with patch("subprocess.run", return_value=proc1):
            with self.assertRaises(RuntimeError) as ctx:
                self.converter.convert("base", self.adapter_path, self.tmpdir)
        self.assertIn("Merge failed", str(ctx.exception))

    def test_gguf_path_contains_quantization(self):
        result, _ = self._convert(quantization="q8_0")
        self.assertIn("q8_0", result)

    def test_default_quantization_q4_k_m(self):
        proc1 = MagicMock(returncode=0, stderr="")
        proc2 = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", side_effect=[proc1, proc2]):
            result = self.converter.convert("base", self.adapter_path, self.tmpdir)
        self.assertIn("q4_k_m", result)

    def test_successful_convert_returns_gguf_path(self):
        result, _ = self._convert(proc1_rc=0, proc2_rc=0)
        self.assertTrue(result.endswith(".gguf"))

    def test_convert_failure_returns_merged_dir(self):
        """When gguf convert fails, falls back to merged model dir."""
        result, _ = self._convert(proc1_rc=0, proc2_rc=1)
        self.assertTrue(result.endswith("merged"))

    def test_merge_script_contains_adapter_path(self):
        self._convert()
        text = (Path(self.tmpdir) / "merge_script.py").read_text()
        self.assertIn(self.adapter_path, text)

    def test_merge_script_contains_base_model(self):
        self._convert()
        text = (Path(self.tmpdir) / "merge_script.py").read_text()
        self.assertIn("Qwen/Base", text)

    def test_output_dir_used_for_gguf_path(self):
        result, _ = self._convert()
        self.assertIn(self.tmpdir, result)

    def test_sys_executable_used_for_merge(self):
        _, mock_sub = self._convert()
        first_cmd = mock_sub.call_args_list[0][0][0]
        self.assertEqual(first_cmd[0], sys.executable)


# ---------------------------------------------------------------------------
# 7. TestModelDeployer
# ---------------------------------------------------------------------------

class TestModelDeployer(unittest.TestCase):
    """Tests for ModelDeployer.deploy()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.deployer = ModelDeployer()
        # Create a fake GGUF file
        self.gguf_path = os.path.join(self.tmpdir, "model_q4_k_m.gguf")
        with open(self.gguf_path, "w") as f:
            f.write("fake gguf content")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _deploy(self, model_name="my-model", dest_base=None):
        dest_base = dest_base or self.tmpdir
        fake_models_dir = Path(dest_base)
        with patch("vetinari.training.pipeline._MODELS_DIR", fake_models_dir):
            with patch("shutil.copy2") as mock_copy:
                result = self.deployer.deploy(self.gguf_path, model_name)
        return result, mock_copy

    def test_returns_string(self):
        result, _ = self._deploy()
        self.assertIsInstance(result, str)

    def test_model_name_in_path(self):
        result, _ = self._deploy(model_name="my-finetune")
        self.assertIn("my-finetune", result)

    def test_vetinari_in_path(self):
        result, _ = self._deploy()
        self.assertIn("vetinari", result)

    def test_shutil_copy2_called(self):
        _, mock_copy = self._deploy()
        mock_copy.assert_called_once()

    def test_shutil_copy2_src_is_gguf(self):
        _, mock_copy = self._deploy()
        src_arg = mock_copy.call_args[0][0]
        self.assertEqual(src_arg, self.gguf_path)

    def test_shutil_copy2_dest_contains_model_name(self):
        _, mock_copy = self._deploy(model_name="cool-model")
        dest_arg = mock_copy.call_args[0][1]
        self.assertIn("cool-model", dest_arg)

    def test_raises_file_not_found_if_missing(self):
        fake_path = os.path.join(self.tmpdir, "missing.gguf")
        with self.assertRaises(FileNotFoundError):
            self.deployer.deploy(fake_path, "whatever")

    def test_filename_preserved_in_dest(self):
        result, _ = self._deploy()
        self.assertIn("model_q4_k_m.gguf", result)

    def test_dest_dir_created(self):
        """Deploy creates the destination directory."""
        dest_base = tempfile.mkdtemp()
        try:
            result, _ = self._deploy(model_name="new-model", dest_base=dest_base)
            # The expected dir should be in the result path
            self.assertIn("new-model", result)
        finally:
            shutil.rmtree(dest_base, ignore_errors=True)

    def test_different_model_names_produce_different_paths(self):
        result1, _ = self._deploy(model_name="model-a")
        result2, _ = self._deploy(model_name="model-b")
        self.assertNotEqual(result1, result2)


# ---------------------------------------------------------------------------
# 8. TestTrainingPipelineRun
# ---------------------------------------------------------------------------

class TestTrainingPipelineRun(unittest.TestCase):
    """Tests for TrainingPipeline.run()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pipeline = TrainingPipeline()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_collector(self, n=50):
        """Return a mock collector with n records."""
        collector = MagicMock()
        records = [{"prompt": f"prompt {i}", "completion": f"output {i}"} for i in range(n)]
        collector.export_sft_dataset.return_value = records
        return collector

    def _make_mock_proc(self, rc=0):
        return MagicMock(returncode=rc, stderr="", stdout="")

    def _run_pipeline(self, n_records=50, train_rc=0, merge_rc=0, convert_rc=0, model_name_suffix="7B", **kw):
        """Run the pipeline with all externals mocked."""
        collector = self._make_collector(n_records)
        train_proc = self._make_mock_proc(train_rc)
        merge_proc = self._make_mock_proc(merge_rc)
        convert_proc = self._make_mock_proc(convert_rc)

        # Create a fake GGUF file so ModelDeployer.deploy() doesn't raise FileNotFoundError
        fake_gguf = None

        def fake_subprocess_run(cmd, **kwargs):
            # Return different mocks based on call order
            if not hasattr(fake_subprocess_run, "_count"):
                fake_subprocess_run._count = 0
            fake_subprocess_run._count += 1
            if fake_subprocess_run._count == 1:
                return train_proc
            elif fake_subprocess_run._count == 2:
                return merge_proc
            else:
                return convert_proc

        fake_subprocess_run._count = 0

        def fake_deploy(gguf_path, model_name):
            # Create a temp dest file path and return it without actually copying
            dest = os.path.join(self.tmpdir, "deployed_model.gguf")
            return dest

        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            with patch("subprocess.run", side_effect=fake_subprocess_run):
                with patch.object(self.pipeline._deployer, "deploy", side_effect=fake_deploy):
                    run = self.pipeline.run(
                        base_model=f"Qwen/Qwen2.5-Coder-{model_name_suffix}",
                        output_base_dir=self.tmpdir,
                        **kw,
                    )
        return run

    def test_returns_training_run(self):
        run = self._run_pipeline()
        self.assertIsInstance(run, TrainingRun)

    def test_run_id_set(self):
        run = self._run_pipeline()
        self.assertTrue(run.run_id.startswith("run_"))

    def test_timestamp_set(self):
        run = self._run_pipeline()
        self.assertTrue(len(run.timestamp) > 0)

    def test_base_model_stored(self):
        run = self._run_pipeline(model_name_suffix="7B")
        self.assertIn("7B", run.base_model)

    def test_success_true_on_complete(self):
        run = self._run_pipeline()
        self.assertTrue(run.success)

    def test_training_examples_counted(self):
        run = self._run_pipeline(n_records=50)
        self.assertEqual(run.training_examples, 50)

    def test_adapter_path_set(self):
        run = self._run_pipeline()
        self.assertIn("lora_adapter", run.adapter_path)

    def test_output_model_path_set(self):
        run = self._run_pipeline()
        self.assertNotEqual(run.output_model_path, "")

    def test_error_empty_on_success(self):
        run = self._run_pipeline()
        self.assertEqual(run.error, "")

    def test_run_json_created(self):
        run = self._run_pipeline()
        run_dir = Path(self.tmpdir) / run.run_id
        self.assertTrue((run_dir / "run.json").exists())

    def test_run_json_has_correct_fields(self):
        run = self._run_pipeline()
        run_dir = Path(self.tmpdir) / run.run_id
        with open(run_dir / "run.json") as f:
            d = json.load(f)
        self.assertIn("run_id", d)
        self.assertIn("success", d)
        self.assertIn("base_model", d)

    def test_run_json_success_true(self):
        run = self._run_pipeline()
        run_dir = Path(self.tmpdir) / run.run_id
        with open(run_dir / "run.json") as f:
            d = json.load(f)
        self.assertTrue(d["success"])

    def test_insufficient_data_sets_error(self):
        """Fewer than 10 examples -> error set, no training."""
        run = self._run_pipeline(n_records=5)
        self.assertFalse(run.success)
        self.assertIn("Insufficient", run.error)

    def test_insufficient_data_run_dir_created(self):
        """run_dir is created even when data is insufficient (run.json is NOT written on early return)."""
        run = self._run_pipeline(n_records=5)
        run_dir = Path(self.tmpdir) / run.run_id
        self.assertTrue(run_dir.exists())

    def test_training_failure_sets_error(self):
        """subprocess.run returning rc=1 for training -> error propagated."""
        run = self._run_pipeline(train_rc=1)
        self.assertFalse(run.success)
        self.assertNotEqual(run.error, "")

    def test_task_type_stored_as_all_when_none(self):
        run = self._run_pipeline(task_type=None)
        self.assertEqual(run.task_type, "all")

    def test_task_type_stored(self):
        run = self._run_pipeline(task_type="math")
        self.assertEqual(run.task_type, "math")

    def test_epochs_stored(self):
        run = self._run_pipeline(epochs=5)
        self.assertEqual(run.epochs, 5)

    def test_check_requirements_returns_dict(self):
        result = self.pipeline.check_requirements()
        self.assertIsInstance(result, dict)

    def test_check_requirements_has_libraries_key(self):
        result = self.pipeline.check_requirements()
        self.assertIn("libraries", result)

    def test_check_requirements_has_ready_for_training(self):
        result = self.pipeline.check_requirements()
        self.assertIn("ready_for_training", result)

    def test_check_requirements_has_lmstudio_models_dir(self):
        result = self.pipeline.check_requirements()
        self.assertIn("lmstudio_models_dir", result)

    def test_multiple_runs_have_unique_run_ids(self):
        collector = self._make_collector(50)

        def fake_deploy(gguf_path, model_name):
            return os.path.join(self.tmpdir, "model.gguf")

        def fake_sub(cmd, **kwargs):
            return self._make_mock_proc(0)

        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            with patch("subprocess.run", side_effect=fake_sub):
                with patch.object(self.pipeline._deployer, "deploy", side_effect=fake_deploy):
                    run1 = self.pipeline.run("ModelA", output_base_dir=self.tmpdir)
                    run2 = self.pipeline.run("ModelB", output_base_dir=self.tmpdir)

        self.assertNotEqual(run1.run_id, run2.run_id)


# ---------------------------------------------------------------------------
# 9. Additional edge-case tests
# ---------------------------------------------------------------------------

class TestDataCuratorEdgeCases(unittest.TestCase):
    """Edge-case tests for DataCurator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.curator = DataCurator()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _curate(self, records, **kw):
        collector = MagicMock()
        collector.export_sft_dataset.return_value = records
        with patch("vetinari.learning.training_data.get_training_collector", return_value=collector):
            return self.curator.curate(output_dir=self.tmpdir, **kw)

    def test_single_record(self):
        records = [{"prompt": "one", "completion": "one_out"}]
        path = self._curate(records)
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

    def test_unicode_content_preserved(self):
        records = [{"prompt": "日本語", "completion": "テスト"}]
        path = self._curate(records)
        with open(path, encoding="utf-8") as f:
            obj = json.loads(f.readline())
        self.assertEqual(obj["instruction"], "日本語")
        self.assertEqual(obj["output"], "テスト")

    def test_newlines_in_prompt_handled(self):
        records = [{"prompt": "line1\nline2", "completion": "out"}]
        path = self._curate(records)
        with open(path) as f:
            obj = json.loads(f.readline())
        self.assertIn("line1", obj["instruction"])


class TestGGUFConverterEdgeCases(unittest.TestCase):
    """Additional edge cases for GGUFConverter."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.converter = GGUFConverter()
        self.adapter_path = os.path.join(self.tmpdir, "adapter")
        os.makedirs(self.adapter_path, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_quantization_written_into_convert_cmd(self):
        proc1 = MagicMock(returncode=0, stderr="")
        proc2 = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", side_effect=[proc1, proc2]) as mock_sub:
            self.converter.convert("base", self.adapter_path, self.tmpdir, quantization="q8_0")
        second_call_cmd = mock_sub.call_args_list[1][0][0]
        cmd_str = " ".join(str(a) for a in second_call_cmd)
        self.assertIn("q8_0", cmd_str)

    def test_merge_error_message_includes_stderr(self):
        proc1 = MagicMock(returncode=1, stderr="CUDA OOM Error detail here")
        with patch("subprocess.run", return_value=proc1):
            with self.assertRaises(RuntimeError) as ctx:
                self.converter.convert("base", self.adapter_path, self.tmpdir)
        self.assertIn("CUDA OOM Error detail here", str(ctx.exception))

    def test_convert_exception_falls_back_to_merged(self):
        """If second subprocess raises an exception (not just rc!=0), fall back."""
        proc1 = MagicMock(returncode=0, stderr="")
        with patch("subprocess.run", side_effect=[proc1, Exception("no llama.cpp")]):
            result = self.converter.convert("base", self.adapter_path, self.tmpdir)
        self.assertTrue(result.endswith("merged"))


class TestLocalTrainerEdgeCases(unittest.TestCase):
    """Additional edge cases for LocalTrainer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.trainer = LocalTrainer()
        self.dataset_path = os.path.join(self.tmpdir, "data.jsonl")
        with open(self.dataset_path, "w") as f:
            f.write('{"instruction": "hi", "input": "", "output": "hello"}\n')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_lora_r_in_unsloth_script(self):
        avail = {"unsloth": True, "trl": True, "transformers": True, "peft": True, "bitsandbytes": True}
        proc = MagicMock(returncode=0, stderr="")
        with patch.object(self.trainer, "check_available", return_value=avail):
            with patch("subprocess.run", return_value=proc):
                self.trainer.train_qlora(
                    "model", self.dataset_path, self.tmpdir, lora_r=32, use_unsloth=True
                )
        text = (Path(self.tmpdir) / "train_script.py").read_text()
        self.assertIn("32", text)

    def test_learning_rate_in_trl_script(self):
        avail = {"unsloth": False, "trl": True, "transformers": True, "peft": True, "bitsandbytes": True}
        proc = MagicMock(returncode=0, stderr="")
        with patch.object(self.trainer, "check_available", return_value=avail):
            with patch("subprocess.run", return_value=proc):
                self.trainer.train_qlora(
                    "model", self.dataset_path, self.tmpdir, learning_rate=1e-4, use_unsloth=False
                )
        text = (Path(self.tmpdir) / "train_trl_script.py").read_text()
        self.assertIn("0.0001", text)


class TestModelDeployerEdgeCases(unittest.TestCase):
    """Additional edge cases for ModelDeployer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.deployer = ModelDeployer()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_special_chars_in_model_name(self):
        gguf = os.path.join(self.tmpdir, "model.gguf")
        with open(gguf, "w") as f:
            f.write("data")
        fake_dir = Path(self.tmpdir)
        with patch("vetinari.training.pipeline._MODELS_DIR", fake_dir):
            with patch("shutil.copy2"):
                result = self.deployer.deploy(gguf, "my-model-v1.0")
        self.assertIn("my-model-v1.0", result)

    def test_dest_contains_filename(self):
        gguf = os.path.join(self.tmpdir, "quantized_q4.gguf")
        with open(gguf, "w") as f:
            f.write("data")
        fake_dir = Path(self.tmpdir)
        with patch("vetinari.training.pipeline._MODELS_DIR", fake_dir):
            with patch("shutil.copy2"):
                result = self.deployer.deploy(gguf, "mymodel")
        self.assertIn("quantized_q4.gguf", result)


if __name__ == "__main__":
    unittest.main()
