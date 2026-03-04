"""Tests for vetinari/utils.py — Phase 7A"""
import logging
import os
import tempfile
import unittest


class TestSetupLogging(unittest.TestCase):
    def test_creates_log_directory(self):
        with tempfile.TemporaryDirectory() as d:
            log_dir = os.path.join(d, "logs", "sub")
            from vetinari.utils import setup_logging
            setup_logging(level=logging.WARNING, log_dir=log_dir)
            self.assertTrue(os.path.isdir(log_dir))

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as d:
            from vetinari.utils import setup_logging
            setup_logging(level=logging.WARNING, log_dir=d)
            self.assertTrue(os.path.exists(os.path.join(d, "vetinari.log")))


class TestLoadYaml(unittest.TestCase):
    def test_load_valid_yaml(self):
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                        delete=False, encoding="utf-8") as f:
            yaml.dump({"key": "value", "num": 42}, f)
            path = f.name
        try:
            from vetinari.utils import load_yaml
            data = load_yaml(path)
            self.assertEqual(data["key"], "value")
            self.assertEqual(data["num"], 42)
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        from vetinari.utils import load_yaml
        with self.assertRaises((FileNotFoundError, OSError)):
            load_yaml("/nonexistent/path/file.yaml")


class TestLoadConfig(unittest.TestCase):
    def test_load_config_is_alias_for_load_yaml(self):
        import yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                        delete=False, encoding="utf-8") as f:
            yaml.dump({"config": True}, f)
            path = f.name
        try:
            from vetinari.utils import load_config
            data = load_config(path)
            self.assertTrue(data["config"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
