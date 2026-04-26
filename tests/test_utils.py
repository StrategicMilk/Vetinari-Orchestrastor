"""Tests for vetinari/utils.py — Phase 7A"""

import logging
import os
import tempfile

import pytest


def _close_file_handlers() -> None:
    """Close and remove all FileHandlers from the root logger.

    Required on Windows: FileHandler keeps the log file open, which
    prevents TemporaryDirectory cleanup (WinError 32).
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root.removeHandler(h)


class TestSetupLogging:
    def test_creates_log_directory(self):
        with tempfile.TemporaryDirectory() as d:
            log_dir = os.path.join(d, "logs", "sub")
            from vetinari.utils import setup_logging

            setup_logging(level=logging.WARNING, log_dir=log_dir)
            try:
                assert os.path.isdir(log_dir)
            finally:
                # Close handlers before TemporaryDirectory.__exit__ to avoid WinError 32
                _close_file_handlers()

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as d:
            from vetinari.utils import setup_logging

            setup_logging(level=logging.WARNING, log_dir=d)
            try:
                assert os.path.exists(os.path.join(d, "vetinari.log"))
            finally:
                # Close handlers before TemporaryDirectory.__exit__ to avoid WinError 32
                _close_file_handlers()


class TestLoadYaml:
    def test_load_valid_yaml(self):
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump({"key": "value", "num": 42}, f)
            path = f.name
        try:
            from vetinari.utils import load_yaml

            data = load_yaml(path)
            assert data["key"] == "value"
            assert data["num"] == 42
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        from vetinari.utils import load_yaml

        with pytest.raises((FileNotFoundError, OSError)):
            load_yaml("/nonexistent/path/file.yaml")


class TestLoadConfig:
    def test_load_config_is_alias_for_load_yaml(self):
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump({"config": True}, f)
            path = f.name
        try:
            from vetinari.utils import load_config

            data = load_config(path)
            assert data["config"]
        finally:
            os.unlink(path)


class TestSetupLoggingReconfigurable:
    """33C.2 D4: second setup_logging() call must activate the new log_dir."""

    def test_second_call_changes_log_dir(self):
        """A second setup_logging(log_dir=X) call must write logs to X, not the first dir."""
        import os

        from vetinari.utils import setup_logging

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            setup_logging(level=logging.WARNING, log_dir=d1)
            setup_logging(level=logging.WARNING, log_dir=d2)
            # After second call, the active file handler must point to d2
            root = logging.getLogger()
            active_dirs = {
                os.path.dirname(h.baseFilename)
                for h in root.handlers
                if isinstance(h, logging.FileHandler)
            }
            # Close handlers before TemporaryDirectory.__exit__ to avoid WinError 32
            _close_file_handlers()

        assert d2 in active_dirs, (
            f"After second setup_logging(log_dir={d2!r}), expected d2 to be active. "
            f"Active dirs: {active_dirs}"
        )
