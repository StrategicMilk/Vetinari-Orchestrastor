"""Regression tests for database lifecycle logging."""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch


def test_database_safe_log_suppresses_closed_stream_handler_noise(capsys) -> None:
    """Late database lifecycle logs must not print logging tracebacks at teardown."""
    import vetinari.database as db

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    original_propagate = db.logger.propagate
    db.logger.addHandler(handler)
    db.logger.propagate = False

    try:
        stream.close()
        db._safe_log(logging.INFO, "schema initialized after capture closed")
    finally:
        db.logger.removeHandler(handler)
        db.logger.propagate = original_propagate

    captured = capsys.readouterr()
    assert "--- Logging error ---" not in captured.err


def test_database_safe_log_skips_low_value_background_thread_records() -> None:
    """Background DB lifecycle INFO/DEBUG records should not outlive test capture."""
    import vetinari.database as db

    handler = MagicMock(spec=logging.Handler)
    original_propagate = db.logger.propagate
    db.logger.addHandler(handler)
    db.logger.propagate = False

    try:
        with (
            patch.object(db.threading, "current_thread", return_value=object()),
            patch.object(db.threading, "main_thread", return_value=object()),
        ):
            db._safe_log(logging.INFO, "late schema init")
    finally:
        db.logger.removeHandler(handler)
        db.logger.propagate = original_propagate

    handler.handle.assert_not_called()
