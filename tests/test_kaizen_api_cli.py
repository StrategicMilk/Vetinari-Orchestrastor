"""Tests for kaizen API endpoints and CLI commands."""

from __future__ import annotations

import argparse

import pytest

from vetinari.cli import cmd_kaizen
from vetinari.kaizen.improvement_log import ImprovementLog


class TestKaizenCLI:
    """Test kaizen CLI commands execute without error."""

    def test_kaizen_report_command(self, tmp_path, monkeypatch, capsys):
        """'vetinari kaizen report' prints a summary without error."""
        db_path = tmp_path / "kaizen_cli_test.db"
        log = ImprovementLog(db_path)
        log.propose(
            hypothesis="Test",
            metric="quality",
            baseline=0.5,
            target=0.7,
            applied_by="test",
            rollback_plan="revert",
        )

        # Monkeypatch the DB path used by cmd_kaizen (now in cli_training)
        monkeypatch.setattr("vetinari.cli_training.KAIZEN_DB_PATH", db_path)

        args = argparse.Namespace(kaizen_action="report")
        # We can't easily monkeypatch the ImprovementLog creation,
        # so just test that the function returns 0 for report
        # by creating the DB at the expected path
        result = cmd_kaizen(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Kaizen Report" in captured.out

    def test_kaizen_gemba_command(self, tmp_path, monkeypatch, capsys):
        """'vetinari kaizen gemba' runs without error."""
        args = argparse.Namespace(kaizen_action="gemba")
        result = cmd_kaizen(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Gemba Walk" in captured.out

    def test_kaizen_no_action(self, capsys):
        """Missing action prints usage."""
        args = argparse.Namespace(kaizen_action=None)
        result = cmd_kaizen(args)
        assert result == 1
