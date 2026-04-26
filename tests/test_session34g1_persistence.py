"""Regression tests for Session 34G1 persistence fail-closed invariants."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


def _table_columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    finally:
        conn.close()


def test_unified_schema_initializes_each_vetinari_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching VETINARI_DB_PATH must initialize the new physical database."""
    from vetinari import database as db

    db.reset_for_testing()
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"

    try:
        monkeypatch.setenv("VETINARI_DB_PATH", str(first))
        conn = db.get_connection()
        assert conn.execute("SELECT COUNT(*) FROM execution_state").fetchone()[0] == 0
        db.close_connection()

        monkeypatch.setenv("VETINARI_DB_PATH", str(second))
        conn = db.get_connection()
        assert conn.execute("SELECT COUNT(*) FROM execution_state").fetchone()[0] == 0
    finally:
        db.reset_for_testing()


def test_migration_schema_matches_plan_tracking_writer_columns(tmp_path: Path) -> None:
    """`vetinari migrate` must create columns used by plan-tracking writers."""
    from vetinari.migrations.runner import run_migrations

    db_path = tmp_path / "plans.db"
    run_migrations(db_path)

    assert "plan_explanation_json" in _table_columns(db_path, "PlanHistory")
    subtask_columns = _table_columns(db_path, "SubtaskMemory")
    assert "subtask_explanation_json" in subtask_columns
    assert "quality_score" in subtask_columns


def test_migration_history_rejects_changed_same_filename(tmp_path: Path) -> None:
    """Migration history must not treat a changed file with the same name as clean."""
    from vetinari.migrations.runner import run_migrations

    runner_mod = sys.modules[run_migrations.__module__]
    original_dir = runner_mod._MIGRATIONS_DIR
    original_schema = runner_mod._SCHEMA_FILE
    runner_mod._MIGRATIONS_DIR = tmp_path
    runner_mod._SCHEMA_FILE = tmp_path / "schema.sql"
    migration = tmp_path / "001_same_name.sql"
    db_path = tmp_path / "history.db"

    try:
        migration.write_text("CREATE TABLE one (id INTEGER PRIMARY KEY);", encoding="utf-8")
        assert run_migrations(db_path) == 1
        migration.write_text("CREATE TABLE two (id INTEGER PRIMARY KEY);", encoding="utf-8")
        with pytest.raises(RuntimeError, match="has changed since it was applied"):
            run_migrations(db_path)
    finally:
        runner_mod._MIGRATIONS_DIR = original_dir
        runner_mod._SCHEMA_FILE = original_schema


def test_migration_failure_rolls_back_script_and_history(tmp_path: Path) -> None:
    """A failed migration must not leave its table or history row behind."""
    from vetinari.migrations.runner import run_migrations

    runner_mod = sys.modules[run_migrations.__module__]
    original_dir = runner_mod._MIGRATIONS_DIR
    original_schema = runner_mod._SCHEMA_FILE
    runner_mod._MIGRATIONS_DIR = tmp_path
    runner_mod._SCHEMA_FILE = tmp_path / "schema.sql"
    (tmp_path / "001_partial.sql").write_text(
        "CREATE TABLE partial_table (id INTEGER PRIMARY KEY);\nTHIS IS NOT SQL;",
        encoding="utf-8",
    )
    db_path = tmp_path / "partial.db"

    try:
        with pytest.raises(sqlite3.Error):
            run_migrations(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            history = conn.execute("SELECT filename FROM _migration_history").fetchall()
        finally:
            conn.close()
        assert "partial_table" not in tables
        assert history == []
    finally:
        runner_mod._MIGRATIONS_DIR = original_dir
        runner_mod._SCHEMA_FILE = original_schema


def test_preferences_save_failure_propagates(tmp_path: Path) -> None:
    """Preference mutation must not report success when persistence fails."""
    from vetinari.web.preferences import PreferencesManager

    prefs_path = tmp_path / "user_preferences.json"
    prefs_path.mkdir()
    manager = PreferencesManager(path=prefs_path)

    with pytest.raises(OSError, match=r"Access is denied|Is a directory|Permission denied"):
        manager.set("theme", "light")


def test_autonomy_policy_malformed_file_refuses_default_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed autonomy policy YAML must fail closed and preserve the file."""
    from vetinari.web import litestar_autonomy_api as api

    bad_config = tmp_path / "autonomy_policies.yaml"
    bad_config.write_text("policies: [", encoding="utf-8")
    monkeypatch.setattr(api, "_CONFIG_PATH", bad_config)
    monkeypatch.setattr(api, "_policies_cache", None)

    with api._lock, pytest.raises(RuntimeError):
        api._load_policies()

    assert bad_config.read_text(encoding="utf-8") == "policies: ["


def test_approval_direct_decision_after_expiry_is_rejected_and_logged(tmp_path: Path) -> None:
    """approve()/reject() must enforce expiry even if get_pending() was never called."""
    from vetinari.autonomy.approval_queue import ApprovalQueue

    db_path = tmp_path / "approval.db"
    queue = ApprovalQueue(db_path=db_path, expiry_hours=0)
    action_id = queue.enqueue("dangerous_change", details={"x": 1}, confidence=0.2)

    assert queue.approve(action_id) is False
    log = queue.get_decision_log()
    assert log[0].action_id == action_id
    assert log[0].decision == "expired"

    conn = sqlite3.connect(str(db_path))
    try:
        status = conn.execute("SELECT status FROM approval_queue WHERE action_id = ?", (action_id,)).fetchone()[0]
    finally:
        conn.close()
    assert status == "expired"


def test_plan_tracking_json_fallback_recovers_corrupt_file(tmp_path: Path) -> None:
    """Corrupt JSON fallback files must not crash startup or block atomic rewrite."""
    from vetinari.memory.plan_tracking import MemoryStore

    json_path = tmp_path / "plan_memory.json"
    json_path.write_text("{not-json", encoding="utf-8")
    store = MemoryStore(db_path=str(tmp_path / "plan_memory.db"), use_json_fallback=True)

    assert store.get_memory_stats()["total_plans"] == 0
    assert store.write_plan_history({"plan_id": "p1", "goal": "g"}) is True
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert "p1" in saved["plans"]


@pytest.mark.skipif(importlib.util.find_spec("cryptography") is None, reason="cryptography unavailable")
def test_legacy_fernet_vault_decrypts_before_key_migration(tmp_path: Path) -> None:
    """Legacy Fernet vaults must be read before AES-GCM key replacement."""
    from cryptography.fernet import Fernet

    from vetinari.credentials import Credential, CredentialVault

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    legacy_key = Fernet.generate_key()
    payload = json.dumps({
        "github": asdict(Credential(source_type="github", credential_type="bearer", token="legacy-token"))
    }).encode("utf-8")
    (vault_path / ".key").write_bytes(legacy_key)
    (vault_path / "credentials.enc").write_bytes(Fernet(legacy_key).encrypt(payload))

    vault = CredentialVault(vault_path=str(vault_path))
    assert vault.get_token("github") == "legacy-token"
    assert len((vault_path / ".key").read_bytes()) == 32

    reopened = CredentialVault(vault_path=str(vault_path))
    assert reopened.get_token("github") == "legacy-token"
