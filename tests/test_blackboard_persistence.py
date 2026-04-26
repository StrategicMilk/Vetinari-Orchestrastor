"""Tests for blackboard mutation persistence."""

from __future__ import annotations

from vetinari.memory.blackboard import Blackboard, EntryState


def test_blackboard_mutations_restore_from_sqlite(tmp_path, monkeypatch) -> None:
    import vetinari.database as dbmod

    monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "blackboard.db"))
    dbmod.reset_for_testing()
    try:
        board = Blackboard(auto_restore=False)
        entry_id = board.post("Persist this work item", "code_search", "worker")
        assert board.complete(entry_id, result={"ok": True})

        restored = Blackboard(auto_restore=True)
        entry = restored.get_entry(entry_id)

        assert entry is not None
        assert entry.state == EntryState.COMPLETED
        assert entry.result == {"ok": True}
    finally:
        dbmod.reset_for_testing()
