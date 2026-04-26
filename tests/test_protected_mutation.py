"""Tests for @protected_mutation decorator and ConfirmedIntent.

Covers: decorator raises without intent; with intent recycles target and emits
receipt; recycle=False path still records receipt; failure inside wrapped
function does NOT leave target half-deleted (recycle record is kept).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.safety.protected_mutation import (
    ConfirmedIntent,
    DestructiveAction,
    UnconfirmedDestructiveAction,
    protected_mutation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small file to use as a recycle target."""
    f = tmp_path / "target.txt"
    f.write_text("delete me", encoding="utf-8")
    return f


@pytest.fixture
def valid_intent() -> ConfirmedIntent:
    """A valid ConfirmedIntent for use in tests."""
    return ConfirmedIntent(confirmed_by="test-user", reason="test deletion")


def _make_recycle_store_patch(tmp_path: Path):
    """Return a context manager that patches RecycleStore with a tmp-backed instance."""
    from vetinari.safety.recycle import RecycleStore

    real_store = RecycleStore(root=tmp_path / "recycle_patch")

    return patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store)


# ---------------------------------------------------------------------------
# ConfirmedIntent validation
# ---------------------------------------------------------------------------


class TestConfirmedIntent:
    """ConfirmedIntent enforces non-empty confirmed_by and reason."""

    def test_valid_intent_constructs(self) -> None:
        """Valid ConfirmedIntent constructs without error."""
        intent = ConfirmedIntent(confirmed_by="alice", reason="clean up")
        assert intent.confirmed_by == "alice"
        assert intent.reason == "clean up"
        assert intent.confirmed_at_utc  # auto-set to now

    def test_empty_confirmed_by_raises(self) -> None:
        """Empty confirmed_by raises ValueError."""
        with pytest.raises(ValueError, match="confirmed_by"):
            ConfirmedIntent(confirmed_by="", reason="some reason")

    def test_whitespace_confirmed_by_raises(self) -> None:
        """Whitespace-only confirmed_by raises ValueError."""
        with pytest.raises(ValueError, match="confirmed_by"):
            ConfirmedIntent(confirmed_by="   ", reason="some reason")

    def test_empty_reason_raises(self) -> None:
        """Empty reason raises ValueError."""
        with pytest.raises(ValueError, match="reason"):
            ConfirmedIntent(confirmed_by="alice", reason="")

    def test_explicit_confirmed_at_is_preserved(self) -> None:
        """Explicitly supplied confirmed_at_utc is not overwritten."""
        ts = "2026-01-01T12:00:00+00:00"
        intent = ConfirmedIntent(confirmed_by="bob", reason="test", confirmed_at_utc=ts)
        assert intent.confirmed_at_utc == ts


# ---------------------------------------------------------------------------
# Decorator raises without intent
# ---------------------------------------------------------------------------


class TestDecoratorRaisesWithoutIntent:
    """Missing or wrong-type intent triggers UnconfirmedDestructiveAction."""

    def test_raises_without_intent(self, sample_file: Path, tmp_path: Path) -> None:
        """Calling without intent raises UnconfirmedDestructiveAction."""

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=False)
        def do_delete(path: Path) -> str:
            return "deleted"

        with pytest.raises(UnconfirmedDestructiveAction):
            do_delete(path=sample_file)

    def test_raises_with_wrong_intent_type(self, sample_file: Path) -> None:
        """Passing a non-ConfirmedIntent as intent raises UnconfirmedDestructiveAction."""

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=False)
        def do_delete(path: Path) -> str:
            return "deleted"

        with pytest.raises(UnconfirmedDestructiveAction):
            do_delete(path=sample_file, intent={"confirmed_by": "user", "reason": "oops"})

    def test_raises_with_none_intent(self, sample_file: Path) -> None:
        """Passing intent=None raises UnconfirmedDestructiveAction."""

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=False)
        def do_delete(path: Path) -> str:
            return "deleted"

        with pytest.raises(UnconfirmedDestructiveAction):
            do_delete(path=sample_file, intent=None)


# ---------------------------------------------------------------------------
# With intent: recycles target and emits receipt
# ---------------------------------------------------------------------------


class TestDecoratorWithIntent:
    """With valid intent, target is recycled and receipt is emitted."""

    def test_recycles_target_before_function(
        self, sample_file: Path, valid_intent: ConfirmedIntent, tmp_path: Path
    ) -> None:
        """The target path is moved to recycle before the function runs."""
        from vetinari.safety.recycle import RecycleStore

        real_store = RecycleStore(root=tmp_path / "recycle_patch")

        calls: list[str] = []

        @protected_mutation(
            DestructiveAction.DELETE_PROJECT,
            recycle=True,
            recycle_target_param="path",
        )
        def do_delete(path: Path) -> str:
            calls.append("fn_called")
            # By the time fn runs, path should already be gone (recycled)
            assert not path.exists(), "target must be recycled before fn executes"
            return "done"

        with patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store):
            with patch("vetinari.safety.protected_mutation._emit_receipt"):
                result = do_delete(path=sample_file, intent=valid_intent)

        assert result == "done"
        assert calls == ["fn_called"]
        assert not sample_file.exists()

    def test_function_return_value_is_preserved(
        self, sample_file: Path, valid_intent: ConfirmedIntent, tmp_path: Path
    ) -> None:
        """The wrapped function's return value passes through unchanged."""
        from vetinari.safety.recycle import RecycleStore

        real_store = RecycleStore(root=tmp_path / "recycle_patch")

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=True)
        def do_delete(path: Path) -> dict:
            return {"status": "ok", "path": str(path)}

        with patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store):
            with patch("vetinari.safety.protected_mutation._emit_receipt"):
                result = do_delete(path=sample_file, intent=valid_intent)

        assert result["status"] == "ok"

    def test_receipt_emitted_on_success(self, sample_file: Path, valid_intent: ConfirmedIntent, tmp_path: Path) -> None:
        """_emit_receipt is called with success=True after a clean run."""
        from vetinari.safety.recycle import RecycleStore

        real_store = RecycleStore(root=tmp_path / "recycle_patch")

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=True)
        def do_delete(path: Path) -> str:
            return "done"

        with patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store):
            with patch("vetinari.safety.protected_mutation._emit_receipt") as mock_emit:
                do_delete(path=sample_file, intent=valid_intent)

        mock_emit.assert_called_once()
        _, kwargs = mock_emit.call_args
        assert kwargs["success"] is True
        assert kwargs["action"] == DestructiveAction.DELETE_PROJECT

    def test_receipt_emitted_on_failure(self, sample_file: Path, valid_intent: ConfirmedIntent, tmp_path: Path) -> None:
        """_emit_receipt is called with success=False when fn raises."""
        from vetinari.safety.recycle import RecycleStore

        real_store = RecycleStore(root=tmp_path / "recycle_patch")

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=True)
        def do_delete(path: Path) -> str:
            raise RuntimeError("something went wrong")

        with patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store):
            with patch("vetinari.safety.protected_mutation._emit_receipt") as mock_emit:
                with pytest.raises(RuntimeError):
                    do_delete(path=sample_file, intent=valid_intent)

        mock_emit.assert_called_once()
        _, kwargs = mock_emit.call_args
        assert kwargs["success"] is False
        assert "something went wrong" in kwargs["error_msg"]


# ---------------------------------------------------------------------------
# recycle=False path still records receipt
# ---------------------------------------------------------------------------


class TestRecycleFalseStillEmitsReceipt:
    """When recycle=False the receipt is still emitted."""

    def test_no_recycle_but_receipt_emitted(self, sample_file: Path, valid_intent: ConfirmedIntent) -> None:
        """recycle=False skips recycling but still emits a receipt."""

        @protected_mutation(DestructiveAction.CLEAR_OUTPUTS, recycle=False)
        def do_clear(path: Path) -> str:
            return "cleared"

        with patch("vetinari.safety.protected_mutation._emit_receipt") as mock_emit:
            result = do_clear(path=sample_file, intent=valid_intent)

        assert result == "cleared"
        mock_emit.assert_called_once()
        _, kwargs = mock_emit.call_args
        assert kwargs["success"] is True
        # No recycle happened, so recycle_record_id is None
        assert kwargs["recycle_record_id"] is None

    def test_no_recycle_leaves_file_in_place(self, sample_file: Path, valid_intent: ConfirmedIntent) -> None:
        """When recycle=False, the file is not moved before fn executes."""

        @protected_mutation(DestructiveAction.CLEAR_OUTPUTS, recycle=False)
        def do_clear(path: Path) -> str:
            assert path.exists(), "file should still be there when recycle=False"
            return "cleared"

        with patch("vetinari.safety.protected_mutation._emit_receipt"):
            do_clear(path=sample_file, intent=valid_intent)


# ---------------------------------------------------------------------------
# Failure inside wrapped function does NOT lose the recycled entity
# ---------------------------------------------------------------------------


class TestFailureInsideFunctionKeepsRecycleRecord:
    """Recycle record is kept when the wrapped function raises."""

    def test_recycle_record_persists_after_fn_failure(
        self, sample_file: Path, valid_intent: ConfirmedIntent, tmp_path: Path
    ) -> None:
        """When fn raises, the recycle record is NOT rolled back."""
        from vetinari.safety.recycle import RecycleStore

        real_store = RecycleStore(root=tmp_path / "recycle_patch")

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=True)
        def do_delete(path: Path) -> str:
            raise RuntimeError("fn failed mid-way")

        with patch("vetinari.safety.protected_mutation.RecycleStore", return_value=real_store):
            with patch("vetinari.safety.protected_mutation._emit_receipt"):
                with pytest.raises(RuntimeError):
                    do_delete(path=sample_file, intent=valid_intent)

        # File was recycled before fn ran; it should be restorable
        records = real_store.list_all()
        assert len(records) == 1, "recycle record must persist after fn failure"
        assert str(sample_file) == records[0].original_path


# ---------------------------------------------------------------------------
# Async callable support
# ---------------------------------------------------------------------------


class TestAsyncDecorator:
    """@protected_mutation wraps async functions correctly."""

    def test_async_raises_without_intent(self, sample_file: Path) -> None:
        """Async decorated function raises without intent."""
        import asyncio

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=False)
        async def async_delete(path: Path) -> str:
            return "done"

        with pytest.raises(UnconfirmedDestructiveAction):
            asyncio.get_event_loop().run_until_complete(async_delete(path=sample_file))

    def test_async_succeeds_with_intent(self, sample_file: Path, valid_intent: ConfirmedIntent) -> None:
        """Async decorated function succeeds with valid intent."""
        import asyncio

        @protected_mutation(DestructiveAction.DELETE_PROJECT, recycle=False)
        async def async_delete(path: Path) -> str:
            return "done"

        with patch("vetinari.safety.protected_mutation._emit_receipt"):
            result = asyncio.get_event_loop().run_until_complete(async_delete(path=sample_file, intent=valid_intent))
        assert result == "done"
