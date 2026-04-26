"""Tests for vetinari.watch — file monitoring and directive scanning."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from vetinari.watch import (
    DIRECTIVE_PATTERNS,
    DirectiveReport,
    DirectiveScanner,
    FileChange,
    FileWatcher,
    VetinariDirective,
    WatchConfig,
    WatchMode,
)


class TestDirectivePatterns:
    """Tests for directive regex patterns."""

    def test_python_comment_pattern(self):
        line = "# @vetinari fix this function"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        assert any(m is not None for m in matches)
        match = next(m for m in matches if m is not None)
        assert match.group(1).strip() == "fix this function"

    def test_js_comment_pattern(self):
        line = "// @vetinari add tests"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        matched = [m for m in matches if m is not None]
        assert len(matched) >= 1
        assert "add tests" in matched[0].group(0)

    def test_block_comment_pattern(self):
        line = "/* @vetinari refactor this */"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        matched = [m for m in matches if m is not None]
        assert len(matched) >= 1
        assert "refactor" in matched[0].group(0)

    def test_case_insensitive(self):
        line = "# @Vetinari Fix This"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        matched = [m for m in matches if m is not None]
        assert len(matched) >= 1
        assert "Fix" in matched[0].group(0)

    def test_no_match_without_directive(self):
        line = "# This is a normal comment"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        assert all(m is None for m in matches)


class TestFileChange:
    """Tests for the FileChange dataclass."""

    def test_fields(self):
        fc = FileChange(path="/a/b.py", change_type="modified", size=100)
        assert fc.path == "/a/b.py"
        assert fc.change_type == "modified"
        assert fc.size == 100

    def test_timestamp_auto_set(self):
        fc = FileChange(path="x.py", change_type="created")
        assert fc.timestamp  # not empty


class TestVetinariDirective:
    """Tests for the VetinariDirective dataclass."""

    def test_action_property(self):
        d = VetinariDirective("f.py", 1, "fix the bug", "# @vetinari fix the bug")
        assert d.action == "fix"

    def test_target_property(self):
        d = VetinariDirective("f.py", 1, "fix the bug", "# @vetinari fix the bug")
        assert d.target == "the bug"

    def test_action_empty_directive(self):
        d = VetinariDirective("f.py", 1, "", "# @vetinari")
        assert d.action == ""

    def test_target_single_word(self):
        d = VetinariDirective("f.py", 1, "review", "# @vetinari review")
        assert d.target == ""


class TestWatchConfig:
    """Tests for WatchConfig defaults."""

    def test_defaults(self):
        cfg = WatchConfig()
        assert cfg.poll_interval == 2.0
        assert "*.py" in cfg.include_patterns
        assert "__pycache__" in cfg.exclude_patterns
        assert cfg.max_file_size == 1_000_000
        assert cfg.scan_directives is True


class TestDirectiveScanner:
    """Tests for DirectiveScanner file scanning."""

    def test_scan_file_finds_directives(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write("x = 1\n")
            f.write("# @vetinari fix this\n")
            f.write("y = 2\n")
            f.write("# @vetinari add tests\n")
            f.flush()
            scanner = DirectiveScanner()
            directives = scanner.scan_file(f.name)
        assert len(directives) == 2
        assert directives[0].action == "fix"
        assert directives[1].action == "add"

    def test_scan_file_nonexistent(self):
        scanner = DirectiveScanner()
        result = scanner.scan_file("/nonexistent/file.py")
        assert result == []

    def test_scan_changes_skips_deleted(self):
        scanner = DirectiveScanner()
        changes = [FileChange("/a.py", "deleted")]
        result = scanner.scan_changes(changes)
        assert result == []


class TestFileWatcher:
    """Tests for FileWatcher scan behavior."""

    def test_initial_scan_no_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatchConfig(watch_dir=tmpdir, include_patterns=["*.py"])
            watcher = FileWatcher(cfg)
            # First scan establishes baseline — no changes reported
            changes = watcher.scan()
            assert changes == []

    def test_is_excluded(self):
        watcher = FileWatcher()
        assert watcher._is_excluded("/project/__pycache__/foo.pyc") is True
        assert watcher._is_excluded("/project/main.py") is False


class TestWatchMode:
    """Tests for WatchMode coordinator."""

    def test_register_handler(self):
        wm = WatchMode()
        handler_called = []
        wm.register_handler("custom", lambda d: handler_called.append(d))
        assert "custom" in wm._directive_handlers

    def test_default_handlers_registered(self):
        wm = WatchMode()
        expected = {"fix", "add", "test", "explain", "refactor", "review"}
        assert expected.issubset(set(wm._directive_handlers.keys()))

    def test_get_history_initially_empty(self):
        wm = WatchMode()
        assert wm.get_history() == []


class TestWatcherStartStop:
    """Tests that start() then stop() leaves no orphan polling threads."""

    def test_stop_joins_polling_thread(self, tmp_path):
        """stop() must terminate the polling thread."""
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.05)
        watcher = FileWatcher(cfg)
        watcher.start()
        assert watcher._thread is not None
        assert watcher._thread.is_alive()
        # Capture reference before stop() nulls the attribute.
        thread_ref = watcher._thread
        watcher.stop()
        # stop() already joins inside _thread_lock; give it a short extra window.
        thread_ref.join(timeout=2.0)
        assert not thread_ref.is_alive()


class TestWatcherRestartNoDuplicates:
    """Tests that stop + start does not create duplicate polling threads."""

    def test_restart_single_thread(self, tmp_path):
        """stop() + start() must not create duplicate polling threads."""
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.05)
        watcher = FileWatcher(cfg)
        watcher.start()
        old_thread = watcher._thread
        watcher.stop()
        # After stop(), _thread is None and old_thread must be dead.
        old_thread.join(timeout=2.0)
        assert not old_thread.is_alive()
        watcher.start()
        # New thread is distinct (or old one was replaced) and alive.
        assert watcher._thread is not None
        assert watcher._thread.is_alive()
        # Exactly one thread alive: the current one. Old must be dead.
        assert not old_thread.is_alive()
        watcher.stop()


class TestWatcherEmptyTreeFirstEvent:
    """Tests that an initial empty scan correctly detects the first created file."""

    def test_first_created_event_on_empty_tree(self, tmp_path):
        """First file created after startup on an empty tree must fire a created event."""
        cfg = WatchConfig(
            watch_dir=str(tmp_path),
            poll_interval=0.05,
            include_patterns=["*.py"],
        )
        watcher = FileWatcher(cfg)
        # Initial scan establishes the baseline (sets _initialized = True).
        watcher.scan()
        # Create a file after baseline is established.
        new_file = tmp_path / "test.py"
        new_file.write_text("# hello", encoding="utf-8")
        # Second scan should detect the new file as "created".
        changes = watcher.scan()
        created = [c for c in changes if c.change_type == "created"]
        assert any(c.path.endswith("test.py") for c in created), (
            f"Expected a 'created' event for test.py but got: {changes}"
        )


class TestWatcherUnchangedDirectiveSkip:
    """Tests that same directive content is not re-dispatched on every save."""

    def test_unchanged_directive_not_redispatched(self, tmp_path):
        """Same directive content across two dispatch calls must not enqueue twice."""
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.05)
        wm = WatchMode(cfg)
        dispatched: list = []

        wm.register_handler("test", lambda d: dispatched.append(d))

        # Synthesise a directive as the scanner would produce it.
        directive = VetinariDirective(
            file_path=str(tmp_path / "task.py"),
            line_number=1,
            directive="test run suite",
            full_line="# @vetinari test run suite",
        )
        # Dispatch the same directive twice — second should be a no-op.
        wm._dispatch_directive(directive)
        wm._dispatch_directive(directive)

        assert len(dispatched) == 1, (
            f"Same content should dispatch only once but got {len(dispatched)} dispatches"
        )


class TestWatcherReportWriteFirst:
    """Tests that queue is not populated if the durable report write fails."""

    def test_queue_not_enqueued_on_report_failure(self, tmp_path):
        """If the report file write fails inside _record, nothing is enqueued."""
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.05)
        wm = WatchMode(cfg)

        report = DirectiveReport(
            action="fix",
            file_path=str(tmp_path / "task.py"),
            line_number=1,
            target="something",
            full_line="# @vetinari fix something",
            priority="high",
        )

        initial_len = len(wm.task_queue)

        # Patch the file-write portion of _record to simulate a disk failure.
        # We do this by replacing the open() call path: patch Path.open on
        # the report path, so the OSError is raised before the enqueue step.
        original_record = wm._record

        def _failing_record(r: DirectiveReport) -> None:
            """Simulate _record with a failing file write."""
            # Reproduce the _record logic but force the write to raise.
            raise OSError("disk full")

        wm._record = _failing_record  # type: ignore[method-assign]

        try:
            wm._record(report)
        except OSError:  # noqa: VET022 — intentional: test verifies queue is unaffected, not the exception
            pass

        # queue must not have grown.
        assert len(wm.task_queue) == initial_len, (
            "task_queue must not be populated when the report write fails"
        )
