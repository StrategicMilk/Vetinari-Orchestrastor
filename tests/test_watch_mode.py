"""Tests for vetinari.watch — Watch Mode file monitoring and directive scanning."""

import os
import time
import textwrap
from pathlib import Path

import pytest

from vetinari.watch import (
    DIRECTIVE_PATTERNS,
    DirectiveScanner,
    FileChange,
    FileWatcher,
    VetinariDirective,
    WatchConfig,
    WatchMode,
)


# ── FileChange dataclass ──────────────────────────────────────────────

class TestFileChange:
    def test_basic_creation(self):
        fc = FileChange(path="/tmp/foo.py", change_type="modified", size=42)
        assert fc.path == "/tmp/foo.py"
        assert fc.change_type == "modified"
        assert fc.size == 42

    def test_default_timestamp(self):
        fc = FileChange(path="a.py", change_type="created")
        assert fc.timestamp  # non-empty ISO string
        assert "T" in fc.timestamp

    def test_default_size(self):
        fc = FileChange(path="a.py", change_type="deleted")
        assert fc.size == 0


# ── VetinariDirective properties ──────────────────────────────────────

class TestVetinariDirective:
    def test_action_property(self):
        d = VetinariDirective(
            file_path="x.py", line_number=1,
            directive="fix the broken parser", full_line="# @vetinari fix the broken parser",
        )
        assert d.action == "fix"

    def test_target_property(self):
        d = VetinariDirective(
            file_path="x.py", line_number=1,
            directive="fix the broken parser", full_line="# @vetinari fix the broken parser",
        )
        assert d.target == "the broken parser"

    def test_action_single_word(self):
        d = VetinariDirective(
            file_path="x.py", line_number=5,
            directive="review", full_line="# @vetinari review",
        )
        assert d.action == "review"
        assert d.target == ""

    def test_empty_directive(self):
        d = VetinariDirective(
            file_path="x.py", line_number=1,
            directive="", full_line="# @vetinari",
        )
        assert d.action == ""
        assert d.target == ""


# ── WatchConfig defaults ─────────────────────────────────────────────

class TestWatchConfig:
    def test_defaults(self):
        cfg = WatchConfig()
        assert cfg.watch_dir == "."
        assert cfg.poll_interval == 2.0
        assert "*.py" in cfg.include_patterns
        assert "__pycache__" in cfg.exclude_patterns
        assert cfg.max_file_size == 1_000_000
        assert cfg.scan_directives is True

    def test_custom_values(self):
        cfg = WatchConfig(watch_dir="/src", poll_interval=0.5, include_patterns=["*.rs"])
        assert cfg.watch_dir == "/src"
        assert cfg.poll_interval == 0.5
        assert cfg.include_patterns == ["*.rs"]


# ── FileWatcher scan ─────────────────────────────────────────────────

class TestFileWatcher:
    def test_initial_scan_no_changes(self, tmp_path):
        """First scan establishes baseline — no changes reported."""
        (tmp_path / "hello.py").write_text("x = 1\n")
        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        watcher = FileWatcher(cfg)
        changes = watcher.scan()
        assert changes == []  # first scan is baseline

    def test_modification_detected(self, tmp_path):
        """Second scan after file edit reports 'modified'."""
        f = tmp_path / "hello.py"
        f.write_text("x = 1\n")
        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        watcher = FileWatcher(cfg)
        watcher.scan()  # baseline

        time.sleep(0.05)
        f.write_text("x = 2\n")
        # Force mtime difference on fast filesystems
        os.utime(str(f), (time.time() + 1, time.time() + 1))

        changes = watcher.scan()
        assert len(changes) == 1
        assert changes[0].change_type == "modified"
        assert "hello.py" in changes[0].path

    def test_creation_detected(self, tmp_path):
        """New file after baseline is reported as 'created'."""
        (tmp_path / "existing.py").write_text("a = 1\n")
        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        watcher = FileWatcher(cfg)
        watcher.scan()  # baseline

        (tmp_path / "new_file.py").write_text("b = 2\n")
        changes = watcher.scan()
        assert any(c.change_type == "created" and "new_file.py" in c.path for c in changes)

    def test_deletion_detected(self, tmp_path):
        """Removed file after baseline is reported as 'deleted'."""
        f = tmp_path / "doomed.py"
        f.write_text("gone = True\n")
        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        watcher = FileWatcher(cfg)
        watcher.scan()  # baseline

        f.unlink()
        changes = watcher.scan()
        assert any(c.change_type == "deleted" and "doomed.py" in c.path for c in changes)

    def test_exclusion_logic(self, tmp_path):
        """Files matching exclude_patterns are ignored."""
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "mod.py").write_text("cached\n")
        (tmp_path / "good.py").write_text("ok\n")

        cfg = WatchConfig(
            watch_dir=str(tmp_path),
            include_patterns=["*.py"],
            exclude_patterns=["__pycache__"],
        )
        watcher = FileWatcher(cfg)
        watcher.scan()

        # Only good.py should be tracked, not the __pycache__ file
        assert any("good.py" in p for p in watcher._file_states)
        assert not any("__pycache__" in p for p in watcher._file_states)

    def test_start_stop(self, tmp_path):
        """Start/stop lifecycle works without errors."""
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.1)
        watcher = FileWatcher(cfg)
        assert not watcher.is_running

        watcher.start()
        assert watcher.is_running

        watcher.stop()
        assert not watcher.is_running

    def test_callback_invoked(self, tmp_path):
        """Registered callbacks are called when changes are detected."""
        f = tmp_path / "cb.py"
        f.write_text("v = 1\n")

        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"], poll_interval=0.1)
        watcher = FileWatcher(cfg)

        received = []
        watcher.on_change(lambda changes: received.extend(changes))

        watcher.scan()  # baseline
        f.write_text("v = 2\n")
        os.utime(str(f), (time.time() + 1, time.time() + 1))

        # Manually trigger a scan + callback dispatch (like the loop would)
        changes = watcher.scan()
        if changes:
            for cb in watcher._callbacks:
                cb(changes)

        assert len(received) >= 1
        assert received[0].change_type == "modified"


# ── DirectiveScanner ─────────────────────────────────────────────────

class TestDirectiveScanner:
    def test_scan_python_comment(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("# @vetinari fix the broken parser\nx = 1\n")

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))

        assert len(directives) == 1
        assert directives[0].action == "fix"
        assert directives[0].target == "the broken parser"
        assert directives[0].line_number == 1

    def test_scan_js_comment(self, tmp_path):
        f = tmp_path / "code.js"
        f.write_text("// @vetinari add error handling\nconst x = 1;\n")

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))

        assert len(directives) == 1
        assert directives[0].action == "add"
        assert directives[0].target == "error handling"

    def test_scan_block_comment(self, tmp_path):
        f = tmp_path / "code.css"
        f.write_text("/* @vetinari review this section */\n.cls { color: red; }\n")

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))

        assert len(directives) == 1
        assert directives[0].action == "review"
        assert directives[0].target == "this section"

    def test_case_insensitive(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("# @VETINARI refactor this function\n")

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))
        assert len(directives) == 1
        assert directives[0].action == "refactor"

    def test_multiple_directives(self, tmp_path):
        f = tmp_path / "multi.py"
        f.write_text(textwrap.dedent("""\
            # @vetinari fix memory leak
            x = 1
            # @vetinari add logging
            y = 2
            # @vetinari test edge cases
        """))

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))
        assert len(directives) == 3
        actions = [d.action for d in directives]
        assert actions == ["fix", "add", "test"]

    def test_no_directives(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("# Regular comment\nx = 1\n")

        scanner = DirectiveScanner()
        directives = scanner.scan_file(str(f))
        assert directives == []

    def test_scan_changes_skips_deleted(self, tmp_path):
        f = tmp_path / "exists.py"
        f.write_text("# @vetinari fix bug\n")

        changes = [
            FileChange(str(f), "modified"),
            FileChange("/gone.py", "deleted"),
        ]
        scanner = DirectiveScanner()
        directives = scanner.scan_changes(changes)

        assert len(directives) == 1
        assert directives[0].action == "fix"

    def test_skip_large_files(self, tmp_path):
        f = tmp_path / "big.py"
        f.write_text("# @vetinari fix\n")

        scanner = DirectiveScanner(max_file_size=5)  # 5 bytes — file is bigger
        directives = scanner.scan_file(str(f))
        assert directives == []

    def test_nonexistent_file(self):
        scanner = DirectiveScanner()
        directives = scanner.scan_file("/does/not/exist.py")
        assert directives == []


# ── Directive patterns ───────────────────────────────────────────────

class TestDirectivePatterns:
    @pytest.mark.parametrize("line,expected_action", [
        ("# @vetinari fix this", "fix"),
        ("  # @vetinari add tests", "add"),
        ("// @vetinari test edge cases", "test"),
        ("/* @vetinari explain logic */", "explain"),
        ("# @Vetinari refactor code", "refactor"),
        ("// @VETINARI review module", "review"),
    ])
    def test_pattern_matching(self, line, expected_action):
        matched = False
        for pattern in DIRECTIVE_PATTERNS:
            m = pattern.search(line)
            if m:
                action = m.group(1).strip().split()[0].lower()
                assert action == expected_action
                matched = True
                break
        assert matched, f"No pattern matched: {line!r}"


# ── WatchMode ────────────────────────────────────────────────────────

class TestWatchMode:
    def test_creation(self):
        wm = WatchMode()
        assert not wm.is_running
        assert wm.get_history() == []

    def test_default_handlers(self):
        wm = WatchMode()
        expected = {"fix", "add", "test", "explain", "refactor", "review"}
        assert set(wm._directive_handlers.keys()) == expected

    def test_register_custom_handler(self):
        wm = WatchMode()
        calls = []
        wm.register_handler("deploy", lambda d: calls.append(d))
        assert "deploy" in wm._directive_handlers

    def test_custom_handler_dispatch(self, tmp_path):
        """Custom handler is called when a matching directive is found."""
        f = tmp_path / "deploy.py"
        f.write_text("# @vetinari deploy to staging\n")

        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        wm = WatchMode(cfg)

        dispatched = []
        wm.register_handler("deploy", lambda d: dispatched.append(d))

        # Simulate changes
        changes = [FileChange(str(f), "modified")]
        wm._handle_changes(changes)

        assert len(dispatched) == 1
        assert dispatched[0].action == "deploy"
        assert dispatched[0].target == "to staging"

    def test_history_tracking(self, tmp_path):
        """Each change batch is recorded in history."""
        f = tmp_path / "hist.py"
        f.write_text("# @vetinari fix leak\nx = 1\n")

        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        wm = WatchMode(cfg)

        changes = [FileChange(str(f), "modified")]
        wm._handle_changes(changes)

        history = wm.get_history()
        assert len(history) == 1
        assert len(history[0]["changes"]) == 1
        assert history[0]["changes"][0]["type"] == "modified"
        assert len(history[0]["directives"]) == 1
        assert history[0]["directives"][0]["action"] == "fix"

    def test_history_no_directives(self, tmp_path):
        """Changes without directives still appear in history."""
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")

        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        wm = WatchMode(cfg)

        changes = [FileChange(str(f), "modified")]
        wm._handle_changes(changes)

        history = wm.get_history()
        assert len(history) == 1
        assert history[0]["directives"] == []

    def test_start_stop(self, tmp_path):
        cfg = WatchConfig(watch_dir=str(tmp_path), poll_interval=0.1)
        wm = WatchMode(cfg)
        wm.start()
        assert wm.is_running
        wm.stop()
        assert not wm.is_running

    def test_directive_scan_disabled(self, tmp_path):
        """When scan_directives=False, no directives are extracted."""
        f = tmp_path / "nodir.py"
        f.write_text("# @vetinari fix bug\n")

        cfg = WatchConfig(
            watch_dir=str(tmp_path),
            include_patterns=["*.py"],
            scan_directives=False,
        )
        wm = WatchMode(cfg)

        changes = [FileChange(str(f), "modified")]
        wm._handle_changes(changes)

        history = wm.get_history()
        assert history[0]["directives"] == []

    def test_unknown_action_logged(self, tmp_path, caplog):
        """Unknown directive actions produce a warning."""
        f = tmp_path / "weird.py"
        f.write_text("# @vetinari frobulate the widget\n")

        cfg = WatchConfig(watch_dir=str(tmp_path), include_patterns=["*.py"])
        wm = WatchMode(cfg)

        changes = [FileChange(str(f), "modified")]
        import logging
        with caplog.at_level(logging.WARNING):
            wm._handle_changes(changes)

        assert any("Unknown directive action" in r.message for r in caplog.records)
