"""Tests for vetinari.watch — file monitoring and directive scanning."""

from __future__ import annotations

import tempfile
from pathlib import Path

from vetinari.watch import (
    DIRECTIVE_PATTERNS,
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
        assert any(m is not None for m in matches)

    def test_block_comment_pattern(self):
        line = "/* @vetinari refactor this */"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        assert any(m is not None for m in matches)

    def test_case_insensitive(self):
        line = "# @Vetinari Fix This"
        matches = [p.search(line) for p in DIRECTIVE_PATTERNS]
        assert any(m is not None for m in matches)

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
            f.name
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
