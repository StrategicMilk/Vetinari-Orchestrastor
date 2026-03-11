"""Watch Mode — file monitoring and automatic task execution.

Monitors a codebase for file changes and reacts to special comments
like `# @vetinari fix this` or `# @vetinari add tests`.

Does NOT use the watchdog library (to avoid extra dependencies).
Uses polling-based file monitoring with configurable intervals.
"""

import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Patterns for vetinari directives in comments
DIRECTIVE_PATTERNS = [
    re.compile(r'#\s*@vetinari\s+(.+)', re.IGNORECASE),
    re.compile(r'//\s*@vetinari\s+(.+)', re.IGNORECASE),
    re.compile(r'/\*\s*@vetinari\s+(.+?)\s*\*/', re.IGNORECASE),
]


@dataclass
class FileChange:
    """A detected file change."""
    path: str
    change_type: str  # "modified", "created", "deleted"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    size: int = 0


@dataclass
class VetinariDirective:
    """A @vetinari directive found in source code."""
    file_path: str
    line_number: int
    directive: str  # The command after @vetinari
    full_line: str

    @property
    def action(self) -> str:
        """Extract action verb from directive."""
        parts = self.directive.strip().split()
        return parts[0].lower() if parts else ""

    @property
    def target(self) -> str:
        """Extract target from directive."""
        parts = self.directive.strip().split(None, 1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class WatchConfig:
    """Configuration for watch mode."""
    watch_dir: str = "."
    poll_interval: float = 2.0  # seconds
    include_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", ".git", "node_modules", ".venv", "*.pyc", ".vetinari",
    ])
    max_file_size: int = 1_000_000  # 1MB max for scanning
    scan_directives: bool = True


class FileWatcher:
    """Polling-based file watcher."""

    def __init__(self, config: WatchConfig = None):
        self._config = config or WatchConfig()
        self._file_states: Dict[str, float] = {}  # path -> mtime
        self._running = False
        self._callbacks: List[Callable[[List[FileChange]], None]] = []
        self._thread: Optional[threading.Thread] = None

    def on_change(self, callback: Callable[[List[FileChange]], None]) -> None:
        """Register a callback for file changes."""
        self._callbacks.append(callback)

    def scan(self) -> List[FileChange]:
        """Scan for file changes since last scan."""
        changes: List[FileChange] = []
        current_files: Set[str] = set()

        watch_path = Path(self._config.watch_dir)
        for pattern in self._config.include_patterns:
            for file_path in watch_path.rglob(pattern):
                str_path = str(file_path)

                # Check exclusions
                if self._is_excluded(str_path):
                    continue

                current_files.add(str_path)
                try:
                    mtime = file_path.stat().st_mtime
                    size = file_path.stat().st_size
                except OSError:
                    continue

                if str_path not in self._file_states:
                    if self._file_states:  # Only report creates after initial scan
                        changes.append(FileChange(str_path, "created", size=size))
                elif mtime > self._file_states[str_path]:
                    changes.append(FileChange(str_path, "modified", size=size))

                self._file_states[str_path] = mtime

        # Check for deleted files
        for old_path in set(self._file_states.keys()) - current_files:
            changes.append(FileChange(old_path, "deleted"))
            del self._file_states[old_path]

        return changes

    def _is_excluded(self, path: str) -> bool:
        """Check if a path matches any exclusion pattern."""
        for pattern in self._config.exclude_patterns:
            if pattern in path:
                return True
        return False

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        self._running = True
        # Initial scan to establish baseline
        self.scan()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            changes = self.scan()
            if changes:
                for callback in self._callbacks:
                    try:
                        callback(changes)
                    except Exception as e:
                        logger.error("Watch callback error: %s", e)
            time.sleep(self._config.poll_interval)


class DirectiveScanner:
    """Scans files for @vetinari directives."""

    def __init__(self, max_file_size: int = 1_000_000):
        self._max_file_size = max_file_size

    def scan_file(self, file_path: str) -> List[VetinariDirective]:
        """Scan a single file for directives."""
        directives = []
        try:
            path = Path(file_path)
            if path.stat().st_size > self._max_file_size:
                return []

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in DIRECTIVE_PATTERNS:
                        match = pattern.search(line)
                        if match:
                            directives.append(VetinariDirective(
                                file_path=file_path,
                                line_number=line_num,
                                directive=match.group(1).strip(),
                                full_line=line.strip(),
                            ))
        except (OSError, UnicodeDecodeError) as e:
            logger.debug("Could not scan %s: %s", file_path, e)
        return directives

    def scan_changes(self, changes: List[FileChange]) -> List[VetinariDirective]:
        """Scan changed files for directives."""
        directives = []
        for change in changes:
            if change.change_type != "deleted":
                directives.extend(self.scan_file(change.path))
        return directives


class WatchMode:
    """Main watch mode coordinator."""

    def __init__(self, config: WatchConfig = None):
        self._config = config or WatchConfig()
        self._watcher = FileWatcher(self._config)
        self._scanner = DirectiveScanner(self._config.max_file_size)
        self._directive_handlers: Dict[str, Callable] = {}
        self._history: List[Dict[str, Any]] = []

        # Register default handlers
        self._register_defaults()

        # Wire up change callback
        self._watcher.on_change(self._handle_changes)

    def _register_defaults(self) -> None:
        """Register default directive handlers."""
        self._directive_handlers = {
            "fix": self._handle_fix,
            "add": self._handle_add,
            "test": self._handle_test,
            "explain": self._handle_explain,
            "refactor": self._handle_refactor,
            "review": self._handle_review,
        }

    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a custom directive handler."""
        self._directive_handlers[action.lower()] = handler

    def start(self) -> None:
        """Start watch mode."""
        logger.info("Watch mode started: monitoring %s", self._config.watch_dir)
        self._watcher.start()

    def stop(self) -> None:
        """Stop watch mode."""
        self._watcher.stop()
        logger.info("Watch mode stopped")

    @property
    def is_running(self) -> bool:
        return self._watcher.is_running

    def _handle_changes(self, changes: List[FileChange]) -> None:
        """Process file changes."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "changes": [{"path": c.path, "type": c.change_type} for c in changes],
            "directives": [],
        }

        if self._config.scan_directives:
            directives = self._scanner.scan_changes(changes)
            for d in directives:
                event["directives"].append({
                    "file": d.file_path,
                    "line": d.line_number,
                    "action": d.action,
                    "target": d.target,
                })
                self._dispatch_directive(d)

        self._history.append(event)

    def _dispatch_directive(self, directive: VetinariDirective) -> None:
        """Dispatch a directive to its handler."""
        handler = self._directive_handlers.get(directive.action)
        if handler:
            try:
                handler(directive)
            except Exception as e:
                logger.error("Directive handler error for '%s': %s", directive.action, e)
        else:
            logger.warning("Unknown directive action: %s", directive.action)

    # Directive handlers — log the detected directive for observability
    def _handle_fix(self, d: VetinariDirective) -> None:
        logger.info("Fix directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def _handle_add(self, d: VetinariDirective) -> None:
        logger.info("Add directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def _handle_test(self, d: VetinariDirective) -> None:
        logger.info("Test directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def _handle_explain(self, d: VetinariDirective) -> None:
        logger.info("Explain directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def _handle_refactor(self, d: VetinariDirective) -> None:
        logger.info("Refactor directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def _handle_review(self, d: VetinariDirective) -> None:
        logger.info("Review directive: %s (line %d in %s)", d.target, d.line_number, d.file_path)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get watch event history."""
        return list(self._history)
