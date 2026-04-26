#!/usr/bin/env python3
"""Git-history-derived file context accelerator.

This is maintainer tooling for coding agents. It summarizes a file's recent git
history, co-change patterns, and test companions without requiring runtime RAG.

Usage:
    python scripts/dev/file_context.py vetinari/notifications/manager.py
    python scripts/dev/file_context.py vetinari.notifications.manager --json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIT_EXE = shutil.which("git") or "git"
_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"
_TEST_FILE_RE = re.compile(r"^(tests/|.+(?:_test|test_).+\.py$)")
_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "bugfix": ("fix", "bug", "regression", "crash", "deadlock", "hang", "broken"),
    "security": ("security", "auth", "permission", "secret", "token", "injection", "xss"),
    "migration": ("migrate", "migration", "rename", "deprecate", "schema", "upgrade"),
    "refactor": ("refactor", "cleanup", "restructure", "simplify", "extract"),
    "performance": ("perf", "performance", "optimize", "latency", "slow", "cache"),
    "test": ("test", "coverage", "pytest", "fixture"),
}
_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "timeout": ("timeout", "timed out"),
    "race": ("race", "deadlock", "lock", "thread"),
    "rollback": ("rollback", "revert"),
    "breaking": ("breaking", "deprecate", "rename", "migration"),
}


@dataclass(frozen=True)
class HistoryCommit:
    """A parsed git history record for a file or repo-level scan."""

    commit: str
    date: str
    subject: str
    files: tuple[str, ...]
    category: str
    signals: tuple[str, ...]


def _git(*args: str, timeout: int = 60) -> str:
    """Run a git command at repo root and return stdout."""

    result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
        [GIT_EXE, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed")
    return result.stdout


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _is_test_file(path: str) -> bool:
    return bool(_TEST_FILE_RE.match(path))


def _classify_subject(subject: str) -> str:
    lowered = subject.lower()
    for category, keywords in _CATEGORY_PATTERNS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _detect_signals(subject: str) -> tuple[str, ...]:
    lowered = subject.lower()
    signals = [name for name, keywords in _SIGNAL_PATTERNS.items() if any(keyword in lowered for keyword in keywords)]
    return tuple(sorted(signals))


def _parse_log_output(output: str) -> list[HistoryCommit]:
    commits: list[HistoryCommit] = []
    for raw_record in output.split(_RECORD_SEP):
        record = raw_record.strip()
        if not record:
            continue
        lines = [line.strip() for line in record.splitlines()]
        if not lines:
            continue
        header = lines[0]
        try:
            commit, date, subject = header.split(_FIELD_SEP, 2)
        except ValueError:
            continue
        files = tuple(line for line in lines[1:] if line)
        commits.append(
            HistoryCommit(
                commit=commit,
                date=date,
                subject=subject,
                files=files,
                category=_classify_subject(subject),
                signals=_detect_signals(subject),
            )
        )
    return commits


def _resolve_target(target: str) -> Path:
    candidate = Path(target)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    repo_candidate = (PROJECT_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate

    if target.startswith("vetinari."):
        module_candidate = PROJECT_ROOT / Path(target.replace(".", "/")).with_suffix(".py")
        if module_candidate.exists():
            return module_candidate.resolve()
        init_candidate = PROJECT_ROOT / target.replace(".", "/") / "__init__.py"
        if init_candidate.exists():
            return init_candidate.resolve()

    raise FileNotFoundError(f"Could not resolve file target: {target}")


def _history_for_target(path: str, max_commits: int) -> list[HistoryCommit]:
    output = _git(
        "log",
        "--follow",
        f"--max-count={max_commits}",
        "--date=short",
        f"--format={_RECORD_SEP}%H{_FIELD_SEP}%ad{_FIELD_SEP}%s",
        "--name-only",
        "--",
        path,
    )
    return _parse_log_output(output)


def _repo_history(max_commits: int) -> list[HistoryCommit]:
    output = _git(
        "log",
        f"--max-count={max_commits}",
        "--date=short",
        f"--format={_RECORD_SEP}%H{_FIELD_SEP}%ad{_FIELD_SEP}%s",
        "--name-only",
        "--",
        "vetinari",
        "tests",
        "scripts",
        "ui",
    )
    return _parse_log_output(output)


def collect_file_context(target: str, max_commits: int = 40) -> dict[str, object]:
    """Collect git-history-derived context for one file."""

    target_path = _resolve_target(target)
    rel_path = _repo_rel(target_path)
    commits = _history_for_target(rel_path, max_commits=max_commits)

    categories = Counter(commit.category for commit in commits)
    signals = Counter(signal for commit in commits for signal in commit.signals)
    cochanged = Counter()
    test_companions = Counter()

    for commit in commits:
        for other in commit.files:
            if other == rel_path:
                continue
            cochanged[other] += 1
            if _is_test_file(other):
                test_companions[other] += 1

    recent_commits = [
        {
            "commit": commit.commit[:8],
            "date": commit.date,
            "subject": commit.subject,
            "category": commit.category,
            "signals": list(commit.signals),
        }
        for commit in commits[:10]
    ]
    risk_bearing = [
        record
        for record in recent_commits
        if record["category"] in {"bugfix", "security", "migration", "performance"}
        or record["signals"]
    ]

    return {
        "path": rel_path,
        "total_commits": len(commits),
        "categories": dict(categories),
        "signals": dict(signals),
        "recent_commits": recent_commits,
        "risk_bearing_commits": risk_bearing[:8],
        "cochanged_files": [{"path": path, "count": count} for path, count in cochanged.most_common(8)],
        "test_companions": [{"path": path, "count": count} for path, count in test_companions.most_common(6)],
    }


def collect_history_hotspots(max_commits: int = 250) -> dict[str, object]:
    """Collect repo-level git-history hotspots for static accelerator indexes."""

    commits = _repo_history(max_commits=max_commits)
    file_commits = Counter()
    file_categories: dict[str, Counter[str]] = defaultdict(Counter)
    file_signals: dict[str, Counter[str]] = defaultdict(Counter)
    file_test_companions: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts = Counter()

    for commit in commits:
        files = sorted({path for path in commit.files if path})
        source_files = [path for path in files if not _is_test_file(path)]
        test_files = [path for path in files if _is_test_file(path)]

        for path in source_files:
            file_commits[path] += 1
            file_categories[path][commit.category] += 1
            for signal in commit.signals:
                file_signals[path][signal] += 1
            for test_path in test_files:
                file_test_companions[path][test_path] += 1

        for left, right in itertools.combinations(source_files[:10], 2):
            pair_counts[left, right] += 1

    hotspots: list[dict[str, object]] = []
    for path, total in file_commits.items():
        categories = file_categories[path]
        score = (
            (categories.get("bugfix", 0) * 4)
            + (categories.get("security", 0) * 5)
            + (categories.get("migration", 0) * 3)
            + (categories.get("performance", 0) * 2)
            + total
        )
        hotspots.append(
            {
                "path": path,
                "commits": total,
                "bugfix": categories.get("bugfix", 0),
                "security": categories.get("security", 0),
                "migration": categories.get("migration", 0),
                "refactor": categories.get("refactor", 0),
                "top_tests": [
                    {"path": test_path, "count": count}
                    for test_path, count in file_test_companions[path].most_common(3)
                ],
                "signals": dict(file_signals[path]),
                "score": score,
            }
        )

    hotspots.sort(key=lambda item: (-int(item["score"]), item["path"]))
    cochange_pairs = [
        {"left": left, "right": right, "count": count}
        for (left, right), count in pair_counts.most_common(15)
    ]

    return {
        "hotspots": hotspots[:25],
        "cochange_pairs": cochange_pairs,
    }


def render_file_context_text(data: dict[str, object]) -> str:
    """Render a file-context summary as human-readable text."""

    lines = [
        f"File: {data['path']}",
        f"Commits scanned: {data['total_commits']}",
        "",
        "History Signals:",
    ]

    categories = data.get("categories", {})
    if isinstance(categories, dict) and categories:
        for key in ("bugfix", "security", "migration", "performance", "refactor", "test"):
            value = categories.get(key)
            if value:
                lines.append(f"  - {key}: {value}")
    else:
        lines.append("  - none")

    signals = data.get("signals", {})
    if isinstance(signals, dict) and signals:
        lines.extend(["", "Risk Markers:"])
        for name, count in sorted(signals.items()):
            lines.append(f"  - {name}: {count}")

    lines.extend(["", "Recent Risk-Bearing Commits:"])
    risk_commits = data.get("risk_bearing_commits", [])
    if isinstance(risk_commits, list) and risk_commits:
        for record in risk_commits:
            lines.append(
                f"  - {record['date']} [{record['category']}] {record['commit']} {record['subject']}"
            )
    else:
        lines.append("  - none")

    lines.extend(["", "Top Co-Changed Files:"])
    cochanged = data.get("cochanged_files", [])
    if isinstance(cochanged, list) and cochanged:
        for record in cochanged:
            lines.append(f"  - {record['path']} ({record['count']})")
    else:
        lines.append("  - none")

    lines.extend(["", "Test Companions:"])
    companions = data.get("test_companions", [])
    if isinstance(companions, list) and companions:
        for record in companions:
            lines.append(f"  - {record['path']} ({record['count']})")
    else:
        lines.append("  - none")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize git-history-derived context for a file")
    parser.add_argument("target", help="Repo-relative file path or dotted module path")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--max-commits",
        type=int,
        default=40,
        help="Maximum number of commits to scan for the target file (default: 40)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        data = collect_file_context(args.target, max_commits=args.max_commits)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(render_file_context_text(data))


if __name__ == "__main__":
    main()
