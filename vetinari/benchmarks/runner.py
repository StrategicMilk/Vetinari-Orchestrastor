"""
Multi-Layer Benchmark Runner for Vetinari
==========================================

Core framework for running benchmarks across three layers:

  Layer 1 (Agent):         Individual agent tool-calling   — seconds
  Layer 2 (Orchestration): Multi-agent tool chains         — minutes
  Layer 3 (Pipeline):      Full end-to-end pipelines       — hours

Tracks per-run: pass@1, pass^k, tokens, latency, cost, error recovery rate.
Persists metrics to SQLite for historical comparison.

Usage::

    from vetinari.benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    report = runner.run_suite("toolbench")
    runner.compare_runs(report.run_id, previous_run_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Enums and constants
# ============================================================

class BenchmarkLayer(Enum):
    """Three-layer testing hierarchy."""
    AGENT = 1          # Individual agent tool-calling
    ORCHESTRATION = 2  # Multi-agent tool chains
    PIPELINE = 3       # Full end-to-end


class BenchmarkTier(Enum):
    """Speed tier for benchmark scheduling."""
    FAST = "fast"        # seconds
    MEDIUM = "medium"    # minutes
    SLOW = "slow"        # hours


# ============================================================
# Data classes
# ============================================================

@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    case_id: str
    suite_name: str
    description: str
    input_data: Dict[str, Any]
    expected: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark case."""
    case_id: str
    suite_name: str
    run_id: str
    passed: bool
    score: float                          # 0.0 - 1.0
    latency_ms: float = 0.0
    tokens_consumed: int = 0
    cost_usd: float = 0.0
    error_recovery_count: int = 0
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class BenchmarkReport:
    """Aggregated report for a benchmark suite run."""
    run_id: str
    suite_name: str
    layer: BenchmarkLayer
    tier: BenchmarkTier
    results: List[BenchmarkResult]
    started_at: str
    finished_at: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    pass_at_1: float = 0.0               # fraction passing on first try
    pass_k: float = 0.0                   # consistency (Tau-bench style)
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_recovery_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_aggregates(self) -> None:
        """Recompute aggregate metrics from individual results."""
        if not self.results:
            return
        self.total_cases = len(self.results)
        self.passed_cases = sum(1 for r in self.results if r.passed)
        self.pass_at_1 = self.passed_cases / self.total_cases
        self.avg_score = sum(r.score for r in self.results) / self.total_cases
        self.avg_latency_ms = (
            sum(r.latency_ms for r in self.results) / self.total_cases
        )
        self.total_tokens = sum(r.tokens_consumed for r in self.results)
        self.total_cost_usd = sum(r.cost_usd for r in self.results)
        recoveries = sum(r.error_recovery_count for r in self.results)
        errors_possible = sum(
            1 for r in self.results if r.error_recovery_count > 0 or r.error
        )
        self.error_recovery_rate = (
            recoveries / errors_possible if errors_possible > 0 else 0.0
        )

    def summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "run_id": self.run_id,
            "suite": self.suite_name,
            "layer": self.layer.name,
            "tier": self.tier.value,
            "total": self.total_cases,
            "passed": self.passed_cases,
            "pass@1": round(self.pass_at_1, 4),
            "pass^k": round(self.pass_k, 4),
            "avg_score": round(self.avg_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "error_recovery_rate": round(self.error_recovery_rate, 4),
        }


@dataclass
class ComparisonReport:
    """Comparison between two benchmark runs."""
    run_a: str
    run_b: str
    suite_name: str
    delta_pass_at_1: float = 0.0
    delta_avg_score: float = 0.0
    delta_avg_latency_ms: float = 0.0
    delta_total_tokens: int = 0
    delta_cost_usd: float = 0.0
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)


# ============================================================
# Abstract base: BenchmarkSuiteAdapter
# ============================================================

class BenchmarkSuiteAdapter(ABC):
    """
    Base class for all benchmark adapters.

    Subclasses must define:
      - name: str
      - layer: BenchmarkLayer
      - tier: BenchmarkTier
      - load_cases() -> List[BenchmarkCase]
      - run_case(case) -> BenchmarkResult
      - evaluate(result) -> float
    """

    name: str = ""
    layer: BenchmarkLayer = BenchmarkLayer.AGENT
    tier: BenchmarkTier = BenchmarkTier.FAST

    @abstractmethod
    def load_cases(self, limit: Optional[int] = None) -> List[BenchmarkCase]:
        """Load benchmark cases. Optional limit for quick runs."""
        ...

    @abstractmethod
    def run_case(self, case: BenchmarkCase, run_id: str) -> BenchmarkResult:
        """Execute a single benchmark case and return its result."""
        ...

    @abstractmethod
    def evaluate(self, result: BenchmarkResult) -> float:
        """Score a result from 0.0 to 1.0."""
        ...

    def description(self) -> str:
        """Human-readable description of this benchmark suite."""
        return f"{self.name} (Layer {self.layer.value}, {self.tier.value})"


# ============================================================
# SQLite metric store
# ============================================================

_DEFAULT_DATA_DIR = Path(os.environ.get("VETINARI_DATA_DIR", Path.home() / ".vetinari"))
_DEFAULT_DB = _DEFAULT_DATA_DIR / "vetinari_benchmark_metrics.db"


class MetricStore:
    """Persists benchmark results and reports to SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DEFAULT_DB
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id       TEXT PRIMARY KEY,
                    suite_name   TEXT NOT NULL,
                    layer        TEXT NOT NULL,
                    tier         TEXT NOT NULL,
                    started_at   TEXT NOT NULL,
                    finished_at  TEXT,
                    total_cases  INTEGER DEFAULT 0,
                    passed_cases INTEGER DEFAULT 0,
                    pass_at_1    REAL DEFAULT 0.0,
                    pass_k       REAL DEFAULT 0.0,
                    avg_score    REAL DEFAULT 0.0,
                    avg_latency  REAL DEFAULT 0.0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost   REAL DEFAULT 0.0,
                    error_recovery_rate REAL DEFAULT 0.0,
                    metadata     TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id        TEXT NOT NULL,
                    case_id       TEXT NOT NULL,
                    suite_name    TEXT NOT NULL,
                    passed        INTEGER NOT NULL,
                    score         REAL NOT NULL,
                    latency_ms    REAL DEFAULT 0.0,
                    tokens        INTEGER DEFAULT 0,
                    cost_usd      REAL DEFAULT 0.0,
                    error_recovery INTEGER DEFAULT 0,
                    error         TEXT,
                    output        TEXT,
                    timestamp     TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_results_run
                ON benchmark_results(run_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_suite
                ON benchmark_runs(suite_name)
            """)

    def save_report(self, report: BenchmarkReport) -> None:
        """Persist a full benchmark report."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_runs
                   (run_id, suite_name, layer, tier, started_at, finished_at,
                    total_cases, passed_cases, pass_at_1, pass_k,
                    avg_score, avg_latency, total_tokens, total_cost,
                    error_recovery_rate, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    report.run_id,
                    report.suite_name,
                    report.layer.name,
                    report.tier.value,
                    report.started_at,
                    report.finished_at,
                    report.total_cases,
                    report.passed_cases,
                    report.pass_at_1,
                    report.pass_k,
                    report.avg_score,
                    report.avg_latency_ms,
                    report.total_tokens,
                    report.total_cost_usd,
                    report.error_recovery_rate,
                    json.dumps(report.metadata),
                ),
            )
            for r in report.results:
                conn.execute(
                    """INSERT INTO benchmark_results
                       (run_id, case_id, suite_name, passed, score,
                        latency_ms, tokens, cost_usd, error_recovery,
                        error, output, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        r.run_id,
                        r.case_id,
                        r.suite_name,
                        int(r.passed),
                        r.score,
                        r.latency_ms,
                        r.tokens_consumed,
                        r.cost_usd,
                        r.error_recovery_count,
                        r.error,
                        json.dumps(r.output) if r.output else None,
                        r.timestamp,
                    ),
                )

    def load_report(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a benchmark run summary by run_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def load_results(self, run_id: str) -> List[Dict[str, Any]]:
        """Load individual results for a run."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_runs(
        self, suite_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List recent benchmark runs."""
        with self._connect() as conn:
            if suite_name:
                rows = conn.execute(
                    """SELECT * FROM benchmark_runs
                       WHERE suite_name = ?
                       ORDER BY started_at DESC LIMIT ?""",
                    (suite_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM benchmark_runs
                       ORDER BY started_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def last_two_run_ids(self, suite_name: str) -> List[str]:
        """Return the two most recent run IDs for a suite."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT run_id FROM benchmark_runs
                   WHERE suite_name = ?
                   ORDER BY started_at DESC LIMIT 2""",
                (suite_name,),
            ).fetchall()
            return [r["run_id"] for r in rows]


# ============================================================
# BenchmarkRunner — main entry point
# ============================================================

class BenchmarkRunner:
    """Run benchmarks against Vetinari's orchestration pipeline."""

    def __init__(self, db_path: Optional[Path] = None):
        self._store = MetricStore(db_path)
        self._suites: Dict[str, BenchmarkSuiteAdapter] = {}

    # -- Registry --

    def register_suite(self, adapter: BenchmarkSuiteAdapter) -> None:
        """Register a benchmark suite adapter."""
        self._suites[adapter.name] = adapter

    def list_suites(self) -> List[Dict[str, str]]:
        """Return metadata for all registered suites."""
        return [
            {
                "name": a.name,
                "layer": a.layer.name,
                "tier": a.tier.value,
                "description": a.description(),
            }
            for a in self._suites.values()
        ]

    # -- Execution --

    def run_suite(
        self,
        suite_name: str,
        limit: Optional[int] = None,
        trials: int = 1,
    ) -> BenchmarkReport:
        """Run a named benchmark suite.

        Args:
            suite_name: Registered suite name.
            limit: Max cases to run (None = all).
            trials: Number of trials per case (for pass^k computation).

        Returns:
            BenchmarkReport with aggregated metrics.
        """
        adapter = self._suites.get(suite_name)
        if adapter is None:
            raise ValueError(
                f"Unknown suite '{suite_name}'. "
                f"Available: {list(self._suites.keys())}"
            )

        run_id = f"{suite_name}-{uuid.uuid4().hex[:8]}"
        started = datetime.now(timezone.utc).isoformat()
        cases = adapter.load_cases(limit=limit)

        all_results: List[BenchmarkResult] = []
        pass_counts: Dict[str, int] = {}  # case_id -> pass count across trials

        for case in cases:
            pass_counts[case.case_id] = 0
            for trial in range(trials):
                try:
                    result = adapter.run_case(case, run_id)
                    result.score = adapter.evaluate(result)
                    result.passed = result.score >= 0.5
                except Exception as exc:
                    result = BenchmarkResult(
                        case_id=case.case_id,
                        suite_name=suite_name,
                        run_id=run_id,
                        passed=False,
                        score=0.0,
                        error=str(exc),
                    )
                if result.passed:
                    pass_counts[case.case_id] += 1
                all_results.append(result)

        finished = datetime.now(timezone.utc).isoformat()

        # Compute pass^k: fraction of cases that passed ALL k trials
        if trials > 0 and cases:
            fully_passed = sum(
                1 for c in cases if pass_counts[c.case_id] == trials
            )
            pass_k = fully_passed / len(cases)
        else:
            pass_k = 0.0

        report = BenchmarkReport(
            run_id=run_id,
            suite_name=suite_name,
            layer=adapter.layer,
            tier=adapter.tier,
            results=all_results,
            started_at=started,
            finished_at=finished,
        )
        report.compute_aggregates()
        report.pass_k = pass_k

        self.track_metrics(report)
        return report

    def run_single(self, benchmark_id: str) -> BenchmarkResult:
        """Run a single benchmark case by its ID (suite:case_id format)."""
        if ":" not in benchmark_id:
            raise ValueError(
                "benchmark_id must be 'suite_name:case_id' format"
            )
        suite_name, case_id = benchmark_id.split(":", 1)
        adapter = self._suites.get(suite_name)
        if adapter is None:
            raise ValueError(f"Unknown suite '{suite_name}'")

        cases = adapter.load_cases()
        case = next((c for c in cases if c.case_id == case_id), None)
        if case is None:
            raise ValueError(
                f"Case '{case_id}' not found in suite '{suite_name}'"
            )

        run_id = f"{suite_name}-single-{uuid.uuid4().hex[:8]}"
        result = adapter.run_case(case, run_id)
        result.score = adapter.evaluate(result)
        result.passed = result.score >= 0.5
        return result

    # -- Comparison --

    def compare_runs(self, run_a: str, run_b: str) -> ComparisonReport:
        """Compare two benchmark runs."""
        a = self._store.load_report(run_a)
        b = self._store.load_report(run_b)

        if a is None or b is None:
            missing = run_a if a is None else run_b
            raise ValueError(f"Run '{missing}' not found in metric store")

        suite = a.get("suite_name", b.get("suite_name", "unknown"))
        comp = ComparisonReport(run_a=run_a, run_b=run_b, suite_name=suite)
        comp.delta_pass_at_1 = (
            (b.get("pass_at_1", 0) or 0) - (a.get("pass_at_1", 0) or 0)
        )
        comp.delta_avg_score = (
            (b.get("avg_score", 0) or 0) - (a.get("avg_score", 0) or 0)
        )
        comp.delta_avg_latency_ms = (
            (b.get("avg_latency", 0) or 0) - (a.get("avg_latency", 0) or 0)
        )
        comp.delta_total_tokens = (
            (b.get("total_tokens", 0) or 0) - (a.get("total_tokens", 0) or 0)
        )
        comp.delta_cost_usd = (
            (b.get("total_cost", 0) or 0) - (a.get("total_cost", 0) or 0)
        )

        # Identify regressions and improvements
        THRESHOLD = 0.05
        if comp.delta_avg_score < -THRESHOLD:
            comp.regressions.append(
                f"avg_score dropped by {abs(comp.delta_avg_score):.4f}"
            )
        elif comp.delta_avg_score > THRESHOLD:
            comp.improvements.append(
                f"avg_score improved by {comp.delta_avg_score:.4f}"
            )

        if comp.delta_pass_at_1 < -THRESHOLD:
            comp.regressions.append(
                f"pass@1 dropped by {abs(comp.delta_pass_at_1):.4f}"
            )
        elif comp.delta_pass_at_1 > THRESHOLD:
            comp.improvements.append(
                f"pass@1 improved by {comp.delta_pass_at_1:.4f}"
            )

        return comp

    # -- Metrics persistence --

    def track_metrics(self, report: BenchmarkReport) -> None:
        """Persist a benchmark report to SQLite."""
        self._store.save_report(report)

    def list_runs(
        self, suite_name: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """List recent benchmark runs from the metric store."""
        return self._store.list_runs(suite_name, limit)

    def get_last_comparison(self, suite_name: str) -> Optional[ComparisonReport]:
        """Compare the two most recent runs for a suite."""
        ids = self._store.last_two_run_ids(suite_name)
        if len(ids) < 2:
            return None
        return self.compare_runs(ids[1], ids[0])  # older vs newer


# ============================================================
# Convenience: default runner with all adapters registered
# ============================================================

def get_default_runner(db_path: Optional[Path] = None) -> BenchmarkRunner:
    """Create a BenchmarkRunner with all built-in adapters registered."""
    runner = BenchmarkRunner(db_path=db_path)

    # Import adapters lazily to avoid circular imports
    try:
        from vetinari.benchmarks.swe_bench import SWEBenchAdapter
        runner.register_suite(SWEBenchAdapter())
    except Exception as exc:
        logger.debug(f"Could not load SWE-bench adapter: {exc}")

    try:
        from vetinari.benchmarks.tau_bench import TauBenchAdapter
        runner.register_suite(TauBenchAdapter())
    except Exception as exc:
        logger.debug(f"Could not load Tau-bench adapter: {exc}")

    try:
        from vetinari.benchmarks.toolbench import ToolBenchAdapter
        runner.register_suite(ToolBenchAdapter())
    except Exception as exc:
        logger.debug(f"Could not load ToolBench adapter: {exc}")

    try:
        from vetinari.benchmarks.taskbench import TaskBenchAdapter
        runner.register_suite(TaskBenchAdapter())
    except Exception as exc:
        logger.debug(f"Could not load TaskBench adapter: {exc}")

    try:
        from vetinari.benchmarks.api_bank import APIBankAdapter
        runner.register_suite(APIBankAdapter())
    except Exception as exc:
        logger.debug(f"Could not load API-Bank adapter: {exc}")

    return runner
