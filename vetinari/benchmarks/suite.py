"""
Vetinari Comprehensive Benchmark Suite
========================================
Standardized evaluation tasks per agent type.

Tracks quality over time and alerts on regressions after prompt/model changes.
All benchmarks run offline (no network calls) using mocked agent execution.

Usage::

    from vetinari.benchmarks.suite import BenchmarkSuite, run_benchmark

    suite = BenchmarkSuite()
    results = suite.run_all()
    suite.print_report(results)

    # Or run a single agent
    agent_results = suite.run_agent("BUILDER")
    logger.debug("Builder: %.3f", agent_results.avg_score)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESULTS_PATH = Path("./vetinari_benchmarks.jsonl")


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""

    case_id: str
    agent_type: str
    task_type: str
    description: str
    input: str
    evaluator: Callable[[Any], float]  # Returns score 0.0-1.0
    expected_keys: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of running a set of benchmark cases."""

    agent_type: str
    timestamp: str
    cases_run: int
    cases_passed: int
    avg_score: float
    scores: list[float] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str = ""


def _score_by_keys(output: Any, required_keys: list[str]) -> float:
    """Score output based on whether required keys are present."""
    if not isinstance(output, dict) and isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return 0.3
    if not isinstance(output, dict):
        return 0.1
    found = sum(1 for k in required_keys if output.get(k))
    return found / max(len(required_keys), 1)


class BenchmarkSuite:
    """Runs standardized benchmarks across all Vetinari agents."""

    PASS_THRESHOLD = 0.6  # Score >= this is considered passing

    def __init__(self):
        self._cases: list[BenchmarkCase] = self._build_cases()

    # ------------------------------------------------------------------
    # Case definitions
    # ------------------------------------------------------------------

    def _build_cases(self) -> list[BenchmarkCase]:
        """Define benchmark cases for all 21 agents."""
        cases = []

        # PLANNER: Task decomposition
        cases.append(
            BenchmarkCase(
                case_id="planner_decompose_001",
                agent_type="PLANNER",
                task_type="planning",
                description="Decompose: build a REST API",
                input="Build a REST API for a todo list application with authentication",
                evaluator=lambda o: _score_by_keys(o, ["tasks", "dependencies"]),
                expected_keys=["tasks", "dependencies"],
            )
        )

        # BUILDER: Code generation
        cases.append(
            BenchmarkCase(
                case_id="builder_scaffold_001",
                agent_type="BUILDER",
                task_type="coding",
                description="Scaffold a Python class",
                input="Generate a UserRepository class with CRUD operations for SQLite",
                evaluator=lambda o: _score_by_keys(o, ["scaffold_code", "tests"]),
                expected_keys=["scaffold_code", "tests"],
            )
        )

        # QUALITY: Code review
        cases.append(
            BenchmarkCase(
                case_id="evaluator_review_001",
                agent_type="QUALITY",
                task_type="review",
                description="Review code with eval/exec",
                input="Review: def run(code): eval(code)",
                evaluator=lambda o: _score_by_keys(o, ["issues", "score"]),
                expected_keys=["issues", "score"],
            )
        )

        # RESEARCHER: Research query
        cases.append(
            BenchmarkCase(
                case_id="researcher_query_001",
                agent_type="RESEARCHER",
                task_type="research",
                description="Research exponential backoff",
                input="Research best practices for implementing exponential backoff in Python",
                evaluator=lambda o: _score_by_keys(o, ["findings", "recommendations"]),
                expected_keys=["findings", "recommendations"],
            )
        )

        # QUALITY: Security scan
        cases.append(
            BenchmarkCase(
                case_id="security_audit_001",
                agent_type="QUALITY",
                task_type="analysis",
                description="Audit SQL injection pattern",
                input="Review: query = f'SELECT * FROM users WHERE id = {user_id}'",
                evaluator=lambda o: _score_by_keys(o, ["vulnerabilities", "remediation"]),
                expected_keys=["vulnerabilities", "remediation"],
            )
        )

        # TEST_AUTOMATION: Test generation
        cases.append(
            BenchmarkCase(
                case_id="test_gen_001",
                agent_type="TEST_AUTOMATION",
                task_type="testing",
                description="Generate tests for add function",
                input="Generate pytest tests for: def add(a, b): return a + b",
                evaluator=lambda o: _score_by_keys(o, ["test_scripts", "test_files"]),
                expected_keys=["test_scripts", "test_files"],
            )
        )

        # OPERATIONS: Doc generation
        cases.append(
            BenchmarkCase(
                case_id="docs_gen_001",
                agent_type="OPERATIONS",
                task_type="documentation",
                description="Generate API docs",
                input="Document this API endpoint: POST /api/users (creates a user)",
                evaluator=lambda o: _score_by_keys(o, ["documentation", "examples"]),
                expected_keys=["documentation", "examples"],
            )
        )

        # DEVOPS: Pipeline design
        cases.append(
            BenchmarkCase(
                case_id="devops_ci_001",
                agent_type="DEVOPS",
                task_type="coding",
                description="Design GitHub Actions CI pipeline",
                input="Design a GitHub Actions CI/CD pipeline for a Python FastAPI application",
                evaluator=lambda o: _score_by_keys(o, ["pipeline", "stages"]),
                expected_keys=["pipeline", "stages"],
            )
        )

        # VERSION_CONTROL: Commit message
        cases.append(
            BenchmarkCase(
                case_id="vc_commit_001",
                agent_type="VERSION_CONTROL",
                task_type="general",
                description="Generate commit messages",
                input="Generate conventional commit messages for: added user authentication",
                evaluator=lambda o: _score_by_keys(o, ["commit_messages", "recommendations"]),
                expected_keys=["commit_messages", "recommendations"],
            )
        )

        # ERROR_RECOVERY: Error analysis
        cases.append(
            BenchmarkCase(
                case_id="error_recovery_001",
                agent_type="ERROR_RECOVERY",
                task_type="analysis",
                description="Analyse ConnectionRefusedError",
                input="Error: ConnectionRefusedError: [Errno 111] Connection refused on port 5432",
                evaluator=lambda o: _score_by_keys(o, ["root_cause", "recovery_strategies"]),
                expected_keys=["root_cause", "recovery_strategies"],
            )
        )

        # CONTEXT_MANAGER: Memory consolidation
        cases.append(
            BenchmarkCase(
                case_id="ctx_mgr_001",
                agent_type="CONTEXT_MANAGER",
                task_type="general",
                description="Consolidate session context",
                input='Consolidate these entries: [\'{"task": "build API", "result": "done"}\']',
                evaluator=lambda o: _score_by_keys(o, ["summary", "key_facts"]),
                expected_keys=["summary", "key_facts"],
            )
        )

        return cases

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self, agent_types: list[str] | None = None) -> list[BenchmarkResult]:
        """Run all benchmark cases, optionally filtered to specific agents.

        Returns:
            List of results.
        """
        results = []
        types_to_test = agent_types or list({c.agent_type for c in self._cases})

        for agent_type in types_to_test:
            result = self.run_agent(agent_type)
            results.append(result)
            self._persist(result)

        return results

    def run_agent(self, agent_type: str) -> BenchmarkResult:
        """Run all benchmark cases for a specific agent.

        Returns:
            The BenchmarkResult result.
        """
        cases = [c for c in self._cases if c.agent_type == agent_type]
        if not cases:
            return BenchmarkResult(
                agent_type=agent_type,
                timestamp=datetime.now().isoformat(),
                cases_run=0,
                cases_passed=0,
                avg_score=0.0,
                error="No benchmark cases defined",
            )

        scores = []
        details = []
        start = time.time()

        for case in cases:
            score, detail = self._run_case(case)
            scores.append(score)
            details.append(detail)

        duration = (time.time() - start) * 1000
        avg = sum(scores) / max(len(scores), 1)
        passed = sum(1 for s in scores if s >= self.PASS_THRESHOLD)

        return BenchmarkResult(
            agent_type=agent_type,
            timestamp=datetime.now().isoformat(),
            cases_run=len(cases),
            cases_passed=passed,
            avg_score=round(avg, 3),
            scores=scores,
            details=details,
            duration_ms=round(duration, 1),
        )

    def _run_case(self, case: BenchmarkCase) -> tuple:
        """Execute a single benchmark case. Returns (score, detail_dict)."""
        try:
            from vetinari.agents.contracts import AgentTask
            from vetinari.orchestration.agent_graph import get_agent_graph
            from vetinari.types import AgentType

            graph = get_agent_graph()
            agent_type_enum = AgentType(case.agent_type)
            agent = graph.get_agent(agent_type_enum)

            if agent is None:
                return 0.0, {
                    "case_id": case.case_id,
                    "score": 0.0,
                    "error": "Agent not available in graph",
                }

            task = AgentTask(
                task_id=f"bench_{case.case_id}",
                agent_type=agent_type_enum,
                description=case.input,
                prompt=case.input,
            )

            result = agent.execute(task)
            if not result.success:
                return 0.2, {
                    "case_id": case.case_id,
                    "score": 0.2,
                    "error": f"Agent returned failure: {result.errors}",
                }

            score = case.evaluator(result.output)
            return score, {
                "case_id": case.case_id,
                "score": round(score, 3),
                "passed": score >= self.PASS_THRESHOLD,
                "output_keys": list(result.output.keys())
                if isinstance(result.output, dict)
                else type(result.output).__name__,
            }

        except Exception as e:
            logger.debug("[Benchmark] Case %s failed: %s", case.case_id, e)
            return 0.0, {"case_id": case.case_id, "score": 0.0, "error": str(e)[:200]}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: list[BenchmarkResult]) -> None:
        """Print a human-readable benchmark report."""
        logger.info("\n" + "=" * 60)
        logger.info("VETINARI BENCHMARK REPORT — %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
        logger.info("=" * 60)
        for r in sorted(results, key=lambda x: -x.avg_score):
            status = "PASS" if r.avg_score >= self.PASS_THRESHOLD else "FAIL"
            logger.info(
                f"  [{status}] {r.agent_type:<25} "
                f"score={r.avg_score:.3f}  "
                f"passed={r.cases_passed}/{r.cases_run}  "
                f"({r.duration_ms:.0f}ms)"
            )
        overall = sum(r.avg_score for r in results) / max(len(results), 1)
        logger.info("=" * 60)
        logger.info("  OVERALL AVG: %.3f", overall)
        logger.info("=" * 60 + "\n")

    def check_regression(self, new_results: list[BenchmarkResult], threshold: float = 0.05) -> list[str]:
        """Compare new results against historical baseline.

        Returns list of agents whose scores regressed by more than threshold.

        Args:
            new_results: The new results.
            threshold: The threshold.

        Returns:
            The result string.
        """
        regressions = []
        historical = self._load_historical()

        for result in new_results:
            baseline = historical.get(result.agent_type)
            if baseline and (baseline - result.avg_score) > threshold:
                regressions.append(
                    f"{result.agent_type}: {baseline:.3f} -> {result.avg_score:.3f} "
                    f"(delta=-{baseline - result.avg_score:.3f})"
                )
        return regressions

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, result: BenchmarkResult) -> None:
        try:
            import dataclasses

            with open(_RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(dataclasses.asdict(result)) + "\n")
        except Exception as e:
            logger.debug("[Benchmark] Persist failed: %s", e)

    def _load_historical(self) -> dict[str, float]:
        """Load per-agent average scores from historical results."""
        if not _RESULTS_PATH.exists():
            return {}
        by_agent: dict[str, list[float]] = {}
        try:
            with open(_RESULTS_PATH, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line.strip())
                    agent = r.get("agent_type", "")
                    score = r.get("avg_score", 0.0)
                    by_agent.setdefault(agent, []).append(score)
            return {k: sum(v) / len(v) for k, v in by_agent.items()}
        except Exception:
            return {}


def run_benchmark(agent_types: list[str] | None = None) -> list[BenchmarkResult]:
    """Convenience function to run the benchmark suite.

    Returns:
        List of results.
    """
    suite = BenchmarkSuite()
    results = suite.run_all(agent_types)
    suite.print_report(results)
    regressions = suite.check_regression(results)
    if regressions:
        logger.warning("[Benchmark] REGRESSIONS DETECTED:\n%s", "\n".join(regressions))
    return results
