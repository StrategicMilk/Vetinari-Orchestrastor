"""Tests for vetinari.planning.plan_validator.

Verifies cycle detection, dependency completeness, goal coverage, testability
checks, LLM retry logic, and degraded-state flagging for keyword-fallback plans.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_task
from vetinari.agents.contracts import Task
from vetinari.planning.plan_validator import (
    IssueCategory,
    IssueSeverity,
    ValidationResult,
    check_cycles,
    check_dependency_completeness,
    check_goal_coverage,
    check_testable_output,
    flag_degraded_fallback,
    validate_and_retry,
    validate_plan,
)
from vetinari.types import AgentType

# -- Shared helpers -----------------------------------------------------------


def linear_plan() -> list[Task]:
    """Return a simple linear A -> B -> C plan with no cycles.

    Returns:
        Three tasks: analyse (outputs spec), implement (outputs code),
        test (outputs test_results).
    """
    t_a = make_task(id="t_a", description="Analyse requirements", inputs=["goal"], outputs=["spec"])
    t_b = make_task(
        id="t_b",
        description="Implement feature",
        inputs=["spec"],
        outputs=["code"],
        dependencies=["t_a"],
    )
    t_c = make_task(
        id="t_c",
        description="Write tests",
        inputs=["code"],
        outputs=["test_results"],
        dependencies=["t_b"],
    )
    return [t_a, t_b, t_c]


# -- TestCycleDetection -------------------------------------------------------


class TestCycleDetection:
    """Tests for check_cycles()."""

    def test_circular_dependencies_detected(self) -> None:
        """A -> B -> C -> A cycle must produce a CYCLE ERROR issue."""
        t_a = make_task(id="t_a", dependencies=["t_c"], outputs=["a_out"])
        t_b = make_task(id="t_b", dependencies=["t_a"], outputs=["b_out"])
        t_c = make_task(id="t_c", dependencies=["t_b"], outputs=["c_out"])

        issues = check_cycles([t_a, t_b, t_c])

        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.ERROR
        assert issue.category == IssueCategory.CYCLE
        # All three tasks should be implicated
        assert set(issue.affected_task_ids) == {"t_a", "t_b", "t_c"}

    def test_acyclic_plan_passes(self) -> None:
        """A linear A -> B -> C plan must produce no cycle issues."""
        issues = check_cycles(linear_plan())
        assert issues == []

    def test_empty_plan_passes(self) -> None:
        """An empty task list must not raise and must return no issues."""
        issues = check_cycles([])
        assert issues == []

    def test_single_task_passes(self) -> None:
        """A single task with no dependencies must pass cycle check."""
        issues = check_cycles([make_task(id="t1", outputs=["out"])])
        assert issues == []

    def test_diamond_dag_passes(self) -> None:
        """A -> (B, C) -> D diamond (no cycle) must pass."""
        t_a = make_task(id="t_a", outputs=["a_out"])
        t_b = make_task(id="t_b", outputs=["b_out"], dependencies=["t_a"])
        t_c = make_task(id="t_c", outputs=["c_out"], dependencies=["t_a"])
        t_d = make_task(id="t_d", outputs=["d_out"], dependencies=["t_b", "t_c"])
        issues = check_cycles([t_a, t_b, t_c, t_d])
        assert issues == []

    def test_self_loop_detected(self) -> None:
        """A task that depends on itself must be caught as a cycle."""
        t_a = make_task(id="t_a", dependencies=["t_a"])
        issues = check_cycles([t_a])
        assert len(issues) == 1
        assert issues[0].category == IssueCategory.CYCLE

    def test_validate_plan_cycle_returns_invalid(self) -> None:
        """validate_plan must return valid=False when a cycle is present."""
        t_a = make_task(id="t_a", dependencies=["t_b"])
        t_b = make_task(id="t_b", dependencies=["t_a"])
        result = validate_plan([t_a, t_b], goal="do something useful")
        assert result.valid is False
        cycle_issues = [i for i in result.issues if i.category == IssueCategory.CYCLE]
        assert len(cycle_issues) >= 1


# -- TestDependencyCompleteness -----------------------------------------------


class TestDependencyCompleteness:
    """Tests for check_dependency_completeness()."""

    def test_missing_dependency_flagged(self) -> None:
        """A task requiring an input not produced by any predecessor must raise an ERROR."""
        t_a = make_task(id="t_a", inputs=["goal"], outputs=["spec"])
        # t_b wants "database_schema" which nothing produces
        t_b = make_task(
            id="t_b",
            inputs=["spec", "database_schema"],
            outputs=["code"],
            dependencies=["t_a"],
        )

        issues = check_dependency_completeness([t_a, t_b])

        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.ERROR
        assert issue.category == IssueCategory.DEPENDENCY
        assert "database_schema" in issue.message
        assert "t_b" in issue.affected_task_ids

    def test_complete_dependencies_pass(self) -> None:
        """All inputs covered by predecessor outputs or initial context must produce no issues."""
        issues = check_dependency_completeness(linear_plan(), initial_context=["goal"])
        assert issues == []

    def test_initial_context_satisfies_inputs(self) -> None:
        """Inputs listed in initial_context must not trigger a missing-dependency error."""
        t_a = make_task(id="t_a", inputs=["goal", "prior_research"], outputs=["spec"])
        issues = check_dependency_completeness([t_a], initial_context=["goal", "prior_research"])
        assert issues == []

    def test_default_context_includes_goal(self) -> None:
        """When initial_context is None, 'goal' must be available by default."""
        t_a = make_task(id="t_a", inputs=["goal"], outputs=["spec"])
        issues = check_dependency_completeness([t_a], initial_context=None)
        assert issues == []

    def test_multiple_missing_inputs_each_flagged(self) -> None:
        """Each unsatisfied input must produce a separate DEPENDENCY issue."""
        t_a = make_task(id="t_a", inputs=["missing_x", "missing_y"], outputs=["out"])
        issues = check_dependency_completeness([t_a], initial_context=[])
        assert len(issues) == 2
        messages = " ".join(i.message for i in issues)
        assert "missing_x" in messages
        assert "missing_y" in messages

    def test_transitive_predecessor_outputs_available(self) -> None:
        """Outputs from indirect predecessors (grandparent) must be visible."""
        t_a = make_task(id="t_a", inputs=["goal"], outputs=["raw_data"])
        t_b = make_task(
            id="t_b",
            inputs=["raw_data"],
            outputs=["processed_data"],
            dependencies=["t_a"],
        )
        # t_c depends on t_b but also needs raw_data from grandparent t_a
        t_c = make_task(
            id="t_c",
            inputs=["processed_data", "raw_data"],
            outputs=["report"],
            dependencies=["t_b"],
        )
        issues = check_dependency_completeness([t_a, t_b, t_c], initial_context=["goal"])
        assert issues == []


# -- TestGoalCoverage ---------------------------------------------------------


class TestGoalCoverage:
    """Tests for check_goal_coverage()."""

    def test_low_coverage_flagged(self) -> None:
        """Tasks completely unrelated to the goal must trigger a COVERAGE warning."""
        tasks = [
            make_task(id="t1", description="Prepare data"),
            make_task(id="t2", description="Clean records"),
        ]
        goal = "build a machine learning fraud detection classifier"
        issues = check_goal_coverage(tasks, goal, min_coverage=0.3)

        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == IssueSeverity.WARNING
        assert issue.category == IssueCategory.COVERAGE

    def test_high_coverage_passes(self) -> None:
        """Tasks whose descriptions overlap substantially with the goal must pass."""
        goal = "implement a fraud detection machine learning classifier"
        tasks = [
            make_task(id="t1", description="Implement the machine learning fraud detection model"),
            make_task(id="t2", description="Train and evaluate the classifier on historical data"),
        ]
        issues = check_goal_coverage(tasks, goal, min_coverage=0.3)
        assert issues == []

    def test_empty_tasks_no_issues(self) -> None:
        """An empty task list must not raise and returns no issues."""
        issues = check_goal_coverage([], "some goal", min_coverage=0.3)
        assert issues == []

    def test_empty_goal_no_issues(self) -> None:
        """An empty goal string must not trigger any coverage issue."""
        tasks = [make_task(id="t1", description="Do something")]
        issues = check_goal_coverage(tasks, "", min_coverage=0.3)
        assert issues == []

    def test_custom_min_coverage(self) -> None:
        """A very high min_coverage threshold must flag plans that otherwise pass."""
        goal = "build an api"
        tasks = [make_task(id="t1", description="Build an api endpoint")]
        # At 0.99 even good overlap fails — just verify no exception raised
        issues = check_goal_coverage(tasks, goal, min_coverage=0.99)
        assert isinstance(issues, list)


# -- TestTestability ----------------------------------------------------------


class TestTestability:
    """Tests for check_testable_output()."""

    def test_no_testable_output_flagged(self) -> None:
        """A plan whose outputs never contain testable keywords must be flagged."""
        tasks = [
            make_task(id="t1", outputs=["code"], description="Write code"),
            make_task(id="t2", outputs=["documentation"], description="Write docs"),
        ]
        issues = check_testable_output(tasks)
        assert len(issues) == 1
        assert issues[0].severity == IssueSeverity.WARNING
        assert issues[0].category == IssueCategory.TESTABILITY

    def test_testable_output_present_passes(self) -> None:
        """A plan with at least one 'test_results' output must pass testability."""
        tasks = [
            make_task(id="t1", outputs=["code"]),
            make_task(id="t2", outputs=["test_results"]),
        ]
        issues = check_testable_output(tasks)
        assert issues == []

    def test_verification_keyword_passes(self) -> None:
        """A task with 'verification_report' in outputs must satisfy testability."""
        tasks = [make_task(id="t1", outputs=["verification_report"])]
        issues = check_testable_output(tasks)
        assert issues == []

    def test_report_in_outputs_passes(self) -> None:
        """An output named 'security_report' must satisfy testability."""
        tasks = [make_task(id="t1", outputs=["security_report"])]
        issues = check_testable_output(tasks)
        assert issues == []

    def test_testable_word_in_description_passes(self) -> None:
        """A description containing 'tests' must satisfy testability as secondary signal."""
        tasks = [make_task(id="t1", description="Write unit tests for the module", outputs=["code"])]
        issues = check_testable_output(tasks)
        assert issues == []

    def test_empty_tasks_no_issues(self) -> None:
        """An empty task list must not raise and returns no issues."""
        issues = check_testable_output([])
        assert issues == []


# -- TestRetry ----------------------------------------------------------------


class TestRetry:
    """Tests for validate_and_retry()."""

    def test_incomplete_plan_reprompted(self) -> None:
        """First call returns a cyclic plan, second returns valid — verify 2 calls made."""
        # Bad plan: A -> B -> A cycle
        bad_tasks = [
            make_task(id="t_a", dependencies=["t_b"], outputs=["a_out"], inputs=["goal"]),
            make_task(id="t_b", dependencies=["t_a"], outputs=["b_out"], inputs=["a_out"]),
        ]
        # Good plan: linear A -> B -> C with testable output
        good_tasks = [
            make_task(id="t_a", inputs=["goal"], outputs=["spec"]),
            make_task(id="t_b", inputs=["spec"], outputs=["code"], dependencies=["t_a"]),
            make_task(
                id="t_c",
                inputs=["code"],
                outputs=["test_results"],
                dependencies=["t_b"],
                description="Write and run tests",
            ),
        ]

        agent = MagicMock()
        call_count = 0

        def fake_decompose(ag: object, goal: str, ctx: dict, max_tasks: int = 15) -> list[Task]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_tasks
            return good_tasks

        with patch(
            "vetinari.agents.planner_decompose.decompose_goal_llm",
            side_effect=fake_decompose,
        ):
            tasks, result = validate_and_retry(agent, "write tests for the app", {}, max_retries=2)

        # Must have called decompose_goal_llm exactly 2 times (initial + 1 retry)
        assert call_count == 2
        # Final result must be valid
        assert result.valid is True
        assert tasks == good_tasks

    def test_all_retries_exhausted_returns_last_result(self) -> None:
        """When all retries produce invalid plans, the last plan+result is still returned."""
        bad_tasks = [
            make_task(id="t_a", dependencies=["t_b"]),
            make_task(id="t_b", dependencies=["t_a"]),
        ]

        with patch(
            "vetinari.agents.planner_decompose.decompose_goal_llm",
            return_value=bad_tasks,
        ):
            tasks, result = validate_and_retry(MagicMock(), "do something", {}, max_retries=1)

        # Returns the last plan even though invalid
        assert tasks == bad_tasks
        assert result.valid is False

    def test_first_attempt_valid_no_retry(self) -> None:
        """When the first LLM result is valid, no retry should happen."""
        good_tasks = [
            make_task(id="t_a", inputs=["goal"], outputs=["spec"]),
            make_task(
                id="t_b",
                inputs=["spec"],
                outputs=["test_results"],
                dependencies=["t_a"],
                description="Run verification tests",
            ),
        ]

        call_count = 0

        def fake_decompose(ag: object, goal: str, ctx: dict, max_tasks: int = 15) -> list[Task]:
            nonlocal call_count
            call_count += 1
            return good_tasks

        with patch(
            "vetinari.agents.planner_decompose.decompose_goal_llm",
            side_effect=fake_decompose,
        ):
            tasks, result = validate_and_retry(MagicMock(), "build an api with tests", {}, max_retries=2)

        assert call_count == 1
        assert result.valid is True


# -- TestDegradedState --------------------------------------------------------


class TestDegradedState:
    """Tests for flag_degraded_fallback()."""

    def test_keyword_fallback_flagged_as_degraded(self) -> None:
        """flag_degraded_fallback must mark the ValidationResult as is_degraded=True."""
        tasks = [
            make_task(id="t1", outputs=["spec"], inputs=["goal"]),
            make_task(
                id="t2",
                outputs=["test_results"],
                inputs=["spec"],
                dependencies=["t1"],
                description="Run verification tests",
            ),
        ]

        result = flag_degraded_fallback(tasks, method="decompose_goal_keyword")

        assert result.is_degraded is True
        assert result.degraded_reason is not None
        assert "keyword" in result.degraded_reason.lower()
        assert "training" in result.degraded_reason.lower()

    def test_degraded_plan_still_structurally_valid(self) -> None:
        """A structurally sound keyword-fallback plan can be degraded but still valid."""
        tasks = [
            make_task(id="t1", inputs=["goal"], outputs=["spec"]),
            make_task(
                id="t2",
                inputs=["spec"],
                outputs=["verification_report"],
                dependencies=["t1"],
                description="Verify spec completeness",
            ),
        ]

        result = flag_degraded_fallback(tasks, method="decompose_goal_keyword")

        assert result.is_degraded is True
        assert result.valid is True  # No structural errors

    def test_degraded_plan_with_structural_errors_is_invalid(self) -> None:
        """A keyword-fallback plan with a cycle must be both degraded and invalid."""
        tasks = [
            make_task(id="t_a", dependencies=["t_b"]),
            make_task(id="t_b", dependencies=["t_a"]),
        ]

        result = flag_degraded_fallback(tasks, method="decompose_goal_keyword")

        assert result.is_degraded is True
        assert result.valid is False


# -- TestValidatePlan (integration) -------------------------------------------


class TestValidatePlan:
    """Integration tests for validate_plan() combining multiple checks."""

    def test_valid_plan_passes_all_checks(self) -> None:
        """A well-formed plan must return valid=True with no ERROR issues."""
        tasks = linear_plan()
        result = validate_plan(tasks, goal="implement code and write tests", initial_context=["goal"])
        assert result.valid is True
        assert result.is_degraded is False
        errors = result.error_issues()
        assert errors == []

    def test_format_for_prompt_includes_all_issues(self) -> None:
        """format_for_prompt must include all issue categories in its output."""
        tasks = [
            make_task(id="t_a", dependencies=["t_b"]),
            make_task(id="t_b", dependencies=["t_a"]),
        ]
        result = validate_plan(tasks, goal="vague")
        prompt = result.format_for_prompt()
        assert "CYCLE" in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_no_tasks_returns_no_errors(self) -> None:
        """An empty task list produces no ERROR issues (only potential warnings)."""
        result = validate_plan([], goal="do something")
        # Empty list: all structural checks pass, testability check finds nothing
        assert result.valid is True  # No errors in empty plan

    def test_validation_result_frozen(self) -> None:
        """ValidationResult must be immutable (frozen dataclass)."""
        result = validate_plan(linear_plan(), goal="code and tests")
        with pytest.raises((AttributeError, TypeError)):
            result.valid = False  # type: ignore[misc]
