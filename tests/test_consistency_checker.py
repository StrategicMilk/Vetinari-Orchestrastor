"""Tests for vetinari.verification.consistency_checker.

Verifies pattern extraction from Python source, inconsistency detection
within a single file and across multiple files, severity enforcement,
and suggested pattern recommendations.

Part of US-014: Implementation Consistency Checking.
"""

from __future__ import annotations

import pytest

from vetinari.verification.consistency_checker import (
    ConsistencyIssue,
    PatternCategory,
    PatternInstance,
    check_consistency,
    check_consistency_across_files,
    extract_patterns,
)

# -- Pattern extraction: FILE_TYPE_CHECK -------------------------------------


class TestFileTypeCheckPatterns:
    """Tests for detecting file type check patterns."""

    def test_endswith_detected(self) -> None:
        """filename.endswith('.py') must be detected as FILE_TYPE_CHECK/endswith."""
        source = 'if filename.endswith(".py"):\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        file_type = [p for p in patterns if p.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type) >= 1
        assert any(p.implementation == "endswith" for p in file_type)

    def test_in_set_detected(self) -> None:
        """ext in {'.py', '.js'} must be detected as FILE_TYPE_CHECK/in_set."""
        source = 'if ext in {".py", ".js"}:\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        file_type = [p for p in patterns if p.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type) >= 1
        assert any(p.implementation == "in_set" for p in file_type)

    def test_in_list_detected(self) -> None:
        """ext in ['.py'] must be detected as FILE_TYPE_CHECK/in_list."""
        source = 'if ext in [".py", ".js"]:\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        file_type = [p for p in patterns if p.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type) >= 1
        assert any(p.implementation == "in_list" for p in file_type)

    def test_endswith_tuple_detected(self) -> None:
        """filename.endswith(('.py', '.js')) must be detected."""
        source = 'if filename.endswith((".py", ".js")):\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        file_type = [p for p in patterns if p.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type) >= 1


# -- Pattern extraction: STRING_MATCHING -------------------------------------


class TestStringMatchingPatterns:
    """Tests for detecting string matching patterns."""

    def test_startswith_detected(self) -> None:
        """x.startswith('prefix') must be detected as STRING_MATCHING/startswith."""
        source = 'if name.startswith("test_"):\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        string_match = [p for p in patterns if p.category == PatternCategory.STRING_MATCHING]
        assert len(string_match) >= 1
        assert any(p.implementation == "startswith" for p in string_match)

    def test_regex_detected(self) -> None:
        """re.match(...) must be detected as STRING_MATCHING/regex."""
        source = 'import re\nif re.match(r"^test_", name):\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        string_match = [p for p in patterns if p.category == PatternCategory.STRING_MATCHING]
        assert len(string_match) >= 1
        assert any(p.implementation == "regex" for p in string_match)


# -- Pattern extraction: COLLECTION_MEMBERSHIP --------------------------------


class TestCollectionMembershipPatterns:
    """Tests for detecting collection membership patterns."""

    def test_in_set_non_extension(self) -> None:
        """x in {'a', 'b'} (non-extension) must be COLLECTION_MEMBERSHIP/in_set."""
        source = 'if status in {"active", "pending"}:\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        membership = [p for p in patterns if p.category == PatternCategory.COLLECTION_MEMBERSHIP]
        assert len(membership) >= 1
        assert any(p.implementation == "in_set" for p in membership)

    def test_in_list_non_extension(self) -> None:
        """x in ['a', 'b'] (non-extension) must be COLLECTION_MEMBERSHIP/in_list."""
        source = 'if status in ["active", "pending"]:\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        membership = [p for p in patterns if p.category == PatternCategory.COLLECTION_MEMBERSHIP]
        assert len(membership) >= 1
        assert any(p.implementation == "in_list" for p in membership)

    def test_extension_set_not_counted_as_collection(self) -> None:
        """ext in {'.py', '.js'} must be FILE_TYPE_CHECK, not COLLECTION_MEMBERSHIP."""
        source = 'if ext in {".py", ".js"}:\n    pass\n'
        patterns = extract_patterns(source, "test.py")

        membership = [p for p in patterns if p.category == PatternCategory.COLLECTION_MEMBERSHIP]
        assert len(membership) == 0


# -- Pattern extraction: NULL_CHECK ------------------------------------------


class TestNullCheckPatterns:
    """Tests for detecting null check patterns."""

    def test_is_none_detected(self) -> None:
        """x is None must be detected as NULL_CHECK/is_none."""
        source = "if x is None:\n    pass\n"
        patterns = extract_patterns(source, "test.py")

        null_checks = [p for p in patterns if p.category == PatternCategory.NULL_CHECK]
        assert len(null_checks) >= 1
        assert any(p.implementation == "is_none" for p in null_checks)

    def test_eq_none_detected(self) -> None:
        """x == None must be detected as NULL_CHECK/eq_none."""
        source = "if x == None:\n    pass\n"
        patterns = extract_patterns(source, "test.py")

        null_checks = [p for p in patterns if p.category == PatternCategory.NULL_CHECK]
        assert len(null_checks) >= 1
        assert any(p.implementation == "eq_none" for p in null_checks)

    def test_not_x_detected(self) -> None:
        """not x must be detected as NULL_CHECK/not_x."""
        source = "if not value:\n    pass\n"
        patterns = extract_patterns(source, "test.py")

        null_checks = [p for p in patterns if p.category == PatternCategory.NULL_CHECK]
        assert len(null_checks) >= 1
        assert any(p.implementation == "not_x" for p in null_checks)


# -- Pattern extraction: ITERATION_PATTERN ------------------------------------


class TestIterationPatterns:
    """Tests for detecting iteration patterns."""

    def test_for_append_detected(self) -> None:
        """for x in y: result.append(x) must be ITERATION_PATTERN/for_append."""
        source = "result = []\nfor item in items:\n    result.append(item)\n"
        patterns = extract_patterns(source, "test.py")

        iteration = [p for p in patterns if p.category == PatternCategory.ITERATION_PATTERN]
        assert len(iteration) >= 1
        assert any(p.implementation == "for_append" for p in iteration)

    def test_list_comprehension_detected(self) -> None:
        """[x for x in y] must be ITERATION_PATTERN/list_comprehension."""
        source = "result = [item for item in items]\n"
        patterns = extract_patterns(source, "test.py")

        iteration = [p for p in patterns if p.category == PatternCategory.ITERATION_PATTERN]
        assert len(iteration) >= 1
        assert any(p.implementation == "list_comprehension" for p in iteration)

    def test_map_call_detected(self) -> None:
        """map(func, iterable) must be ITERATION_PATTERN/map_call."""
        source = "result = list(map(str, items))\n"
        patterns = extract_patterns(source, "test.py")

        iteration = [p for p in patterns if p.category == PatternCategory.ITERATION_PATTERN]
        assert len(iteration) >= 1
        assert any(p.implementation == "map_call" for p in iteration)


# -- Consistency checking: single file ----------------------------------------


class TestSingleFileConsistency:
    """Tests for check_consistency() on a single file."""

    def test_consistent_file_no_issues(self) -> None:
        """A file using only endswith for file checks must produce no issues."""
        source = 'if name.endswith(".py"):\n    pass\nif other.endswith(".js"):\n    pass\n'
        issues = check_consistency(source, "test.py")
        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 0

    def test_inconsistent_file_flagged(self) -> None:
        """A file mixing endswith and in_set for extensions must produce an issue."""
        source = 'if name.endswith(".py"):\n    pass\nif ext in {".js", ".ts"}:\n    pass\n'
        issues = check_consistency(source, "test.py")

        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 1
        assert file_type_issues[0].severity == "medium"
        assert len(file_type_issues[0].instances) >= 2

    def test_inconsistent_null_checks_flagged(self) -> None:
        """A file mixing is_none and eq_none must produce an issue."""
        source = "if x is None:\n    pass\nif y == None:\n    pass\n"
        issues = check_consistency(source, "test.py")

        null_issues = [i for i in issues if i.category == PatternCategory.NULL_CHECK]
        assert len(null_issues) == 1
        assert null_issues[0].severity == "medium"

    def test_syntax_error_returns_empty(self) -> None:
        """Unparseable source must return an empty issue list, not raise."""
        issues = check_consistency("def broken(\n", "test.py")
        assert issues == []

    def test_issue_has_suggested_pattern(self) -> None:
        """Inconsistency issues must include a suggested_pattern recommendation."""
        source = 'if name.endswith(".py"):\n    pass\nif ext in {".js", ".ts"}:\n    pass\n'
        issues = check_consistency(source, "test.py")

        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 1
        assert file_type_issues[0].suggested_pattern == "endswith"

    def test_issue_message_describes_inconsistency(self) -> None:
        """The issue message must list the conflicting implementations."""
        source = 'if name.endswith(".py"):\n    pass\nif ext in {".js", ".ts"}:\n    pass\n'
        issues = check_consistency(source, "test.py")

        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 1
        assert "endswith" in file_type_issues[0].message
        assert "in_set" in file_type_issues[0].message


# -- Consistency checking: cross-file ----------------------------------------


class TestCrossFileConsistency:
    """Tests for check_consistency_across_files()."""

    def test_cross_file_inconsistency_detected(self) -> None:
        """Different extension check patterns across files must be flagged."""
        sources = {
            "file_a.py": 'if name.endswith(".py"):\n    pass\n',
            "file_b.py": 'if ext in {".py", ".js"}:\n    pass\n',
        }
        issues = check_consistency_across_files(sources)

        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 1

        # Instances should reference both files
        files_in_issue = {i.file_path for i in file_type_issues[0].instances}
        assert "file_a.py" in files_in_issue
        assert "file_b.py" in files_in_issue

    def test_consistent_across_files_no_issues(self) -> None:
        """Files all using the same pattern must produce no cross-file issues for that category."""
        sources = {
            "file_a.py": 'if name.endswith(".py"):\n    pass\n',
            "file_b.py": 'if other.endswith(".js"):\n    pass\n',
        }
        issues = check_consistency_across_files(sources)

        file_type_issues = [i for i in issues if i.category == PatternCategory.FILE_TYPE_CHECK]
        assert len(file_type_issues) == 0

    def test_empty_sources_returns_empty(self) -> None:
        """Empty source dict must return no issues."""
        assert check_consistency_across_files({}) == []

    def test_cross_file_iteration_inconsistency(self) -> None:
        """One file using for+append and another using comprehension must be flagged."""
        sources = {
            "file_a.py": "result = []\nfor item in items:\n    result.append(item)\n",
            "file_b.py": "result = [item for item in items]\n",
        }
        issues = check_consistency_across_files(sources)

        iter_issues = [i for i in issues if i.category == PatternCategory.ITERATION_PATTERN]
        assert len(iter_issues) == 1


# -- Data structure tests ----------------------------------------------------


class TestDataStructures:
    """Tests for PatternInstance, ConsistencyIssue, and PatternCategory."""

    def test_pattern_instance_frozen(self) -> None:
        """PatternInstance must be immutable."""
        inst = PatternInstance(
            category=PatternCategory.NULL_CHECK,
            implementation="is_none",
            file_path="test.py",
            line_number=1,
            code_snippet="if x is None:",
        )
        with pytest.raises((AttributeError, TypeError)):
            inst.implementation = "eq_none"  # type: ignore[misc]

    def test_pattern_instance_repr(self) -> None:
        """PatternInstance repr must show category, implementation, and file."""
        inst = PatternInstance(
            category=PatternCategory.NULL_CHECK,
            implementation="is_none",
            file_path="test.py",
            line_number=5,
            code_snippet="if x is None:",
        )
        r = repr(inst)
        assert "null_check" in r
        assert "is_none" in r
        assert "test.py" in r

    def test_consistency_issue_frozen(self) -> None:
        """ConsistencyIssue must be immutable."""
        issue = ConsistencyIssue(
            category=PatternCategory.NULL_CHECK,
            instances=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            issue.severity = "high"  # type: ignore[misc]

    def test_consistency_issue_repr(self) -> None:
        """ConsistencyIssue repr must show category and severity."""
        issue = ConsistencyIssue(
            category=PatternCategory.NULL_CHECK,
            instances=(),
            severity="medium",
        )
        r = repr(issue)
        assert "null_check" in r
        assert "medium" in r

    def test_all_severity_is_medium(self) -> None:
        """Per US-014 AC, all issues must have severity='medium'."""
        source = "if x is None:\n    pass\nif y == None:\n    pass\n"
        issues = check_consistency(source, "test.py")
        for issue in issues:
            assert issue.severity == "medium"


# -- Module wiring -----------------------------------------------------------


class TestModuleWiring:
    """Verify the module is importable and exports are correct."""

    def test_imports_from_consistency_checker(self) -> None:
        """Key types must be importable from vetinari.verification.consistency_checker."""
        from vetinari.verification.consistency_checker import (
            ConsistencyIssue,
            PatternCategory,
            PatternInstance,
            check_consistency,
            check_consistency_across_files,
            extract_patterns,
        )

        assert ConsistencyIssue is not None
        assert PatternCategory is not None
        assert PatternInstance is not None
        assert check_consistency is not None
        assert check_consistency_across_files is not None
        assert extract_patterns is not None

    def test_imports_from_verification_init(self) -> None:
        """Key types must be importable from vetinari.verification."""
        from vetinari.verification import (
            ConsistencyIssue,
            PatternCategory,
            check_consistency,
            extract_patterns,
        )

        assert ConsistencyIssue is not None
        assert PatternCategory is not None
        assert check_consistency is not None
        assert extract_patterns is not None
