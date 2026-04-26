"""Tests for vetinari.validation.entailment_checker.EntailmentChecker (Tier 2 cascade)."""

from __future__ import annotations

import pytest

from vetinari.validation.entailment_checker import EntailmentChecker, EntailmentResult


class TestEntailmentResult:
    """EntailmentResult dataclass behaves correctly."""

    def test_defaults_to_empty_missing_keywords(self) -> None:
        """missing_keywords defaults to [] when not supplied."""
        result = EntailmentResult(entailed=True, coverage=1.0)
        assert result.missing_keywords == []

    def test_explicit_missing_keywords_preserved(self) -> None:
        """Explicitly provided missing_keywords are stored."""
        result = EntailmentResult(entailed=False, coverage=0.2, missing_keywords=["search", "binary"])
        assert "search" in result.missing_keywords
        assert "binary" in result.missing_keywords

    def test_similarity_defaults_to_none(self) -> None:
        """similarity defaults to None when sentence-transformers is not used."""
        result = EntailmentResult(entailed=True, coverage=0.8)
        assert result.similarity is None

    def test_repr_contains_key_fields(self) -> None:
        """__repr__ shows entailed, coverage, and similarity."""
        result = EntailmentResult(entailed=True, coverage=0.75, similarity=0.82)
        r = repr(result)
        assert "True" in r
        assert "0.750" in r
        assert "0.82" in r


class TestExtractKeywords:
    """_extract_keywords returns meaningful content words."""

    def test_filters_short_words(self) -> None:
        """Words shorter than 4 characters are excluded."""
        checker = EntailmentChecker()
        keywords = checker._extract_keywords("add two numbers together")
        assert "add" not in keywords
        assert "two" not in keywords

    def test_filters_stop_words(self) -> None:
        """Common stop words are excluded from keyword list."""
        checker = EntailmentChecker()
        keywords = checker._extract_keywords("implement a function that will sort data")
        # "that" and "will" are stop words
        assert "that" not in keywords
        assert "will" not in keywords

    def test_extracts_domain_words(self) -> None:
        """Domain-relevant words >= 4 characters are extracted."""
        checker = EntailmentChecker()
        keywords = checker._extract_keywords("implement a binary search function")
        assert "implement" in keywords
        assert "binary" in keywords
        assert "search" in keywords
        assert "function" in keywords

    def test_lowercases_words(self) -> None:
        """Keywords are returned in lowercase."""
        checker = EntailmentChecker()
        keywords = checker._extract_keywords("Implement Binary Search")
        assert "implement" in keywords
        assert "binary" in keywords

    def test_deduplicates_words(self) -> None:
        """Repeated words appear only once in the keyword list."""
        checker = EntailmentChecker()
        keywords = checker._extract_keywords("search search search for the value")
        assert keywords.count("search") == 1

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty string yields an empty keyword list."""
        checker = EntailmentChecker()
        assert checker._extract_keywords("") == []


class TestCheckEmptyInputs:
    """check() handles empty/missing inputs gracefully."""

    def test_empty_task_description_returns_not_entailed(self) -> None:
        """Empty task description yields entailed=False and coverage=0.0."""
        checker = EntailmentChecker()
        result = checker.check("", "def something(): pass")
        assert result.entailed is False
        assert result.coverage == 0.0

    def test_empty_response_returns_not_entailed(self) -> None:
        """Empty response text yields entailed=False and coverage=0.0."""
        checker = EntailmentChecker()
        result = checker.check("implement a sort function", "")
        assert result.entailed is False
        assert result.coverage == 0.0

    def test_both_empty_returns_not_entailed(self) -> None:
        """Both inputs empty yields EntailmentResult with entailed=False."""
        checker = EntailmentChecker()
        result = checker.check("", "")
        assert result.entailed is False

    def test_returns_entailment_result(self) -> None:
        """check() always returns an EntailmentResult instance."""
        checker = EntailmentChecker()
        result = checker.check("", "")
        assert isinstance(result, EntailmentResult)


class TestKeywordCoverage:
    """Coverage score reflects how many task keywords appear in the response."""

    def test_full_coverage_passes(self) -> None:
        """Response containing all task keywords yields high coverage and entailed=True."""
        checker = EntailmentChecker()
        task = "implement a binary search function"
        # Use prose that explicitly includes every keyword as a standalone word
        response = (
            "Here is how to implement binary search.\n"
            "The search function scans a sorted array.\n"
            "def binary_search_fn(arr, target): pass\n"
        )
        result = checker.check(task, response)
        # All four keywords (implement, binary, search, function) present as standalone words
        assert result.coverage == pytest.approx(1.0, abs=0.01)
        assert result.entailed is True

    def test_zero_coverage_fails(self) -> None:
        """Response with no matching keywords yields low coverage and entailed=False."""
        checker = EntailmentChecker()
        task = "implement a database migration with schema versioning"
        response = "The weather today is sunny and warm."
        result = checker.check(task, response)
        assert result.coverage < 0.2
        assert result.entailed is False

    def test_partial_coverage_below_threshold_fails(self) -> None:
        """Partial keyword coverage below _MIN_KEYWORD_COVERAGE fails entailment."""
        checker = EntailmentChecker()
        # Craft a task with many keywords, response covers only a few
        task = (
            "implement a complete authentication system with password hashing token validation and session management"
        )
        response = "Here is something about tokens."
        result = checker.check(task, response)
        # Should fail unless coverage hits the 0.4 threshold
        assert result.entailed is False or result.coverage >= 0.4

    def test_coverage_stored_in_result(self) -> None:
        """Result.coverage is between 0.0 and 1.0 inclusive."""
        checker = EntailmentChecker()
        result = checker.check("sort the items list", "def sort(items): return sorted(items)")
        assert 0.0 <= result.coverage <= 1.0

    def test_missing_keywords_listed(self) -> None:
        """Keywords present in task but absent from response are in missing_keywords."""
        checker = EntailmentChecker()
        task = "implement binary search algorithm"
        response = "Here is some code."
        result = checker.check(task, response)
        # "binary", "search", "algorithm", "implement" should be missing
        assert isinstance(result.missing_keywords, list)
        assert len(result.missing_keywords) > 0

    def test_no_missing_keywords_when_full_coverage(self) -> None:
        """When response covers all keywords, missing_keywords is empty."""
        checker = EntailmentChecker()
        task = "sort data"  # short enough that response can cover all meaningful keywords
        response = "def sort_data(data): return sorted(data)"
        result = checker.check(task, response)
        # If all keywords found, missing list should be empty
        if result.coverage == 1.0:
            assert result.missing_keywords == []

    def test_no_meaningful_keywords_returns_not_entailed(self) -> None:
        """If task has no extractable keywords, check returns entailed=False with coverage=0.0.

        Returning entailed=True here would certify any response as valid — the
        default-pass verifier anti-pattern.  Zero trust is the correct conservative default.
        """
        checker = EntailmentChecker()
        # "do it" — "do" and "it" are both < 4 chars, so no keywords extracted
        result = checker.check("do it", "okay sure")
        # No keywords to check — must NOT default to passing
        assert result.entailed is False
        assert result.coverage == 0.0

    def test_word_boundary_prevents_substring_false_positive(self) -> None:
        """'research' in the response must NOT satisfy the keyword 'search' from the task.

        Without whole-word matching, 'research' contains 'search' and would produce
        a false positive coverage hit.
        """
        checker = EntailmentChecker()
        # Task requires "search" (4 chars, passes keyword filter)
        # Response only contains "research" — a different word that happens to include "search"
        result = checker.check("implement a search function", "this is a research function")
        # "search" must NOT be matched inside "research"
        assert "search" in result.missing_keywords


class TestHighCoverageExamples:
    """Real-world examples where response clearly satisfies the task."""

    def test_binary_search_implementation(self) -> None:
        """A proper binary search implementation entails the task."""
        checker = EntailmentChecker()
        task = "implement a binary search function that finds an element in a sorted list"
        response = (
            "def binary_search(sorted_list, element):\n"
            "    low, high = 0, len(sorted_list) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if sorted_list[mid] == element:\n"
            "            return mid\n"
            "        elif sorted_list[mid] < element:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        )
        result = checker.check(task, response)
        assert result.entailed is True
        assert result.coverage >= 0.4

    def test_sort_function_entails_sort_task(self) -> None:
        """A sort function response entails a sort task."""
        checker = EntailmentChecker()
        task = "implement a function to sort a list of integers"
        response = "def sort_integers(integers):\n    return sorted(integers)\n"
        result = checker.check(task, response)
        assert result.entailed is True


class TestLowCoverageExamples:
    """Real-world examples where response clearly does not satisfy the task."""

    def test_prose_does_not_entail_code_task(self) -> None:
        """Prose answer does not entail a code implementation task."""
        checker = EntailmentChecker()
        task = "implement a database migration with schema versioning"
        response = "The weather today is sunny and warm with a light breeze from the north."
        result = checker.check(task, response)
        assert result.entailed is False
        assert result.coverage < 0.2

    def test_unrelated_code_does_not_entail_task(self) -> None:
        """Code about a completely different topic does not entail the task."""
        checker = EntailmentChecker()
        task = "implement a linear regression model for stock price prediction"
        response = "def greet(name):\n    return f'Hello {name}'\n"
        result = checker.check(task, response)
        assert result.entailed is False


class TestSemanticSimilarity:
    """_semantic_similarity gracefully handles missing sentence-transformers."""

    def test_returns_none_or_float_when_unavailable(self) -> None:
        """When sentence-transformers is not installed, returns None without error."""
        checker = EntailmentChecker()
        result = checker._semantic_similarity("implement search", "def search(): pass")
        # Either None (not installed) or a float in [0, 1]
        assert result is None or (isinstance(result, float) and 0.0 <= result <= 1.0)

    def test_check_does_not_raise_when_transformers_missing(self) -> None:
        """check() completes without raising even when sentence-transformers is absent."""
        checker = EntailmentChecker()
        # Should not raise regardless of whether transformers is installed
        result = checker.check("implement binary search", "def binary_search(): pass")
        assert isinstance(result, EntailmentResult)

    def test_similarity_field_none_when_unavailable(self) -> None:
        """When sentence-transformers is not installed, result.similarity is None."""
        checker = EntailmentChecker()
        result = checker.check("implement something", "def something(): pass")
        # If transformers not installed, similarity will be None
        assert result.similarity is None or isinstance(result.similarity, float)


class TestCoverageRounding:
    """Coverage values are rounded to 3 decimal places."""

    def test_coverage_is_rounded(self) -> None:
        """coverage field is rounded to 3 decimal places."""
        checker = EntailmentChecker()
        result = checker.check(
            "implement a function to search data structures efficiently",
            "def implement_search(data):\n    pass",
        )
        # Round to 3 dp means at most 3 digits after decimal
        coverage_str = str(result.coverage)
        if "." in coverage_str:
            _, decimals = coverage_str.split(".")
            assert len(decimals) <= 3


class TestResultFieldTypes:
    """check() always returns correctly-typed fields."""

    def test_entailed_is_bool(self) -> None:
        """entailed field is always a bool."""
        checker = EntailmentChecker()
        result = checker.check("implement search", "def search(): pass")
        assert isinstance(result.entailed, bool)

    def test_coverage_is_float(self) -> None:
        """coverage field is always a float."""
        checker = EntailmentChecker()
        result = checker.check("implement search", "def search(): pass")
        assert isinstance(result.coverage, float)

    def test_missing_keywords_is_list(self) -> None:
        """missing_keywords is always a list."""
        checker = EntailmentChecker()
        result = checker.check("implement binary search", "Here is a response.")
        assert isinstance(result.missing_keywords, list)
