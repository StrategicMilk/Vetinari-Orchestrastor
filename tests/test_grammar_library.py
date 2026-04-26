"""Tests for vetinari/adapters/grammar_library.py — US-005 grammar enforcement."""

from __future__ import annotations

import logging

import pytest

from vetinari.adapters.grammar_library import (
    GRAMMAR_LIBRARY,
    TASK_TYPE_TO_GRAMMAR,
    _brackets_balanced,
    _rfind_outside_strings,
    get_grammar,
    get_grammar_for_task_type,
    truncate_at_grammar_boundary,
    validate_grammar,
)

# -- Grammar library content tests -------------------------------------------


class TestGrammarLibraryContent:
    """Each named grammar must exist and be a non-empty valid GBNF string."""

    @pytest.mark.parametrize("name", ["json_object", "json_array", "yaml_document", "structured_review", "plan_dag"])
    def test_grammar_exists_and_non_empty(self, name: str) -> None:
        grammar = GRAMMAR_LIBRARY.get(name)
        assert grammar is not None, f"Grammar '{name}' missing from GRAMMAR_LIBRARY"
        assert isinstance(grammar, str), f"Grammar '{name}' must be a str"
        assert len(grammar.strip()) > 0, f"Grammar '{name}' must not be empty"

    @pytest.mark.parametrize("name", ["json_object", "json_array", "yaml_document", "structured_review", "plan_dag"])
    def test_grammar_has_root_rule(self, name: str) -> None:
        grammar = GRAMMAR_LIBRARY[name]
        # All grammars must define a 'root' rule as the entry point for llama-cpp
        assert "root" in grammar, f"Grammar '{name}' must define a 'root' rule"
        assert "::=" in grammar, f"Grammar '{name}' must contain GBNF rule definitions"

    @pytest.mark.parametrize("name", ["json_object", "json_array", "yaml_document", "structured_review", "plan_dag"])
    def test_grammar_passes_validation(self, name: str) -> None:
        grammar = GRAMMAR_LIBRARY[name]
        assert validate_grammar(grammar) is True, f"Grammar '{name}' did not pass validate_grammar"


# -- get_grammar tests --------------------------------------------------------


class TestGetGrammar:
    def test_known_name_returns_grammar(self) -> None:
        result = get_grammar("json_object")
        assert result is not None
        assert "root" in result

    def test_unknown_name_returns_none(self) -> None:
        result = get_grammar("nonexistent_grammar_xyz")
        assert result is None

    def test_empty_name_returns_none(self) -> None:
        result = get_grammar("")
        assert result is None

    @pytest.mark.parametrize("name", list(GRAMMAR_LIBRARY.keys()))
    def test_all_library_names_resolve(self, name: str) -> None:
        result = get_grammar(name)
        assert result is not None


# -- get_grammar_for_task_type tests ------------------------------------------


class TestGetGrammarForTaskType:
    @pytest.mark.parametrize(
        "task_type, expected_grammar_name",
        [
            ("json", "json_object"),
            ("json_object", "json_object"),
            ("json_array", "json_array"),
            ("plan", "plan_dag"),
            ("planning", "plan_dag"),
            ("plan_dag", "plan_dag"),
            ("review", "structured_review"),
            ("inspection", "structured_review"),
            ("quality_review", "structured_review"),
            ("structured_review", "structured_review"),
            ("yaml", "yaml_document"),
            ("yaml_document", "yaml_document"),
            ("config", "yaml_document"),
        ],
    )
    def test_known_task_types_return_correct_grammar(self, task_type: str, expected_grammar_name: str) -> None:
        result = get_grammar_for_task_type(task_type)
        expected = GRAMMAR_LIBRARY[expected_grammar_name]
        assert result == expected, f"task_type='{task_type}' should map to grammar '{expected_grammar_name}'"

    def test_unknown_task_type_returns_none(self) -> None:
        result = get_grammar_for_task_type("totally_unknown_task_xyz")
        assert result is None

    def test_empty_task_type_returns_none(self) -> None:
        result = get_grammar_for_task_type("")
        assert result is None

    def test_general_task_type_returns_none(self) -> None:
        # 'general' is intentionally not mapped — no grammar constraint for free-form tasks
        result = get_grammar_for_task_type("general")
        assert result is None

    def test_task_type_to_grammar_keys_all_map_to_existing_grammars(self) -> None:
        for task_type, grammar_name in TASK_TYPE_TO_GRAMMAR.items():
            assert grammar_name in GRAMMAR_LIBRARY, (
                f"task_type='{task_type}' maps to '{grammar_name}' which is not in GRAMMAR_LIBRARY"
            )


# -- validate_grammar tests ---------------------------------------------------


class TestValidateGrammar:
    def test_valid_grammar_accepted(self) -> None:
        grammar = 'root ::= "hello"'
        assert validate_grammar(grammar) is True

    def test_empty_string_rejected(self) -> None:
        assert validate_grammar("") is False

    def test_whitespace_only_rejected(self) -> None:
        assert validate_grammar("   \n\t  ") is False

    def test_grammar_without_rule_definitions_rejected(self) -> None:
        # No ::= operator — not a valid GBNF grammar
        malformed = "just some text without any rule definitions here"
        assert validate_grammar(malformed) is False

    def test_unbalanced_brackets_rejected(self) -> None:
        # Bracket outside a quoted string — the [ character class is never closed
        malformed = "root ::= [ a-z"
        assert validate_grammar(malformed) is False

    def test_extra_closing_bracket_rejected(self) -> None:
        # Extra ] outside any quoted string
        malformed = "root ::= ws ]"
        assert validate_grammar(malformed) is False

    def test_quoted_bracket_terminal_accepted(self) -> None:
        # A grammar using "[" or "]" as a quoted terminal must pass validation.
        # Previously _brackets_balanced counted quoted brackets as structural,
        # causing false rejections.
        grammar_with_quoted_bracket = 'root ::= "[" [a-z]+ "]"'
        assert validate_grammar(grammar_with_quoted_bracket) is True

    def test_all_library_grammars_pass_validation(self) -> None:
        for name, grammar in GRAMMAR_LIBRARY.items():
            assert validate_grammar(grammar) is True, f"Library grammar '{name}' failed validation"


# -- truncate_at_grammar_boundary tests ---------------------------------------


class TestTruncateAtGrammarBoundary:
    def test_text_within_budget_not_truncated(self) -> None:
        text = '{"key": "value"}'
        result = truncate_at_grammar_boundary(text, max_tokens=100)
        assert result == text

    def test_empty_text_returns_empty(self) -> None:
        result = truncate_at_grammar_boundary("", max_tokens=10)
        assert result == ""

    def test_json_object_truncated_at_closing_brace(self) -> None:
        # Build a JSON string that exceeds the budget
        # Budget: 5 tokens * 4 chars = 20 chars
        json_text = '{"a": 1}{"b": 2}{"c": 3} extra text that overflows'
        result = truncate_at_grammar_boundary(json_text, max_tokens=5, grammar_name="json_object")
        # Must end at } and be no longer than 20 chars
        assert result.endswith("}")
        assert len(result) <= 20

    def test_json_array_truncated_at_closing_bracket(self) -> None:
        # 5 tokens = 20 chars budget
        array_text = "[1, 2, 3][4, 5, 6] overflow content here extra"
        result = truncate_at_grammar_boundary(array_text, max_tokens=5, grammar_name="json_array")
        assert result.endswith("]")
        assert len(result) <= 20

    def test_yaml_truncated_at_newline(self) -> None:
        yaml_text = "key1: value1\nkey2: value2\nkey3: very long value that overflows the budget limit"
        # 5 tokens * 4 = 20 chars budget
        result = truncate_at_grammar_boundary(yaml_text, max_tokens=5, grammar_name="yaml_document")
        assert result.endswith("\n")
        assert len(result) <= 20

    def test_plain_text_truncated_at_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence that is very long and overflows."
        # 5 tokens * 4 = 20 chars
        result = truncate_at_grammar_boundary(text, max_tokens=5, grammar_name=None)
        # Should end at a period (sentence boundary) or be the hard limit
        assert len(result) <= 20

    def test_unknown_grammar_falls_back_to_hard_truncation(self) -> None:
        # Text without sentence boundaries or newlines
        text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # 3 tokens * 4 = 12 chars
        result = truncate_at_grammar_boundary(text, max_tokens=3, grammar_name="unknown_grammar")
        assert len(result) <= 12

    def test_structured_review_truncated_at_brace(self) -> None:
        review_text = '{"score": 8, "findings": [], "summary": "ok"}{"extra": "data"} overflow text'
        result = truncate_at_grammar_boundary(review_text, max_tokens=12, grammar_name="structured_review")
        assert result.endswith("}")
        assert len(result) <= 48  # 12 tokens * 4 chars

    def test_plan_dag_truncated_at_brace(self) -> None:
        plan_text = '{"tasks": [{"id": "t1", "description": "step one"}]}  extra overflow goes here'
        result = truncate_at_grammar_boundary(plan_text, max_tokens=13, grammar_name="plan_dag")
        assert result.endswith("}")
        assert len(result) <= 52  # 13 tokens * 4 chars


# -- _brackets_balanced helper tests ------------------------------------------


class TestBracketsBalanced:
    def test_empty_string_is_balanced(self) -> None:
        assert _brackets_balanced("") is True

    def test_balanced_braces(self) -> None:
        assert _brackets_balanced("root ::= { ws }") is True

    def test_balanced_brackets(self) -> None:
        assert _brackets_balanced("[a b c]") is True

    def test_balanced_parens(self) -> None:
        assert _brackets_balanced("(a | b)") is True

    def test_unmatched_open_brace(self) -> None:
        assert _brackets_balanced("root ::= { ws") is False

    def test_unmatched_close_bracket(self) -> None:
        assert _brackets_balanced("root ::= ws ]") is False

    def test_mismatched_brackets(self) -> None:
        assert _brackets_balanced("root ::= { ws ]") is False

    def test_quoted_brackets_are_balanced(self) -> None:
        # Brackets inside double-quoted GBNF string literals must NOT be counted
        # as structural brackets — "["  is a valid terminal, not an open char class.
        assert _brackets_balanced('root ::= "["') is True
        assert _brackets_balanced('root ::= "]"') is True
        assert _brackets_balanced('root ::= "["  "]"') is True
        assert _brackets_balanced('root ::= "[]"') is True
        # An unquoted naked "[" (e.g. an unclosed char class) is still unbalanced
        assert _brackets_balanced("root ::= [ a-z") is False

    def test_escape_inside_quoted_string(self) -> None:
        # Escaped quote inside a string must not prematurely end the string context
        assert _brackets_balanced(r'root ::= "a\"b"') is True


# -- _rfind_outside_strings helper tests --------------------------------------


class TestRfindOutsideStrings:
    """Verify that structural characters inside string literals are not matched."""

    def test_finds_brace_outside_string(self) -> None:
        assert _rfind_outside_strings('{"key": "val"}', "}") == 13

    def test_ignores_brace_inside_string_value(self) -> None:
        # The } inside the string literal "val with {braces}" must not be
        # returned; only the outermost structural } at the end should match.
        text = '{"key": "val with {braces}"}'
        idx = _rfind_outside_strings(text, "}")
        assert idx == len(text) - 1

    def test_ignores_bracket_inside_string(self) -> None:
        text = '{"arr": "[1,2,3]"}'
        # No structural ] in this text — the only ] is inside a string literal
        idx = _rfind_outside_strings(text, "]")
        assert idx == -1

    def test_not_found_returns_minus_one(self) -> None:
        assert _rfind_outside_strings('{"a": "b"}', "]") == -1

    def test_empty_string(self) -> None:
        assert _rfind_outside_strings("", "}") == -1

    def test_escaped_quote_in_string_value(self) -> None:
        # The escaped \" must not end the string context prematurely.
        text = r'{"key": "val\"}"}'
        # The } at position -1 is the structural close; the } inside the
        # escaped string is not structural.
        idx = _rfind_outside_strings(text, "}")
        assert idx == len(text) - 1


class TestTruncateAtGrammarBoundaryStringAware:
    """Verify that truncation does not split at braces inside string literals."""

    def test_json_truncation_skips_brace_inside_string(self) -> None:
        # The { and } inside "value with {braces}" must not be treated as
        # structural boundaries.  The last real structural } is at the end.
        # Budget: 10 tokens * 4 = 40 chars
        text = '{"key": "value with {braces}"}  extra overflow text here'
        result = truncate_at_grammar_boundary(text, max_tokens=10, grammar_name="json_object")
        # Result must end at the structural } (position 29), not at the one inside the string
        assert result == '{"key": "value with {braces}"}'
        assert result.endswith("}")

    def test_json_array_truncation_skips_bracket_inside_string(self) -> None:
        # The ] inside the string literal "[nested]" must not be used as a boundary.
        # Budget: 10 tokens * 4 = 40 chars
        text = '["first", "second [nested]", "third"] extra overflow'
        result = truncate_at_grammar_boundary(text, max_tokens=10, grammar_name="json_array")
        assert result == '["first", "second [nested]", "third"]'
        assert result.endswith("]")


# -- Auto-selection wiring test -----------------------------------------------


class TestAutoSelectionWiring:
    """Verify that LlamaCppProviderAdapter wires grammar auto-selection correctly."""

    def test_task_type_triggers_grammar_in_completion_kwargs(self) -> None:
        """When task_type is set and grammar is None, grammar is auto-selected."""
        from vetinari.adapters.base import InferenceRequest
        from vetinari.adapters.llama_cpp_adapter import _build_completion_kwargs

        request = InferenceRequest(
            model_id="test-model",
            prompt="Write a plan",
            task_type="plan",
            grammar=None,
        )
        messages = [{"role": "user", "content": request.prompt}]

        kwargs = _build_completion_kwargs(request, messages)

        # Grammar for 'plan' task_type should have been auto-selected
        assert "grammar" in kwargs, "Expected grammar to be auto-selected for task_type='plan'"
        plan_grammar = get_grammar_for_task_type("plan")
        assert kwargs["grammar"] == plan_grammar

    def test_explicit_grammar_takes_precedence_over_task_type(self) -> None:
        """When grammar is set explicitly, it overrides task_type-derived grammar."""
        from vetinari.adapters.base import InferenceRequest
        from vetinari.adapters.llama_cpp_adapter import _build_completion_kwargs

        explicit_grammar = 'root ::= "explicit"'
        request = InferenceRequest(
            model_id="test-model",
            prompt="Test",
            task_type="plan",
            grammar=explicit_grammar,
        )
        messages = [{"role": "user", "content": request.prompt}]

        kwargs = _build_completion_kwargs(request, messages)

        assert kwargs["grammar"] == explicit_grammar

    def test_unknown_task_type_does_not_set_grammar(self) -> None:
        """task_type with no grammar mapping leaves grammar key absent."""
        from vetinari.adapters.base import InferenceRequest
        from vetinari.adapters.llama_cpp_adapter import _build_completion_kwargs

        request = InferenceRequest(
            model_id="test-model",
            prompt="Do something general",
            task_type="general",
            grammar=None,
        )
        messages = [{"role": "user", "content": request.prompt}]

        kwargs = _build_completion_kwargs(request, messages)

        assert "grammar" not in kwargs

    def test_no_task_type_no_grammar_key_absent(self) -> None:
        """When neither grammar nor task_type is set, grammar key is absent."""
        from vetinari.adapters.base import InferenceRequest
        from vetinari.adapters.llama_cpp_adapter import _build_completion_kwargs

        request = InferenceRequest(
            model_id="test-model",
            prompt="Hello",
        )
        messages = [{"role": "user", "content": request.prompt}]

        kwargs = _build_completion_kwargs(request, messages)

        assert "grammar" not in kwargs

    def test_invalid_grammar_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A malformed grammar is rejected before reaching llama-cpp-python."""
        from vetinari.adapters.base import InferenceRequest
        from vetinari.adapters.llama_cpp_adapter import _build_completion_kwargs

        malformed_grammar = "not a valid grammar at all"
        request = InferenceRequest(
            model_id="test-model",
            prompt="Test",
            grammar=malformed_grammar,
        )
        messages = [{"role": "user", "content": request.prompt}]

        with caplog.at_level(logging.WARNING):
            kwargs = _build_completion_kwargs(request, messages)

        assert "grammar" not in kwargs
        assert "Invalid grammar" in caplog.text
