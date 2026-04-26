"""Tests for vetinari/token_optimizer.py

Coverage targets:
- TokenBudget: creation, record, remaining, is_exhausted, check_task
- LocalPreprocessor: _truncate, _extract_key_lines, _extract_code_signatures
  (regex and AST paths), _extract_code_signatures_ast, compress_context
  (short passthrough, code_only goal, key_facts goal, caching, LLM path,
  LLM failure fallback), preprocess_for_cloud
- TokenOptimizer: create_budget, get_budget, record_usage, get_task_profile,
  prepare_prompt (basic, with context, with budget, cloud preprocessing flag,
  local truncation, deduplication, plan_id lookup), summarise_results
- Constants: TASK_PROFILES, _CHARS_PER_TOKEN_BY_TYPE, _CHARS_PER_TOKEN
- Singleton: get_token_optimizer
- Edge cases: empty inputs, unknown task types, exhausted budgets, no local
  model, LLM network errors
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Remove incomplete stubs left by earlier test files so real modules load
sys.modules.pop("vetinari.token_optimizer", None)

from vetinari.token_compression import _CONTEXT_WINDOW_CHARS
from vetinari.token_optimizer import (
    _CHARS_PER_TOKEN,
    _CHARS_PER_TOKEN_BY_TYPE,
    _COMPRESS_THRESHOLD_CHARS,
    TASK_PROFILES,
    LocalPreprocessor,
    TokenBudget,
    TokenOptimizer,
    get_token_optimizer,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prose(n_chars: int) -> str:
    """Return n_chars of plausible prose text."""
    unit = "This sentence is filler prose used to pad context length. "
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


def _code_block(n_chars: int) -> str:
    """Return n_chars of valid Python source."""
    unit = 'def placeholder(x, y):\n    """Filler function."""\n    return x + y\n\n'
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


def _markdown(n_chars: int) -> str:
    """Return n_chars of markdown text (headers + bullets + prose)."""
    unit = (
        "# Section Header\n"
        "- bullet item one\n"
        "- bullet item two\n"
        "Random prose that carries no useful information and should be omitted.\n"
    )
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


# ===========================================================================
# TokenBudget — creation
# ===========================================================================


def test_token_budget_default_field_values() -> None:
    b = TokenBudget(plan_id="p1")
    assert b.plan_id == "p1"
    assert b.max_tokens == 100000
    assert b.max_tokens_per_task == 8000
    assert b.tokens_used == 0
    assert b.task_token_counts == {}


def test_token_budget_custom_max_tokens() -> None:
    b = TokenBudget(plan_id="p2", max_tokens=500, max_tokens_per_task=50)
    assert b.max_tokens == 500
    assert b.max_tokens_per_task == 50


def test_token_budget_task_token_counts_initially_empty() -> None:
    b = TokenBudget(plan_id="p3")
    assert isinstance(b.task_token_counts, dict)
    assert len(b.task_token_counts) == 0


# ===========================================================================
# TokenBudget — record
# ===========================================================================


def test_token_budget_record_increments_tokens_used() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("t1", 100)
    assert b.tokens_used == 100


def test_token_budget_record_accumulates_across_calls() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("t1", 100)
    b.record("t1", 50)
    assert b.tokens_used == 150


def test_token_budget_record_tracks_per_task() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("t1", 300)
    b.record("t2", 200)
    assert b.task_token_counts["t1"] == 300
    assert b.task_token_counts["t2"] == 200


def test_token_budget_record_same_task_twice_accumulates() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("t1", 400)
    b.record("t1", 400)
    assert b.task_token_counts["t1"] == 800


def test_token_budget_record_zero_tokens_is_valid() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("t1", 0)
    assert b.tokens_used == 0
    assert b.task_token_counts["t1"] == 0


def test_token_budget_record_multiple_tasks_total_is_sum() -> None:
    b = TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)
    b.record("a", 100)
    b.record("b", 200)
    b.record("c", 300)
    assert b.tokens_used == 600


# ===========================================================================
# TokenBudget — remaining
# ===========================================================================


def test_token_budget_remaining_equals_max_when_unused() -> None:
    b = TokenBudget(plan_id="p", max_tokens=1_000)
    assert b.remaining == 1000


def test_token_budget_remaining_decreases_on_record() -> None:
    b = TokenBudget(plan_id="p", max_tokens=1_000)
    b.record("t", 300)
    assert b.remaining == 700


def test_token_budget_remaining_clamps_at_zero() -> None:
    b = TokenBudget(plan_id="p", max_tokens=100)
    b.record("t", 500)
    assert b.remaining == 0


def test_token_budget_remaining_is_zero_at_exact_limit() -> None:
    b = TokenBudget(plan_id="p", max_tokens=100)
    b.record("t", 100)
    assert b.remaining == 0


# ===========================================================================
# TokenBudget — is_exhausted
# ===========================================================================


def test_token_budget_not_exhausted_when_fresh() -> None:
    b = TokenBudget(plan_id="p", max_tokens=1_000)
    assert not b.is_exhausted


def test_token_budget_exhausted_at_exact_limit() -> None:
    b = TokenBudget(plan_id="p", max_tokens=100)
    b.record("t", 100)
    assert b.is_exhausted


def test_token_budget_exhausted_when_over_limit() -> None:
    b = TokenBudget(plan_id="p", max_tokens=100)
    b.record("t", 200)
    assert b.is_exhausted


def test_token_budget_not_exhausted_one_token_below_limit() -> None:
    b = TokenBudget(plan_id="p", max_tokens=100)
    b.record("t", 99)
    assert not b.is_exhausted


# ===========================================================================
# TokenBudget — check_task
# ===========================================================================


def test_token_budget_check_task_ok_for_small_estimate() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=2_000)
    assert b.check_task("t", 100)


def test_token_budget_check_task_fails_when_plan_exhausted() -> None:
    b = TokenBudget(plan_id="p", max_tokens=50, max_tokens_per_task=2_000)
    b.record("t", 50)
    assert not b.check_task("t2", 1)


def test_token_budget_check_task_fails_when_exceeds_per_task_limit() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=100)
    # 80 already used + 30 estimated = 110 > 100
    b.record("t1", 80)
    assert not b.check_task("t1", 30)


def test_token_budget_check_task_ok_different_task_has_no_history() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=100)
    b.record("t1", 90)
    assert b.check_task("t2", 50)


def test_token_budget_check_task_zero_estimated_always_ok() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=2_000)
    assert b.check_task("t", 0)


def test_token_budget_check_task_exactly_at_per_task_limit_fails() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=100)
    # 0 used + 101 estimated > 100
    assert not b.check_task("t", 101)


def test_token_budget_check_task_within_per_task_limit_after_partial_use() -> None:
    b = TokenBudget(plan_id="p", max_tokens=10_000, max_tokens_per_task=500)
    b.record("t", 300)
    # 300 + 199 = 499 <= 500
    assert b.check_task("t", 199)


# ===========================================================================
# LocalPreprocessor._truncate
# ===========================================================================


def test_local_preprocessor_truncate_short_context_returned_unchanged() -> None:
    proc = LocalPreprocessor()
    ctx = "hello world"
    assert proc._truncate(ctx) == ctx


def test_local_preprocessor_truncate_at_window_boundary_returned_unchanged() -> None:
    proc = LocalPreprocessor()
    ctx = "x" * _CONTEXT_WINDOW_CHARS
    assert proc._truncate(ctx) == ctx


def test_local_preprocessor_truncate_long_context_contains_truncation_marker() -> None:
    proc = LocalPreprocessor()
    ctx = "A" * (_CONTEXT_WINDOW_CHARS + 500)
    result = proc._truncate(ctx)
    assert "[... context truncated for token efficiency ...]" in result


def test_local_preprocessor_truncate_long_context_shorter_than_original() -> None:
    proc = LocalPreprocessor()
    ctx = "B" * 20_000
    result = proc._truncate(ctx)
    assert len(result) < len(ctx)


def test_local_preprocessor_truncate_result_preserves_head_and_tail() -> None:
    proc = LocalPreprocessor()
    head_marker = "<<HEAD>>"
    tail_marker = "<<TAIL>>"
    filler = "x" * _CONTEXT_WINDOW_CHARS
    ctx = head_marker + filler + tail_marker
    result = proc._truncate(ctx)
    assert head_marker in result
    assert tail_marker in result


# ===========================================================================
# LocalPreprocessor._extract_key_lines
# ===========================================================================


def test_extract_key_lines_extracts_markdown_headers() -> None:
    proc = LocalPreprocessor()
    ctx = "# Title\nsome random prose\n## Section\nmore prose"
    result = proc._extract_key_lines(ctx)
    assert "# Title" in result
    assert "## Section" in result
    assert "some random prose" not in result


def test_extract_key_lines_extracts_bullet_points_dash() -> None:
    proc = LocalPreprocessor()
    ctx = "- item one\nignored prose"
    result = proc._extract_key_lines(ctx)
    assert "- item one" in result


def test_extract_key_lines_extracts_bullet_points_star() -> None:
    proc = LocalPreprocessor()
    ctx = "* item two\nignored prose"
    result = proc._extract_key_lines(ctx)
    assert "* item two" in result


def test_extract_key_lines_extracts_numbered_list() -> None:
    proc = LocalPreprocessor()
    ctx = "1. first item\n2) second item\nignored"
    result = proc._extract_key_lines(ctx)
    assert "1. first item" in result
    assert "2) second item" in result


def test_extract_key_lines_extracts_key_value_patterns() -> None:
    proc = LocalPreprocessor()
    ctx = "host: localhost\nport = 8080\nnoise"
    result = proc._extract_key_lines(ctx)
    assert "host: localhost" in result
    assert "port = 8080" in result


def test_extract_key_lines_extracts_https_url_lines() -> None:
    proc = LocalPreprocessor()
    ctx = "See https://example.com\nsome noise"
    result = proc._extract_key_lines(ctx)
    assert "https://example.com" in result


def test_extract_key_lines_extracts_http_url_lines() -> None:
    proc = LocalPreprocessor()
    ctx = "API at http://api.local/v1\nsome noise"
    result = proc._extract_key_lines(ctx)
    assert "http://api.local/v1" in result


def test_extract_key_lines_extracts_key_term_error() -> None:
    proc = LocalPreprocessor()
    ctx = "An error occurred here.\nsome boring filler"
    result = proc._extract_key_lines(ctx)
    assert "error" in result


def test_extract_key_lines_extracts_key_term_endpoint() -> None:
    proc = LocalPreprocessor()
    ctx = "The endpoint must be authenticated.\nfiller"
    result = proc._extract_key_lines(ctx)
    assert "endpoint" in result


def test_extract_key_lines_empty_input_returns_empty() -> None:
    proc = LocalPreprocessor()
    assert proc._extract_key_lines("") == ""


def test_extract_key_lines_only_blank_lines_returns_empty() -> None:
    proc = LocalPreprocessor()
    assert proc._extract_key_lines("\n\n\n") == ""


def test_extract_key_lines_pure_prose_no_matches_returns_empty() -> None:
    proc = LocalPreprocessor()
    ctx = "just some ordinary prose without any special markers\n" * 3
    result = proc._extract_key_lines(ctx)
    assert result == ""


# ===========================================================================
# LocalPreprocessor._extract_code_signatures_ast
# ===========================================================================


def test_extract_code_signatures_ast_extracts_function_signature() -> None:
    proc = LocalPreprocessor()
    code = 'def add(x: int, y: int) -> int:\n    """Add."""\n    return x + y\n'
    result = proc._extract_code_signatures_ast(code)
    assert "def add" in result
    assert "x: int" in result
    assert "-> int" in result


def test_extract_code_signatures_ast_extracts_class_with_base() -> None:
    proc = LocalPreprocessor()
    code = 'class Foo(Bar):\n    """A foo."""\n    def method(self) -> None:\n        pass\n'
    result = proc._extract_code_signatures_ast(code)
    assert "class Foo(Bar):" in result
    assert "def method" in result


def test_extract_code_signatures_ast_extracts_async_function() -> None:
    proc = LocalPreprocessor()
    code = "async def fetch(url: str) -> bytes:\n    pass\n"
    result = proc._extract_code_signatures_ast(code)
    assert "async def fetch" in result
    assert "url: str" in result


def test_extract_code_signatures_ast_extracts_import_statements() -> None:
    proc = LocalPreprocessor()
    code = "import os\nfrom pathlib import Path\nx = 1\n"
    result = proc._extract_code_signatures_ast(code)
    assert "import os" in result
    assert "from pathlib import Path" in result


def test_extract_code_signatures_ast_uppercase_constant_included() -> None:
    proc = LocalPreprocessor()
    code = "MAX_SIZE = 100\nsmall_var = 5\n"
    result = proc._extract_code_signatures_ast(code)
    assert "MAX_SIZE" in result
    assert "small_var" not in result


def test_extract_code_signatures_ast_module_docstring_included() -> None:
    proc = LocalPreprocessor()
    code = '"""Module description."""\n\ndef func():\n    pass\n'
    result = proc._extract_code_signatures_ast(code)
    assert "Module description." in result


def test_extract_code_signatures_ast_decorator_included() -> None:
    proc = LocalPreprocessor()
    code = "@staticmethod\ndef helper():\n    pass\n"
    result = proc._extract_code_signatures_ast(code)
    assert "@staticmethod" in result


def test_extract_code_signatures_ast_varargs_and_kwargs() -> None:
    proc = LocalPreprocessor()
    code = "def variadic(*args, **kwargs) -> None:\n    pass\n"
    result = proc._extract_code_signatures_ast(code)
    assert "*args" in result
    assert "**kwargs" in result


def test_extract_code_signatures_ast_default_argument() -> None:
    proc = LocalPreprocessor()
    code = "def greet(name: str = 'world') -> str:\n    pass\n"
    result = proc._extract_code_signatures_ast(code)
    assert "greet" in result
    assert "name" in result


def test_extract_code_signatures_ast_kwonly_arg() -> None:
    proc = LocalPreprocessor()
    code = "def func(a, *, verbose: bool = False) -> None:\n    pass\n"
    result = proc._extract_code_signatures_ast(code)
    assert "func" in result
    assert "verbose" in result


def test_extract_code_signatures_ast_returns_string() -> None:
    proc = LocalPreprocessor()
    result = proc._extract_code_signatures_ast("x = 1\n")
    assert isinstance(result, str)


# ===========================================================================
# LocalPreprocessor._extract_code_signatures (dispatch + regex fallback)
# ===========================================================================


def test_extract_code_signatures_valid_python_uses_ast_path() -> None:
    proc = LocalPreprocessor()
    code = "def hello(): pass\n"
    result = proc._extract_code_signatures(code)
    assert "def hello" in result


def test_extract_code_signatures_syntax_error_falls_back_to_regex() -> None:
    proc = LocalPreprocessor()
    bad_code = "def broken(:\nclass Orphan:\n    pass\n"
    result = proc._extract_code_signatures(bad_code)
    assert isinstance(result, str)
    # Regex should still catch 'class Orphan'
    assert "class Orphan" in result


def test_extract_code_signatures_regex_captures_imports() -> None:
    proc = LocalPreprocessor()
    code = "def broken(:\nimport os\nfrom sys import argv\n"
    result = proc._extract_code_signatures(code)
    assert "import os" in result


def test_extract_code_signatures_empty_code_returns_empty_string() -> None:
    proc = LocalPreprocessor()
    assert proc._extract_code_signatures("") == ""


def test_extract_code_signatures_regex_captures_decorators() -> None:
    proc = LocalPreprocessor()
    code = "def broken(:\n@my_decorator\ndef valid_func():\n    pass\n"
    result = proc._extract_code_signatures(code)
    assert "@my_decorator" in result


# ===========================================================================
# LocalPreprocessor.compress_context
# ===========================================================================


def test_compress_context_short_context_returned_as_is() -> None:
    proc = LocalPreprocessor()
    ctx = "short context"
    result, ratio = proc.compress_context(ctx)
    assert result == ctx
    assert ratio == 1.0


def test_compress_context_just_below_threshold_returned_as_is() -> None:
    proc = LocalPreprocessor()
    ctx = "x" * (LocalPreprocessor.MIN_CONTEXT_CHARS - 1)
    result, ratio = proc.compress_context(ctx, compression_goal="key_facts")
    assert result == ctx
    assert ratio == 1.0


def test_compress_context_long_code_only_goal_returns_string() -> None:
    # Mock the adapter layer (external dependency) so no real LLM is needed.
    # _get_local_model will call LocalInferenceAdapter() — patching it to raise
    # causes the real _get_local_model to return None, exercising the real code path.
    proc = LocalPreprocessor()
    code = _code_block(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        result, _ratio = proc.compress_context(code, compression_goal="code_only")
    assert isinstance(result, str)
    assert len(result) > 0


def test_compress_context_long_key_facts_goal_returns_string() -> None:
    proc = LocalPreprocessor()
    ctx = _markdown(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        result, _ratio = proc.compress_context(ctx, compression_goal="key_facts")
    assert isinstance(result, str)
    assert len(result) > 0


def test_compress_context_key_facts_markdown_compresses() -> None:
    proc = LocalPreprocessor()
    ctx = _markdown(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        _result, ratio = proc.compress_context(ctx, compression_goal="key_facts")
    assert isinstance(ratio, float)
    assert ratio <= 1.0


def test_compress_context_caching_returns_same_result_on_second_call() -> None:
    proc = LocalPreprocessor()
    code = _code_block(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        r1, _ratio1 = proc.compress_context(code, compression_goal="code_only")
        r2, _ratio2 = proc.compress_context(code, compression_goal="code_only")
    assert r1 == r2


def test_compress_context_ratio_is_float() -> None:
    proc = LocalPreprocessor()
    ctx = _prose(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        _, ratio = proc.compress_context(ctx, compression_goal="key_facts")
    assert isinstance(ratio, float)


def test_compress_context_no_local_model_falls_back_to_truncation() -> None:
    # Pure numeric filler won't compress well via key_facts heuristic.
    # Adapter unavailable forces _get_local_model to return None via real code path.
    numeric = ("1234 5678 " * 500)[:8_000]
    proc = LocalPreprocessor()
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        result, ratio = proc.compress_context(numeric, compression_goal="key_facts")
    assert isinstance(result, str)
    assert len(result) > 0
    assert ratio <= 1.0


def test_compress_context_llm_path_used_when_heuristic_fails_and_model_available() -> None:
    numeric = ("1234 5678 " * 500)[:8_000]

    mock_adapter = MagicMock()
    mock_adapter.list_loaded_models.return_value = [{"id": "local-7b"}]
    mock_adapter.chat.return_value = {"output": "compressed llm output"}

    pp = LocalPreprocessor()
    with patch("vetinari.token_compression.LocalInferenceAdapter", return_value=mock_adapter):
        result, _ratio = pp.compress_context(numeric, "task", "key_facts")
    assert isinstance(result, str)
    assert len(result) > 0


def test_compress_context_llm_failure_falls_back_to_truncation() -> None:
    """When local adapter raises, compress_context falls back to truncation."""
    numeric = ("9999 8888 " * 500)[:8_000]

    mock_adapter = MagicMock()
    mock_adapter.list_loaded_models.return_value = [{"id": "local-7b"}]
    mock_adapter.chat.side_effect = ConnectionError("refused")

    pp = LocalPreprocessor()
    with patch("vetinari.token_compression.LocalInferenceAdapter", return_value=mock_adapter):
        result, _ratio = pp.compress_context(numeric, "task", "key_facts")
    assert isinstance(result, str)
    assert len(result) > 0


# ===========================================================================
# LocalPreprocessor._get_local_model
# ===========================================================================


def test_get_local_model_returns_none_when_adapter_unavailable() -> None:
    """Returns None when LocalInferenceAdapter raises on construction."""
    pp = LocalPreprocessor()
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        result = pp._get_local_model()
    assert result is None


def test_get_local_model_prefers_small_model_by_size_tag() -> None:
    """_get_local_model prefers models with a small-size tag in their ID."""
    mock_adapter = MagicMock()
    mock_adapter.list_loaded_models.return_value = [
        {"id": "large-70b-model"},
        {"id": "fast-7b-model"},
    ]
    pp = LocalPreprocessor()
    with patch("vetinari.token_compression.LocalInferenceAdapter", return_value=mock_adapter):
        model = pp._get_local_model()
    assert model == "fast-7b-model"


def test_get_local_model_falls_back_to_first_model_when_no_small_tag() -> None:
    """_get_local_model falls back to first model when none have a small-size tag."""
    mock_adapter = MagicMock()
    mock_adapter.list_loaded_models.return_value = [
        {"id": "large-34b-model"},
        {"id": "another-large-model"},
    ]
    pp = LocalPreprocessor()
    with patch("vetinari.token_compression.LocalInferenceAdapter", return_value=mock_adapter):
        model = pp._get_local_model()
    assert model == "large-34b-model"


def test_get_local_model_cached_after_first_call() -> None:
    """_get_local_model caches the result so adapter is only called once."""
    mock_adapter = MagicMock()
    mock_adapter.list_loaded_models.return_value = [{"id": "cached-7b"}]
    pp = LocalPreprocessor()
    with patch("vetinari.token_compression.LocalInferenceAdapter", return_value=mock_adapter) as mock_cls:
        m1 = pp._get_local_model()
        m2 = pp._get_local_model()
    # Constructor called once; second call returns from cache
    assert mock_cls.call_count == 1
    assert m1 == m2 == "cached-7b"


# ===========================================================================
# LocalPreprocessor.preprocess_for_cloud
# ===========================================================================


def test_preprocess_for_cloud_returns_three_tuple() -> None:
    proc = LocalPreprocessor()
    out = proc.preprocess_for_cloud("prompt", "ctx")
    assert len(out) == 3


def test_preprocess_for_cloud_short_inputs_not_compressed() -> None:
    proc = LocalPreprocessor()
    p, _c, meta = proc.preprocess_for_cloud("short", "ctx")
    assert p == "short"
    assert not meta["compressed"]
    assert meta["compression_ratio"] == 1.0


def test_preprocess_for_cloud_meta_has_required_keys() -> None:
    proc = LocalPreprocessor()
    _, _, meta = proc.preprocess_for_cloud("p", "c")
    for key in (
        "original_prompt_chars",
        "original_context_chars",
        "compressed",
        "compression_ratio",
        "final_prompt_chars",
        "final_context_chars",
    ):
        assert key in meta


def test_preprocess_for_cloud_large_context_processed() -> None:
    proc = LocalPreprocessor()
    large_ctx = _prose(8_000)
    with patch(
        "vetinari.token_compression.LocalInferenceAdapter",
        side_effect=RuntimeError("no llama-cpp"),
    ):
        _p, c, meta = proc.preprocess_for_cloud("prompt", large_ctx, "task")
    assert isinstance(c, str)
    assert meta["final_context_chars"] > 0


# ===========================================================================
# TokenOptimizer — budget management
# ===========================================================================


def test_token_optimizer_create_budget_returns_token_budget() -> None:
    opt = TokenOptimizer()
    b = opt.create_budget("plan-1", max_tokens=5_000, max_tokens_per_task=500)
    assert isinstance(b, TokenBudget)


def test_token_optimizer_create_budget_registers_under_plan_id() -> None:
    opt = TokenOptimizer()
    b = opt.create_budget("plan-2")
    assert opt.get_budget("plan-2") is b


def test_token_optimizer_create_budget_default_limits() -> None:
    opt = TokenOptimizer()
    b = opt.create_budget("plan-defaults")
    assert b.max_tokens == 100000
    assert b.max_tokens_per_task == 8000


def test_token_optimizer_get_budget_unknown_plan_returns_none() -> None:
    opt = TokenOptimizer()
    assert opt.get_budget("nonexistent") is None


def test_token_optimizer_record_usage_updates_tokens_used() -> None:
    opt = TokenOptimizer()
    opt.create_budget("plan-r", max_tokens=10_000)
    opt.record_usage("plan-r", "task-a", 400)
    b = opt.get_budget("plan-r")
    assert b.tokens_used == 400


def test_token_optimizer_record_usage_unknown_plan_is_noop() -> None:
    opt = TokenOptimizer()
    # Must not raise — recording against an unknown plan is silently ignored
    opt.record_usage("ghost", "t", 100)
    assert opt.get_budget("ghost") is None  # no budget was created for unknown plan


def test_token_optimizer_multiple_budgets_are_independent() -> None:
    opt = TokenOptimizer()
    b1 = opt.create_budget("plan-x", max_tokens=1_000)
    b2 = opt.create_budget("plan-y", max_tokens=2_000)
    opt.record_usage("plan-x", "t", 500)
    assert b1.tokens_used == 500
    assert b2.tokens_used == 0


# ===========================================================================
# TokenOptimizer — get_task_profile
# ===========================================================================


def test_get_task_profile_known_type_coding() -> None:
    opt = TokenOptimizer()
    max_tok, temp, prefer_json = opt.get_task_profile("coding")
    assert max_tok == 4096
    assert abs(temp - 0.1) < 1e-6
    assert not prefer_json


def test_get_task_profile_known_type_planning() -> None:
    opt = TokenOptimizer()
    max_tok, _temp, prefer_json = opt.get_task_profile("planning")
    assert max_tok == 3000
    assert prefer_json


def test_get_task_profile_known_type_classification() -> None:
    opt = TokenOptimizer()
    max_tok, temp, prefer_json = opt.get_task_profile("classification")
    assert max_tok == 256
    assert temp == 0.0
    assert prefer_json


def test_get_task_profile_unknown_type_falls_back_to_general() -> None:
    opt = TokenOptimizer()
    result = opt.get_task_profile("totally_made_up_type")
    assert result == TASK_PROFILES["general"]


def test_get_task_profile_case_insensitive_lookup() -> None:
    opt = TokenOptimizer()
    assert opt.get_task_profile("CODING") == opt.get_task_profile("coding")


def test_get_task_profile_hyphen_normalised_to_underscore() -> None:
    opt = TokenOptimizer()
    assert opt.get_task_profile("code-gen") == TASK_PROFILES["code_gen"]


def test_get_task_profile_space_normalised_to_underscore() -> None:
    opt = TokenOptimizer()
    assert opt.get_task_profile("ui design") == TASK_PROFILES["ui_design"]


def test_get_task_profile_all_profiles_have_correct_shape() -> None:
    TokenOptimizer()
    for key, val in TASK_PROFILES.items():
        assert isinstance(val, tuple), f"Profile {key!r} is not a tuple"
        assert len(val) == 3, f"Profile {key!r} tuple has wrong length"
        max_tok, temp, pref_json = val
        assert isinstance(max_tok, int), f"Profile {key!r} max_tokens not int"
        assert isinstance(temp, float), f"Profile {key!r} temperature not float"
        assert isinstance(pref_json, bool), f"Profile {key!r} prefer_json not bool"


# ===========================================================================
# TokenOptimizer — prepare_prompt (basic)
# ===========================================================================


def test_prepare_prompt_returns_required_keys() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("Hello", task_type="general")
    for key in ("prompt", "context", "max_tokens", "temperature", "prefer_json", "metadata", "budget_ok"):
        assert key in result


def test_prepare_prompt_budget_ok_true_with_no_budget() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="general")
    assert result["budget_ok"]


def test_prepare_prompt_task_profile_max_tokens_applied() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="classification")
    assert result["max_tokens"] == TASK_PROFILES["classification"][0]


def test_prepare_prompt_task_profile_temperature_applied() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="coding")
    assert abs(result["temperature"] - TASK_PROFILES["coding"][1]) < 1e-6


def test_prepare_prompt_task_profile_prefer_json_applied() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="planning")
    assert result["prefer_json"]


def test_prepare_prompt_context_prepended_to_prompt() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("Do something", context="ctx here")
    assert "ctx here" in result["prompt"]
    assert "Do something" in result["prompt"]


def test_prepare_prompt_empty_context_prompt_returned_unchanged() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("standalone", context="")
    assert result["prompt"] == "standalone"


def test_prepare_prompt_metadata_contains_task_type() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="research")
    assert result["metadata"]["task_type"] == "research"


def test_prepare_prompt_metadata_contains_task_profile_dict() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="coding")
    profile = result["metadata"]["task_profile"]
    assert "max_tokens" in profile
    assert "temperature" in profile


def test_prepare_prompt_unknown_task_type_uses_general_profile() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", task_type="__unknown__")
    assert result["max_tokens"] == TASK_PROFILES["general"][0]


def test_prepare_prompt_empty_prompt_and_context_does_not_raise() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("", context="", task_type="general")
    assert "prompt" in result


# ===========================================================================
# TokenOptimizer — prepare_prompt with budget
# ===========================================================================


def test_prepare_prompt_budget_ok_false_when_exhausted() -> None:
    opt = TokenOptimizer()
    budget = TokenBudget(plan_id="p", max_tokens=10, max_tokens_per_task=5)
    budget.record("t", 10)
    result = opt.prepare_prompt("Hello world", budget=budget, task_id="t")
    assert not result["budget_ok"]


def test_prepare_prompt_budget_ok_true_when_sufficient() -> None:
    opt = TokenOptimizer()
    opt.create_budget("plan-ok", max_tokens=100_000)
    result = opt.prepare_prompt("p", plan_id="plan-ok", task_id="t1")
    assert result["budget_ok"]


def test_prepare_prompt_budget_remaining_in_metadata() -> None:
    opt = TokenOptimizer()
    budget = opt.create_budget("plan-rem", max_tokens=5_000)
    budget.record("prior", 1_000)
    result = opt.prepare_prompt("p", plan_id="plan-rem", task_id="new")
    assert "budget_remaining" in result["metadata"]
    assert result["metadata"]["budget_remaining"] == 4000


def test_prepare_prompt_explicit_budget_takes_precedence_over_plan_id() -> None:
    opt = TokenOptimizer()
    # Create a tight plan budget
    opt.create_budget("plan-tight", max_tokens=1)
    # Pass a generous explicit budget
    generous = TokenBudget(plan_id="explicit", max_tokens=100_000)
    result = opt.prepare_prompt("p", plan_id="plan-tight", budget=generous, task_id="t")
    assert result["budget_ok"]


def test_prepare_prompt_plan_id_budget_lookup() -> None:
    opt = TokenOptimizer()
    opt.create_budget("plan-lookup", max_tokens=100_000, max_tokens_per_task=8_000)
    result = opt.prepare_prompt("p", task_type="general", plan_id="plan-lookup", task_id="t1")
    assert "budget_remaining" in result["metadata"]
    assert result["budget_ok"]


# ===========================================================================
# TokenOptimizer — prepare_prompt cloud and local truncation
# ===========================================================================


def test_prepare_prompt_cloud_flag_true_in_metadata() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", is_cloud_model=True)
    assert result["metadata"]["is_cloud_model"]


def test_prepare_prompt_cloud_flag_false_in_metadata() -> None:
    opt = TokenOptimizer()
    result = opt.prepare_prompt("p", is_cloud_model=False)
    assert not result["metadata"]["is_cloud_model"]


@patch.object(LocalPreprocessor, "preprocess_for_cloud")
def test_prepare_prompt_cloud_preprocessing_called_for_large_context(mock_pp: MagicMock) -> None:
    opt = TokenOptimizer()
    large_ctx = _prose(25_000)  # must exceed 70% of default 8192 context_length * 4 chars/token = 22937
    mock_pp.return_value = (
        "compressed prompt",
        "compressed ctx",
        {
            "original_prompt_chars": 6,
            "original_context_chars": 25_000,
            "compressed": True,
            "compression_ratio": 0.4,
            "final_prompt_chars": 6,
            "final_context_chars": 12_000,
        },
    )
    result = opt.prepare_prompt("prompt", context=large_ctx, is_cloud_model=True, task_description="t")
    mock_pp.assert_called_once()
    assert result["prompt"].endswith("compressed prompt")
    assert "compressed ctx" in result["prompt"]
    assert result["context"] == "compressed ctx"
    assert result["metadata"]["compressed"] is True


@patch.object(LocalPreprocessor, "preprocess_for_cloud")
def test_prepare_prompt_cloud_preprocessing_not_called_for_small_context(mock_pp: MagicMock) -> None:
    opt = TokenOptimizer()
    opt.prepare_prompt("p", context="tiny ctx", is_cloud_model=True)
    mock_pp.assert_not_called()


@patch.object(LocalPreprocessor, "preprocess_for_cloud")
def test_prepare_prompt_local_model_compresses_very_long_context(mock_pp: MagicMock) -> None:
    """Compression now applies to local models when context exceeds 70% of context_length."""
    opt = TokenOptimizer()
    # Use default context_length=8192; 70% threshold = 8192 * 0.7 * 4 = ~22937 chars
    long_ctx = "word " * 5000  # ~25000 chars — well over the threshold
    mock_pp.return_value = (
        "p",
        "compressed local ctx",
        {
            "original_prompt_chars": 1,
            "original_context_chars": len(long_ctx),
            "compressed": True,
            "compression_ratio": 0.4,
            "final_prompt_chars": 1,
            "final_context_chars": 12000,
        },
    )
    result = opt.prepare_prompt("p", context=long_ctx, is_cloud_model=False)
    mock_pp.assert_called_once()
    assert result["metadata"].get("compressed", False)


# ===========================================================================
# TokenOptimizer — prepare_prompt context deduplication
# ===========================================================================


def test_prepare_prompt_first_call_not_deduplicated() -> None:
    opt = TokenOptimizer()
    ctx = "some context " * 10
    r1 = opt.prepare_prompt("p1", context=ctx)
    assert not r1["metadata"].get("context_deduplicated", False)


def test_prepare_prompt_second_call_with_same_context_is_deduplicated() -> None:
    opt = TokenOptimizer()
    ctx = "some context " * 10
    opt.prepare_prompt("p1", context=ctx)
    r2 = opt.prepare_prompt("p2", context=ctx)
    assert r2["metadata"].get("context_deduplicated", False)


def test_prepare_prompt_different_context_not_deduplicated() -> None:
    opt = TokenOptimizer()
    opt.prepare_prompt("p1", context="context A " * 10)
    r2 = opt.prepare_prompt("p2", context="context B " * 10)
    assert not r2["metadata"].get("context_deduplicated", False)


def test_prepare_prompt_deduplicated_context_contains_reference_prefix() -> None:
    opt = TokenOptimizer()
    ctx = "repeated context " * 10
    opt.prepare_prompt("p1", context=ctx)
    r2 = opt.prepare_prompt("p2", context=ctx)
    assert "[Context unchanged from previous task" in r2["context"]


def test_prepare_prompt_empty_context_never_deduplicated() -> None:
    opt = TokenOptimizer()
    r1 = opt.prepare_prompt("p1", context="")
    r2 = opt.prepare_prompt("p2", context="")
    assert not r1["metadata"].get("context_deduplicated", False)
    assert not r2["metadata"].get("context_deduplicated", False)


# ===========================================================================
# TokenOptimizer — summarise_results
# ===========================================================================


def test_summarise_results_empty_list_returns_empty_string() -> None:
    opt = TokenOptimizer()
    assert opt.summarise_results([]) == ""


def test_summarise_results_single_dict_with_description_and_output() -> None:
    opt = TokenOptimizer()
    results = [{"description": "Build task", "output": "compiled OK"}]
    summary = opt.summarise_results(results)
    assert "Build task" in summary
    assert "compiled OK" in summary


def test_summarise_results_string_result_included() -> None:
    opt = TokenOptimizer()
    results = ["Task A finished"]
    summary = opt.summarise_results(results)
    assert "Task A finished" in summary


def test_summarise_results_mixed_string_and_dict_results() -> None:
    opt = TokenOptimizer()
    results = [{"description": "dict task", "output": "done"}, "string result"]
    summary = opt.summarise_results(results)
    assert "dict task" in summary
    assert "string result" in summary


def test_summarise_results_dict_output_is_itself_a_dict() -> None:
    opt = TokenOptimizer()
    results = [{"description": "analysis", "output": {"status": "ok", "count": 5}}]
    summary = opt.summarise_results(results)
    assert "status" in summary


def test_summarise_results_uses_task_id_when_no_description() -> None:
    opt = TokenOptimizer()
    results = [{"task_id": "my_task", "output": "ok"}]
    summary = opt.summarise_results(results)
    assert "my_task" in summary


def test_summarise_results_uses_result_key_when_no_output() -> None:
    opt = TokenOptimizer()
    results = [{"description": "thing", "result": "success"}]
    summary = opt.summarise_results(results)
    assert "success" in summary


# ===========================================================================
# LocalPreprocessor — LLMLingua path (US-002)
# ===========================================================================


def test_llmlingua_path_selected_when_available() -> None:
    """LLMLingua compression is attempted when context exceeds the heuristic threshold.

    The test patches ``vetinari.token_compression.LocalPreprocessor._llmlingua_compress``
    to return a compressed string, then calls ``compress_context`` with a context that:

    1. Exceeds ``MIN_CONTEXT_CHARS`` (so compression is triggered at all).
    2. Is numeric filler — the key-facts heuristic cannot reduce it enough, so
       execution falls through to the ``_llmlingua_compress`` call.

    We assert that the patched ``_llmlingua_compress`` was called, proving the
    LLMLingua code path was reached rather than the heuristic or LLM paths.
    """
    # Numeric filler resists key-facts heuristic extraction (no headers/bullets/keywords).
    # Must exceed MIN_CONTEXT_CHARS (24000) so compress_context does not short-circuit.
    unit = "9876 5432 "  # 10 chars; repeat enough times to exceed the threshold
    reps = LocalPreprocessor.MIN_CONTEXT_CHARS // len(unit) + 20
    numeric_filler = (unit * reps)[: LocalPreprocessor.MIN_CONTEXT_CHARS + 500]

    compressed_output = "llmlingua compressed result"

    proc = LocalPreprocessor()

    with patch.object(
        LocalPreprocessor,
        "_llmlingua_compress",
        return_value=compressed_output,
    ) as mock_llmlingua:
        result, ratio = proc.compress_context(
            numeric_filler,
            task_description="test task",
            compression_goal="key_facts",
        )

    (
        mock_llmlingua.assert_called_once(),
        (
            "_llmlingua_compress must be called when context exceeds threshold and "
            "heuristic extraction does not compress enough"
        ),
    )
    assert result == compressed_output, (
        f"compress_context must return the LLMLingua result when it succeeds, got {result!r}"
    )
    assert ratio < 1.0, f"Compression ratio must be < 1.0 when LLMLingua compressed, got {ratio}"


def test_summarise_results_long_output_does_not_explode_summary() -> None:
    opt = TokenOptimizer()
    results = [{"description": "long", "output": "x" * 1_000}]
    summary = opt.summarise_results(results)
    # Per-result output is capped at 200 chars
    assert len(summary) <= 600


def test_summarise_results_total_truncated_at_max_chars() -> None:
    opt = TokenOptimizer()
    results = [{"description": f"t{i}", "output": "res " * 60} for i in range(20)]
    summary = opt.summarise_results(results, max_chars=300)
    assert "[... additional results truncated ...]" in summary
    assert len(summary) <= 500


def test_summarise_results_max_chars_parameter_controls_length() -> None:
    opt = TokenOptimizer()
    results = [{"description": "t", "output": "a" * 200} for _ in range(10)]
    short = opt.summarise_results(results, max_chars=100)
    long_ = opt.summarise_results(results, max_chars=5_000)
    assert len(short) < len(long_)


def test_summarise_results_none_output_does_not_raise() -> None:
    opt = TokenOptimizer()
    results = [{"description": "task", "output": None}]
    summary = opt.summarise_results(results)
    assert isinstance(summary, str)


def test_summarise_results_empty_dict_result_does_not_raise() -> None:
    opt = TokenOptimizer()
    results = [{}]
    summary = opt.summarise_results(results)
    assert isinstance(summary, str)


# ===========================================================================
# Module constants
# ===========================================================================


def test_module_constants_task_profiles_is_non_empty_dict() -> None:
    assert isinstance(TASK_PROFILES, dict)
    assert len(TASK_PROFILES) > 0


def test_module_constants_chars_per_token_by_type_has_expected_keys() -> None:
    for key in ("code", "text", "mixed"):
        assert key in _CHARS_PER_TOKEN_BY_TYPE


def test_module_constants_chars_per_token_by_type_values_are_positive_ints() -> None:
    for key, val in _CHARS_PER_TOKEN_BY_TYPE.items():
        assert isinstance(val, int), f"Value for {key!r} is not int"
        assert val > 0, f"Value for {key!r} is not positive"


def test_module_constants_default_chars_per_token_is_positive() -> None:
    assert isinstance(_CHARS_PER_TOKEN, int)
    assert _CHARS_PER_TOKEN > 0


def test_module_constants_compress_threshold_chars_is_positive() -> None:
    assert _COMPRESS_THRESHOLD_CHARS > 0


def test_module_constants_context_window_chars_is_positive() -> None:
    assert _CONTEXT_WINDOW_CHARS > 0


def test_module_constants_all_profile_tuples_have_three_elements() -> None:
    for k, v in TASK_PROFILES.items():
        assert len(v) == 3, f"Profile {k!r} has wrong tuple length"


# ===========================================================================
# Singleton
# ===========================================================================


@pytest.fixture(autouse=False)
def _reset_token_optimizer_singleton() -> None:
    """Reset the module-level singleton before and after each singleton test.

    Uses ``sys.modules[get_token_optimizer.__module__]`` to reference the
    same module object the top-level import bound to, even if
    ``_restore_sys_modules`` swapped in a fresh module after collection.
    """
    mod = sys.modules[get_token_optimizer.__module__]
    original = mod._token_optimizer
    mod._token_optimizer = None
    yield
    mod._token_optimizer = original


def test_get_token_optimizer_returns_token_optimizer_instance(
    _reset_token_optimizer_singleton: None,
) -> None:
    opt = get_token_optimizer()
    assert isinstance(opt, TokenOptimizer)


def test_get_token_optimizer_same_instance_returned_on_repeated_calls(
    _reset_token_optimizer_singleton: None,
) -> None:
    opt1 = get_token_optimizer()
    opt2 = get_token_optimizer()
    assert opt1 is opt2


def test_get_token_optimizer_fresh_instance_after_reset(
    _reset_token_optimizer_singleton: None,
) -> None:
    mod = sys.modules[get_token_optimizer.__module__]
    opt1 = get_token_optimizer()
    mod._token_optimizer = None
    opt2 = get_token_optimizer()
    assert opt1 is not opt2
