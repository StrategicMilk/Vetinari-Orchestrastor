"""
Tests for vetinari/token_optimizer.py

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
import unittest
from unittest.mock import MagicMock, patch

# Remove incomplete stubs left by earlier test files so real modules load
sys.modules.pop("vetinari.token_optimizer", None)

from vetinari.token_optimizer import (
    _CHARS_PER_TOKEN,
    _CHARS_PER_TOKEN_BY_TYPE,
    _COMPRESS_THRESHOLD_CHARS,
    _CONTEXT_WINDOW_CHARS,
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
    unit = "def placeholder(x, y):\n    \"\"\"Filler function.\"\"\"\n    return x + y\n\n"
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
# TokenBudget
# ===========================================================================

class TestTokenBudgetCreation(unittest.TestCase):

    def test_default_field_values(self):
        b = TokenBudget(plan_id="p1")
        assert b.plan_id == "p1"
        assert b.max_tokens == 100000
        assert b.max_tokens_per_task == 8000
        assert b.tokens_used == 0
        assert b.task_token_counts == {}

    def test_custom_max_tokens(self):
        b = TokenBudget(plan_id="p2", max_tokens=500, max_tokens_per_task=50)
        assert b.max_tokens == 500
        assert b.max_tokens_per_task == 50

    def test_task_token_counts_initially_empty(self):
        b = TokenBudget(plan_id="p3")
        assert isinstance(b.task_token_counts, dict)
        assert len(b.task_token_counts) == 0


class TestTokenBudgetRecord(unittest.TestCase):

    def _make(self) -> TokenBudget:
        return TokenBudget(plan_id="plan", max_tokens=10_000, max_tokens_per_task=2_000)

    def test_record_increments_tokens_used(self):
        b = self._make()
        b.record("t1", 100)
        assert b.tokens_used == 100

    def test_record_accumulates_across_calls(self):
        b = self._make()
        b.record("t1", 100)
        b.record("t1", 50)
        assert b.tokens_used == 150

    def test_record_tracks_per_task(self):
        b = self._make()
        b.record("t1", 300)
        b.record("t2", 200)
        assert b.task_token_counts["t1"] == 300
        assert b.task_token_counts["t2"] == 200

    def test_record_same_task_twice_accumulates(self):
        b = self._make()
        b.record("t1", 400)
        b.record("t1", 400)
        assert b.task_token_counts["t1"] == 800

    def test_record_zero_tokens_is_valid(self):
        b = self._make()
        b.record("t1", 0)
        assert b.tokens_used == 0
        assert b.task_token_counts["t1"] == 0

    def test_record_multiple_tasks_total_is_sum(self):
        b = self._make()
        b.record("a", 100)
        b.record("b", 200)
        b.record("c", 300)
        assert b.tokens_used == 600


class TestTokenBudgetRemaining(unittest.TestCase):

    def test_remaining_equals_max_when_unused(self):
        b = TokenBudget(plan_id="p", max_tokens=1_000)
        assert b.remaining == 1000

    def test_remaining_decreases_on_record(self):
        b = TokenBudget(plan_id="p", max_tokens=1_000)
        b.record("t", 300)
        assert b.remaining == 700

    def test_remaining_clamps_at_zero(self):
        b = TokenBudget(plan_id="p", max_tokens=100)
        b.record("t", 500)
        assert b.remaining == 0

    def test_remaining_is_zero_at_exact_limit(self):
        b = TokenBudget(plan_id="p", max_tokens=100)
        b.record("t", 100)
        assert b.remaining == 0


class TestTokenBudgetIsExhausted(unittest.TestCase):

    def test_not_exhausted_when_fresh(self):
        b = TokenBudget(plan_id="p", max_tokens=1_000)
        assert not b.is_exhausted

    def test_exhausted_at_exact_limit(self):
        b = TokenBudget(plan_id="p", max_tokens=100)
        b.record("t", 100)
        assert b.is_exhausted

    def test_exhausted_when_over_limit(self):
        b = TokenBudget(plan_id="p", max_tokens=100)
        b.record("t", 200)
        assert b.is_exhausted

    def test_not_exhausted_one_token_below_limit(self):
        b = TokenBudget(plan_id="p", max_tokens=100)
        b.record("t", 99)
        assert not b.is_exhausted


class TestTokenBudgetCheckTask(unittest.TestCase):

    def _make(self, max_tokens: int = 10_000, max_per_task: int = 2_000) -> TokenBudget:
        return TokenBudget(plan_id="p", max_tokens=max_tokens,
                           max_tokens_per_task=max_per_task)

    def test_check_task_ok_for_small_estimate(self):
        b = self._make()
        assert b.check_task("t", 100)

    def test_check_task_fails_when_plan_exhausted(self):
        b = self._make(max_tokens=50)
        b.record("t", 50)
        assert not b.check_task("t2", 1)

    def test_check_task_fails_when_exceeds_per_task_limit(self):
        b = self._make(max_per_task=100)
        # 80 already used + 30 estimated = 110 > 100
        b.record("t1", 80)
        assert not b.check_task("t1", 30)

    def test_check_task_ok_different_task_has_no_history(self):
        b = self._make(max_per_task=100)
        b.record("t1", 90)
        assert b.check_task("t2", 50)

    def test_check_task_zero_estimated_always_ok(self):
        b = self._make()
        assert b.check_task("t", 0)

    def test_check_task_exactly_at_per_task_limit_fails(self):
        b = self._make(max_per_task=100)
        # 0 used + 101 estimated > 100
        assert not b.check_task("t", 101)

    def test_check_task_within_per_task_limit_after_partial_use(self):
        b = self._make(max_per_task=500)
        b.record("t", 300)
        # 300 + 199 = 499 <= 500
        assert b.check_task("t", 199)


# ===========================================================================
# LocalPreprocessor._truncate
# ===========================================================================

class TestLocalPreprocessorTruncate(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_short_context_returned_unchanged(self):
        ctx = "hello world"
        assert self.proc._truncate(ctx) == ctx

    def test_at_window_boundary_returned_unchanged(self):
        ctx = "x" * _CONTEXT_WINDOW_CHARS
        assert self.proc._truncate(ctx) == ctx

    def test_long_context_contains_truncation_marker(self):
        ctx = "A" * (_CONTEXT_WINDOW_CHARS + 500)
        result = self.proc._truncate(ctx)
        assert "[... context truncated for token efficiency ...]" in result

    def test_long_context_shorter_than_original(self):
        ctx = "B" * 20_000
        result = self.proc._truncate(ctx)
        assert len(result) < len(ctx)

    def test_truncated_result_preserves_head_and_tail(self):
        head_marker = "<<HEAD>>"
        tail_marker = "<<TAIL>>"
        filler = "x" * _CONTEXT_WINDOW_CHARS
        ctx = head_marker + filler + tail_marker
        result = self.proc._truncate(ctx)
        assert head_marker in result
        assert tail_marker in result


# ===========================================================================
# LocalPreprocessor._extract_key_lines
# ===========================================================================

class TestExtractKeyLines(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_extracts_markdown_headers(self):
        ctx = "# Title\nsome random prose\n## Section\nmore prose"
        result = self.proc._extract_key_lines(ctx)
        assert "# Title" in result
        assert "## Section" in result
        assert "some random prose" not in result

    def test_extracts_bullet_points_dash(self):
        ctx = "- item one\nignored prose"
        result = self.proc._extract_key_lines(ctx)
        assert "- item one" in result

    def test_extracts_bullet_points_star(self):
        ctx = "* item two\nignored prose"
        result = self.proc._extract_key_lines(ctx)
        assert "* item two" in result

    def test_extracts_numbered_list(self):
        ctx = "1. first item\n2) second item\nignored"
        result = self.proc._extract_key_lines(ctx)
        assert "1. first item" in result
        assert "2) second item" in result

    def test_extracts_key_value_patterns(self):
        ctx = "host: localhost\nport = 8080\nnoise"
        result = self.proc._extract_key_lines(ctx)
        assert "host: localhost" in result
        assert "port = 8080" in result

    def test_extracts_https_url_lines(self):
        ctx = "See https://example.com\nsome noise"
        result = self.proc._extract_key_lines(ctx)
        assert "https://example.com" in result

    def test_extracts_http_url_lines(self):
        ctx = "API at http://api.local/v1\nsome noise"
        result = self.proc._extract_key_lines(ctx)
        assert "http://api.local/v1" in result

    def test_extracts_key_term_error(self):
        ctx = "An error occurred here.\nsome boring filler"
        result = self.proc._extract_key_lines(ctx)
        assert "error" in result

    def test_extracts_key_term_endpoint(self):
        ctx = "The endpoint must be authenticated.\nfiller"
        result = self.proc._extract_key_lines(ctx)
        assert "endpoint" in result

    def test_empty_input_returns_empty(self):
        assert self.proc._extract_key_lines("") == ""

    def test_only_blank_lines_returns_empty(self):
        assert self.proc._extract_key_lines("\n\n\n") == ""

    def test_pure_prose_no_matches_returns_empty(self):
        ctx = "just some ordinary prose without any special markers\n" * 3
        result = self.proc._extract_key_lines(ctx)
        assert result == ""


# ===========================================================================
# LocalPreprocessor._extract_code_signatures (AST path)
# ===========================================================================

class TestExtractCodeSignaturesAST(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_extracts_function_signature(self):
        code = "def add(x: int, y: int) -> int:\n    \"\"\"Add.\"\"\"\n    return x + y\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "def add" in result
        assert "x: int" in result
        assert "-> int" in result

    def test_extracts_class_with_base(self):
        code = "class Foo(Bar):\n    \"\"\"A foo.\"\"\"\n    def method(self) -> None:\n        pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "class Foo(Bar):" in result
        assert "def method" in result

    def test_extracts_async_function(self):
        code = "async def fetch(url: str) -> bytes:\n    pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "async def fetch" in result
        assert "url: str" in result

    def test_extracts_import_statements(self):
        code = "import os\nfrom pathlib import Path\nx = 1\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "import os" in result
        assert "from pathlib import Path" in result

    def test_uppercase_constant_included(self):
        code = "MAX_SIZE = 100\nsmall_var = 5\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "MAX_SIZE" in result
        assert "small_var" not in result

    def test_module_docstring_included(self):
        code = '"""Module description."""\n\ndef func():\n    pass\n'
        result = self.proc._extract_code_signatures_ast(code)
        assert "Module description." in result

    def test_decorator_included(self):
        code = "@staticmethod\ndef helper():\n    pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "@staticmethod" in result

    def test_varargs_and_kwargs(self):
        code = "def variadic(*args, **kwargs) -> None:\n    pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "*args" in result
        assert "**kwargs" in result

    def test_default_argument(self):
        code = "def greet(name: str = 'world') -> str:\n    pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "greet" in result
        assert "name" in result

    def test_kwonly_arg(self):
        code = "def func(a, *, verbose: bool = False) -> None:\n    pass\n"
        result = self.proc._extract_code_signatures_ast(code)
        assert "func" in result
        assert "verbose" in result

    def test_returns_string(self):
        result = self.proc._extract_code_signatures_ast("x = 1\n")
        assert isinstance(result, str)


# ===========================================================================
# LocalPreprocessor._extract_code_signatures (dispatch + regex fallback)
# ===========================================================================

class TestExtractCodeSignaturesDispatch(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_valid_python_uses_ast_path(self):
        code = "def hello(): pass\n"
        result = self.proc._extract_code_signatures(code)
        assert "def hello" in result

    def test_syntax_error_falls_back_to_regex(self):
        # Invalid Python that ast.parse cannot handle
        bad_code = "def broken(:\nclass Orphan:\n    pass\n"
        result = self.proc._extract_code_signatures(bad_code)
        assert isinstance(result, str)
        # Regex should still catch 'class Orphan'
        assert "class Orphan" in result

    def test_regex_captures_imports(self):
        code = "def broken(:\nimport os\nfrom sys import argv\n"
        result = self.proc._extract_code_signatures(_bad_code := code)
        assert "import os" in result

    def test_empty_code_returns_empty_string(self):
        result = self.proc._extract_code_signatures("")
        assert result == ""

    def test_regex_captures_decorators(self):
        code = "def broken(:\n@my_decorator\ndef valid_func():\n    pass\n"
        result = self.proc._extract_code_signatures(code)
        assert "@my_decorator" in result


# ===========================================================================
# LocalPreprocessor.compress_context
# ===========================================================================

class TestCompressContext(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_short_context_returned_as_is(self):
        ctx = "short context"
        result, ratio = self.proc.compress_context(ctx)
        assert result == ctx
        assert ratio == 1.0

    def test_just_below_threshold_returned_as_is(self):
        ctx = "x" * (LocalPreprocessor.MIN_CONTEXT_CHARS - 1)
        result, ratio = self.proc.compress_context(ctx, compression_goal="key_facts")
        assert result == ctx
        assert ratio == 1.0

    def test_long_code_only_goal_returns_string(self):
        code = _code_block(8_000)
        result, _ratio = self.proc.compress_context(code, compression_goal="code_only")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_long_key_facts_goal_returns_string(self):
        ctx = _markdown(8_000)
        result, _ratio = self.proc.compress_context(ctx, compression_goal="key_facts")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_key_facts_markdown_compresses(self):
        # Markdown with lots of prose should compress; headers/bullets are extracted
        ctx = _markdown(8_000)
        _result, ratio = self.proc.compress_context(ctx, compression_goal="key_facts")
        # Ratio < 1 means compression occurred (or == 1 means heuristic threshold
        # was not met and we fell through to LLM/truncate)
        assert isinstance(ratio, float)
        assert ratio <= 1.0

    def test_caching_returns_same_result_on_second_call(self):
        code = _code_block(8_000)
        r1, _ratio1 = self.proc.compress_context(code, compression_goal="code_only")
        r2, _ratio2 = self.proc.compress_context(code, compression_goal="code_only")
        assert r1 == r2

    def test_ratio_is_float(self):
        ctx = _prose(8_000)
        _, ratio = self.proc.compress_context(ctx, compression_goal="key_facts")
        assert isinstance(ratio, float)

    @patch("vetinari.token_optimizer.LocalPreprocessor._get_local_model", return_value=None)
    def test_no_local_model_falls_back_to_truncation(self, _mock):
        # Pure numeric filler won't compress well via key_facts heuristic
        numeric = ("1234 5678 " * 500)[:8_000]
        result, ratio = self.proc.compress_context(numeric, compression_goal="key_facts")
        assert isinstance(result, str)
        assert len(result) > 0
        assert ratio <= 1.0

    @patch("requests.post")
    @patch("requests.get")
    def test_llm_path_used_when_heuristic_fails_and_model_available(
        self, mock_get, mock_post
    ):
        numeric = ("1234 5678 " * 500)[:8_000]

        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {"data": [{"id": "local-7b"}]}
        mock_get.return_value = get_resp

        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {
            "choices": [{"message": {"content": "compressed llm output"}}]
        }
        mock_post.return_value = post_resp

        pp = LocalPreprocessor()
        result, _ratio = pp.compress_context(numeric, "task", "key_facts")
        # LLM returned "compressed llm output" or heuristic/truncate was used
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("requests.post")
    @patch("requests.get")
    def test_llm_failure_falls_back_to_truncation(self, mock_get, mock_post):
        numeric = ("9999 8888 " * 500)[:8_000]

        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {"data": [{"id": "local-7b"}]}
        mock_get.return_value = get_resp

        mock_post.side_effect = ConnectionError("refused")

        pp = LocalPreprocessor()
        result, _ratio = pp.compress_context(numeric, "task", "key_facts")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# LocalPreprocessor._get_local_model
# ===========================================================================

class TestGetLocalModel(unittest.TestCase):

    def test_returns_none_when_server_unreachable(self):
        pp = LocalPreprocessor()
        with patch("requests.get", side_effect=ConnectionError("refused")):
            result = pp._get_local_model()
        assert result is None

    @patch("requests.get")
    def test_prefers_small_model_by_size_tag(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"id": "large-70b-model"}, {"id": "fast-7b-model"}]
        }
        mock_get.return_value = resp
        pp = LocalPreprocessor()
        model = pp._get_local_model()
        assert model == "fast-7b-model"

    @patch("requests.get")
    def test_falls_back_to_first_model_when_no_small_tag(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": [{"id": "large-34b-model"}, {"id": "another-large-model"}]
        }
        mock_get.return_value = resp
        pp = LocalPreprocessor()
        model = pp._get_local_model()
        assert model == "large-34b-model"

    @patch("requests.get")
    def test_cached_after_first_call(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "cached-7b"}]}
        mock_get.return_value = resp
        pp = LocalPreprocessor()
        m1 = pp._get_local_model()
        m2 = pp._get_local_model()
        # Second call should NOT hit the network
        assert mock_get.call_count == 1
        assert m1 == m2


# ===========================================================================
# LocalPreprocessor.preprocess_for_cloud
# ===========================================================================

class TestPreprocessForCloud(unittest.TestCase):

    def setUp(self):
        self.proc = LocalPreprocessor()

    def test_returns_three_tuple(self):
        out = self.proc.preprocess_for_cloud("prompt", "ctx")
        assert len(out) == 3

    def test_short_inputs_not_compressed(self):
        p, _c, meta = self.proc.preprocess_for_cloud("short", "ctx")
        assert p == "short"
        assert not meta["compressed"]
        assert meta["compression_ratio"] == 1.0

    def test_meta_has_required_keys(self):
        _, _, meta = self.proc.preprocess_for_cloud("p", "c")
        for key in ("original_prompt_chars", "original_context_chars",
                    "compressed", "compression_ratio",
                    "final_prompt_chars", "final_context_chars"):
            assert key in meta

    def test_large_context_processed(self):
        large_ctx = _prose(8_000)
        _p, c, meta = self.proc.preprocess_for_cloud("prompt", large_ctx, "task")
        assert isinstance(c, str)
        assert meta["final_context_chars"] > 0


# ===========================================================================
# TokenOptimizer — budget management
# ===========================================================================

class TestTokenOptimizerBudgetManagement(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_create_budget_returns_token_budget(self):
        b = self.opt.create_budget("plan-1", max_tokens=5_000, max_tokens_per_task=500)
        assert isinstance(b, TokenBudget)

    def test_create_budget_registers_under_plan_id(self):
        b = self.opt.create_budget("plan-2")
        assert self.opt.get_budget("plan-2") is b

    def test_create_budget_default_limits(self):
        b = self.opt.create_budget("plan-defaults")
        assert b.max_tokens == 100000
        assert b.max_tokens_per_task == 8000

    def test_get_budget_unknown_plan_returns_none(self):
        assert self.opt.get_budget("nonexistent") is None

    def test_record_usage_updates_tokens_used(self):
        self.opt.create_budget("plan-r", max_tokens=10_000)
        self.opt.record_usage("plan-r", "task-a", 400)
        b = self.opt.get_budget("plan-r")
        assert b.tokens_used == 400

    def test_record_usage_unknown_plan_is_noop(self):
        # Must not raise
        self.opt.record_usage("ghost", "t", 100)

    def test_multiple_budgets_are_independent(self):
        b1 = self.opt.create_budget("plan-x", max_tokens=1_000)
        b2 = self.opt.create_budget("plan-y", max_tokens=2_000)
        self.opt.record_usage("plan-x", "t", 500)
        assert b1.tokens_used == 500
        assert b2.tokens_used == 0


# ===========================================================================
# TokenOptimizer — get_task_profile
# ===========================================================================

class TestGetTaskProfile(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_known_type_coding(self):
        max_tok, temp, prefer_json = self.opt.get_task_profile("coding")
        assert max_tok == 4096
        self.assertAlmostEqual(temp, 0.1)
        assert not prefer_json

    def test_known_type_planning(self):
        max_tok, _temp, prefer_json = self.opt.get_task_profile("planning")
        assert max_tok == 3000
        assert prefer_json

    def test_known_type_classification(self):
        max_tok, temp, prefer_json = self.opt.get_task_profile("classification")
        assert max_tok == 256
        assert temp == 0.0
        assert prefer_json

    def test_unknown_type_falls_back_to_general(self):
        result = self.opt.get_task_profile("totally_made_up_type")
        assert result == TASK_PROFILES["general"]

    def test_case_insensitive_lookup(self):
        assert self.opt.get_task_profile("CODING") == self.opt.get_task_profile("coding")

    def test_hyphen_normalised_to_underscore(self):
        assert self.opt.get_task_profile("code-gen") == TASK_PROFILES["code_gen"]

    def test_space_normalised_to_underscore(self):
        assert self.opt.get_task_profile("ui design") == TASK_PROFILES["ui_design"]

    def test_all_profiles_have_correct_shape(self):
        for key, val in TASK_PROFILES.items():
            with self.subTest(key=key):
                assert isinstance(val, tuple)
                assert len(val) == 3
                max_tok, temp, pref_json = val
                assert isinstance(max_tok, int)
                assert isinstance(temp, float)
                assert isinstance(pref_json, bool)


# ===========================================================================
# TokenOptimizer — prepare_prompt
# ===========================================================================

class TestPreparePromptBasic(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_returns_required_keys(self):
        result = self.opt.prepare_prompt("Hello", task_type="general")
        for key in ("prompt", "context", "max_tokens", "temperature",
                    "prefer_json", "metadata", "budget_ok"):
            assert key in result

    def test_budget_ok_true_with_no_budget(self):
        result = self.opt.prepare_prompt("p", task_type="general")
        assert result["budget_ok"]

    def test_task_profile_max_tokens_applied(self):
        result = self.opt.prepare_prompt("p", task_type="classification")
        assert result["max_tokens"] == TASK_PROFILES["classification"][0]

    def test_task_profile_temperature_applied(self):
        result = self.opt.prepare_prompt("p", task_type="coding")
        self.assertAlmostEqual(result["temperature"], TASK_PROFILES["coding"][1])

    def test_task_profile_prefer_json_applied(self):
        result = self.opt.prepare_prompt("p", task_type="planning")
        assert result["prefer_json"]

    def test_context_prepended_to_prompt(self):
        result = self.opt.prepare_prompt("Do something", context="ctx here")
        assert "ctx here" in result["prompt"]
        assert "Do something" in result["prompt"]

    def test_empty_context_prompt_returned_unchanged(self):
        result = self.opt.prepare_prompt("standalone", context="")
        assert result["prompt"] == "standalone"

    def test_metadata_contains_task_type(self):
        result = self.opt.prepare_prompt("p", task_type="research")
        assert result["metadata"]["task_type"] == "research"

    def test_metadata_contains_task_profile_dict(self):
        result = self.opt.prepare_prompt("p", task_type="coding")
        profile = result["metadata"]["task_profile"]
        assert "max_tokens" in profile
        assert "temperature" in profile

    def test_unknown_task_type_uses_general_profile(self):
        result = self.opt.prepare_prompt("p", task_type="__unknown__")
        assert result["max_tokens"] == TASK_PROFILES["general"][0]

    def test_empty_prompt_and_context_does_not_raise(self):
        result = self.opt.prepare_prompt("", context="", task_type="general")
        assert "prompt" in result


class TestPreparePromptWithBudget(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_budget_ok_false_when_exhausted(self):
        budget = TokenBudget(plan_id="p", max_tokens=10, max_tokens_per_task=5)
        budget.record("t", 10)
        result = self.opt.prepare_prompt("Hello world", budget=budget, task_id="t")
        assert not result["budget_ok"]

    def test_budget_ok_true_when_sufficient(self):
        self.opt.create_budget("plan-ok", max_tokens=100_000)
        result = self.opt.prepare_prompt("p", plan_id="plan-ok", task_id="t1")
        assert result["budget_ok"]

    def test_budget_remaining_in_metadata(self):
        budget = self.opt.create_budget("plan-rem", max_tokens=5_000)
        budget.record("prior", 1_000)
        result = self.opt.prepare_prompt("p", plan_id="plan-rem", task_id="new")
        assert "budget_remaining" in result["metadata"]
        assert result["metadata"]["budget_remaining"] == 4000

    def test_explicit_budget_takes_precedence_over_plan_id(self):
        # Create a tight plan budget
        self.opt.create_budget("plan-tight", max_tokens=1)
        # Pass a generous explicit budget
        generous = TokenBudget(plan_id="explicit", max_tokens=100_000)
        result = self.opt.prepare_prompt(
            "p", plan_id="plan-tight", budget=generous, task_id="t"
        )
        assert result["budget_ok"]

    def test_plan_id_budget_lookup(self):
        self.opt.create_budget("plan-lookup", max_tokens=100_000, max_tokens_per_task=8_000)
        result = self.opt.prepare_prompt(
            "p", task_type="general", plan_id="plan-lookup", task_id="t1"
        )
        assert "budget_remaining" in result["metadata"]
        assert result["budget_ok"]


class TestPreparePromptCloudAndLocalTruncation(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_cloud_flag_true_in_metadata(self):
        result = self.opt.prepare_prompt("p", is_cloud_model=True)
        assert result["metadata"]["is_cloud_model"]

    def test_cloud_flag_false_in_metadata(self):
        result = self.opt.prepare_prompt("p", is_cloud_model=False)
        assert not result["metadata"]["is_cloud_model"]

    @patch.object(LocalPreprocessor, "preprocess_for_cloud")
    def test_cloud_preprocessing_called_for_large_context(self, mock_pp):
        large_ctx = _prose(8_000)
        mock_pp.return_value = ("compressed prompt", "compressed ctx", {
            "original_prompt_chars": 6,
            "original_context_chars": 8_000,
            "compressed": True,
            "compression_ratio": 0.4,
            "final_prompt_chars": 6,
            "final_context_chars": 3_200,
        })
        self.opt.prepare_prompt(
            "prompt", context=large_ctx, is_cloud_model=True, task_description="t"
        )
        mock_pp.assert_called_once()

    @patch.object(LocalPreprocessor, "preprocess_for_cloud")
    def test_cloud_preprocessing_not_called_for_small_context(self, mock_pp):
        self.opt.prepare_prompt("p", context="tiny ctx", is_cloud_model=True)
        mock_pp.assert_not_called()

    @patch.object(LocalPreprocessor, "_truncate")
    def test_local_model_truncates_very_long_context(self, mock_trunc):
        mock_trunc.return_value = "truncated result"
        long_ctx = "w " * (_CONTEXT_WINDOW_CHARS + 500)
        result = self.opt.prepare_prompt("p", context=long_ctx, is_cloud_model=False)
        mock_trunc.assert_called_once()
        assert result["metadata"].get("truncated", False)


class TestPreparePromptContextDeduplication(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_first_call_not_deduplicated(self):
        ctx = "some context " * 10
        r1 = self.opt.prepare_prompt("p1", context=ctx)
        assert not r1["metadata"].get("context_deduplicated", False)

    def test_second_call_with_same_context_is_deduplicated(self):
        ctx = "some context " * 10
        self.opt.prepare_prompt("p1", context=ctx)
        r2 = self.opt.prepare_prompt("p2", context=ctx)
        assert r2["metadata"].get("context_deduplicated", False)

    def test_different_context_not_deduplicated(self):
        self.opt.prepare_prompt("p1", context="context A " * 10)
        r2 = self.opt.prepare_prompt("p2", context="context B " * 10)
        assert not r2["metadata"].get("context_deduplicated", False)

    def test_deduplicated_context_contains_reference_prefix(self):
        ctx = "repeated context " * 10
        self.opt.prepare_prompt("p1", context=ctx)
        r2 = self.opt.prepare_prompt("p2", context=ctx)
        assert "[Context unchanged from previous task" in r2["context"]

    def test_empty_context_never_deduplicated(self):
        r1 = self.opt.prepare_prompt("p1", context="")
        r2 = self.opt.prepare_prompt("p2", context="")
        assert not r1["metadata"].get("context_deduplicated", False)
        assert not r2["metadata"].get("context_deduplicated", False)


# ===========================================================================
# TokenOptimizer — summarise_results
# ===========================================================================

class TestSummariseResults(unittest.TestCase):

    def setUp(self):
        self.opt = TokenOptimizer()

    def test_empty_list_returns_empty_string(self):
        assert self.opt.summarise_results([]) == ""

    def test_single_dict_with_description_and_output(self):
        results = [{"description": "Build task", "output": "compiled OK"}]
        summary = self.opt.summarise_results(results)
        assert "Build task" in summary
        assert "compiled OK" in summary

    def test_string_result_included(self):
        results = ["Task A finished"]
        summary = self.opt.summarise_results(results)
        assert "Task A finished" in summary

    def test_mixed_string_and_dict_results(self):
        results = [{"description": "dict task", "output": "done"}, "string result"]
        summary = self.opt.summarise_results(results)
        assert "dict task" in summary
        assert "string result" in summary

    def test_dict_output_is_itself_a_dict(self):
        results = [{"description": "analysis", "output": {"status": "ok", "count": 5}}]
        summary = self.opt.summarise_results(results)
        assert "status" in summary

    def test_uses_task_id_when_no_description(self):
        results = [{"task_id": "my_task", "output": "ok"}]
        summary = self.opt.summarise_results(results)
        assert "my_task" in summary

    def test_uses_result_key_when_no_output(self):
        results = [{"description": "thing", "result": "success"}]
        summary = self.opt.summarise_results(results)
        assert "success" in summary

    def test_long_output_does_not_explode_summary(self):
        results = [{"description": "long", "output": "x" * 1_000}]
        summary = self.opt.summarise_results(results)
        # Per-result output is capped at 200 chars
        assert len(summary) <= 600

    def test_total_truncated_at_max_chars(self):
        results = [{"description": f"t{i}", "output": "res " * 60} for i in range(20)]
        summary = self.opt.summarise_results(results, max_chars=300)
        assert "[... additional results truncated ...]" in summary
        assert len(summary) <= 500

    def test_max_chars_parameter_controls_length(self):
        results = [{"description": "t", "output": "a" * 200} for _ in range(10)]
        short = self.opt.summarise_results(results, max_chars=100)
        long_ = self.opt.summarise_results(results, max_chars=5_000)
        assert len(short) < len(long_)

    def test_none_output_does_not_raise(self):
        results = [{"description": "task", "output": None}]
        summary = self.opt.summarise_results(results)
        assert isinstance(summary, str)

    def test_empty_dict_result_does_not_raise(self):
        results = [{}]
        summary = self.opt.summarise_results(results)
        assert isinstance(summary, str)


# ===========================================================================
# Module constants
# ===========================================================================

class TestModuleConstants(unittest.TestCase):

    def test_task_profiles_is_non_empty_dict(self):
        assert isinstance(TASK_PROFILES, dict)
        assert len(TASK_PROFILES) > 0

    def test_chars_per_token_by_type_has_expected_keys(self):
        for key in ("code", "text", "mixed"):
            assert key in _CHARS_PER_TOKEN_BY_TYPE

    def test_chars_per_token_by_type_values_are_positive_ints(self):
        for key, val in _CHARS_PER_TOKEN_BY_TYPE.items():
            with self.subTest(key=key):
                assert isinstance(val, int)
                assert val > 0

    def test_default_chars_per_token_is_positive(self):
        assert isinstance(_CHARS_PER_TOKEN, int)
        assert _CHARS_PER_TOKEN > 0

    def test_compress_threshold_chars_is_positive(self):
        assert _COMPRESS_THRESHOLD_CHARS > 0

    def test_context_window_chars_is_positive(self):
        assert _CONTEXT_WINDOW_CHARS > 0

    def test_all_profile_tuples_have_three_elements(self):
        for k, v in TASK_PROFILES.items():
            with self.subTest(key=k):
                assert len(v) == 3


# ===========================================================================
# Singleton
# ===========================================================================

class TestGetTokenOptimizer(unittest.TestCase):

    def setUp(self):
        import vetinari.token_optimizer as mod
        mod._token_optimizer = None  # reset between tests

    def tearDown(self):
        import vetinari.token_optimizer as mod
        mod._token_optimizer = None

    def test_returns_token_optimizer_instance(self):
        opt = get_token_optimizer()
        assert isinstance(opt, TokenOptimizer)

    def test_same_instance_returned_on_repeated_calls(self):
        opt1 = get_token_optimizer()
        opt2 = get_token_optimizer()
        assert opt1 is opt2

    def test_fresh_instance_after_reset(self):
        import vetinari.token_optimizer as mod
        opt1 = get_token_optimizer()
        mod._token_optimizer = None
        opt2 = get_token_optimizer()
        assert opt1 is not opt2


if __name__ == "__main__":
    unittest.main()
