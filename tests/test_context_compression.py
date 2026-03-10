"""
Tests for context compression engine.

Run with: python -m pytest tests/test_context_compression.py -x -q --tb=short
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from vetinari.context_compression import (
    CompressionConfig,
    CompressionResult,
    ContextCompressor,
    get_context_compressor,
)


def make_messages(pairs):
    """Helper: build message list from (role, content) tuples."""
    return [{"role": r, "content": c} for r, c in pairs]


class TestEstimateTokens:
    def test_estimate_tokens(self):
        """Rough 4-chars-per-token estimation."""
        compressor = ContextCompressor()
        # 40 chars -> 10 tokens
        assert compressor.estimate_tokens("a" * 40) == 10
        # 1 char -> min 1 token
        assert compressor.estimate_tokens("x") == 1
        # empty -> min 1
        assert compressor.estimate_tokens("") == 1


class TestNeedsCompression:
    def test_needs_compression_below_threshold(self):
        """Returns False when total tokens are under the threshold."""
        config = CompressionConfig(max_context_tokens=1000, compress_threshold=0.75)
        compressor = ContextCompressor(config)
        # 100 chars * 2 msgs = 200 chars -> 50 tokens, well below 750 threshold
        messages = make_messages([("user", "a" * 100), ("assistant", "b" * 100)])
        assert compressor.needs_compression(messages) is False

    def test_needs_compression_above_threshold(self):
        """Returns True when total tokens exceed the threshold."""
        config = CompressionConfig(max_context_tokens=100, compress_threshold=0.75)
        compressor = ContextCompressor(config)
        # 400 chars -> 100 tokens, threshold is 75
        messages = make_messages([("user", "a" * 400)])
        assert compressor.needs_compression(messages) is True


class TestCompressNoChangeNeeded:
    def test_compress_no_change_needed(self):
        """Returns messages unchanged when already under budget."""
        config = CompressionConfig(max_context_tokens=10000)
        compressor = ContextCompressor(config)
        messages = make_messages([("user", "hello"), ("assistant", "world")])
        result = compressor.compress(messages)
        assert result.messages == messages
        assert result.original_tokens == result.compressed_tokens
        assert result.compression_ratio == 0.0


class TestCompressTruncatesCodeBlocks:
    def test_compress_truncates_code_blocks(self):
        """Long code blocks (>20 lines) are truncated."""
        compressor = ContextCompressor()
        # Build a code block with 30 lines
        code_lines = [f"    line_{i} = {i}" for i in range(30)]
        code_block = "```python\n" + "\n".join(code_lines) + "\n```"
        messages = make_messages([("user", code_block)])
        result = compressor._truncate_verbose(messages)
        compressed_content = result[0]["content"]
        assert "truncated" in compressed_content
        assert len(compressed_content) < len(code_block)


class TestCompressTruncatesOutputs:
    def test_compress_truncates_outputs(self):
        """Long output blocks (>30 lines of short text) are truncated."""
        compressor = ContextCompressor()
        # 40 short lines that look like command output
        lines = [f"output line {i}" for i in range(40)]
        content = "\n".join(lines)
        result = compressor._truncate_outputs(content)
        assert "truncated" in result
        assert result.count("\n") < content.count("\n")


class TestCompressSummarizesHistory:
    def test_compress_summarizes_history(self):
        """Old messages get replaced with a summary; recent messages are kept intact."""
        config = CompressionConfig(
            max_context_tokens=10,   # tiny budget to force summarization (messages total ~31 tokens)
            preserve_recent=2,
        )
        compressor = ContextCompressor(config)
        messages = make_messages([
            ("user", "First old message about topic A"),
            ("assistant", "Reply about topic A"),
            ("user", "Second old message about topic B"),
            ("assistant", "Reply about topic B"),
            ("user", "Recent message 1"),          # keep
            ("assistant", "Recent response 1"),    # keep
        ])
        result = compressor.compress(messages)
        roles = [m["role"] for m in result.messages]
        contents = [m["content"] for m in result.messages]
        # A summary system message should have been injected
        assert any("Context Summary" in c for c in contents)
        # Recent messages should still appear verbatim
        assert any("Recent message 1" in c for c in contents)
        assert any("Recent response 1" in c for c in contents)


class TestPreserveSystemMessages:
    def test_preserve_system_messages(self):
        """System messages are always preserved even under heavy compression."""
        config = CompressionConfig(
            max_context_tokens=20,   # force heavy compression
            preserve_system=True,
            preserve_recent=1,
        )
        compressor = ContextCompressor(config)
        system_content = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_content},
            *make_messages([
                ("user", "Message " * 20),
                ("assistant", "Response " * 20),
                ("user", "Recent question"),
            ]),
        ]
        result = compressor.compress(messages)
        system_msgs = [m for m in result.messages if m["role"] == "system" and m["content"] == system_content]
        assert len(system_msgs) >= 1


class TestExtractKeyDecisions:
    def test_extract_key_decisions(self):
        """Finds decision patterns (decided, will, the approach is) in message text."""
        compressor = ContextCompressor()
        messages = make_messages([
            ("assistant", "We decided to use PostgreSQL for the database."),
            ("assistant", "The approach is to split the work into three phases."),
            ("user", "We will implement the auth module first."),
        ])
        decisions = compressor.extract_key_decisions(messages)
        assert len(decisions) > 0
        full_text = " ".join(decisions).lower()
        assert any(
            keyword in full_text
            for keyword in ("postgresql", "phase", "auth", "approach", "implement")
        )

    def test_extract_key_decisions_deduplicates(self):
        """Duplicate decision phrases are not repeated in results."""
        compressor = ContextCompressor()
        repeated = "use microservices architecture for scalability"
        messages = make_messages([
            ("assistant", f"We decided to {repeated}."),
            ("assistant", f"We decided to {repeated}."),
            ("assistant", f"We decided to {repeated}."),
        ])
        decisions = compressor.extract_key_decisions(messages)
        lower_decisions = [d.lower() for d in decisions]
        # All entries should be unique (by first 50 chars of lowercased text)
        keys = [d[:50] for d in lower_decisions]
        assert len(keys) == len(set(keys))


class TestCompressionResultRatio:
    def test_compression_result_ratio(self):
        """compression_ratio is computed as 1 - (compressed / original)."""
        result = CompressionResult(
            original_tokens=100,
            compressed_tokens=60,
            messages=[],
            decisions_preserved=[],
        )
        assert result.compression_ratio == pytest.approx(0.4, abs=0.001)

    def test_compression_result_ratio_zero_when_unchanged(self):
        """compression_ratio is 0.0 when no compression occurred."""
        result = CompressionResult(
            original_tokens=50,
            compressed_tokens=50,
            messages=[],
            decisions_preserved=[],
        )
        assert result.compression_ratio == 0.0


class TestSummarizeHistoryPublicApi:
    def test_summarize_history_public_api(self):
        """summarize_history() returns a non-empty text summary."""
        compressor = ContextCompressor()
        messages = make_messages([
            ("user", "What is the capital of France?"),
            ("assistant", "The capital of France is Paris."),
            ("user", "Tell me more about Paris."),
        ])
        summary = compressor.summarize_history(messages)
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain role labels
        assert "[user]" in summary or "[assistant]" in summary

    def test_summarize_history_empty_messages(self):
        """summarize_history() handles empty message list gracefully."""
        compressor = ContextCompressor()
        summary = compressor.summarize_history([])
        assert isinstance(summary, str)


class TestGetContextCompressor:
    def test_get_context_compressor_returns_instance(self):
        """get_context_compressor() returns a ContextCompressor instance."""
        # Reset singleton for test isolation
        import vetinari.context_compression as mod
        mod._compressor = None
        compressor = get_context_compressor()
        assert isinstance(compressor, ContextCompressor)

    def test_get_context_compressor_singleton(self):
        """get_context_compressor() returns the same instance on repeated calls."""
        import vetinari.context_compression as mod
        mod._compressor = None
        c1 = get_context_compressor()
        c2 = get_context_compressor()
        assert c1 is c2
