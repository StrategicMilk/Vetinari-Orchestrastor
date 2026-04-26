"""Tests for context compression — tiered context compression."""

from __future__ import annotations

from vetinari.context import (
    CompressionConfig,
    CompressionResult,
    ContextCompressor,
    get_context_compressor,
)


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_ratio_calculated(self):
        r = CompressionResult(
            original_tokens=100,
            compressed_tokens=60,
            messages=[],
            decisions_preserved=[],
        )
        assert r.compression_ratio == 0.4

    def test_zero_original_tokens(self):
        r = CompressionResult(
            original_tokens=0,
            compressed_tokens=0,
            messages=[],
            decisions_preserved=[],
        )
        assert r.compression_ratio == 0.0


class TestCompressionConfig:
    """Tests for CompressionConfig defaults."""

    def test_defaults(self):
        cfg = CompressionConfig()
        assert cfg.max_context_tokens == 8192
        assert cfg.compress_threshold == 0.75
        assert cfg.preserve_recent == 5
        assert cfg.preserve_system is True


class TestContextCompressor:
    """Tests for the ContextCompressor class."""

    def setup_method(self):
        self.compressor = ContextCompressor()

    def test_estimate_tokens(self):
        # ~4 chars per token
        assert self.compressor.estimate_tokens("hello world!") == 3

    def test_needs_compression_below_threshold(self):
        msgs = [{"role": "user", "content": "short"}]
        assert self.compressor.needs_compression(msgs) is False

    def test_needs_compression_above_threshold(self):
        cfg = CompressionConfig(max_context_tokens=100, compress_threshold=0.5)
        comp = ContextCompressor(cfg)
        msgs = [{"role": "user", "content": "x" * 400}]  # ~100 tokens
        assert comp.needs_compression(msgs) is True

    def test_compress_no_op_when_under_limit(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = self.compressor.compress(msgs, max_tokens=1000)
        assert result.compressed_tokens == result.original_tokens
        assert len(result.messages) == 1

    def test_compress_reduces_verbose_messages(self):
        cfg = CompressionConfig(max_context_tokens=50)
        comp = ContextCompressor(cfg)
        # Create messages with long code blocks
        code = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(30)) + "\n```"
        msgs = [{"role": "assistant", "content": code}]
        result = comp.compress(msgs, max_tokens=50)
        assert result.compressed_tokens <= result.original_tokens

    def test_compress_preserves_system_messages(self):
        cfg = CompressionConfig(max_context_tokens=20, preserve_recent=1)
        comp = ContextCompressor(cfg)
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "x" * 200},
            {"role": "assistant", "content": "y" * 200},
            {"role": "user", "content": "latest question"},
        ]
        result = comp.compress(msgs, max_tokens=20)
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1

    def test_extract_key_decisions(self):
        msgs = [
            {"role": "assistant", "content": "We decided to use PostgreSQL for storage."},
            {"role": "assistant", "content": "The approach is to use REST APIs."},
        ]
        decisions = self.compressor.extract_key_decisions(msgs)
        assert len(decisions) >= 1

    def test_extract_key_decisions_deduplicates(self):
        msgs = [
            {"role": "user", "content": "We decided to use Redis for caching layer."},
            {"role": "user", "content": "We decided to use Redis for caching layer."},
        ]
        decisions = self.compressor.extract_key_decisions(msgs)
        # Both messages have the same decision — should deduplicate
        assert len(decisions) >= 1
        # Should not have duplicates
        lowered = [d.lower()[:50] for d in decisions]
        assert len(lowered) == len(set(lowered))

    def test_summarize_history(self):
        msgs = [
            {"role": "user", "content": "Please help me fix the bug"},
            {"role": "assistant", "content": "I found the issue in line 42"},
        ]
        summary = self.compressor.summarize_history(msgs)
        assert "[user]" in summary
        assert "[assistant]" in summary

    def test_truncate_code_blocks(self):
        long_block = "```python\n" + "\n".join(f"line {i}" for i in range(50)) + "\n```"
        result = self.compressor._truncate_code_blocks(long_block)
        assert "truncated" in result


class TestGetContextCompressor:
    """Tests for the singleton accessor."""

    def test_returns_context_compressor(self):
        comp = get_context_compressor()
        assert isinstance(comp, ContextCompressor)
