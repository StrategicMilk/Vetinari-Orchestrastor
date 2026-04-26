"""Tests for PerplexityCompressor and compress_for_rag (Story 43)."""

from __future__ import annotations

import pytest

from vetinari.optimization.prompt_compressor import (
    PerplexityCompressor,
    TextSegment,
    compress_for_rag,
)


class TestCompressionRatio:
    """compress() should reduce text length toward the target ratio."""

    def test_compression_reduces_length(self):
        compressor = PerplexityCompressor()
        # Provide a longer repetitive text that has compressible content
        text = (
            "This is a very informative section.\n"
            "aaa aaa aaa aaa aaa aaa aaa aaa aaa.\n"
            "bbb bbb bbb bbb bbb bbb bbb bbb bbb.\n"
            "ccc ccc ccc ccc ccc ccc ccc ccc ccc.\n"
            "ddd ddd ddd ddd ddd ddd ddd ddd ddd.\n"
        )
        result = compressor.compress(text, target_ratio=0.5)
        assert len(result) < len(text)

    def test_ratio_gte_one_returns_original(self):
        compressor = PerplexityCompressor()
        text = "Some text that should not be touched."
        assert compressor.compress(text, target_ratio=1.0) == text
        assert compressor.compress(text, target_ratio=2.0) == text

    def test_empty_input_returns_empty(self):
        compressor = PerplexityCompressor()
        assert compressor.compress("") == ""

    def test_compression_preserves_some_content(self):
        compressor = PerplexityCompressor()
        text = "\n".join(["line number " + str(i) + " with some content here" for i in range(20)])
        result = compressor.compress(text, target_ratio=0.6)
        assert len(result) > 0


class TestStructurePreservationHeaders:
    """Headers must never be removed regardless of entropy."""

    def test_markdown_headers_preserved(self):
        compressor = PerplexityCompressor()
        text = (
            "# Main Title\n"
            "aaa aaa aaa aaa aaa.\n"
            "## Section Two\n"
            "bbb bbb bbb bbb bbb.\n"
            "### Subsection\n"
            "ccc ccc ccc ccc ccc.\n"
        )
        result = compressor.compress(text, target_ratio=0.3)
        assert "# Main Title" in result
        assert "## Section Two" in result
        assert "### Subsection" in result

    def test_header_not_scored_for_removal(self):
        compressor = PerplexityCompressor()
        segments = compressor._split_into_segments("# My Header\nsome content\n")
        header_segs = [s for s in segments if s.text.strip().startswith("#")]
        assert all(s.is_structural for s in header_segs)


class TestCodeBlockPreservation:
    """Fenced code blocks must be preserved in their entirety."""

    def test_code_block_preserved(self):
        compressor = PerplexityCompressor()
        text = (
            "Some introductory prose here.\n"
            "aaa aaa aaa aaa aaa aaa aaa.\n"
            "```python\n"
            "def hello():\n"
            "    return 'world'\n"
            "```\n"
            "bbb bbb bbb bbb bbb bbb.\n"
        )
        result = compressor.compress(text, target_ratio=0.4)
        assert "```python" in result
        assert "def hello():" in result

    def test_code_block_lines_are_structural(self):
        compressor = PerplexityCompressor()
        text = "```\ncode here\n```\n"
        segments = compressor._split_into_segments(text)
        structural = [s for s in segments if s.is_structural]
        assert len(structural) >= 1  # at least the fence lines should be structural


class TestEmptyInput:
    """Edge cases around empty or trivial input."""

    def test_empty_string(self):
        compressor = PerplexityCompressor()
        assert compressor.compress("") == ""

    def test_whitespace_only(self):
        compressor = PerplexityCompressor()
        result = compressor.compress("   \n  \n  ", target_ratio=0.5)
        assert isinstance(result, str)

    def test_single_line_no_crash(self):
        compressor = PerplexityCompressor()
        result = compressor.compress("Just one line.", target_ratio=0.5)
        assert isinstance(result, str)

    def test_compute_entropy_short_string(self):
        compressor = PerplexityCompressor()
        assert compressor._compute_entropy("ab") == 0.0

    def test_compute_entropy_longer_string(self):
        compressor = PerplexityCompressor()
        entropy = compressor._compute_entropy("the quick brown fox jumps over the lazy dog")
        assert entropy > 0.0


class TestCompressForRagTokenBudget:
    """compress_for_rag should compress when token budget is exceeded."""

    def test_short_context_returned_unchanged(self):
        short = "This is a short context."
        result = compress_for_rag(short, max_tokens=2048)
        assert result == short

    def test_long_context_compressed(self):
        # Create a text that exceeds the token budget (len // 4 > max_tokens)
        long_text = "word " * 10000  # ~50000 chars, ~12500 estimated tokens
        result = compress_for_rag(long_text, max_tokens=1000)
        assert len(result) < len(long_text)

    def test_compress_for_rag_returns_string(self):
        result = compress_for_rag("Some context text here.", max_tokens=100)
        assert isinstance(result, str)

    def test_compress_for_rag_empty_input(self):
        result = compress_for_rag("", max_tokens=100)
        assert result == ""

    def test_preserve_patterns_honored(self):
        compressor = PerplexityCompressor(preserve_patterns=[r"IMPORTANT:"])
        text = (
            "aaa aaa aaa aaa aaa aaa.\nIMPORTANT: keep this line.\nbbb bbb bbb bbb bbb bbb.\nccc ccc ccc ccc ccc ccc.\n"
        )
        result = compressor.compress(text, target_ratio=0.3)
        assert "IMPORTANT: keep this line." in result
