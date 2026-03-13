"""Context Compression Engine — auto-compress context when approaching token limits.

Critical for local LLMs with 4-32K context windows where every saved token matters.
Uses a tiered strategy: remove duplicates → truncate verbose outputs → summarize history.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    messages: list[dict[str, str]]
    decisions_preserved: list[str]
    compression_ratio: float = 0.0

    def __post_init__(self):
        if self.original_tokens > 0:
            self.compression_ratio = round(1 - (self.compressed_tokens / self.original_tokens), 3)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""

    max_context_tokens: int = 8192  # model's context window
    compress_threshold: float = 0.75  # compress when context reaches this % of max
    preserve_recent: int = 5  # always keep last N messages uncompressed
    preserve_system: bool = True  # always keep system messages
    min_message_tokens: int = 10  # don't try to compress very short messages
    summary_max_tokens: int = 200  # max tokens per summarized chunk


class ContextCompressor:
    """Auto-compress conversation context when approaching token limits.

    Tiered compression strategy:
    1. Remove duplicate/redundant information
    2. Truncate verbose tool outputs and code blocks
    3. Summarize older conversation history
    4. Extract and preserve key decisions
    """

    def __init__(self, config: CompressionConfig | None = None):
        self.config = config or CompressionConfig()
        self._decision_patterns = [
            r"(?:decided|chosen|selected|agreed|confirmed|approved)\s+(?:to\s+)?(.+?)(?:\.|$)",
            r"(?:will|going to|plan to)\s+(.+?)(?:\.|$)",
            r"(?:the approach|the solution|the fix)\s+(?:is|was)\s+(.+?)(?:\.|$)",
        ]

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return max(1, len(text) // 4)

    def estimate_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate total tokens across all messages."""
        return sum(self.estimate_tokens(m.get("content", "")) for m in messages)

    def needs_compression(self, messages: list[dict[str, str]]) -> bool:
        """Check if context needs compression."""
        total = self.estimate_messages_tokens(messages)
        threshold = int(self.config.max_context_tokens * self.config.compress_threshold)
        return total > threshold

    def compress(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> CompressionResult:
        """Compress messages to fit within token budget.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Override max context tokens

        Returns:
            CompressionResult with compressed messages and metadata
        """
        max_tokens = max_tokens or self.config.max_context_tokens
        original_tokens = self.estimate_messages_tokens(messages)

        if original_tokens <= max_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                messages=list(messages),
                decisions_preserved=self.extract_key_decisions(messages),
            )

        # Extract decisions before compression
        decisions = self.extract_key_decisions(messages)

        # Tier 1: Truncate verbose outputs
        compressed = self._truncate_verbose(messages)

        # Check if sufficient
        current_tokens = self.estimate_messages_tokens(compressed)
        if current_tokens <= max_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=current_tokens,
                messages=compressed,
                decisions_preserved=decisions,
            )

        # Tier 2: Summarize older messages (keep recent ones intact)
        compressed = self._summarize_history(compressed, max_tokens)
        current_tokens = self.estimate_messages_tokens(compressed)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=current_tokens,
            messages=compressed,
            decisions_preserved=decisions,
        )

    def _truncate_verbose(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Tier 1: Truncate code blocks and tool outputs."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            # Don't truncate system messages or short messages
            if role == "system" or self.estimate_tokens(content) < self.config.min_message_tokens:
                result.append(msg)
                continue

            # Truncate long code blocks (keep first and last 5 lines)
            content = self._truncate_code_blocks(content)

            # Truncate long tool/command outputs (keep first 10 lines)
            content = self._truncate_outputs(content)

            result.append({"role": role, "content": content})

        return result

    def _truncate_code_blocks(self, text: str) -> str:
        """Truncate code blocks longer than 20 lines."""

        def truncate_block(match):
            block = match.group(0)
            lines = block.split("\n")
            if len(lines) <= 20:
                return block
            lang = lines[0]  # ```python etc
            closing = lines[-1]  # ```
            kept = [*lines[1:6], f"... ({len(lines) - 12} lines truncated) ...", *lines[-6:-1]]
            return lang + "\n" + "\n".join(kept) + "\n" + closing

        return re.sub(r"```[\s\S]*?```", truncate_block, text)

    def _truncate_outputs(self, text: str) -> str:
        """Truncate command/tool output blocks."""
        lines = text.split("\n")
        if len(lines) <= 30:
            return text

        # If it looks like command output (many short lines), truncate middle
        avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)  # noqa: E741
        if avg_line_len < 80 and len(lines) > 30:
            kept = [*lines[:10], f"... ({len(lines) - 15} lines truncated) ...", *lines[-5:]]
            return "\n".join(kept)

        return text

    def _summarize_history(self, messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
        """Tier 2: Summarize older messages, keep recent ones intact."""
        if len(messages) <= self.config.preserve_recent:
            return messages

        # Split into old (to summarize) and recent (to keep)
        system_msgs = [m for m in messages if m.get("role") == "system"] if self.config.preserve_system else []
        non_system = [m for m in messages if m.get("role") != "system"]

        preserve_count = min(self.config.preserve_recent, len(non_system))
        old_msgs = non_system[:-preserve_count] if preserve_count > 0 else non_system
        recent_msgs = non_system[-preserve_count:] if preserve_count > 0 else []

        # Create summary of old messages
        summary = self._create_summary(old_msgs)

        summary_msg = {"role": "system", "content": f"[Context Summary]\n{summary}"}

        return [*system_msgs, summary_msg, *recent_msgs]

    def _create_summary(self, messages: list[dict[str, str]]) -> str:
        """Create a concise summary of messages without LLM (extractive)."""
        summaries = []

        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")

            if not content.strip():
                continue

            # Extract first sentence or first line as summary
            first_line = content.split("\n")[0].strip()
            if len(first_line) > 150:
                first_line = first_line[:150] + "..."

            if first_line:
                summaries.append(f"- [{role}] {first_line}")

        # Cap total summary length
        result = "\n".join(summaries)
        max_chars = self.config.summary_max_tokens * 4  # rough estimate
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (earlier context truncated)"

        return result

    def extract_key_decisions(self, messages: list[dict[str, str]]) -> list[str]:
        """Extract key decisions from conversation history."""
        decisions = []

        for msg in messages:
            content = msg.get("content", "")
            for pattern in self._decision_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    clean = match.strip()
                    if len(clean) > 10 and len(clean) < 200:
                        decisions.append(clean)

        # Deduplicate
        seen = set()
        unique = []
        for d in decisions:
            key = d.lower()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique[:20]  # cap at 20 decisions

    def summarize_history(self, messages: list[dict[str, str]]) -> str:
        """Public API: get a text summary of message history."""
        return self._create_summary(messages)


_compressor: ContextCompressor | None = None


def get_context_compressor(config: CompressionConfig | None = None) -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor(config)
    return _compressor
