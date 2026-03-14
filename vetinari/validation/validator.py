"""Validator module."""

from __future__ import annotations

import ast
import json
import re


class Validator:
    """Validates agent outputs against quality and safety rules."""
    def is_valid_text(self, text: str) -> bool:
        # Check if text is empty
        """Check if valid text.

        Returns:
            True if successful, False otherwise.
        """
        if not text or len(text.strip()) == 0:
            return False

        # Try to parse as JSON first (some tasks return JSON)
        try:
            json.loads(text)
            # If it's valid JSON, it's probably a structured response
            return True
        except (json.JSONDecodeError, TypeError, ValueError):  # noqa: VET022
            pass

        # Check if it looks like code
        if self._looks_like_code(text):
            return self._validate_python_code(text)

        # Default: if there's content and it's not empty, accept it
        return len(text.strip()) > 0

    def _looks_like_code(self, text: str) -> bool:
        # Check for code patterns, including markdown code blocks
        # Remove markdown code blocks first (handles both opening and closing fences)
        cleaned = re.sub(r"```[\w]*\n?", "", text)
        cleaned = cleaned.strip()
        return bool(re.search(r"^(def|class|import|from|@|#|async|with|\s+=\s+)", cleaned, re.M))

    def _validate_python_code(self, code: str) -> bool:
        # Remove markdown code blocks before validation
        cleaned = re.sub(r"```[\w]*\n", "\n", code)
        cleaned = re.sub(r"```$", "", cleaned)
        cleaned = cleaned.strip()

        # Remove JSON wrapper only when the entire content is a single JSON object
        # (not valid Python code that starts/ends with braces)
        if cleaned.startswith("{") and cleaned.endswith("}"):
            # Check if this looks like a JSON code wrapper rather than Python dict/set
            # Only strip if there is no Python code structure inside
            inner = cleaned[1:-1].strip()
            if not re.search(r"^(def|class|import|from|@|async|with|\s+=)", inner, re.M):
                cleaned = inner

        try:
            ast.parse(cleaned)
            return True
        except (SyntaxError, ValueError):
            # Real syntax error - return False so the executor can flag it
            return False
