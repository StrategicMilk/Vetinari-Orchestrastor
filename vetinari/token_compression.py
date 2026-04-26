"""Token compression — heuristic and LLM-based context reduction.

Contains ``LocalPreprocessor``, which compresses long context strings before
they are sent to inference. Three strategies are used in order of preference:

1. **Heuristic extraction** — AST-based code signature extraction and
   keyword-line filtering (no LLM required, near-instant).
2. **LLMLingua-2 compression** — BERT-level encoder (optional dependency,
   runs on CPU, no LLM server needed).  Installed via ``pip install llmlingua``.  # noqa: VET301 — user guidance string
3. **LLM compression** — a small local llama-cpp-python model rewrites the
   context to retain only technically relevant content (~70% token savings).
4. **Truncation fallback** — sliding-window head+tail truncation when no
   strategy is available.

``LocalPreprocessor`` is used by ``TokenOptimizer.prepare_prompt()`` and is
not intended to be called directly by most callers.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from typing import Any

from vetinari.constants import TRUNCATE_RESPONSE

logger = logging.getLogger(__name__)

# Maximum prompt character length before triggering local summarisation
_COMPRESS_THRESHOLD_CHARS = 24000  # ~6K tokens — modern models handle 32K-64K context natively

# Sliding window for context: keep latest N chars when truncating
_CONTEXT_WINDOW_CHARS = 16000  # ~4K tokens — enough for meaningful code context


def LocalInferenceAdapter(*args: Any, **kwargs: Any) -> Any:
    """Lazily construct the local inference adapter.

    Kept as a module-level callable so tests can patch the historical
    ``vetinari.token_compression.LocalInferenceAdapter`` seam without forcing
    an eager llama.cpp import during module import.

    Returns:
        A lazily-imported ``LocalInferenceAdapter`` instance.
    """
    from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter as _LocalInferenceAdapter

    return _LocalInferenceAdapter(*args, **kwargs)


class LocalPreprocessor:
    """Uses a local LLM to compress and distil context before cloud API calls.

    This is the key cost-reduction feature:
    - Input: verbose context (code, docs, prior results)  ~3000 tokens
    - Output: compressed key points                        ~800 tokens
    - Savings: ~70% of cloud input tokens

    Only activates when:
    1. The target model is a cloud model (has cost > 0)
    2. The context length exceeds the threshold
    3. A local llama-cpp-python model is available
    """

    # Minimum context length (chars) to justify preprocessing overhead
    MIN_CONTEXT_CHARS = _COMPRESS_THRESHOLD_CHARS

    def __init__(self) -> None:
        self._local_model: str | None = None
        # Bounded cache: evict oldest entries when full to prevent memory leaks
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []
        self._max_cache_size: int = 1024

    def _get_local_model(self) -> str | None:
        """Discover the best available local model for preprocessing.

        Returns:
            Model identifier string, or None if no model is available.
        """
        if self._local_model:
            return self._local_model
        try:
            adapter = LocalInferenceAdapter()
            models = adapter.list_loaded_models()
            if models:
                # Prefer smaller/faster models for preprocessing
                for m in models:
                    mid = m.get("id", "")
                    if any(x in mid.lower() for x in ["7b", "8b", "3b", "1b"]):
                        self._local_model = mid
                        return mid
                # Fall back to first available model
                self._local_model = models[0].get("id", "")
                return self._local_model
        except Exception:
            logger.warning("Failed to discover local model for token optimization", exc_info=True)
        return None

    def _cache_put(self, key: str, value: str) -> None:
        """Insert into the bounded cache, evicting oldest entries if full."""
        if key not in self._cache:
            if len(self._cache_order) >= self._max_cache_size:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            self._cache_order.append(key)
        self._cache[key] = value

    def compress_context(
        self,
        context: str,
        task_description: str = "",
        compression_goal: str = "key_facts",
    ) -> tuple[str, float]:
        """Compress verbose context using heuristic extraction first, LLM as fallback.

        Args:
            context: The verbose context to compress.
            task_description: What the context is for (guides compression).
            compression_goal: "key_facts" | "summary" | "code_only"

        Returns:
            Tuple of (compressed_text, compression_ratio) where ratio < 1.0 means smaller.
        """
        if len(context) < self.MIN_CONTEXT_CHARS:
            return context, 1.0

        # Check cache
        cache_key = hashlib.md5(
            f"{context}{task_description}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            ratio = len(cached) / max(len(context), 1)
            return cached, ratio

        # --- Heuristic compression (no LLM needed) ---
        if compression_goal == "code_only":
            compressed = self._extract_code_signatures(context)
            if compressed and len(compressed) < len(context) * 0.8:
                ratio = len(compressed) / max(len(context), 1)
                self._cache_put(cache_key, compressed)
                logger.info(
                    "[LocalPreprocessor] Heuristic code extraction: %s -> %s chars (%.0f%%)",
                    len(context),
                    len(compressed),
                    ratio * 100,
                )
                return compressed, ratio

        if compression_goal == "key_facts":
            compressed = self._extract_key_lines(context)
            if compressed and len(compressed) < len(context) * 0.7:
                ratio = len(compressed) / max(len(context), 1)
                self._cache_put(cache_key, compressed)
                logger.info(
                    "[LocalPreprocessor] Heuristic key-facts extraction: %s -> %s chars (%.0f%%)",
                    len(context),
                    len(compressed),
                    ratio * 100,
                )
                return compressed, ratio

        # --- LLMLingua-2 compression (BERT-level, CPU, optional dep) ---
        llmlingua_result = self._llmlingua_compress(context, task_description)
        if llmlingua_result and len(llmlingua_result) < len(context) * 0.8:
            ratio = len(llmlingua_result) / max(len(context), 1)
            self._cache_put(cache_key, llmlingua_result)
            logger.info(
                "[LocalPreprocessor] LLMLingua-2 compression: %s -> %s chars (%.0f%%)",
                len(context),
                len(llmlingua_result),
                ratio * 100,
            )
            return llmlingua_result, ratio

        # --- LLM compression fallback ---
        local_model = self._get_local_model()
        if not local_model:
            # No local model — fall back to truncation
            return self._truncate(context), len(context[:_CONTEXT_WINDOW_CHARS]) / max(len(context), 1)

        try:
            goals = {
                "key_facts": "Extract ONLY the key facts, function signatures, API endpoints, and critical constraints. Remove examples, explanations, and repetition.",
                "summary": "Write a concise summary preserving all actionable information and technical specifics.",
                "code_only": "Extract ONLY the function/class definitions, signatures, and docstrings. Remove all prose.",
            }
            goal_instruction = goals.get(compression_goal, goals["key_facts"])

            prompt = (
                f"{goal_instruction}\n\n"
                f"Task context: {task_description[:200]}\n\n"
                f"Content to compress:\n{context[:TRUNCATE_RESPONSE]}\n\n"
                "Provide the compressed version. Be as concise as possible while preserving ALL technically relevant information."
            )

            adapter = LocalInferenceAdapter()
            result = adapter.chat(
                local_model,
                "You are a context compression specialist. Compress text while preserving all technically critical information.",
                prompt,
            )
            compressed = result.get("output", "").strip()
            if compressed:
                ratio = len(compressed) / max(len(context), 1)
                self._cache_put(cache_key, compressed)
                logger.info(
                    "[LocalPreprocessor] Compressed %s -> %s chars (%.0f%%) for task: %s",
                    len(context),
                    len(compressed),
                    ratio * 100,
                    task_description[:40],
                )
                return compressed, ratio
        except Exception as e:
            logger.warning("[LocalPreprocessor] Compression failed: %s", e)

        return self._truncate(context), len(context[:_CONTEXT_WINDOW_CHARS]) / max(len(context), 1)

    def _llmlingua_compress(self, context: str, task_description: str) -> str | None:
        """Compress context using LLMLingua-2 BERT-level encoder (optional dependency).

        Runs on CPU, no LLM server needed. Returns None if llmlingua not installed
        or if compression fails for any reason.

        Args:
            context: The context text to compress.
            task_description: Used as instruction hint for guided compression.

        Returns:
            Compressed text string, or None if llmlingua is unavailable or fails.
        """
        try:
            from llmlingua import PromptCompressor

            compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
            result = compressor.compress_prompt(
                context,
                instruction=f"Task: {task_description[:200]}",
                rate=0.5,
            )
            return result.get("compressed_prompt")
        except ImportError:
            logger.warning("LLMLingua-2 not installed — token compression unavailable, returning None")
            return None
        except Exception:
            logger.warning("[LocalPreprocessor] LLMLingua-2 compression failed — falling back to LLM summarization")
            return None

    def _extract_code_signatures_ast(self, code: str) -> str:
        """Extract code structure using ast.parse() for accurate Python parsing.

        Extracts imports, class/function signatures, docstrings, and module-level
        constants. Falls back to the regex method if AST parsing fails.
        """
        tree = ast.parse(code)
        lines: list[str] = []

        def _annotation_str(node: ast.expr) -> str:
            return ast.unparse(node)

        def _default_str(node: ast.expr) -> str:
            return ast.unparse(node)

        def _get_docstring(node: ast.AST) -> str | None:
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                return body[0].value.value
            return None

        def _format_arg(arg: ast.arg) -> str:
            if arg.annotation:
                return f"{arg.arg}: {_annotation_str(arg.annotation)}"
            return arg.arg

        def _format_arguments(args: ast.arguments) -> str:
            parts: list[str] = []
            for i, arg in enumerate(args.posonlyargs):
                n_defaults = len(args.posonlyargs) - len(args.defaults)
                default_idx = i - n_defaults
                if default_idx >= 0:
                    parts.append(f"{_format_arg(arg)} = {_default_str(args.defaults[default_idx])}")
                else:
                    parts.append(_format_arg(arg))
            if args.posonlyargs:
                parts.append("/")
            n_pos = len(args.posonlyargs)
            n_reg = len(args.args)
            n_defaults = len(args.defaults)
            for i, arg in enumerate(args.args):
                default_idx = n_pos + i - (n_reg + n_pos - n_defaults)
                if default_idx >= 0:
                    parts.append(f"{_format_arg(arg)} = {_default_str(args.defaults[default_idx])}")
                else:
                    parts.append(_format_arg(arg))
            if args.vararg:
                parts.append(f"*{_format_arg(args.vararg)}")
            elif args.kwonlyargs:
                parts.append("*")
            for i, arg in enumerate(args.kwonlyargs):
                kw_default = args.kw_defaults[i]
                if kw_default is not None:
                    parts.append(f"{_format_arg(arg)} = {_default_str(kw_default)}")
                else:
                    parts.append(_format_arg(arg))
            if args.kwarg:
                parts.append(f"**{_format_arg(args.kwarg)}")
            return ", ".join(parts)

        def _emit_func(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> None:
            lines.extend(f"{indent}@{ast.unparse(dec)}" for dec in node.decorator_list)
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            sig = f"{indent}{prefix} {node.name}({_format_arguments(node.args)})"
            if node.returns:
                sig += f" -> {_annotation_str(node.returns)}"
            sig += ":"
            lines.append(sig)
            doc = _get_docstring(node)
            if doc:
                short_doc = doc.strip().split("\n")[0]
                lines.append(f'{indent}    """{short_doc}"""')
            else:
                lines.append(f"{indent}    ...")
            lines.append("")

        def _emit_class(node: ast.ClassDef, indent: str) -> None:
            lines.extend(f"{indent}@{ast.unparse(dec)}" for dec in node.decorator_list)
            bases = [ast.unparse(b) for b in node.bases]
            keywords = [f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords]
            all_bases = bases + keywords
            if all_bases:
                lines.append(f"{indent}class {node.name}({', '.join(all_bases)}):")
            else:
                lines.append(f"{indent}class {node.name}:")
            doc = _get_docstring(node)
            if doc:
                short_doc = doc.strip().split("\n")[0]
                lines.append(f'{indent}    """{short_doc}"""')
                lines.append("")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _emit_func(child, indent + "    ")
                elif isinstance(child, ast.ClassDef):
                    _emit_class(child, indent + "    ")
            lines.append("")

        mod_doc = _get_docstring(tree)
        if mod_doc:
            short_doc = mod_doc.strip().split("\n")[0]
            lines.extend((f'"""{short_doc}"""', ""))

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lines.append(ast.unparse(node))
            elif isinstance(node, ast.ClassDef):
                lines.append("")
                _emit_class(node, "")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.append("")
                _emit_func(node, "")
            elif isinstance(node, ast.Assign):
                lines.extend(
                    f"{target.id} = {ast.unparse(node.value)}"
                    for target in node.targets
                    if (
                        isinstance(target, ast.Name)
                        and target.id == target.id.upper()
                        and target.id.replace("_", "").isalpha()
                    )
                    or (isinstance(target, ast.Name) and re.match(r"^[A-Z][A-Z_0-9]*$", target.id))
                )
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and re.match(r"^[A-Z][A-Z_0-9]*$", node.target.id)
            ):
                ann = f": {_annotation_str(node.annotation)}"
                val = f" = {ast.unparse(node.value)}" if node.value else ""
                lines.append(f"{node.target.id}{ann}{val}")

        return "\n".join(lines).strip()

    def _extract_code_signatures(self, context: str) -> str:
        """Extract function/class definitions and docstrings from code.

        Tries AST-based extraction first (accurate for Python).
        Falls back to regex-based extraction for non-Python content or
        code with syntax errors.
        """
        try:
            result = self._extract_code_signatures_ast(context)
            if result:
                return result
        except Exception:
            logger.warning("AST extraction failed, falling back to regex")

        lines = context.split("\n")
        extracted = []
        in_docstring = False
        docstring_delim = None

        for line in lines:
            stripped = line.strip()

            if re.match(r"^(class |def |async def )", stripped):
                extracted.append(line)
                continue

            if stripped.startswith("@"):
                extracted.append(line)
                continue

            if stripped.startswith(("import ", "from ")):
                extracted.append(line)
                continue

            if in_docstring:
                extracted.append(line)
                if docstring_delim and docstring_delim in stripped:
                    in_docstring = False
                continue

            if stripped.startswith(('"""', "'''")):
                in_docstring = True
                docstring_delim = stripped[:3]
                extracted.append(line)
                if stripped.count(docstring_delim) >= 2:
                    in_docstring = False
                continue

            if re.match(r"^[A-Z_][A-Z_0-9]*\s*[:=]", stripped):
                extracted.append(line)
                continue

        return "\n".join(extracted) if extracted else ""

    def _extract_key_lines(self, context: str) -> str:
        """Extract key factual lines: headers, bullet points, definitions, URLs."""
        lines = context.split("\n")
        key_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Headers (markdown or comment headers)
            if stripped.startswith(("#", "##", "###", "//", "/*", "* ")):
                key_lines.append(line)
                continue

            # Bullet points and numbered items
            if re.match(r"^[-*•]\s+|^\d+[.)]\s+", stripped):
                key_lines.append(line)
                continue

            # Lines with key-value patterns (config, API, constraints)
            if re.match(r"^\w[\w_]*\s*[:=]", stripped):
                key_lines.append(line)
                continue

            # Lines containing URLs
            if "http://" in stripped or "https://" in stripped:
                key_lines.append(line)
                continue

            # Lines with key terms (error, warning, constraint, require, must, endpoint)
            if re.search(
                r"\b(error|warning|constraint|require[ds]?|must|endpoint|api|url|port|host|timeout|limit)\b",
                stripped,
                re.IGNORECASE,
            ):
                key_lines.append(line)
                continue

        return "\n".join(key_lines) if key_lines else ""

    def _truncate(self, context: str) -> str:
        """Simple truncation fallback — keep head and tail to preserve framing."""
        if len(context) <= _CONTEXT_WINDOW_CHARS:
            return context
        head = context[: _CONTEXT_WINDOW_CHARS // 3]
        tail = context[-(2 * _CONTEXT_WINDOW_CHARS // 3) :]
        return f"{head}\n\n[... context truncated for token efficiency ...]\n\n{tail}"

    def preprocess_for_cloud(
        self,
        prompt: str,
        context: str = "",
        task_description: str = "",
    ) -> tuple[str, str, dict[str, Any]]:
        """Full preprocessing pipeline for cloud API calls.

        Compresses both context and prompt if they exceed size thresholds,
        reducing cloud token consumption by 30-70%.

        Args:
            prompt: The prompt text to preprocess.
            context: Additional context string to compress.
            task_description: Human-readable description forwarded to the
                compressor for guided key-fact extraction.

        Returns:
            Tuple of (processed_prompt, processed_context, metadata_dict).
        """
        meta: dict[str, Any] = {
            "original_prompt_chars": len(prompt),
            "original_context_chars": len(context),
            "compressed": False,
            "compression_ratio": 1.0,
        }

        if context and len(context) >= self.MIN_CONTEXT_CHARS:
            compressed_context, ratio = self.compress_context(context, task_description, "key_facts")
            meta["compressed"] = ratio < 0.9
            meta["compression_ratio"] = ratio
            context = compressed_context

        if len(prompt) > _COMPRESS_THRESHOLD_CHARS * 2:
            compressed_prompt, ratio = self.compress_context(prompt, task_description, "summary")
            meta["prompt_compressed"] = ratio < 0.9
            prompt = compressed_prompt

        meta["final_prompt_chars"] = len(prompt)
        meta["final_context_chars"] = len(context)
        return prompt, context, meta
