"""Chat API handlers — file attachments, conversation export, feedback, and retry.

Native Litestar equivalents of the routes previously registered by ``chat_api``.
Part of Flask->Litestar migration (ADR-0066). URL paths identical to Flask originals.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import msgspec

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post
    from litestar.datastructures import UploadFile
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# -- Constants ----------------------------------------------------------------

MAX_ATTACHMENT_SIZE_BYTES = 32 * 1024 * 1024  # 32 MB (matches Claude's per-file limit)

# Explicit allowlist of safe extensions — anything not listed is rejected.
# Prefer allowlist over blocklist so new risky types are blocked by default.
ALLOWED_ATTACHMENT_EXTENSIONS = frozenset({
    # Source code
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".kt",
    ".swift",
    ".rb",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".r",
    ".lua",
    ".scala",
    ".clj",
    # Web / markup
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".less",
    ".svg",
    # Data / config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".csv",
    ".tsv",
    ".xml",
    ".sql",
    # Documentation
    ".md",
    ".rst",
    ".txt",
    ".tex",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".tiff",
    # Archives (read-only; no execution risk in upload context)
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    # Notebooks / misc
    ".ipynb",
    ".pdf",
})

# MIME type map for attachment serving
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
}

# -- Private helpers ----------------------------------------------------------


def _attachments_dir(project_id: str) -> Path:
    """Return the attachments directory for a project, creating it if absent.

    Args:
        project_id: The project identifier used as the subdirectory name.

    Returns:
        A ``pathlib.Path`` pointing to the attachments directory.
    """
    from vetinari.web import shared

    target = shared.PROJECT_ROOT / "outputs" / "attachments" / project_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _feedback_dir() -> Path:
    """Return the feedback output directory, creating it if absent.

    Returns:
        A ``pathlib.Path`` pointing to ``outputs/feedback/``.
    """
    from vetinari.web import shared

    target = shared.PROJECT_ROOT / "outputs" / "feedback"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_conversation(project_id: str) -> list[dict[str, Any]]:
    """Load the conversation history for a project from disk.

    Reads ``projects/<project_id>/conversation.json`` if it exists, otherwise
    returns an empty list.

    Args:
        project_id: The project identifier.

    Returns:
        A list of message dictionaries as stored in conversation.json.

    Raises:
        FileNotFoundError: If the project directory does not exist.
        json.JSONDecodeError: If conversation.json is malformed.
    """
    from vetinari.web import shared

    project_dir = shared.PROJECT_ROOT / "projects" / project_id
    if not project_dir.exists():
        raise FileNotFoundError(f"Project '{project_id}' not found")

    conv_file = project_dir / "conversation.json"
    if not conv_file.exists():
        return []

    with Path(conv_file).open(encoding="utf-8") as fh:
        return json.load(fh)


def _safe_download_name(filename: str) -> str:
    """Return an RFC-safe filename for Content-Disposition headers."""
    name = Path(filename or "download").name.replace("\\", "_").replace("/", "_")
    name = "".join("_" if ord(ch) < 32 or ch in {'"', ";", "\x7f"} else ch for ch in name).strip(" .")
    name = re.sub(r"_+", "_", name)
    return name or "download"


def _content_disposition(disposition: str, filename: str) -> str:
    """Build a safe Content-Disposition value with RFC 5987 UTF-8 filename."""
    safe_name = _safe_download_name(filename)
    ascii_name = safe_name.encode("ascii", errors="ignore").decode("ascii") or "download"
    encoded = quote(safe_name, safe="")
    return f'{disposition}; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'


def _render_conversation_markdown(messages: list[dict[str, Any]], project_id: str) -> str:
    """Render a conversation history as a Markdown document.

    User messages are rendered as blockquotes. Agent responses are rendered
    under ``##`` headings with code blocks preserved.

    Args:
        messages: List of message dicts with at least ``role`` and ``content``.
        project_id: The project identifier, used in the document title.

    Returns:
        A Markdown string representing the full conversation.
    """
    lines: list[str] = [
        f"# Conversation Export — {project_id}",
        "",
        f"Exported at: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
    ]
    from vetinari.security.redaction import redact_text
    from vetinari.web.litestar_projects_formatters import markdown_code_block

    for msg in messages:
        role = redact_text(str(msg.get("role", "unknown")))
        content = str(msg.get("content", ""))
        agent = redact_text(str(msg.get("agent", "")))
        timestamp = redact_text(str(msg.get("timestamp", "")))
        ts_suffix = f" _{timestamp}_" if timestamp else ""
        agent_label = "User" if role == "user" else agent or role.capitalize()
        lines.extend((f"## {agent_label}{ts_suffix}", "", markdown_code_block(content), ""))
    return "\n".join(lines)


def _render_conversation_text(messages: list[dict[str, Any]], project_id: str) -> str:
    """Render a conversation history as plain text.

    Each message is prefixed with a role label and optional timestamp.

    Args:
        messages: List of message dicts with at least ``role`` and ``content``.
        project_id: The project identifier, used in the document header.

    Returns:
        A plain-text string representing the full conversation.
    """
    lines: list[str] = [
        f"Conversation Export — {project_id}",
        f"Exported at: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        "=" * 60,
        "",
    ]
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        agent = msg.get("agent", "")
        timestamp = msg.get("timestamp", "")
        label = agent or role.capitalize()
        ts_suffix = f"  [{timestamp}]" if timestamp else ""
        lines.extend((f"[{label}]{ts_suffix}", content, ""))
    return "\n".join(lines)


def _render_conversation_json(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise conversation messages into a structured export format.

    Args:
        messages: Raw message dicts from conversation.json.

    Returns:
        A list of dicts each containing ``role``, ``content``, ``agent``,
        and ``timestamp`` keys.
    """
    return [
        {
            "role": msg.get("role", "unknown"),
            "content": msg.get("content", ""),
            "agent": msg.get("agent", ""),
            "timestamp": msg.get("timestamp", ""),
        }
        for msg in messages
    ]


# -- Request body models -------------------------------------------------------


class RetryTaskRequest(msgspec.Struct, forbid_unknown_fields=True):
    """Typed request body for the retry-task endpoint.

    Uses msgspec.Struct with forbid_unknown_fields=True so Litestar returns 400
    for bodies that contain any key other than ``model`` (e.g. deeply nested
    structures, garbage payloads, or attempt to inject unexpected parameters).

    ``model`` is required (may be ``null``) so that an empty ``{}`` body is
    rejected with 422 — callers must explicitly declare their model intent
    rather than relying on a silent default.

    Attributes:
        model: Model identifier to use for the retry, or ``null`` to use the
            default model for the project.
    """

    model: str | None  # required — pass null to use project default


# -- Handler factory ----------------------------------------------------------


def create_chat_api_handlers() -> list[Any]:
    """Return Litestar route handler instances for the chat API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all chat endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response
    from vetinari.web.shared import validate_path_param

    @post("/api/v1/chat/attachments", guards=[admin_guard])
    async def upload_attachment(data: UploadFile) -> Response:
        """Accept a multipart file upload and persist it in the project attachments store.

        Validates the file extension against an allowlist and rejects files larger
        than 32 MB. Saves the file under
        ``outputs/attachments/<project_id>/<attachment_id>_<original_name>``.

        Args:
            data: The uploaded file provided as a Litestar UploadFile.

        Returns:
            JSON object with ``ok``, ``attachment_id``, ``filename``, ``size``,
            and ``type`` on success.
            Returns 400 for missing filename, disallowed extension, or oversized
            file. Returns 500 on unexpected errors.
        """
        if not data.filename:
            return litestar_error_response("Uploaded file has no filename", code=400)

        # Attachments are intentionally global-scoped: this endpoint stores all
        # uploads under outputs/attachments/default/ regardless of the calling
        # project.  There is no per-project isolation here by design — a future
        # endpoint can add project-scoped uploads if required.
        attachment_scope = "default"
        original_name = Path(data.filename).name
        extension = Path(original_name).suffix.lower()

        if extension not in ALLOWED_ATTACHMENT_EXTENSIONS:
            return litestar_error_response(f"File type '{extension}' is not allowed.", code=400)

        file_data = await data.read()
        file_size = len(file_data)

        if file_size > MAX_ATTACHMENT_SIZE_BYTES:
            return litestar_error_response(
                f"File size {file_size} bytes exceeds the 32 MB limit ({MAX_ATTACHMENT_SIZE_BYTES} bytes)",
                code=400,
            )

        attachment_id = str(uuid.uuid4())
        save_name = f"{attachment_id}_{original_name}"
        dest = _attachments_dir(attachment_scope) / save_name
        dest.write_bytes(file_data)

        logger.info(
            "Attachment %s saved to global scope (%d bytes, type=%s)",
            attachment_id,
            file_size,
            extension,
        )

        return Response(
            content={
                "ok": True,
                "attachment_id": attachment_id,
                "filename": original_name,
                "size": file_size,
                "type": extension,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/chat/attachments/{attachment_id:str}")
    async def serve_attachment(attachment_id: str) -> Response:
        """Serve an uploaded attachment file by its UUID prefix.

        Searches all project attachment directories for a file whose name starts
        with the given attachment_id and returns it with the correct MIME type.

        Args:
            attachment_id: The UUID assigned when the file was uploaded.

        Returns:
            The file contents with appropriate Content-Type header.
            Returns 400 if attachment_id is invalid, 404 if not found.
        """
        from vetinari.web import shared

        clean_id = (validate_path_param(attachment_id) or "").strip()
        if not clean_id or len(clean_id) < 8:
            return litestar_error_response("Invalid attachment_id", code=400)

        attachments_root = shared.PROJECT_ROOT / "outputs" / "attachments"
        if not attachments_root.exists():
            return litestar_error_response("Attachment not found", code=404)

        for project_dir in attachments_root.iterdir():
            if not project_dir.is_dir():
                continue
            for candidate in project_dir.iterdir():
                if candidate.name.startswith(clean_id):
                    ext = candidate.suffix.lower()
                    mimetype = _MIME_MAP.get(ext, "application/octet-stream")
                    download_name = candidate.name.split("_", 1)[-1] if "_" in candidate.name else candidate.name
                    return Response(
                        content=candidate.read_bytes(),
                        status_code=200,
                        media_type=mimetype,
                        headers={"Content-Disposition": _content_disposition("inline", download_name)},
                    )

        return litestar_error_response("Attachment not found", code=404)

    @get("/api/v1/chat/export/{project_id:str}")
    async def export_conversation(
        project_id: str,
        format: str = Parameter(query="format", default="md"),
    ) -> Response:
        """Export a project's conversation history as a downloadable file.

        Supports Markdown, plain text, and JSON output formats.

        Args:
            project_id: The project whose conversation history to export.
            format: Output format — ``md`` (default), ``json``, or ``txt``.
                Also accepts ``markdown`` and ``text`` as aliases.

        Returns:
            A file download response with ``Content-Disposition: attachment``.
            Returns 400 for an unsupported format, 404 when the project
            does not exist.
        """
        from vetinari.web.shared import validate_path_param

        safe_id = validate_path_param(project_id)
        if safe_id is None:
            return litestar_error_response("Invalid project ID", code=400)

        fmt = format.strip().lower()
        _fmt_aliases = {"markdown": "md", "text": "txt", "plaintext": "txt"}
        fmt = _fmt_aliases.get(fmt, fmt)
        if fmt not in ("md", "json", "txt"):
            return litestar_error_response("Unsupported format. Use 'md' (or 'markdown'), 'json', or 'txt'", code=400)

        try:
            messages = _load_conversation(safe_id)
        except FileNotFoundError:
            logger.warning("Conversation export requested for unknown project '%s' — returning 404", safe_id)
            return litestar_error_response(f"Project '{safe_id}' not found", code=404)

        if fmt == "md":
            content_str = _render_conversation_markdown(messages, project_id)
            mime = "text/markdown"
            filename = f"conversation_{safe_id}.md"
            body = content_str.encode("utf-8")
        elif fmt == "txt":
            content_str = _render_conversation_text(messages, project_id)
            mime = "text/plain"
            filename = f"conversation_{safe_id}.txt"
            body = content_str.encode("utf-8")
        else:
            payload = _render_conversation_json(messages)
            content_str = json.dumps(payload, indent=2, ensure_ascii=False)
            mime = "application/json"
            filename = f"conversation_{safe_id}.json"
            body = content_str.encode("utf-8")

        return Response(
            content=body,
            status_code=200,
            media_type=mime,
            headers={"Content-Disposition": _content_disposition("attachment", filename)},
        )

    @post("/api/v1/chat/feedback", guards=[admin_guard])
    async def submit_feedback(data: dict[str, Any]) -> Response:
        """Store user feedback for a specific task result.

        Appends a feedback record to ``outputs/feedback/feedback.jsonl`` and
        wires the rating into the Thompson sampling and rule-learning systems.

        Args:
            data: Request body with ``project_id``, ``task_id``, ``rating``
                (``"up"`` or ``"down"``), optional ``score``, and optional
                ``comment``.

        Returns:
            JSON object with ``ok`` (``true``) on success.
            Returns 400 when required fields are missing or rating is invalid.
        """
        from vetinari.types import AgentType
        from vetinari.web.shared import validate_path_param

        project_id = str(data.get("project_id") or "").strip()
        task_id = str(data.get("task_id") or "").strip()
        rating = str(data.get("rating") or "").strip()

        if not project_id:
            return litestar_error_response("'project_id' is required", code=400)
        if validate_path_param(project_id) is None:
            return litestar_error_response("Invalid project ID", code=400)
        if not task_id:
            return litestar_error_response("'task_id' is required", code=400)
        if rating not in ("up", "down"):
            return litestar_error_response("'rating' must be 'up' or 'down'", code=400)

        score = data.get("score")
        raw_comment = data.get("comment")
        comment = unicodedata.normalize("NFC", raw_comment) if isinstance(raw_comment, str) else raw_comment

        record: dict[str, Any] = {
            "project_id": project_id,
            "task_id": task_id,
            "rating": rating,
            "score": score,
            "comment": comment,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        feedback_file = _feedback_dir() / "feedback.jsonl"
        with Path(feedback_file).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Wire feedback into learning system — thumbs-up/down updates Thompson
        # arms and triggers rule learning on rejections.
        try:
            from vetinari.learning.feedback_loop import get_feedback_loop
            from vetinari.learning.model_selector import get_thompson_selector

            _fb = get_feedback_loop()
            _thompson = get_thompson_selector()
            _quality = 0.9 if rating == "up" else 0.2

            _model_id = "default"
            _task_type = "general"
            _agent_type = AgentType.WORKER.value
            try:
                conv = _load_conversation(project_id)
                for msg in conv:
                    _meta = msg.get("metadata", {})
                    if _meta.get("task_id") == task_id:
                        _model_id = _meta.get("model_id") or _model_id
                        _task_type = _meta.get("task_type") or _task_type
                        _agent_type = _meta.get("agent_type") or _agent_type
                        break
            except Exception:
                logger.warning("Could not look up task %s in project %s history", task_id, project_id)

            _thompson.update(_model_id, _task_type, _quality, success=(rating == "up"))

            if rating == "down":
                _fb.record_quality_rejection(
                    agent_type=_agent_type,
                    mode="user_feedback",
                    violation_description=str(comment or f"User rejected task {task_id}"),
                    model_name=_model_id,
                )
        except Exception:
            logger.warning("Failed to wire feedback into learning system", exc_info=True)

        logger.info(
            "Feedback recorded for project=%s task=%s rating=%s",
            project_id,
            task_id,
            rating,
        )
        return Response(content={"ok": True}, status_code=200, media_type=MediaType.JSON)

    @post("/api/v1/chat/retry/{project_id:str}/{task_id:str}", guards=[admin_guard])
    async def retry_task(project_id: str, task_id: str, data: RetryTaskRequest) -> Response:
        """Queue a retry for a previously executed task via the orchestrator.

        Enqueues a retry request through the orchestration RequestQueue so the
        task is re-executed by the pipeline. Accepts an optional model override.
        Litestar validates the body against ``RetryTaskRequest`` and returns 422
        for unexpected shapes (e.g. deeply nested objects).

        Args:
            project_id: The project that owns the task.
            task_id: The task to retry.
            data: Validated request body with an optional ``model`` override.

        Returns:
            JSON object with ``ok``, ``message``, and ``exec_id`` on success.
            Returns 503 when the orchestrator is unavailable.
        """
        from vetinari.web.shared import validate_path_param

        if validate_path_param(project_id) is None:
            return litestar_error_response("Invalid project ID", code=400)
        if validate_path_param(task_id) is None:
            return litestar_error_response("Invalid task ID", code=400)

        model_override = data.model or None

        try:
            from vetinari.orchestration.request_routing import PRIORITY_STANDARD, RequestQueue

            queue = RequestQueue()
            context = {
                "project_id": project_id,
                "task_id": task_id,
                "retry": True,
                "model_override": model_override,
            }
            exec_id = queue.enqueue(
                goal=f"Retry task {task_id} in project {project_id}",
                context=context,
                priority=PRIORITY_STANDARD,
            )
        except Exception:
            logger.exception("Failed to enqueue retry for project=%s task=%s", project_id, task_id)
            return litestar_error_response("Orchestrator unavailable — retry not queued", code=503)

        logger.info(
            "Retry enqueued exec_id=%s for project=%s task=%s model_override=%s",
            exec_id,
            project_id,
            task_id,
            model_override,
        )
        return Response(
            content={"ok": True, "message": "Retry queued", "exec_id": exec_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    return [
        upload_attachment,
        serve_attachment,
        export_conversation,
        submit_feedback,
        retry_task,
    ]
