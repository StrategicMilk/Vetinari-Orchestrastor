"""File Operations Tool.

====================
Provides safe, sandboxed file operations for Vetinari agents.

All operations are restricted to a configurable project root to prevent
path traversal attacks.  Agents use this tool instead of raw ``os``/``pathlib``
calls so that every I/O operation is auditable and permission-gated.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_resolve(path: str, root: Path) -> Path:
    """Resolve *path* and ensure it stays inside *root*.

    Raises ``PermissionError`` on traversal attempts.
    """
    resolved = (root / path).resolve()
    root_resolved = root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise PermissionError(f"Path traversal blocked: {path!r} resolves outside project root")
    return resolved


@dataclass
class FileInfo:
    """Metadata about a single file or directory."""

    path: str
    name: str
    is_file: bool
    is_dir: bool
    size_bytes: int = 0
    modified: str = ""
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "size_bytes": self.size_bytes,
            "modified": self.modified,
            "created": self.created,
        }


# ---------------------------------------------------------------------------
# Core operations (stateless, testable)
# ---------------------------------------------------------------------------


class FileOperations:
    """Low-level file operations scoped to a project root."""

    def __init__(self, project_root: str | Path | None = None):
        self.root = Path(project_root or os.getcwd()).resolve()

    # -- read ---------------------------------------------------------------

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read and return the text content of *path*."""
        target = _safe_resolve(path, self.root)
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return target.read_text(encoding=encoding)

    def file_exists(self, path: str) -> bool:
        target = _safe_resolve(path, self.root)
        return target.exists()

    def get_file_info(self, path: str) -> FileInfo:
        target = _safe_resolve(path, self.root)
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        st = target.stat()
        return FileInfo(
            path=str(target.relative_to(self.root)),
            name=target.name,
            is_file=target.is_file(),
            is_dir=target.is_dir(),
            size_bytes=st.st_size,
            modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
            created=datetime.fromtimestamp(st.st_ctime).isoformat(),
        )

    def list_directory(self, path: str = ".") -> list[FileInfo]:
        target = _safe_resolve(path, self.root)
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        entries: list[FileInfo] = []
        for child in sorted(target.iterdir()):
            try:
                st = child.stat()
                entries.append(
                    FileInfo(
                        path=str(child.relative_to(self.root)),
                        name=child.name,
                        is_file=child.is_file(),
                        is_dir=child.is_dir(),
                        size_bytes=st.st_size,
                        modified=datetime.fromtimestamp(st.st_mtime).isoformat(),
                        created=datetime.fromtimestamp(st.st_ctime).isoformat(),
                    )
                )
            except OSError:  # noqa: VET022
                pass  # skip inaccessible entries
        return entries

    # -- write --------------------------------------------------------------

    def write_file(self, path: str, content: str, encoding: str = "utf-8") -> str:
        """Write *content* to *path*, creating parent dirs as needed.

        Returns the resolved path relative to the project root.
        """
        target = _safe_resolve(path, self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
        return str(target.relative_to(self.root))

    def create_directory(self, path: str) -> str:
        target = _safe_resolve(path, self.root)
        target.mkdir(parents=True, exist_ok=True)
        return str(target.relative_to(self.root))

    def move_file(self, src: str, dst: str) -> str:
        src_path = _safe_resolve(src, self.root)
        dst_path = _safe_resolve(dst, self.root)
        if not src_path.exists():
            raise FileNotFoundError(f"Source does not exist: {src}")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        return str(dst_path.relative_to(self.root))

    def delete_file(self, path: str) -> bool:
        target = _safe_resolve(path, self.root)
        if not target.exists():
            return False
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return True


# ---------------------------------------------------------------------------
# Tool wrapper (for ToolRegistry integration)
# ---------------------------------------------------------------------------


class FileOperationsTool(Tool):
    """Vetinari Tool wrapper around :class:`FileOperations`."""

    def __init__(self, project_root: str | None = None):
        self._ops = FileOperations(project_root)
        metadata = ToolMetadata(
            name="file_operations",
            description="Safe file I/O restricted to the project directory",
            version="1.0.0",
            category=ToolCategory.FILE_OPERATIONS,
            required_permissions=[ToolPermission.FILE_WRITE],
            parameters=[
                ToolParameter(
                    name="operation",
                    type=str,
                    description="Operation: read, write, list, info, exists, mkdir, move, delete",
                    required=True,
                ),
                ToolParameter(
                    name="path", type=str, description="Target path (relative to project root)", required=True
                ),
                ToolParameter(name="content", type=str, description="File content (for write)", required=False),
                ToolParameter(name="destination", type=str, description="Destination path (for move)", required=False),
            ],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:
        op = kwargs.get("operation", "")
        path = kwargs.get("path", ".")

        try:
            if op == "read":
                content = self._ops.read_file(path)
                return ToolResult(success=True, output=content)

            elif op == "write":
                content = kwargs.get("content", "")
                rel = self._ops.write_file(path, content)
                return ToolResult(success=True, output=f"Written to {rel}")

            elif op == "list":
                entries = self._ops.list_directory(path)
                return ToolResult(success=True, output=[e.to_dict() for e in entries])

            elif op == "info":
                info = self._ops.get_file_info(path)
                return ToolResult(success=True, output=info.to_dict())

            elif op == "exists":
                return ToolResult(success=True, output=self._ops.file_exists(path))

            elif op == "mkdir":
                rel = self._ops.create_directory(path)
                return ToolResult(success=True, output=f"Created {rel}")

            elif op == "move":
                dst = kwargs.get("destination", "")
                if not dst:
                    return ToolResult(success=False, output="", error="destination is required for move")
                rel = self._ops.move_file(path, dst)
                return ToolResult(success=True, output=f"Moved to {rel}")

            elif op == "delete":
                deleted = self._ops.delete_file(path)
                return ToolResult(success=True, output=f"Deleted: {deleted}")

            else:
                return ToolResult(success=False, output="", error=f"Unknown operation: {op}")

        except PermissionError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except FileNotFoundError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:
            logger.exception("FileOperationsTool error")
            return ToolResult(success=False, output="", error=str(exc))
