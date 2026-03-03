import subprocess
import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum


class SearchBackendStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INDEXING = "indexing"
    ERROR = "error"


@dataclass
class CodeSearchResult:
    file_path: str
    language: str
    content: str
    line_start: int
    line_end: int
    score: float
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> dict:
        return {
            'file_path': self.file_path,
            'language': self.language,
            'content': self.content,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'score': self.score,
            'context_before': self.context_before,
            'context_after': self.context_after
        }


class CodeSearchAdapter(ABC):
    """Base class for code search backends"""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict = None) -> List[CodeSearchResult]:
        pass

    @abstractmethod
    def index_project(self, project_path: str, force: bool = False) -> bool:
        pass

    @abstractmethod
    def get_status(self) -> SearchBackendStatus:
        pass

    @abstractmethod
    def get_indexed_projects(self) -> List[str]:
        pass


class CocoIndexAdapter(CodeSearchAdapter):
    """CocoIndex as default code search backend"""

    name = "cocoindex"

    def __init__(self, root_path: str = None, embedding_model: str = None):
        self.root_path = root_path or os.getcwd()
        self.embedding_model = embedding_model
        self._status = None

    def _check_availability(self) -> bool:
        if not shutil.which('uvx'):
            return False
        try:
            result = subprocess.run(
                ["uvx", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict = None
    ) -> List[CodeSearchResult]:
        cmd = [
            "uvx", "--prerelease=explicit",
            "cocoindex-code@latest",
            "search",
            "--query", query,
            "--limit", str(limit)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.root_path
            )

            if result.returncode != 0:
                return self._fallback_search(query, limit)

            return self._parse_results(result.stdout)

        except Exception as e:
            print(f"CocoIndex search error: {e}")
            return self._fallback_search(query, limit)

    def _parse_results(self, output: str) -> List[CodeSearchResult]:
        results = []

        try:
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    content = item.get('content', '')
                    if content:
                        results.append(CodeSearchResult(
                            file_path=item.get('file', 'unknown'),
                            language=self._detect_language(item.get('file', '')),
                            content=content[:500],
                            line_start=item.get('line_number', 1),
                            line_end=item.get('line_number', 1),
                            score=item.get('score', 0.5)
                        ))
        except json.JSONDecodeError:
            for line in output.strip().split('\n'):
                if line.strip():
                    results.append(CodeSearchResult(
                        file_path="unknown",
                        language="text",
                        content=line[:500],
                        line_start=1,
                        line_end=1,
                        score=0.5
                    ))

        return results[:10]

    def _detect_language(self, file_path: str) -> str:
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'javascript', '.tsx': 'typescript', '.rs': 'rust',
            '.go': 'go', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
            '.cs': 'csharp', '.rb': 'ruby', '.php': 'php',
            '.swift': 'swift', '.kt': 'kotlin', '.sql': 'sql',
            '.sh': 'shell', '.yaml': 'yaml', '.yml': 'yaml',
            '.json': 'json', '.md': 'markdown', '.html': 'html',
            '.css': 'css'
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'unknown')

    def _fallback_search(
        self,
        query: str,
        limit: int
    ) -> List[CodeSearchResult]:
        results = []

        try:
            result = subprocess.run(
                ['grep', '-r', '-n', '-i',
                 '--include=*.py', '--include=*.js', '--include=*.ts',
                 query, self.root_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                for line in result.stdout.split('\n')[:limit]:
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            results.append(CodeSearchResult(
                                file_path=parts[0],
                                language=self._detect_language(parts[0]),
                                content=parts[2][:200],
                                line_start=int(parts[1]) if parts[1].isdigit() else 1,
                                line_end=int(parts[1]) if parts[1].isdigit() else 1,
                                score=0.5
                            ))

        except Exception as e:
            print(f"Fallback search error: {e}")

        return results

    def index_project(self, project_path: str, force: bool = False) -> bool:
        cmd = [
            "uvx", "--prerelease=explicit",
            "cocoindex-code@latest",
            "index",
            "--path", project_path
        ]

        if force:
            cmd.append("--refresh")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                cwd=project_path
            )
            return result.returncode == 0
        except Exception as e:
            print(f"CocoIndex index error: {e}")
            return False

    def get_status(self) -> SearchBackendStatus:
        if self._status:
            return self._status

        if self._check_availability():
            self._status = SearchBackendStatus.AVAILABLE
        else:
            self._status = SearchBackendStatus.UNAVAILABLE

        return self._status

    def get_indexed_projects(self) -> List[str]:
        indexed = []
        for root, dirs, files in os.walk(self.root_path):
            if '.cocoindex_code' in dirs:
                indexed.append(root)
        return indexed


class CodeSearchRegistry:
    """Registry for code search backends"""

    DEFAULT_BACKEND = "cocoindex"

    def __init__(self):
        self.backends: Dict[str, type] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.backends["cocoindex"] = CocoIndexAdapter

    def register(self, name: str, adapter_class: type):
        self.backends[name] = adapter_class

    def unregister(self, name: str):
        if name != self.DEFAULT_BACKEND:
            self.backends.pop(name, None)

    def get_adapter(self, name: str = None, **kwargs) -> CodeSearchAdapter:
        name = name or self.DEFAULT_BACKEND

        if name not in self.backends:
            raise ValueError(f"Unknown backend: {name}")

        return self.backends[name](**kwargs)

    def list_backends(self) -> List[str]:
        return list(self.backends.keys())

    def get_backend_info(self, name: str) -> dict:
        try:
            adapter = self.get_adapter(name)
            status = adapter.get_status()
            return {
                'name': name,
                'status': status.value,
                'indexed_projects': adapter.get_indexed_projects() if status == SearchBackendStatus.AVAILABLE else []
            }
        except:
            return {'name': name, 'status': 'error'}


code_search_registry = CodeSearchRegistry()
