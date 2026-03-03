# Vetinari Phase 1 CocoIndex Adapter Specification

## Overview

This document defines the pluggable code search adapter architecture, with CocoIndex as the default backend.

---

## 1. Adapter Interface

### 1.1 Base Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class SearchBackendStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INDEXING = "indexing"
    ERROR = "error"

@dataclass
class CodeSearchResult:
    """Single search result"""
    file_path: str
    language: str
    content: str
    line_start: int
    line_end: int
    score: float
    context_before: str = ""
    context_after: str = ""
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return {
            'file_path': self.file_path,
            'language': self.language,
            'content': self.content,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'score': self.score,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'metadata': self.metadata or {}
        }

@dataclass
class SearchFilters:
    """Search filters"""
    language: Optional[str] = None
    file_types: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    min_score: float = 0.0

class CodeSearchAdapter(ABC):
    """Base class for code search backends"""
    
    name: str = "base"
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        limit: int = 10, 
        filters: SearchFilters = None
    ) -> List[CodeSearchResult]:
        """
        Search codebase for query
        
        Args:
            query: Natural language or code query
            limit: Maximum results to return
            filters: Optional search filters
            
        Returns:
            List of CodeSearchResult objects
        """
        pass
    
    @abstractmethod
    def index_project(self, project_path: str, force: bool = False) -> bool:
        """
        Index a project for search
        
        Args:
            project_path: Path to project root
            force: Force reindex even if already indexed
            
        Returns:
            True if indexing succeeded
        """
        pass
    
    @abstractmethod
    def get_status(self) -> SearchBackendStatus:
        """
        Get backend status
        
        Returns:
            SearchBackendStatus enum
        """
        pass
    
    @abstractmethod_indexed_projects(self) ->
    def get List[str]:
        """
        Get list of indexed projects
        
        Returns:
            List of project paths
        """
        pass
```

---

## 2. CocoIndex Adapter Implementation

### 2.1 Implementation

```python
import subprocess
import json
import os
from pathlib import Path
from typing import List, Optional
import shutil

class CocoIndexAdapter(CodeSearchAdapter):
    """CocoIndex as default code search backend"""
    
    name = "cocoindex"
    
    def __init__(
        self, 
        root_path: str = None,
        embedding_model: str = "sbert/sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.root_path = root_path or os.getcwd()
        self.embedding_model = embedding_model
        self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if CocoIndex is available"""
        result = shutil.which('uvx')
        if not result:
            return False
        # Try to run cocoindex --version
        try:
            result = subprocess.run(
                ["uvx", "--prerelease=explicit", "cocoindex-code@latest", "--version"],
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
        filters: SearchFilters = None
    ) -> List[CodeSearchResult]:
        """Search using CocoIndex"""
        
        # Build command
        cmd = [
            "uvx", "--prerelease=explicit",
            "cocoindex-code@latest",
            "search",
            "--query", query,
            "--limit", str(limit),
            "--root-path", self.root_path
        ]
        
        # Add filters if provided
        if filters and filters.language:
            cmd.extend(["--language", filters.language])
            
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
            
            # Parse results
            return self._parse_results(result.stdout, filters)
            
        except subprocess.TimeoutExpired:
            return self._fallback_search(query, limit)
        except Exception as e:
            print(f"CocoIndex search error: {e}")
            return self._fallback_search(query, limit)
    
    def _parse_results(
        self, 
        output: str, 
        filters: SearchFilters = None
    ) -> List[CodeSearchResult]:
        """Parse CocoIndex output into results"""
        
        results = []
        
        try:
            # Try JSON parsing first
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    result = self._item_to_result(item)
                    if result:
                        if filters and result.score < filters.min_score:
                            continue
                        results.append(result)
        except json.JSONDecodeError:
            # Fallback to line-by-line parsing
            for line in output.strip().split('\n'):
                if line.strip():
                    item = {'content': line, 'file': 'unknown', 'score': 0.5}
                    result = self._item_to_result(item)
                    if result:
                        results.append(result)
        
        return results[:limit] if limit else results
    
    def _item_to_result(self, item: dict) -> Optional[CodeSearchResult]:
        """Convert CocoIndex item to CodeSearchResult"""
        
        content = item.get('content', '')
        if not content:
            return None
            
        # Determine language from file extension
        file_path = item.get('file', 'unknown')
        language = self._detect_language(file_path)
        
        return CodeSearchResult(
            file_path=file_path,
            language=language,
            content=content[:500],  # Truncate long content
            line_start=item.get('line_number', 1),
            line_end=item.get('line_number', 1),
            score=item.get('score', 0.5),
            context_before=item.get('context_before', ''),
            context_after=item.get('context_after', ''),
            metadata=item.get('metadata', {})
        )
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.sql': 'sql',
            '.sh': 'shell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.md': 'markdown',
            '.html': 'html',
            '.css': 'css'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'unknown')
    
    def _fallback_search(
        self, 
        query: str, 
        limit: int
    ) -> List[CodeSearchResult]:
        """Fallback to basic grep search if CocoIndex fails"""
        
        import subprocess
        
        results = []
        
        try:
            # Use grep as fallback
            result = subprocess.run(
                ['grep', '-r', '-n', '-i', '--include=*.py', 
                 '--include=*.js', '--include=*.ts', 
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
                                line_start=int(parts[1]),
                                line_end=int(parts[1]),
                                score=0.5
                            ))
                            
        except Exception as e:
            print(f"Fallback search error: {e}")
            
        return results
    
    def index_project(self, project_path: str, force: bool = False) -> bool:
        """Index project with CocoIndex"""
        
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
        """Check if CocoIndex is available"""
        
        if not self._check_availability():
            return SearchBackendStatus.UNAVAILABLE
            
        return SearchBackendStatus.AVAILABLE
    
    def get_indexed_projects(self) -> List[str]:
        """Get list of indexed projects"""
        
        # CocoIndex stores index in .cocoindex_code/ directory
        indexed = []
        
        for root, dirs, files in os.walk(self.root_path):
            if '.cocoindex_code' in dirs:
                indexed.append(root)
                
        return indexed
```

---

## 3. Pluggable Registry

### 3.1 Registry Implementation

```python
class CodeSearchRegistry:
    """Registry for code search backends"""
    
    DEFAULT_BACKEND = "cocoindex"
    
    def __init__(self):
        self.backends: Dict[str, Type[CodeSearchAdapter]] = {}
        self._register_defaults()
        
    def _register_defaults(self):
        """Register default backends"""
        from .cocoindex_adapter import CocoIndexAdapter
        self.register("cocoindex", CocoIndexAdapter)
        
    def register(
        self, 
        name: str, 
        adapter_class: Type[CodeSearchAdapter]
    ):
        """Register a new backend"""
        self.backends[name] = adapter_class
        
    def unregister(self, name: str):
        """Unregister a backend"""
        if name != self.DEFAULT_BACKEND:
            self.backends.pop(name, None)
            
    def get_adapter(
        self, 
        name: str = None,
        **kwargs
    ) -> CodeSearchAdapter:
        """Get adapter instance"""
        name = name or self.DEFAULT_BACKEND
        
        if name not in self.backends:
            raise ValueError(
                f"Unknown backend: {name}. "
                f"Available: {list(self.backends.keys())}"
            )
            
        return self.backends[name](**kwargs)
    
    def list_backends(self) -> List[str]:
        """List available backends"""
        return list(self.backends.keys())
    
    def get_backend_info(self, name: str) -> dict:
        """Get backend information"""
        adapter = self.get_adapter(name)
        status = adapter.get_status()
        
        return {
            'name': name,
            'status': status.value,
            'indexed_projects': adapter.get_indexed_projects() if status == SearchBackendStatus.AVAILABLE else []
        }
```

### 3.2 Usage

```python
# Get default adapter
registry = CodeSearchRegistry()
adapter = registry.get_adapter()

# Search
results = adapter.search("authentication", limit=10)

# Use specific backend
adapter = registry.get_adapter("cocoindex")

# Register custom backend
class MyCustomAdapter(CodeSearchAdapter):
    name = "custom"
    ...
    
registry.register("custom", MyCustomAdapter)
```

---

## 4. API Integration

### 4.1 Flask Routes

```python
from flask import Blueprint, request, jsonify

search_bp = Blueprint('search', __name__, url_prefix='/api/search')
registry = CodeSearchRegistry()

@search_bp.route('', methods=['GET'])
def search_code():
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    language = request.args.get('language')
    backend = request.args.get('backend', 'cocoindex')
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    try:
        adapter = registry.get_adapter(backend)
        
        filters = SearchFilters(language=language) if language else None
        results = adapter.search(query, limit=limit, filters=filters)
        
        return jsonify({
            'query': query,
            'backend': backend,
            'results': [r.to_dict() for r in results],
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@search_bp.route('/index', methods=['POST'])
def index_project():
    data = request.json
    project_path = data.get('project_path')
    backend = data.get('backend', 'cocoindex')
    force = data.get('force', False)
    
    if not project_path:
        return jsonify({'error': 'project_path required'}), 400
    
    try:
        adapter = registry.get_adapter(backend)
        success = adapter.index_project(project_path, force=force)
        
        return jsonify({
            'status': 'indexing' if success else 'error',
            'project_path': project_path,
            'backend': backend
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@search_bp.route('/status', methods=['GET'])
def search_status():
    backends = {}
    
    for name in registry.list_backends():
        try:
            backends[name] = registry.get_backend_info(name)
        except:
            backends[name] = {'name': name, 'status': 'error'}
    
    return jsonify({
        'backends': backends,
        'default_backend': registry.DEFAULT_BACKEND
    })
```

---

## 5. Configuration

### 5.1 YAML Config

```yaml
# vetinari.yaml
search:
  backend: cocoindex
  default_limit: 10
  timeout: 60
  
  # Per-backend config
  backends:
    cocoindex:
      enabled: true
      root_path: "."
      embedding_model: "sbert/sentence-transformers/all-MiniLM-L6-v2"
      
    # Future backends
    # semgrep:
    #   enabled: false
    #   
    # grep:
    #   enabled: false
```

---

## 6. Explorer/Librarian Integration

### 6.1 Integration Pattern

```python
class ExplorerAgent:
    """Explorer agent with code search"""
    
    def __init__(self):
        self.registry = CodeSearchRegistry()
        
    def search_code(self, query: str, agent_type: str = None):
        """Search code using available adapter"""
        
        # Determine best backend
        backend = 'cocoindex'  # Default
        
        adapter = self.registry.get_adapter(backend)
        
        # Execute search
        results = adapter.search(query, limit=15)
        
        # Store in memory
        memory = SharedMemory.get_instance()
        memory.add(
            agent_name='explorer',
            memory_type='discovery',
            summary=f"Search for: {query}",
            content=f"Found {len(results)} results",
            tags=['search', query]
        )
        
        return results
```

---

## 7. Success Criteria

- [ ] CodeSearchAdapter interface defined
- [ ] CocoIndexAdapter implemented as default
- [ ] Pluggable registry working
- [ ] API endpoints functional
- [ ] Fallback search implemented
- [ ] Explorer/Librarian integration complete

---

*Document Version: 1.0*
*Phase: 1*
*Last Updated: 2026-03-02*
