import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

class CocoIndexClient:
    """CocoIndex semantic code search client for Vetinari."""
    
    def __init__(self, root_path: Optional[str] = None):
        self.root_path = root_path or os.getcwd()
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if cocoindex is available."""
        try:
            result = subprocess.run(
                ["uvx", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def search(self, query: str, limit: int = 10, refresh: bool = True) -> List[Dict[str, Any]]:
        """
        Search the codebase using semantic similarity.
        
        Args:
            query: Natural language query or code snippet
            limit: Maximum results (1-100)
            refresh: Refresh index before searching
            
        Returns:
            List of matching code chunks with file path, language, content, line numbers, similarity score
        """
        if not self.available:
            return []
        
        try:
            cmd = [
                "uvx", "--prerelease=explicit",
                "cocoindex-code@latest",
                "search",
                "--query", query,
                "--limit", str(limit)
            ]
            
            if refresh:
                cmd.append("--refresh")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.root_path
            )
            
            if result.returncode == 0:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return self._parse_search_output(result.stdout)
            
            return []
            
        except Exception as e:
            print(f"[CocoIndex] Search error: {e}")
            return []
    
    def _parse_search_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse search output if not JSON."""
        results = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if line.strip():
                results.append({
                    "content": line,
                    "file": "unknown",
                    "score": 0.5
                })
        
        return results[:10]
    
    def get_file_context(self, file_path: str, lines: int = 50) -> str:
        """Get context around a specific file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines_list = content.split('\n')
                
                # Get first N lines for context
                context = '\n'.join(lines_list[:lines])
                return context
                
        except Exception as e:
            return f"Error reading file: {e}"
    
    def find_similar_code(self, code_snippet: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find code similar to the given snippet."""
        return self.search(code_snippet, limit=limit, refresh=False)
    
    def get_project_structure(self) -> Dict[str, Any]:
        """Get a summary of the project structure."""
        structure = {
            "root": self.root_path,
            "languages": [],
            "file_count": 0,
            "dirs": []
        }
        
        try:
            # Count files by extension
            extensions = {}
            for root, dirs, files in os.walk(self.root_path):
                # Skip common ignored directories
                dirs[:] = [d for d in dirs if d not in ['__pycache__', 'node_modules', '.git', 'venv', '.venv', 'dist', 'build', 'target']]
                
                for f in files:
                    ext = Path(f).suffix
                    if ext:
                        extensions[ext] = extensions.get(ext, 0) + 1
                        structure["file_count"] += 1
            
            structure["extensions"] = extensions
            
            # Map extensions to languages
            lang_map = {
                ".py": "Python",
                ".js": "JavaScript",
                ".ts": "TypeScript",
                ".tsx": "TypeScript",
                ".jsx": "JavaScript",
                ".rs": "Rust",
                ".go": "Go",
                ".java": "Java",
                ".c": "C",
                ".cpp": "C++",
                ".cs": "C#",
                ".rb": "Ruby",
                ".php": "PHP",
                ".swift": "Swift",
                ".kt": "Kotlin",
                ".sql": "SQL",
                ".sh": "Shell",
                ".yaml": "YAML",
                ".yml": "YAML",
                ".json": "JSON",
                ".xml": "XML",
                ".md": "Markdown"
            }
            
            for ext, count in extensions.items():
                lang = lang_map.get(ext)
                if lang and lang not in structure["languages"]:
                    structure["languages"].append(lang)
            
        except Exception as e:
            print(f"[CocoIndex] Structure error: {e}")
        
        return structure


class ContextAwareSearch:
    """Enhanced context-aware search using CocoIndex."""
    
    def __init__(self, cocoindex: CocoIndexClient):
        self.cocoindex = cocoindex
        self.cache = {}
    
    def search_task_context(self, task_description: str, project_files: List[str]) -> Dict[str, Any]:
        """
        Search for relevant code context for a task.
        
        Args:
            task_description: Description of the task
            project_files: List of files in the project
            
        Returns:
            Dict with relevant code snippets and file references
        """
        context = {
            "snippets": [],
            "files": [],
            "summary": ""
        }
        
        # Search for relevant code
        results = self.cocoindex.search(task_description, limit=15)
        
        for result in results:
            context["snippets"].append({
                "content": result.get("content", ""),
                "file": result.get("file", "unknown"),
                "score": result.get("score", 0),
                "language": result.get("language", "unknown")
            })
            
            if result.get("file") not in context["files"]:
                context["files"].append(result.get("file"))
        
        # Build summary
        if context["snippets"]:
            context["summary"] = f"Found {len(context['snippets'])} relevant code snippets in {len(context['files'])} files"
        
        return context
    
    def find_related_files(self, target_file: str) -> List[Dict[str, Any]]:
        """Find files related to a target file."""
        # Search for imports/dependencies
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract potential search terms
            search_terms = []
            
            # Python imports
            if target_file.endswith('.py'):
                import re
                imports = re.findall(r'^(?:from|import)\s+([\w.]+)', content, re.MULTILINE)
                search_terms.extend(imports)
            
            # Search for related code
            related = []
            for term in search_terms[:5]:
                results = self.cocoindex.search(term, limit=3, refresh=False)
                related.extend(results)
            
            return related[:10]
            
        except Exception as e:
            return []
    
    def generate_code_context(self, task: str) -> str:
        """Generate a context prompt for code generation."""
        results = self.cocoindex.search(task, limit=10)
        
        if not results:
            return ""
        
        context_parts = ["// Relevant code context:"]
        
        for r in results:
            file = r.get("file", "unknown")
            content = r.get("content", "")
            
            # Truncate long snippets
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_parts.append(f"\n// From {file}:\n{content}")
        
        return "\n\n".join(context_parts)


def get_cocoindex(root_path: Optional[str] = None) -> CocoIndexClient:
    """Get or create CocoIndex client instance."""
    return CocoIndexClient(root_path)
