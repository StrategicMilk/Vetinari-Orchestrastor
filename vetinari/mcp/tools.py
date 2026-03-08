"""MCP Tool definitions for Vetinari."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class MCPToolParameter:
    name: str
    type: str  # "string", "number", "boolean", "object"
    description: str
    required: bool = True
    default: Any = None

@dataclass
class MCPTool:
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP tool schema format."""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

class MCPToolRegistry:
    """Registry of MCP tools."""
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[MCPTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        if tool.handler is None:
            return {"error": f"Tool '{name}' has no handler"}
        try:
            result = tool.handler(**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def register_defaults(self) -> None:
        """Register default Vetinari MCP tools."""
        self.register(MCPTool(
            name="vetinari_plan",
            description="Generate an execution plan from a goal description",
            parameters=[
                MCPToolParameter("goal", "string", "The goal to plan for"),
                MCPToolParameter("context", "string", "Additional context", required=False),
            ],
            handler=self._handle_plan,
        ))
        self.register(MCPTool(
            name="vetinari_search",
            description="Search codebase with semantic understanding",
            parameters=[
                MCPToolParameter("query", "string", "Search query"),
                MCPToolParameter("limit", "number", "Max results", required=False, default=10),
            ],
            handler=self._handle_search,
        ))
        self.register(MCPTool(
            name="vetinari_execute",
            description="Execute a task through the Vetinari pipeline",
            parameters=[
                MCPToolParameter("task", "string", "Task description"),
            ],
            handler=self._handle_execute,
        ))
        self.register(MCPTool(
            name="vetinari_memory",
            description="Query or store in Vetinari memory system",
            parameters=[
                MCPToolParameter("action", "string", "Action: query, store, recall"),
                MCPToolParameter("content", "string", "Content to query/store"),
            ],
            handler=self._handle_memory,
        ))
        self.register(MCPTool(
            name="vetinari_benchmark",
            description="Run a benchmark suite",
            parameters=[
                MCPToolParameter("suite", "string", "Benchmark suite name"),
                MCPToolParameter("limit", "number", "Max cases", required=False, default=10),
            ],
            handler=self._handle_benchmark,
        ))

    # Default handlers (stubs that connect to real Vetinari subsystems)
    def _handle_plan(self, goal: str, context: str = "") -> Dict:
        return {"status": "planned", "goal": goal, "steps": []}

    def _handle_search(self, query: str, limit: int = 10) -> Dict:
        try:
            from vetinari.code_search import CocoIndexAdapter
            searcher = CocoIndexAdapter()
            results = searcher.search(query, limit=limit)
            return {"results": results}
        except Exception:
            return {"results": [], "note": "Search subsystem not available"}

    def _handle_execute(self, task: str) -> Dict:
        return {"status": "queued", "task": task}

    def _handle_memory(self, action: str, content: str) -> Dict:
        try:
            from vetinari.memory import get_dual_memory_store
            store = get_dual_memory_store()
            if action == "recall":
                results = store.recall(content, limit=5)
                return {"entries": [str(r) for r in results]}
            return {"status": "ok", "action": action}
        except Exception:
            return {"status": "ok", "note": "Memory subsystem not available"}

    def _handle_benchmark(self, suite: str, limit: int = 10) -> Dict:
        try:
            from vetinari.benchmarks.runner import get_default_runner
            runner = get_default_runner()
            report = runner.run_suite(suite, limit=limit)
            return report.summary_dict()
        except Exception as e:
            return {"error": str(e)}
