# Vetinari Phase 1 Architecture & Design

## Overview

This document outlines the Phase 1 architecture for Vetinari, covering:
- Planning engine with wave-based execution
- Flexible Model Relay for local/cloud routing
- Two-layer Sandbox for safe execution
- Pluggable CocoIndex code search adapter
- Memory tagging and governance scaffolding

---

## 1. Planning Engine

### 1.1 Core Concepts

**Plan**: A high-level container representing a user's goal
**Wave**: A milestone within a plan containing related tasks
**Task**: Atomic unit of work assigned to an agent

### 1.2 Data Model

```python
@dataclass
class Plan:
    plan_id: str              # Unique identifier (e.g., "plan_001")
    title: str                # Human-readable title
    prompt: str               # Original user prompt
    created_by: str           # User or agent who created
    created_at: str           # ISO timestamp
    updated_at: str           # Last modified
    status: PlanStatus        # pending, active, paused, completed, failed
    waves: List[Wave]        # Ordered list of waves
    
    # Computed fields
    total_tasks: int          # Total task count
    completed_tasks: int      # Completed count
    progress_percent: float   # 0.0 - 100.0
    
@dataclass 
class Wave:
    wave_id: str              # Unique within plan
    milestone: str            # Human-readable milestone name
    description: str          # What this wave achieves
    order: int                # Wave sequence (1, 2, 3...)
    status: WaveStatus        # pending, running, completed, failed
    tasks: List[Task]         # Tasks in this wave
    dependencies: List[str]   # IDs of waves this depends on
    
@dataclass
class Task:
    task_id: str              # Unique within plan
    agent_type: AgentType     # Explorer, Librarian, Oracle, etc.
    description: str          # Human-readable description
    prompt: str               # Full prompt for agent
    status: TaskStatus        # pending, assigned, running, completed, failed
    dependencies: List[str]   # Task IDs this depends on
    assigned_agent: str        # Agent instance handling this
    result: Any               # Task output/result
    error: str               # Error message if failed
    planned_start: str        # ISO timestamp
    planned_end: str          # ISO timestamp
    actual_start: str         # ISO timestamp
    actual_end: str          # ISO timestamp
    retry_count: int         # Number of retries
    
    # Priority and weighting
    priority: int            # 1-10, higher = more important
    estimated_effort: float  # Story points
    
# Enums
class PlanStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
class WaveStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    
class AgentType(Enum):
    EXPLORER = "explorer"
    LIBRARIAN = "librarian"
    ORACLE = "oracle"
    UI_PLANNER = "ui_planner"
    BUILDER = "builder"
    RESEARCHER = "researcher"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"
```

### 1.3 Wave Execution Rules

1. **Sequential Waves**: By default, waves execute in order
2. **Parallel Within Wave**: Tasks within a wave can run in parallel
3. **Dependency Blocking**: Wave won't start until dependencies complete
4. **Auto-Continue**: Next wave starts when all tasks in current wave complete
5. **Manual Gate**: Optional manual approval before proceeding to next wave

### 1.4 Plan State Machine

```
PENDING → ACTIVE → (COMPLETED | FAILED | PAUSED)
PAUSED → ACTIVE (resume)
ACTIVE → FAILED (irrecoverable error)
ANY → CANCELLED (user cancellation)
```

---

## 2. Model Relay

### 2.1 Architecture

The Model Relay is a pluggable adapter that routes tasks to appropriate models based on:
- **Policy**: User-defined preferences (local-first, privacy, cost)
- **Availability**: Whether models are currently loaded
- **Capabilities**: Model strengths for the task type

### 2.2 Data Model

```python
@dataclass
class ModelCatalog:
    """Registry of available models"""
    models: List[ModelEntry]
    default_policy: RoutingPolicy
    
@dataclass
class ModelEntry:
    model_id: str             # Unique identifier
    provider: str             # "local", "lmstudio", "openai", "anthropic", etc.
    display_name: str         # Human-readable name
    capabilities: List[str]   # ["coding", "reasoning", "vision", "long-context"]
    context_window: int       # Max tokens
    latency_hint: str        # "fast", "medium", "slow"
    privacy_level: str       # "local", "private", "public"
    memory_requirements_gb: float  # VRAM/RAM needed
    status: str              # "available", "loading", "unavailable"
    endpoint: str            # URL for API models
    
@dataclass
class RoutingPolicy:
    local_first: bool = True           # Prefer local models
    privacy_weight: float = 1.0       # 0.0-1.0 importance of privacy
    latency_weight: float = 0.5       # 0.0-1.0 importance of speed
    cost_weight: float = 0.3          # 0.0-1.0 importance of cost
    max_cost_per_1k_tokens: float = 0.0  # Budget cap
    preferred_providers: List[str] = None  # Priority order
    
@dataclass
class ModelSelection:
    model_id: str
    provider: str
    endpoint: str
    reasoning: str        # Why this model was selected
    confidence: float      # 0.0-1.0
```

### 2.3 Routing Algorithm

```
1. Filter models by:
   - Required capabilities (task type)
   - Availability status
   - Memory requirements (if local)
   
2. Score available models:
   - Privacy score: local=1.0, private=0.7, public=0.3
   - Latency score: fast=1.0, medium=0.6, slow=0.3
   - Cost score: inverse of cost per 1k tokens
   
3. Apply policy weights:
   - final_score = (privacy * policy.privacy_weight) + 
                  (latency * policy.latency_weight) +
                  (cost * policy.cost_weight)
                  
4. Select highest scoring model
   - Fallback to cloud if no local models score well
```

### 2.4 Configuration Format (YAML)

```yaml
# vetinari/models.yaml
models:
  - model_id: "qwen2.5-coder-7b"
    provider: "lmstudio"
    display_name: "Qwen 2.5 Coder 7B"
    capabilities: ["coding", "fast"]
    context_window: 32768
    latency_hint: "fast"
    privacy_level: "local"
    memory_requirements_gb: 8
    status: "available"
    
  - model_id: "qwen2.5-72b"
    provider: "lmstudio"
    display_name: "Qwen 2.5 72B"
    capabilities: ["reasoning", "coding"]
    context_window: 32768
    latency_hint: "medium"
    privacy_level: "local"
    memory_requirements_gb: 48
    status: "available"
    
  - model_id: "gpt-4o"
    provider: "openai"
    display_name: "GPT-4o"
    capabilities: ["reasoning", "vision", "coding"]
    context_window: 128000
    latency_hint: "medium"
    privacy_level: "public"
    cost_per_1k_tokens: 0.005

policy:
  local_first: true
  privacy_weight: 1.0
  latency_weight: 0.5
  cost_weight: 0.3
```

### 2.5 API Surface

```python
class ModelRelay:
    def pick_model_for_task(self, task: Task, context: dict = None) -> ModelSelection:
        """Select best model for a given task"""
        
    def get_available_models(self) -> List[ModelEntry]:
        """List all available models"""
        
    def reload_catalog(self):
        """Hot-reload model catalog from config"""
        
    def get_model_status(self, model_id: str) -> str:
        """Check if model is available/loading"""
        
    def set_policy(self, policy: RoutingPolicy):
        """Update routing policy"""
```

---

## 3. Sandbox (Two-Layer Architecture)

### 3.1 Layer 1: In-Process Safe Sandbox

For agent code execution within the same process:
- **Purpose**: Quick execution of agent-generated code snippets
- **Isolation**: Python `exec()` with restricted globals
- **Timeouts**: Configurable (default 30s)
- **Resource Limits**: Memory cap via tracing

```python
class InProcessSandbox:
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        
    def execute(self, code: str, globals: dict = None) -> SandboxResult:
        """Execute code in restricted environment"""
        # Restricted globals (no file I/O, no network, no os)
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'abs': abs,
                'round': round,
                'pow': pow,
                # BLOCKED: open, eval, exec, __import__, etc.
            }
        }
        # Execution with timeout and memory monitoring
```

### 3.2 Layer 2: External Plugin Sandbox

For loading external plugins/extensions:
- **Purpose**: Run third-party plugins safely
- **Isolation**: Separate process or container
- **Permissions**: Explicit allow-listing
- **Audit**: All operations logged

```python
class ExternalPluginSandbox:
    ALLOWED_HOOKS = [
        'read_file',
        'write_file', 
        'search_code',
        'execute_command',  # Whitelisted commands only
        'http_request',    # Only to allowed domains
    ]
    
    def __init__(self, plugin_dir: str):
        self.plugin_dir = plugin_dir
        self.allowed_permissions = set()
        self.audit_log = []
        
    def load_plugin(self, plugin_name: str) -> PluginInstance:
        """Load and initialize a plugin"""
        
    def execute_hook(self, hook_name: str, params: dict) -> Any:
        """Execute a permitted plugin hook"""
        # Check permissions
        # Log operation
        # Execute with timeout
        
    def get_audit_log(self) -> List[AuditEntry]:
        """Return all sandbox operations"""
```

### 3.3 Sandbox Policy (Phase 1)

```yaml
# vetinari/sandbox_policy.yaml
sandbox:
  in_process:
    timeout_seconds: 30
    max_memory_mb: 512
    allowed_builtins:
      - print
      - len
      - range
      # ... (restricted set)
    blocked_builtins:
      - open
      - eval
      - exec
      - __import__
      
  external:
    plugin_dir: "./plugins"
    isolation: "process"  # or "container"
    timeout_seconds: 300
    max_memory_mb: 2048
    
    allowed_hooks:
      - read_file
      - write_file
      - search_code
      
    blocked_hooks:
      - shell_exec
      - network_raw
      - file_delete
      
    allowed_domains:
      - "api.github.com"
      - "api.openai.com"
      
    audit_enabled: true
    audit_log_dir: "./logs/sandbox"
```

---

## 4. CocoIndex Adapter (Pluggable)

### 4.1 Adapter Interface

```python
class CodeSearchAdapter(ABC):
    """Base class for code search backends"""
    
    @abstractmethod
    def search(self, query: str, limit: int = 10, filters: dict = None) -> List[CodeSearchResult]:
        """Search codebase for query"""
        pass
    
    @abstractmethod
    def index_project(self, project_path: str) -> bool:
        """Index a project for search"""
        pass
    
    @abstractmethod
    def get_status(self) -> SearchBackendStatus:
        """Get backend status"""
        pass

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
```

### 4.2 CocoIndex Adapter

```python
class CocoIndexAdapter(CodeSearchAdapter):
    """CocoIndex as default code search backend"""
    
    def __init__(self, root_path: str = None):
        self.root_path = root_path or os.getcwd()
        self.client = None
        
    def search(self, query: str, limit: int = 10, filters: dict = None) -> List[CodeSearchResult]:
        """Search using CocoIndex"""
        # Call: uvx cocoindex-code search --query "..." --limit N
        # Parse results into CodeSearchResult
        
    def index_project(self, project_path: str) -> bool:
        """Index project with CocoIndex"""
        # Call: uvx cocoindex-code index --path "..."
        
    def get_status(self) -> SearchBackendStatus:
        """Check if CocoIndex is available"""
        # Check if uvx is available
        # Check if cocoindex-code is installed
```

### 4.3 Pluggable Registry

```python
class CodeSearchRegistry:
    """Registry for code search backends"""
    
    DEFAULT_BACKEND = "cocoindex"
    
    def __init__(self):
        self.backends: Dict[str, Type[CodeSearchAdapter]] = {}
        self._register_defaults()
        
    def _register_defaults(self):
        self.register("cocoindex", CocoIndexAdapter)
        
    def register(self, name: str, adapter_class: Type[CodeSearchAdapter]):
        self.backends[name] = adapter_class
        
    def get_adapter(self, name: str = None) -> CodeSearchAdapter:
        name = name or self.DEFAULT_BACKEND
        if name not in self.backends:
            raise ValueError(f"Unknown backend: {name}")
        return self.backends[name]()
```

---

## 5. Memory Tagging

### 5.1 Extended Memory Types

```python
class MemoryType(Enum):
    # Existing types
    INTENT = "intent"
    DISCOVERY = "discovery"
    DECISION = "decision"
    PROBLEM = "problem"
    SOLUTION = "solution"
    PATTERN = "pattern"
    WARNING = "warning"
    SUCCESS = "success"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    FEATURE = "feature"
    
    # New Phase 1 types
    PLAN = "plan"
    WAVE = "wave"
    TASK = "task"
    PLAN_RESULT = "plan_result"
    WAVE_RESULT = "wave_result"
    MODEL_SELECTION = "model_selection"
    SANDBOX_EVENT = "sandbox_event"
    GOVERNANCE = "governance"
```

### 5.2 Memory Schema

```python
@dataclass
class MemoryEntry:
    entry_id: str
    agent_name: str
    memory_type: str
    summary: str
    content: str
    timestamp: str
    tags: List[str]
    project_id: str
    session_id: str
    
    # Phase 1 additions
    plan_id: str = ""          # Link to plan if applicable
    wave_id: str = ""          # Link to wave if applicable
    task_id: str = ""         # Link to task if applicable
    provenance: str = ""       # "user", "agent", "system"
    confidence: float = 1.0    # 0.0-1.0
    metadata: dict = field(default_factory=dict)
```

---

## 6. API Endpoints (Phase 1 Design)

### 6.1 Plan Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/plans | Create a new plan |
| GET | /api/plans | List all plans |
| GET | /api/plans/{plan_id} | Get plan details |
| PUT | /api/plans/{plan_id} | Update plan |
| DELETE | /api/plans/{plan_id} | Delete plan |
| POST | /api/plans/{plan_id}/start | Start plan execution |
| POST | /api/plans/{plan_id}/pause | Pause plan |
| POST | /api/plans/{plan_id}/resume | Resume plan |
| POST | /api/plans/{plan_id}/cancel | Cancel plan |
| GET | /api/plans/{plan_id}/status | Get plan status |
| GET | /api/plans/{plan_id}/waves | Get waves |
| GET | /api/plans/{plan_id}/waves/{wave_id}/tasks | Get tasks in wave |

### 6.2 Model Relay

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/models | List available models |
| GET | /api/models/{model_id} | Get model details |
| POST | /api/models/select | Select model for task |
| GET | /api/models/policy | Get routing policy |
| PUT | /api/models/policy | Update routing policy |
| POST | /api/models/reload | Reload model catalog |

### 6.3 Sandbox

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/sandbox/execute | Execute in sandbox |
| GET | /api/sandbox/status | Sandbox status |
| GET | /api/sandbox/audit | Get audit log |

### 6.4 Code Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/search | Search code |
| POST | /api/search/index | Index project |
| GET | /api/search/status | Search backend status |

---

## 7. Phase 1 Governance Skeleton

### 7.1 AGENTS.md Template

```markdown
# Vetinari Agents

## Overview
This document describes the specialized agents in Vetinari and their responsibilities.

## Agent Types

### Explorer
- **Purpose**: Codebase search and file discovery
- **When to use**: Finding code, exploring project structure
- **Skills**: grep, file discovery, pattern matching

### Librarian
- **Purpose**: Documentation research and example finding
- **When to use**: Learning libraries, API usage
- **Skills**: docs lookup, GitHub examples

### Oracle
- **Purpose**: Strategic decisions and architecture
- **When to use**: Design decisions, trade-offs
- **Skills**: architecture analysis, trade-off evaluation

### UI Planner
- **Purpose**: Frontend design and visual polish
- **When to use**: UI/UX improvements, CSS
- **Skills**: CSS, animations, accessibility

### Builder
- **Purpose**: Code implementation
- **When to use**: Creating features, refactoring
- **Skills**: implementation, testing, error handling

### Researcher
- **Purpose**: Comprehensive investigation
- **When to use**: Deep research, fact-finding
- **Skills**: source verification, analysis

### Evaluator
- **Purpose**: Code review and quality
- **When to use**: Reviews, audits, testing
- **Skills**: security audit, quality assessment

### Synthesizer
- **Purpose**: Result consolidation
- **When to use**: Summarizing, reporting
- **Skills**: synthesis, report generation

## Auto-Update
This file is auto-generated. Manual changes may be overwritten.
```

### 7.2 CLAUDE.md Template

```markdown
# Vetinari Development Guide

## Project Structure
- `/vetinari/` - Core application
- `/skills/` - Agent skill definitions
- `/projects/` - User projects
- `/ui/` - Web interface

## Key Patterns

### Planning
- Plans contain Waves
- Waves contain Tasks
- Tasks assigned to agents

### Memory
- All agent interactions stored in SharedMemory
- Memories tagged by type and agent

### Model Selection
- Model Relay picks best available model
- Local models preferred when available
- Cloud fallback configured

## Configuration
- `vetinari.yaml` - Main config
- `models.yaml` - Model catalog
- `sandbox_policy.yaml` - Security policies

## Auto-Update
This file is auto-generated. Manual changes may be overwritten.
```

---

## 8. Phase 1 SKILL.md Structure

### Required Frontmatter

```yaml
---
name: <agent-name>
description: <what agent does>
version: 1.0.0
agent: vetinari
tags:
  - <tag1>
  - <tag2>
capabilities:
  - <capability1>
  - <capability2>
triggers:
  - <keyword1>
  - <keyword2>
thinking_modes:
  low: <description>
  medium: <description>
  high: <description>
  xhigh: <description>
---
```

### Section Template

1. **Purpose** - What the agent does
2. **When to Use** - Activation conditions  
3. **Capabilities** - Detailed capability list
4. **Workflow** - How to execute tasks
5. **Output Format** - Expected response structure
6. **Tools Available** - What tools can be used
7. **Error Handling** - What to do when things fail
8. **Integration** - How to work with other agents
9. **Examples** - Usage examples
10. **Reference** - Link to reference docs

---

## 9. Phase 1 Success Criteria

- [ ] Plan data model defined and documented
- [ ] Wave execution rules specified
- [ ] Plan API surface designed
- [ ] Model Relay architecture defined
- [ ] Pluggable adapter interface created
- [ ] Two-layer sandbox design documented
- [ ] Sandbox policy skeleton in place
- [ ] CocoIndex adapter interface defined
- [ ] Memory tagging extended for plans/decisions
- [ ] Governance skeleton templates created
- [ ] SKILL.md scaffolds for all 8 agents
- [ ] Reference docs per agent started

---

*Document Version: 1.0*
*Phase: 1*
*Last Updated: 2026-03-02*
