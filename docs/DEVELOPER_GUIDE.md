# Vetinari Developer Guide
## A Practical Guide for Building, Extending, and Maintaining Vetinari's Hierarchical Multi-Agent Orchestration Framework

**Version:** 1.0  
**Status:** Active  
**Last Updated:** March 3, 2026

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Layout](#project-layout)
3. [Agent Implementation Guidelines](#agent-implementation-guidelines)
4. [Tool Interface Migration](#tool-interface-migration)
5. [Testing Strategy](#testing-strategy)
6. [CI/CD and Quality Gates](#cicd-and-quality-gates)
7. [Observability and Safety](#observability-and-safety)
8. [Onboarding Checklist](#onboarding-checklist)
9. [Contribution Guidelines](#contribution-guidelines)

---

## Getting Started

### Prerequisites

- **Python:** 3.11 or higher
- **Virtual Environment:** venv, pyenv, or conda
- **Git:** For version control
- **LM Studio:** For local model inference (optional for development)
- **Basic Knowledge:** LLM-assisted task orchestration, agent patterns

### Quick Start

1. **Clone the repository:**
   ```bash
   cd C:\Users\darst\.lmstudio\projects\Vetinari
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests to verify setup:**
   ```bash
   python -m pytest tests/ -v --tb=short
   ```

5. **Run a simple demonstration:**
   ```bash
   python -m vetinari.example_minimal
   ```

---

## Project Layout

```
Vetinari/
├── docs/                       # Documentation
│   ├── SKILL_MIGRATION_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   ├── MIGRATION_INDEX.md
│   ├── DRIFT_PREVENTION.md
│   ├── ARCHITECTURE.md
│   └── ...
├── vetinari/                   # Core source code
│   ├── __init__.py
│   ├── orchestrator.py         # Main orchestration engine
│   ├── planning_engine.py     # Plan generation
│   ├── decomposition_agent.py # Task decomposition
│   ├── ponder.py              # Model scoring
│   ├── multi_agent_orchestrator.py
│   ├── live_model_search.py   # Model discovery
│   ├── agents/                # Agent implementations
│   │   ├── __init__.py
│   │   ├── planner_agent.py
│   │   ├── explorer_agent.py
│   │   ├── builder_agent.py
│   │   └── ...
│   ├── tools/                 # Tool interfaces
│   │   ├── __init__.py
│   │   ├── builder_skill.py
│   │   └── ...
│   ├── interfaces/           # Contract definitions
│   │   ├── tool_interface.py
│   │   └── provider_interface.py
│   └── utils/                # Utilities
├── tests/                     # Test suite
│   ├── test_ponder.py
│   ├── test_builder_skill.py
│   └── ...
├── examples/                  # Example scripts
├── scripts/                   # Build and utility scripts
└── vetinari.yaml             # Configuration
```

---

## Agent Implementation Guidelines

### Minimal Agent Interface

Each agent must implement the following interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class AgentTask:
    task_id: str
    agent_type: str
    description: str
    prompt: str
    status: str = "idle"
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class AgentResult:
    success: bool
    output: Any
    metadata: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class Agent(ABC):
    """Base class for all Vetinari agents."""
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable agent name."""
        pass
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize agent with context."""
        pass
    
    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the given task and return results."""
        pass
    
    @abstractmethod
    def verify(self, output: Any) -> Dict[str, Any]:
        """Verify the output meets quality standards."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
```

### Agent Lifecycle

1. **Initialization:** Agent receives context (available models, permissions, etc.)
2. **Task Receipt:** Agent receives an AgentTask with description and prompt
3. **Execution:** Agent processes the task and produces output
4. **Verification:** Agent verifies output quality
5. **Reporting:** Agent returns structured AgentResult

### Observability Requirements

All agents must emit structured logs:

```python
import logging

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    def _log(self, level: str, message: str, **kwargs):
        """Emit structured log with agent context."""
        log_data = {
            "agent_type": self.agent_type,
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        getattr(logger, level)(f"{message} | {log_data}")
```

### Safety and Policy Requirements

- Always enforce policy checks at task initiation
- Do not perform side effects without explicit verification
- Report all policy violations to Security Auditor
- Never expose secrets or credentials in outputs

---

## Tool Interface Migration

### Why Migrate to Tool Interface?

The Tool interface provides:
- **Standardization:** Consistent contract across all skills
- **Type Safety:** JSON schema validation
- **Testability:** Easy mocking and unit testing
- **Observability:** Built-in logging and metrics
- **Security:** Centralized permission enforcement

### Step-by-Step Migration Process

#### Step 1: Analyze the Existing Skill

```bash
# Location of skills
ls skills/<skill_name>/
```

Review:
- SKILL.md for capabilities and triggers
- references/ for supporting docs
- Existing tests

#### Step 2: Define Tool Metadata

```python
from dataclasses import dataclass
from enum import Enum
from typing import List

class ToolCategory(Enum):
    CODE_EXECUTION = "code_execution"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    UI = "ui"
    GOVERNANCE = "governance"

@dataclass
class ToolMetadata:
    name: str
    description: str
    category: ToolCategory
    version: str
    required_permissions: List[str]
    allowed_modes: List[str]
    capabilities: List[str]
```

#### Step 3: Create Tool Wrapper

```python
from vetinari.tool_interface import Tool, ToolResult, ToolMetadata
from vetinari.execution_context import ExecutionContext

class MySkillTool(Tool):
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="my_skill",
            description="Description of what this skill does",
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            required_permissions=["FILE_READ", "FILE_WRITE"],
            allowed_modes=["PLANNING", "EXECUTION"],
            capabilities=["capability1", "capability2"]
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        # Implementation
        pass
```

#### Step 4: Add Unit Tests

```python
import pytest
from unittest.mock import Mock, MagicMock

class TestMySkillTool:
    def setup_method(self):
        self.tool = MySkillTool()
        self.mock_context = Mock(spec=ExecutionContext)
    
    def test_metadata(self):
        metadata = self.tool.get_metadata()
        assert metadata.name == "my_skill"
        assert metadata.version == "1.0.0"
    
    def test_execution_success(self):
        result = self.tool.execute(
            self.mock_context,
            capability="capability1",
            input_data="test"
        )
        assert result.success is True
```

#### Step 5: Update Registry

```python
# In vetinari/tools/__init__.py
from vetinari.tools.my_skill_tool import MySkillTool

__all__ = [
    # ... existing tools
    "MySkillTool",
]
```

---

## Testing Strategy

### Unit Tests

**Purpose:** Test individual components in isolation

**Location:** `tests/test_<component>.py`

**Example:**
```python
def test_agent_initialization():
    agent = MyAgent()
    agent.initialize({"models": []})
    assert agent.is_initialized

def test_capability_execution():
    tool = MyTool()
    result = tool.execute(context, capability="test")
    assert result.success
```

### Integration Tests

**Purpose:** Test interactions between components

**Location:** `tests/integration/`

**Example:**
```python
def test_planner_to_explorer_handoff():
    planner = PlannerAgent()
    plan = planner.create_plan("Build a web app")
    
    explorer = ExplorerAgent()
    task = AgentTask(
        task_id="t1",
        agent_type="EXPLORER",
        description=plan.tasks[0].description,
        prompt=plan.tasks[0].description
    )
    result = explorer.execute(task)
    assert result.success
```

### Mock-Based Tests

**Purpose:** Test failure scenarios and edge cases

```python
from unittest.mock import patch, Mock

@patch('vetinari.agents.explorer.requests.get')
def test_api_failure(mock_get):
    mock_get.side_effect = TimeoutError()
    
    explorer = ExplorerAgent()
    result = explorer.execute(failing_task)
    
    assert result.success is False
    assert "timeout" in result.errors[0].lower()
```

### Test Coverage Targets

| Component | Minimum Coverage |
|-----------|-----------------|
| Core orchestration | 80% |
| Agents | 75% |
| Tools | 85% |
| Planning engine | 80% |
| Utilities | 70% |

---

## CI/CD and Quality Gates

### Required Checks

All PRs must pass:

1. **Unit Tests:** `pytest tests/ -v --tb=short`
2. **Lint:** `ruff check vetinari/`
3. **Type Check:** `mypy vetinari/`
4. **Doc Alignment:** Check for docs updates on contract changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Migration
- [ ] Documentation

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Migration Impact
- [ ] Contracts updated (version bump required)
- [ ] MIGRATION_INDEX.md updated
- [ ] Phase acceptance obtained (if applicable)

## Docs Updated
- [ ] SKILL_MIGRATION_GUIDE.md
- [ ] DEVELOPER_GUIDE.md
- [ ] Other relevant docs
```

### Phase Gating

Each migration phase requires:
- Formal acceptance criteria met
- Documentation updated
- MIGRATION_INDEX.md status changed
- All CI checks passing

---

## Observability and Safety

### Structured Logging

```python
import logging
import json

def log_agent_execution(agent: str, task: str, result: dict):
    logging.info(json.dumps({
        "event": "agent_execution",
        "agent": agent,
        "task_id": task,
        "success": result.success,
        "duration_ms": result.duration_ms,
        "metadata": result.metadata
    }))
```

### Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("agent_execution")
def execute_task(agent, task):
    # Processing
    pass
```

### Security Gates

1. **Input Validation:** All inputs validated against schema
2. **Permission Checks:** Verify permissions before execution
3. **Secret Detection:** Scan for exposed credentials
4. **Policy Enforcement:** Security Auditor reviews all outputs

---

## Onboarding Checklist

### Day 1: Environment Setup

- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Run test suite
- [ ] Verify local LM Studio connection (optional)

### Week 1: Understanding the Codebase

- [ ] Read ARCHITECTURE.md
- [ ] Read SKILL_MIGRATION_GUIDE.md
- [ ] Review existing agent implementations
- [ ] Run example scripts

### Week 2: First Contribution

- [ ] Pick a good first issue
- [ ] Implement solution
- [ ] Add unit tests
- [ ] Update documentation
- [ ] Submit PR

### First Month: Deep Dive

- [ ] Understand all 15 agents
- [ ] Complete one full migration
- [ ] Participate in code review
- [ ] Contribute to testing strategy

---

## Contribution Guidelines

### Code Standards

- Follow PEP 8 (enforced by ruff)
- Use type hints throughout
- Write docstrings for all public APIs
- Keep functions under 100 lines

### Commit Messages

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `migration`

### Review Process

1. **Self-review** before PR
2. **Automated checks** pass
3. **Code review** by maintainer
4. **Documentation review** by docs owner
5. **Phase acceptance** (if migration phase)

### Getting Help

- **Documentation:** See docs/ folder
- **Issues:** Open GitHub issue
- **Discussion:** Use discussions board
- **Slack:** #vetinari-dev (internal)

---

## Related Documents

- `SKILL_MIGRATION_GUIDE.md` - Migration process and agent prompts
- `MIGRATION_INDEX.md` - Phase tracking
- `DRIFT_PREVENTION.md` - Code/docs alignment
- `ARCHITECTURE.md` - System architecture
