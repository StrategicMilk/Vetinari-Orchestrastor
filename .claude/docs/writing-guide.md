# Writing Standards

This guide defines quality standards for all written content in the Vetinari project — docstrings, comments, markdown documentation, changelogs, and commit messages.

## Core Principle

Every piece of documentation MUST be **specific**, **accurate**, **actionable**, and **structured**. Vague, generic, or restated content wastes reader time and provides no value.

---

## Docstring Standards

### Required Structure (Google-style)

Every public function, method, and class MUST have a docstring following this template:

```python
def execute_plan(plan: Plan, config: dict[str, Any]) -> AgentResult:
    """Execute a plan by dispatching tasks to agents in wave order.

    Iterates through the plan's task waves, assigns each task to the
    appropriate agent based on the AGENT_REGISTRY, and collects results.
    Failed tasks are retried up to MAX_RETRIES times before the plan
    is marked as FAILED.

    Args:
        plan: The plan to execute, containing ordered task waves.
        config: Runtime configuration including model settings,
            timeout values, and retry policies.

    Returns:
        AgentResult containing the aggregated outputs from all
        completed tasks, with success=True if all tasks passed.

    Raises:
        PlanError: If the plan has no tasks or is in an invalid state.
        AgentError: If an agent fails after exhausting retries.

    Example:
        >>> plan = create_plan("Build auth module")
        >>> result = execute_plan(plan, {"model": "qwen2.5-72b"})
        >>> assert result.success
    """
```

### Quality Requirements

**Summary line** (first line):
- MUST be a single sentence describing what the function does, not what it is
- MUST start with an imperative verb: "Execute", "Return", "Validate", "Parse" — NOT "This function executes..."
- MUST be specific: "Execute a plan by dispatching tasks to agents" NOT "Execute a plan"
- MUST be under 79 characters (fits on one line)

**Extended description** (optional but recommended for complex functions):
- Explain the **algorithm** or **approach**, not just the inputs/outputs
- Mention important side effects (modifies shared state, writes to disk, sends network requests)
- Describe performance characteristics if relevant (O(n) complexity, blocking calls)
- Note thread safety considerations if applicable

**Args section** (required when function has 2+ parameters):
- Every parameter MUST be documented with type and purpose
- Multi-line descriptions indent with 4 spaces
- Include valid ranges, defaults, and constraints: `timeout: Maximum wait time in seconds (1-300, default 30).`
- For dict/object parameters, document expected keys

**Returns section** (required when function returns non-None):
- Describe the return type AND what the value represents
- For complex return types, describe the structure
- Document what None means if the function can return None

**Raises section** (required when function raises exceptions):
- List every exception type the function explicitly raises
- Describe the condition that triggers each exception
- Order from most specific to most general

### Anti-Patterns (NEVER do these)

```python
# WRONG — restates the function name
def get_agent():
    """Get agent."""  # Useless — tells nothing the name didn't already say

# WRONG — too vague
def process_task(task):
    """Process a task."""  # What kind of processing? What happens to the task?

# WRONG — missing Args/Returns
def validate_config(config, strict=False):
    """Validate configuration settings."""  # What's config? What's strict? What's returned?

# WRONG — documents implementation, not purpose
def calculate_score(items):
    """Loop through items and sum their weights."""  # HOW, not WHY
```

### Correct Alternatives

```python
# CORRECT — specific, actionable
def get_agent(agent_type: AgentType) -> BaseAgent:
    """Retrieve an agent instance from the registry by type.

    Args:
        agent_type: The type of agent to retrieve.

    Returns:
        Configured agent instance ready for task execution.

    Raises:
        KeyError: If agent_type is not in AGENT_REGISTRY.
    """

# CORRECT — explains purpose and behavior
def validate_config(config: dict[str, Any], strict: bool = False) -> list[str]:
    """Validate configuration against the schema and return any errors.

    In strict mode, missing optional fields are also flagged as errors.
    In non-strict mode (default), only required fields and type mismatches
    are reported.

    Args:
        config: Configuration dictionary to validate.
        strict: If True, treat missing optional fields as errors.

    Returns:
        List of validation error messages, empty if valid.
    """
```

---

## Comment Standards

### When to Comment

Comments explain **why**, not **what**. The code shows what happens — comments explain intent, constraints, and reasoning.

**MUST comment:**
- Non-obvious business logic: why a specific threshold, formula, or condition exists
- Workarounds and hacks: reference the issue and explain what the ideal solution would be
- Performance-critical sections: explain why a particular approach was chosen over alternatives
- Constants: explain the meaning and source of the value
- Complex regex patterns: explain what the pattern matches in plain language
- Algorithm choices: explain why this algorithm was selected

**NEVER comment:**
- Code that is self-explanatory: `count += 1  # increment count`
- Function calls that already have descriptive names
- Import statements (unless the import is surprising)
- Simple conditionals: `if user.is_admin:  # check if user is admin`

### Comment Quality Examples

```python
# WRONG — restates the code
x = x + 1  # increment x
if len(items) > 0:  # check if items is not empty
results = []  # initialize empty results list

# CORRECT — explains why
x = x + 1  # Compensate for 0-indexed API response (our IDs start at 1)
if len(items) > 0:  # Empty task lists indicate a planning failure upstream
results = []  # Accumulate across waves; order matters for dependency resolution

# CORRECT — constant explanation
MAX_RETRIES = 3  # LM Studio recovers within 3 attempts for transient failures
WAVE_TIMEOUT = 120  # 2 min per wave; longest observed agent execution is ~90s
TOKEN_BUDGET = 4096  # Conservative limit; qwen2.5-72b context is 32k but quality degrades past 4k

# CORRECT — workaround with issue reference
# Workaround for #42 — APScheduler drops jobs when timezone changes during DST.
# Remove this manual reschedule once upstream fixes apscheduler#987.
scheduler.reschedule_job(job_id, trigger="interval", seconds=interval)

# CORRECT — regex explanation
# Match ISO 8601 timestamps with optional timezone: 2024-01-15T10:30:00+05:00
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2})?")
```

### Section Separators

For files over 100 lines, use section separators to organize code:

```python
# ── Public API ───────────────────────────────────────────────────────────────

def create_plan(...):
    ...

def execute_plan(...):
    ...

# ── Internal Helpers ─────────────────────────────────────────────────────────

def _validate_task_graph(...):
    ...
```

---

## Markdown Documentation Standards

### File Structure

Every markdown file MUST follow this structure:

```markdown
# Document Title

Brief introduction (1-2 sentences) explaining the purpose and audience.

---

## Section 1

Content...

## Section 2

Content...

---

*Footer with maintenance notes or cross-references.*
```

### Content Quality Rules

**Be specific and data-rich:**
```markdown
# WRONG — vague
The system has several agents that do different things.

# CORRECT — specific with data
The system uses 6 consolidated agents, each with distinct modes:
Planner (6 modes), Researcher (8 modes), Oracle (4 modes),
Builder (2 modes), Quality (4 modes), Operations (9 modes).
```

**Use tables for structured comparisons:**
```markdown
# WRONG — paragraph listing
The Planner agent handles orchestration. The Researcher agent
handles code discovery. The Oracle handles architecture...

# CORRECT — table
| Agent | Role | Modes |
|-------|------|-------|
| Planner | Orchestration, task decomposition | 6 |
| Researcher | Code discovery, domain research | 8 |
| Oracle | Architecture decisions, risk assessment | 4 |
```

**Use code blocks with language identifiers:**
```markdown
# WRONG — no language identifier
```
from vetinari.types import AgentType
```

# CORRECT — with language identifier
```python
from vetinari.types import AgentType
```
```

**Include actionable examples:**
```markdown
# WRONG — tells without showing
You should use pytest fixtures for test setup.

# CORRECT — shows how
Use `pytest.fixture` for test setup:
```python
@pytest.fixture
def mock_agent():
    return MockAgent(config={"model": "test"})

def test_execute(mock_agent):
    result = mock_agent.execute({"id": "t1"})
    assert result.success
```
```

### Empty Section Prevention

Every section heading MUST be followed by meaningful content. NEVER create placeholder sections:

```markdown
# WRONG — empty sections
## Authentication
## Authorization
## Session Management

# CORRECT — each section has content
## Authentication
Token-based authentication using JWT. Tokens expire after 24 hours.

## Authorization
Role-based access control with three levels: admin, user, viewer.
```

---

## Changelog Standards

Follow Keep a Changelog format (keepachangelog.com):

```markdown
## [Unreleased]

### Added
- ConsolidatedResearcherAgent git_workflow mode for repository analysis

### Changed
- QualityAgent security patterns now detect f-string SQL injection

### Fixed
- PlannerAgent wave ordering when tasks have circular soft dependencies

### Removed
- Deprecated `LegacyAgent` class (use `ConsolidatedOperationsAgent` instead)
```

**Rules:**
- Each entry is a single clear sentence describing user-visible impact
- Start with a noun or noun phrase, not a verb: "ConsolidatedResearcherAgent git_workflow mode" not "Added git_workflow mode"
- Include the component/scope affected
- Reference issue numbers when applicable: `Fixed task timeout handling (#42)`
- Group by category: Added, Changed, Deprecated, Removed, Fixed, Security
- Most recent changes at the top

---

## Enforcement

| Rule | What It Catches | Severity |
|------|----------------|----------|
| VET090 | Public function/class missing docstring | Warning |
| VET091 | Docstring too short (< 10 chars) | Warning |
| VET092 | Missing Args section (2+ params) | Warning |
| VET093 | Missing Returns section (non-None return) | Warning |
| VET094 | Missing Raises section (has raise statements) | Warning |
| VET095 | Module missing module-level docstring | Warning |
| VET096 | Docstring just restates the name | Warning |
| VET100 | Markdown file missing top-level heading | Warning |
| VET101 | Empty section (heading with no content) | Warning |
| VET102 | Markdown file with very little content | Warning |

All documentation rules are warnings (not errors) to avoid blocking existing code. New code MUST satisfy all rules.

---

*This file defines writing standards for the Vetinari project. See CLAUDE.md Section 4.6 for the mandatory rules summary.*
