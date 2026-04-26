# Vetinari Universal Agent Standards

These standards govern all Vetinari agents regardless of type, mode, or model provider. Every rule here is mandatory. Violations trigger rework.

## Core Principles

### Correctness Above All

Do things correctly, even when it is harder or slower. No shortcuts, no workarounds, no placeholder implementations, no superficial patches, no skipping steps that feel tedious. If a task touches ten files, touch all ten. If a fix requires updating every caller, update every caller. Fix root causes — never delete or weaken a test to make code pass.

### Reasoning Protocol

When producing output for complex decisions:
1. Identify the key question or requirement
2. Consider 2-3 approaches with trade-offs
3. Choose the best approach and explain why
4. Execute with verification

### Confidence Reporting

Rate your confidence in each major output:
- HIGH (>80%): Well-understood domain, clear requirements, verified data
- MEDIUM (50-80%): Some ambiguity, partial information, reasonable inference
- LOW (<50%): Significant uncertainty — flag for human review or escalate

### Verification Before Reporting

Before finalizing any output:
- Does this directly address the task requirements?
- Are there logical contradictions or unsupported claims?
- Would a domain expert find obvious errors?
- Is the output format correct and complete?

### Error Handling Protocol

- If requirements are ambiguous, state your assumptions explicitly
- If you lack information, say so rather than fabricating data
- If a subtask fails, provide partial results with clear error context
- Never silently drop errors — always surface them

### Quality Standards

- Cite sources or reasoning for factual claims
- Prefer specific, actionable output over vague generalities
- If output exceeds expected scope, summarize and offer details on request
- Maintain consistent terminology throughout

## Agent Best Practices

### PLAN BEFORE ACT
Before implementing, outline your approach. Identify what you know, what you need to find out, and what could go wrong.

### EXPLORE BEFORE MODIFY
Before changing code, read existing implementations. Search for utilities and patterns that already exist. Never duplicate functionality.

### VERIFY BEFORE REPORT
Before reporting completion, verify your output: run tests, check for logical contradictions, confirm all requirements are addressed.

### CONTEXT DISCIPLINE
Track your token budget. Summarize large findings. Drop stale context. Focus on what is relevant to the current task.

### EVIDENCE OVER ASSUMPTION
Every claim must be backed by evidence: file path + line number, test result, URL, or explicit reasoning chain. Never guess.

### ESCALATE UNCERTAINTY
If confidence is below MEDIUM, surface your uncertainty explicitly. Request additional research or human review rather than guessing.

### MINIMAL SCOPE
Only modify what is directly required. Do not refactor surrounding code, add unrequested features, or improve things outside your task.

### DELEGATION DEPTH
Maximum 3 levels of delegation. No recursive self-delegation. If a subtask needs a different specialist, route through the Planner.

### CHECKPOINT FREQUENTLY
After completing each significant step, persist your progress. Enable resumption if interrupted.

### FAIL INFORMATIVELY
On failure, report: what was attempted, what went wrong, what was tried to fix it, and what the next agent should know.

## Code Generation Rules

When generating Python code for the Vetinari project, follow these rules:

- `from __future__ import annotations` at the top of every file
- Enums from `vetinari.types`, specs from `vetinari.agents.contracts`, interfaces from `vetinari.agents.interfaces` — never redefine locally
- Modern typing: `list[str]`, `dict[str, Any]`, `X | None` — never `List`, `Dict`, `Optional`
- `logger = logging.getLogger(__name__)` per module; %-style formatting; never `print()`
- `encoding='utf-8'` on all `open()` calls; use `pathlib.Path` not `os.path`
- Never bare `except:` — always catch specific types; chain with `from exc`; never swallow silently
- No `TODO`, `FIXME`, `pass` bodies, `NotImplementedError`, `print()`, commented-out code, or magic numbers
- Google-style docstrings for all public APIs with Args/Returns/Raises sections
- f-strings for general use; %-style ONLY in logger calls; never `.format()`
- Every new public function needs at least one test
- All function signatures fully annotated with type hints

## Type System and Data Structures

- All function signatures MUST be fully annotated
- Use `@dataclass` or Pydantic `BaseModel` for structured data — never raw `dict` for domain objects
- Use `Enum` for fixed sets of values — never string literals for status/type fields
- Define constants as `UPPER_SNAKE_CASE` — never use magic numbers
- Prefer composition over inheritance; use dependency injection
- Keep `__init__` simple — validate inputs, store attributes, no heavy work
- Avoid mutable default arguments: `def f(items=None):` then `items = items or []`

## Import Rules

When writing import statements:
1. Standard library imports first
2. Third-party imports second
3. Local imports (`vetinari.*`) third

Canonical sources — always use these, never redefine:
- Enums: `from vetinari.types import AgentType, TaskStatus, ExecutionMode, PlanStatus`
- Agent specs: `from vetinari.agents.contracts import AgentSpec, Task, Plan, AgentResult`
- Agent interface: `from vetinari.agents.interfaces import AgentInterface`
- Never use wildcard imports (`from module import *`)

## Logging Rules

- Every module: `import logging` then `logger = logging.getLogger(__name__)`
- Use `logger.info()`, `logger.warning()`, `logger.error()` — never `print()`
- Use %-style formatting: `logger.info("Processing %s", item)` not f-strings
- In `except` blocks: use `logger.exception()` for automatic traceback inclusion
- `print()` is only acceptable in: `__main__.py`, `cli.py`, `scripts/`, `tests/`

## Documentation Rules

- Every public function, method, and class MUST have a Google-style docstring
- Docstrings MUST be meaningful (minimum 10 characters) — never just restate the name
- Functions with 2+ parameters MUST include an `Args:` section
- Functions that return values MUST include a `Returns:` section
- Functions that raise exceptions MUST include a `Raises:` section
- Every module MUST have a module-level docstring
- Comments explain WHY, not WHAT — the code shows what

## Completeness and Robustness

### Completeness (all forbidden in production code)
- `TODO`, `FIXME`, `HACK`, `XXX`, `TEMP` comments (unless referencing a tracked issue)
- `pass` as a function body (except in abstract methods)
- `...` (Ellipsis) as a function body (except in type stubs)
- `raise NotImplementedError` (except in `@abstractmethod`)
- Placeholder strings: "placeholder", "example", "sample", "lorem ipsum", "foo", "bar"
- Empty function bodies with only a docstring
- Commented-out code blocks (delete dead code, don't comment it)
- Magic numbers without named constants
- Hardcoded file paths, URLs, or credentials

### Robustness
- File I/O: always `encoding="utf-8"` — Windows defaults to cp1252
- Path handling: `pathlib.Path` for all new code, not `os.path.join()`
- No debug code: never commit `breakpoint()`, `import pdb`, `pdb.set_trace()`
- Thread safety: never modify shared mutable state without locks
- Resource cleanup: use context managers (`with` statements) for files, sockets, DB connections
- Idempotency: operations that might be retried should be idempotent
- No silent failures: never return `None` or empty results on errors — raise or log
- No deprecated APIs: check library docs for deprecation notices

## Safety and High-Impact File Rules

These files require extra care when modifying:

| File | Rule |
|------|------|
| `vetinari/types.py` | Only ADD new enum values. Never rename or remove existing values. |
| `vetinari/agents/contracts.py` | Adding fields: use defaults. Never remove fields. |
| `vetinari/agents/interfaces.py` | Adding methods: must have default impl or be optional. |
| `vetinari/exceptions.py` | Never rename exception classes. Only add new ones. |
| `vetinari/adr.py` | Never remove fields from ADR dataclass. Only add new enum values. |
| `conftest.py` | Test all fixture changes with the full test suite. |
| `pyproject.toml` | Never remove dependencies. Adding deps: also add to optional group. |

When modifying any of these files, run the FULL test suite.

## Agent Conventions and ADRs

Vetinari uses a three-agent factory pipeline: Foreman, Worker, Inspector (ADR-0061).

### Architecture Decision Records
- Significant decisions MUST be documented as ADRs
- `context` MUST explain the problem and constraints
- `decision` MUST state the chosen option explicitly
- `consequences` MUST list both positive and negative trade-offs
- High-stakes categories (architecture, security, data_flow) require at least 3 evaluated alternatives
- The Oracle agent is the primary ADR author
- The Planner MUST check existing ADRs before proposing contradictory work

## Communication Protocol

### Requesting Information from Other Agents
- Set `metadata.delegate_to` with the target agent type and required information
- Provide specific questions, not open-ended exploration requests
- Include what you already know to avoid redundant work

### Escalating to User
- Set `metadata.needs_user_input = true` with a specific question
- Triggers: ambiguous requirements, conflicting constraints, scope exceeds estimate by 2x+

### Andon Trigger Protocol
- Halt the pipeline if: security violation detected, repeated failures (3+ rework cycles), data integrity at risk
- Log the trigger reason and affected tasks
- Resume only after the trigger condition is resolved

### Inter-Agent Delegation Matrix
| From | To | When |
|------|----|------|
| Builder | Researcher | Needs context about existing code or external API |
| Builder | Oracle | Architecture question requiring formal decision |
| Quality | Builder | Rework needed — specific issues to fix |
| Planner | Oracle | Needs ADR check before committing to approach |
| Any | Operations | Cost question, documentation need, monitoring request |
