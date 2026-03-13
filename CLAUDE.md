# Vetinari Development Guide

This is the authoritative development guide for the Vetinari codebase. All contributors and AI agents MUST follow these conventions. Rules use consistent severity: **NEVER** (forbidden, enforced by automation), **MUST** (required for Definition of Done), **Prefer** (best practice), **MAY** (allowed in context).

---

## 1. Quick Reference

```bash
# Run tests
python -m pytest tests/ -x -q

# Run tests with coverage
python -m pytest tests/ -x -q --cov=vetinari --cov-report=term-missing

# Verify package imports correctly
python -c "import vetinari; print('OK')"

# Run a specific test file
python -m pytest tests/test_contracts.py -x -q

# Run regression tests
python -m pytest tests/regression/ -x -q

# Lint
python -m ruff check vetinari/ tests/
python -m ruff format --check vetinari/ tests/

# Custom project rules
python scripts/check_vetinari_rules.py

# Start the web UI
python -m vetinari

# Check for type errors
python -m mypy vetinari/ --ignore-missing-imports
```

---

## 2. Build and Test Commands

| Task | Command |
|---|---|
| Run full test suite | `python -m pytest tests/ -x -q` |
| Run with verbose output | `python -m pytest tests/ -v` |
| Run single test file | `python -m pytest tests/test_<module>.py -x -q` |
| Run regression suite | `python -m pytest tests/regression/ -x -q` |
| Coverage report | `python -m pytest tests/ --cov=vetinari --cov-report=term-missing` |
| Verify import | `python -c "import vetinari; print('OK')"` |
| Verify types file | `python -c "from vetinari.types import AgentType; print(AgentType.PLANNER)"` |
| Start server | `python -m vetinari` |
| Lint (ruff) | `python -m ruff check vetinari/ --fix && python -m ruff format vetinari/` |
| Custom rules | `python scripts/check_vetinari_rules.py` |

**Test discovery**: pytest discovers tests in `tests/` matching `test_*.py`. Test functions MUST be prefixed `test_`.

**Test isolation**: Each test MUST be independent. No shared mutable state between tests. Use `pytest.fixture` for setup.

---

## 3. Project Conventions

### 3.1 Python Version

**Python 3.10+** is required. Use Python 3.10+ language features:
- `X | Y` union syntax (not `Union[X, Y]`)
- `list[str]` (not `List[str]`)
- `dict[str, Any]` (not `Dict[str, Any]`)
- `match/case` statements where appropriate
- `from __future__ import annotations` at the top of all new files

### 3.2 Module File Organization

Within each Python file, organize in this order:
1. Module docstring
2. `from __future__ import annotations`
3. Standard library imports
4. Third-party imports
5. Local imports
6. Module-level constants (`UPPER_SNAKE_CASE`)
7. Module-level logger: `logger = logging.getLogger(__name__)`
8. Exception classes (if any)
9. Helper functions (private, prefixed `_`)
10. Public classes and functions
11. `if __name__ == "__main__":` block (if any)

### 3.3 Naming Conventions

- **Variables and functions**: `snake_case` (e.g., `process_task`, `agent_result`)
- **Classes**: `PascalCase` (e.g., `PlannerAgent`, `TaskStatus`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private members**: prefix with single underscore `_` (e.g., `_internal_state`)
- **Module-level constants**: define at the top of the file, after imports
- **Boolean variables**: use `is_`, `has_`, `can_`, `should_` prefixes (e.g., `is_complete`, `has_errors`)
- **Avoid single-letter names** except in comprehensions and short lambdas (`i`, `x`, `k`, `v` are acceptable)

### 3.4 Style

- **PEP 8** compliance is mandatory.
- Line length: soft limit 88 characters, hard limit 120 characters.
- Use 4-space indentation (never tabs).
- Two blank lines between top-level definitions.
- One blank line between methods within a class.
- No trailing whitespace.

### 3.5 String Formatting

- Use **f-strings** for general string formatting: `f"Task {task_id} completed"`
- Use **%-style** ONLY inside logging calls: `logger.info("Task %s completed", task_id)`
- NEVER use `.format()` — f-strings are always preferred
- NEVER use `+` for string concatenation in loops — use `"".join()` or `io.StringIO`

### 3.6 Boolean Expressions

- Use `is None` / `is not None`, NEVER `== None` / `!= None`
- Use truthiness for collections: `if items:` not `if len(items) > 0:`
- Use truthiness for strings: `if name:` not `if name != "":`
- Use explicit comparison for booleans in ambiguous contexts: `if flag is True:` when `flag` could be truthy non-boolean

### 3.7 Type Hints

All new function signatures MUST be fully annotated:
```python
# CORRECT
def process_task(task: Task, config: dict[str, Any]) -> AgentResult:
    ...

# WRONG — missing type hints
def process_task(task, config):
    ...
```

### 3.8 Data Structures

- Use `@dataclass` or Pydantic `BaseModel` for structured data — NEVER raw `dict` for domain objects
- Use `dict` only for genuinely dynamic/unstructured data (e.g., JSON payloads, config)
- Use `Enum` for fixed sets of values — NEVER string literals for status/type fields
- Use `TypedDict` when you need dict compatibility with type safety

### 3.9 Constants and Configuration

- Define constants in the module that owns them, or in `vetinari/constants.py` for shared ones
- NEVER use magic numbers: `if retries > 3:` → `if retries > MAX_RETRIES:`
- Configuration values come from YAML config files or function parameters — NEVER hardcoded in source
- Default values for optional parameters MUST be documented in docstrings

### 3.10 Class Design

- Prefer composition over inheritance (use dependency injection)
- Single Responsibility: each class does one thing well
- Keep `__init__` simple — validate inputs, store attributes, don't do heavy work
- Use `@property` for computed attributes, not getter/setter methods
- Avoid mutable default arguments: `def f(items=None):` then `items = items or []`

### 3.11 Docstrings

Use **Google-style docstrings** for all public functions, methods, and classes:

```python
def verify_token(token: str, secret_key: str) -> dict[str, Any]:
    """Verify a JWT token and return its decoded payload.

    Args:
        token: The JWT token string to verify.
        secret_key: The secret key used to sign the token.

    Returns:
        Decoded token payload as a dictionary.

    Raises:
        ValueError: If the token is invalid or expired.

    Example:
        >>> payload = verify_token("eyJ...", "my-secret")
        >>> print(payload["sub"])
    """
```

Private functions (prefixed `_`) MAY use shorter docstrings but MUST still document non-obvious behaviour.

### 3.12 Error Handling

- NEVER use bare `except:` — always catch specific exception types.
- Always chain exceptions: `raise ValueError("message") from exc`
- NEVER swallow exceptions silently. If you catch and discard, add a log call.
- Use custom exception classes from `vetinari/exceptions.py` where appropriate.

### 3.13 Imports

**Import order** (enforced by ruff isort):
1. Standard library imports
2. Third-party imports
3. Local imports (`vetinari.*`)

**Canonical import sources** — always use these, NEVER redefine locally:
```python
# Enums — ALWAYS from vetinari.types
from vetinari.types import AgentType, TaskStatus, ExecutionMode, PlanStatus

# Agent specs and dataclasses — from vetinari.agents.contracts
from vetinari.agents.contracts import AgentSpec, Task, Plan, AgentResult

# Agent interface — from vetinari.agents.interfaces
from vetinari.agents.interfaces import AgentInterface
```

NEVER use wildcard imports (`from module import *`).

For full import patterns and anti-patterns, see `.claude/docs/import-patterns.md`.

### 3.14 Project Organization

#### File Naming
- **Python source files**: `snake_case.py` (e.g., `planning_engine.py`, `base_agent.py`)
- **Test files**: `test_<source_module>.py` — MUST mirror source module name
- **Config files**: `snake_case.yaml` or `snake_case.json`
- **Documentation**: `UPPER_CASE.md` for root-level docs (`README.md`, `AGENTS.md`), `kebab-case.md` for subdirectories
- **Scripts**: `snake_case.py` in `scripts/`

#### Directory Structure Rules
- New Python modules go in `vetinari/` — NEVER create top-level Python packages
- New agent implementations go in `vetinari/agents/consolidated/` (multi-mode) or `vetinari/agents/` (single-purpose)
- Tests MUST mirror source structure: `vetinari/analytics/cost.py` → `tests/test_cost.py`
- Config files go in `config/` — NEVER embed config in source files
- Every Python directory MUST have an `__init__.py`
- NEVER create nested directories more than 3 levels deep under `vetinari/`
- Generated/temporary files go in `outputs/` (gitignored)

### 3.15 Documentation Standards

- Use ATX-style headers (`#`, `##`, `###`) — NEVER setext-style
- Use fenced code blocks with language identifiers: ` ```python `, ` ```bash `, ` ```yaml `
- Tables MUST have header row and separator row
- Lists use `-` for unordered, `1.` for ordered — NEVER `*` or `+`
- Code references use backticks: `function_name()`, `ClassName`, `file_path.py`

### 3.16 Configuration Standards

- **Project config**: `pyproject.toml` — single source of truth for build, lint, test config
- **Runtime config**: YAML files in `config/` directory
- **YAML keys**: `snake_case` with 2-space indentation; `true`/`false` for booleans
- **TOML**: follow PEP 621 for project metadata
- **JSON**: 2-space indentation, `camelCase` keys (JavaScript convention)

---

## 4. Quality Gates (MANDATORY)

### 4.1 Completeness Rules

All delivered code MUST be fully finished. The following are **forbidden** in production code (`vetinari/`):

- `TODO`, `FIXME`, `HACK`, `XXX`, `TEMP` comments (unless referencing a tracked issue: `TODO(#123)`)
- `pass` as a function/method body (except in abstract methods or `__init__` of exception classes)
- `...` (Ellipsis) as a function/method body (except in type stubs `.pyi` files)
- `raise NotImplementedError` (except in abstract base class methods decorated with `@abstractmethod`)
- Placeholder strings: "placeholder", "example", "sample", "test data", "lorem ipsum", "foo", "bar", "baz"
- Empty function/method bodies with only a docstring and no implementation
- `print()` statements (use `logging` module instead; `print` MAY be used in `__main__.py`, `cli.py`, `scripts/`, `tests/`)
- Commented-out code blocks (dead code MUST be deleted, not commented)
- Magic numbers without named constants
- Hardcoded file paths, URLs, or credentials

Enforced by `scripts/check_vetinari_rules.py` (VET030-036) and ruff T20.

### 4.2 Testing Rules

- Every new public function/method in `vetinari/` MUST have at least one corresponding test in `tests/`
- Test file naming: `tests/test_<module_name>.py` — mirrors the source module
- Test function naming: `test_<function_name>_<scenario>` (e.g., `test_execute_plan_with_empty_tasks`)
- Tests MUST be independent — no shared mutable state, no test ordering dependencies
- Use `pytest.fixture` for setup/teardown, NEVER `setUp()`/`tearDown()` class methods
- Mock external dependencies (LM Studio, network, filesystem) — NEVER make real API calls in tests
- All tests MUST pass before any work is considered complete: `python -m pytest tests/ -x -q`
- NEVER delete or skip existing tests to make new code pass. Fix the code, not the tests
- When fixing a bug, write a regression test FIRST that reproduces the bug, then fix it
- Group related tests in classes: `class TestMyFunction:`
- Order methods: happy path first, then edge cases, then error cases

### 4.3 Logging Rules

**NEVER use `print()` in production code.** Use the `logging` module. See `.claude/docs/logging-guide.md` for full standards.

- Every module: `import logging` then `logger = logging.getLogger(__name__)`
- Use `logger.info()`, `logger.warning()`, `logger.error()` — NOT `print()`
- Use `logging.getLogger(__name__)` — NOT `logging.info()` (root logger)
- Use %-style formatting — NOT f-strings: `logger.info("Processing %s", item)` not `logger.info(f"Processing {item}")`
- In `except` blocks: use `logger.exception()` for automatic traceback inclusion
- `print()` MAY be used in: `__main__.py`, `cli.py`, `scripts/`, `tests/`

Enforced by: ruff T20 (on save) + VET035/VET050/VET051 (on commit/session stop).

### 4.4 Robustness Rules

- **File I/O**: Always specify `encoding="utf-8"` when calling `open()`. Windows defaults to cp1252, causing silent data corruption.
- **Path handling**: Use `pathlib.Path` for all new code, not `os.path.join()`. Paths MUST be cross-platform.
- **No debug code**: NEVER commit `breakpoint()`, `import pdb`, or `pdb.set_trace()`.
- **No blocking sleeps**: Avoid `time.sleep()` with values > 5 seconds in production code.
- **Thread safety**: Flask and APScheduler run in threaded mode. NEVER modify shared mutable state without locks.
- **Resource cleanup**: Use context managers (`with` statements) for files, sockets, and database connections.
- **Idempotency**: Operations that might be retried (agent tasks, API calls) should be idempotent where possible.
- **No silent failures**: Functions MUST NOT return `None` or empty results when they encounter errors — raise exceptions or log warnings.
- **Performance awareness**: Avoid O(n^2) algorithms on collections that could grow. No string concatenation in loops.
- **Error messages**: Make error messages actionable — include what happened, what was expected, and what to do about it.
- **No deprecated APIs**: Check library documentation for deprecation notices. Use current APIs.
- **Environment dependencies**: NEVER hardcode values that should come from config.
- **Scope discipline**: Only modify files directly required by the current task.

Enforced by: VET060-063 (on commit/session stop).

### 4.5 Dependency Management

- Every `import <third_party_package>` in `vetinari/` MUST have a corresponding entry in `pyproject.toml`
  - Core deps go in `[project.dependencies]`
  - Optional deps go in the appropriate group under `[project.optional-dependencies]`
- NEVER pin exact versions unless there's a known incompatibility. Use `>=` minimum version.
- When adding a new optional dependency group, also add it to the `all` extra.
- NEVER remove a dependency that existing code imports. Search for usages first.
- After adding/removing deps, run: `pip install -e ".[dev]"` to verify installation works.

Enforced by: VET070 (on commit/session stop).

### 4.6 Documentation Quality

All documentation — docstrings, comments, markdown files, and inline annotations — MUST be robust, accurate, and content-rich. See `.claude/docs/writing-guide.md` for full standards.

**Docstrings (MANDATORY for all public APIs)**
- Every public function, method, and class MUST have a Google-style docstring
- Docstrings MUST be meaningful (minimum 10 characters) — NEVER just restate the name
- Functions with 2+ parameters MUST include an `Args:` section documenting each parameter
- Functions that return values MUST include a `Returns:` section describing the return type and meaning
- Functions that raise exceptions MUST include a `Raises:` section listing each exception and when it occurs
- Every module MUST have a module-level docstring explaining its purpose and responsibilities

**Comments**
- Comments MUST explain **why**, not **what** — the code shows what, comments explain intent and reasoning
- NEVER write comments that just restate the code: `x = x + 1  # increment x` is useless
- Complex algorithms MUST have a block comment explaining the approach before the code
- All constants MUST have an inline comment explaining their value: `MAX_RETRIES = 3  # LM Studio typical failure recovery window`
- Temporary workarounds MUST reference an issue: `# Workaround for #123 — remove when upstream fixes X`
- Section separators (`# ── Section Name ──`) are encouraged for files > 100 lines

**Markdown Documentation**
- Every markdown file MUST start with a top-level `# Title` heading
- Every section MUST contain meaningful content — NEVER leave empty sections (heading with no content)
- Documentation MUST be structured: use tables for comparisons, lists for sequences, code blocks for examples
- Content MUST be specific and actionable — NEVER use vague language like "do the right thing" or "handle appropriately"
- All code examples in documentation MUST be syntactically correct and follow project conventions
- Cross-references MUST use relative paths: `See .claude/docs/architecture.md` not absolute paths

**Changelog and Commit Messages**
- CHANGELOG.md entries MUST follow Keep a Changelog format (Added, Changed, Fixed, Removed, Security)
- Each entry MUST be a single clear sentence describing the user-visible impact
- Commit messages MUST follow Conventional Commits: `type(scope): description` (see `.claude/docs/workflow.md`)

Enforced by: VET090-096 (docstring quality) + VET100-102 (markdown quality) on commit/session stop.

---

## 5. Safety and Process

### 5.1 Safe Modification Rules

These files are high-impact shared modules. Changes require extra care:

| File | Risk | Rule |
|------|------|------|
| `vetinari/types.py` | All agents depend on these enums | ONLY add new enum values. NEVER rename or remove existing values. |
| `vetinari/agents/contracts.py` | Agent registry, shared dataclasses | Adding fields: use defaults. NEVER remove fields. |
| `vetinari/agents/interfaces.py` | ABC used by all agents | Adding methods: MUST have default impl or be optional. |
| `vetinari/exceptions.py` | Caught throughout codebase | NEVER rename exception classes. Only add new ones. |
| `vetinari/adr.py` | ADR system used by Oracle and web UI | NEVER remove fields from `ADR` dataclass. NEVER change `ADRStatus`/`ADRCategory` enum values. Only add new ones. |
| `conftest.py` | All tests depend on fixtures | Test all fixture changes with full test suite. |
| `pyproject.toml` | Build system + tool config | NEVER remove dependencies. Adding deps: also add to appropriate optional group. |

When modifying any file above, run the FULL test suite, not just related tests.

### 5.2 Definition of Done

A task is NOT complete until ALL of the following are true:

**Code Quality**
1. Code is fully implemented — no stubs, placeholders, or partial implementations
2. All new public functions have type annotations AND Google-style docstrings
3. Imports follow canonical sources (enums from `vetinari.types`)
4. Error handling uses specific exceptions with `from` chaining
5. No `print()` in production code (use `logging`)
6. No hardcoded credentials, magic numbers, or debug code
7. All file I/O uses `encoding="utf-8"`

**Testing**
8. Every new function/method has at least one test
9. `python -m pytest tests/ -x -q` passes with ZERO failures
10. Changes to shared modules tested with FULL test suite

**Integration**
11. `python -c "import vetinari; print('OK')"` succeeds
12. `python -m ruff check vetinari/` reports ZERO errors
13. `python scripts/check_vetinari_rules.py` reports ZERO errors
14. Every new function is called from at least one place (no unwired code)
15. No hallucinated imports — every imported package exists in pyproject.toml

**Documentation**
16. If agent behavior changed → AGENTS.md is updated
17. No hardcoded mock/fake data returned from production functions
18. If a significant architectural, security, or design decision was made → an ADR exists (see 5.3)

If ANY item fails, the task is NOT done.

### 5.3 Architecture Decision Records (ADRs)

Significant decisions MUST be documented as ADRs using the project's ADR system (`vetinari/adr.py`, stored as JSON in `adr/`). ADRs capture **why** a choice was made so future contributors don't re-litigate settled decisions.

**When to create an ADR:**

| Trigger | Category | Example |
|---------|----------|---------|
| New module or subsystem added | `architecture` | Adding a caching layer, new agent type |
| Data model or schema change | `data_flow` | Changing how plans are persisted |
| Security-relevant choice | `security` | Auth mechanism, token storage strategy |
| Public API contract change | `api_design` | New REST endpoint, changing response format |
| Agent behavior or pipeline change | `agent_design` | Changing agent routing, adding a new mode |
| Technology or library adoption | `architecture` | Choosing a new dependency over alternatives |
| Performance trade-off | `performance` | Choosing O(n) scan over index for simplicity |
| Integration pattern choice | `integration` | How Vetinari connects to LM Studio |

**When NOT to create an ADR:**
- Bug fixes, refactors, or style changes that don't alter behavior
- Adding tests or documentation
- Minor config tweaks

**ADR lifecycle:**
1. **Proposed** — created when the decision is being evaluated (Oracle agent's `architecture` mode)
2. **Accepted** — approved and in effect; referenced by implementation code
3. **Deprecated** / **Superseded** — replaced by a newer ADR (link via `related_adrs`)

**Who creates ADRs:**
- The **Oracle agent** is the primary ADR author (via `architecture` and `risk_assessment` modes)
- Human developers MAY create ADRs directly via `ADRSystem.create_adr()`
- The **Planner agent** MUST check existing ADRs before proposing work that contradicts accepted decisions

**ADR quality requirements:**
- `context` MUST explain the problem and constraints that led to the decision
- `decision` MUST state the chosen option explicitly
- `consequences` MUST list both positive and negative trade-offs
- High-stakes categories (`architecture`, `security`, `data_flow`) require at least 3 evaluated alternatives documented in the ADR context before acceptance

**Referencing ADRs in code:**
```python
# Decision: use polling over webhooks for LM Studio health checks (ADR-0012)
```

**Querying ADRs:**
```python
from vetinari.adr import adr_system

# List all accepted architecture decisions
accepted = adr_system.list_adrs(status="accepted", category="architecture")

# Check if a category is high-stakes (requires deeper review)
adr_system.is_high_stakes("security")  # True
```

### 5.4 Common AI Pitfalls

**Import and Dependency Errors**
1. **Redefining enums**: Creating `class AgentType(Enum)` instead of importing from `vetinari.types`
2. **Circular imports**: Use late imports or move shared types to `types.py`
3. **Using legacy typing**: Use `list`, `dict`, `X | None` — NOT `typing.List`, `typing.Dict`, `typing.Optional`
4. **Hallucinating packages**: Importing a package not in pyproject.toml. Verify before adding ANY import.
5. **Missing pyproject.toml sync**: Every `import <third_party>` needs a matching entry in dependencies.

**Code Quality Errors**
6. **Forgetting `from __future__ import annotations`**: MUST be the first import in every `vetinari/` file.
7. **Using print instead of logging**: Use `logger = logging.getLogger(__name__)`. NEVER `print()` in production.
8. **Using f-strings in logger calls**: `logger.info(f"x={x}")` is wrong — use `logger.info("x=%s", x)`.
9. **Swallowing exceptions**: Empty `except` blocks or `except Exception: pass`. Always log or re-raise.
10. **Leaving debug code**: `print()`, `breakpoint()`, `import pdb` MUST NEVER be committed.

**Architecture Errors**
11. **Breaking the single-writer rule**: Only BuilderAgent writes production files.
12. **Ignoring existing utilities**: Search `vetinari/utils.py`, `vetinari/exceptions.py` before writing new helpers.
13. **Code duplication**: ALWAYS search first: `grep -r "def function_name" vetinari/` before writing a new function.
14. **Scope creep**: Only modify files directly required by the current task.

**Completeness Errors**
15. **Unwired features**: Every new function MUST be called from at least one place.
16. **Fake implementations**: Returning hardcoded mock data instead of actual logic.
17. **API drift**: Changing signatures or removing public exports without updating all callers.

**Testing Errors**
18. **Writing dependent tests**: Each test MUST work in isolation. No shared mutable state.
19. **Not running full test suite**: After changing types.py, contracts.py, interfaces.py — run ALL tests.
20. **Using deprecated APIs**: Check library docs for deprecation notices. Use current APIs.

---

## 6. Reference

Detailed reference material is in `.claude/docs/`. Read these when you need specific guidance.

| File | Content | When to Read |
|------|---------|-------------|
| `.claude/docs/architecture.md` | System architecture, components, key files, agent roles | Working on system design or unfamiliar with codebase |
| `.claude/docs/code-patterns.md` | Preferred patterns for I/O, exceptions, config, agents, tests | Implementing new features or unsure of the right pattern |
| `.claude/docs/common-errors.md` | Error recovery guide for import, test, build, and git failures | Encountering errors and need to debug |
| `.claude/docs/import-patterns.md` | Import conventions, anti-patterns, adding new agent types | Writing imports or adding new agents |
| `.claude/docs/logging-guide.md` | Logging setup, levels, formatting, print-to-logging migration | Adding logging or migrating from print() |
| `.claude/docs/workflow.md` | Branch naming, commit conventions, PR requirements | Committing code or creating PRs |
| `.claude/docs/writing-guide.md` | Writing standards for docstrings, comments, markdown, changelogs | Writing documentation or reviewing doc quality |

**Architecture summary**: Vetinari uses a six-agent pipeline (`Planner → Researcher → Oracle → Builder → Quality → Operations`). See `.claude/docs/architecture.md` for full details and `AGENTS.md` for the complete agent specification. Agent prompt specs are in `.claude/agents/`.

---

## 7. Automated Enforcement

Quality is enforced at multiple levels — no violations can slip through:

| Layer | What Runs | When | Config |
|-------|-----------|------|--------|
| **PostToolUse hook** | `ruff check --fix` + `ruff format` on saved .py file | Every Python file save | `.claude/settings.json` |
| **PreToolUse hook** | `pytest` + `ruff check` + `check_vetinari_rules.py --errors-only` | Before `git commit` / `git push` | `.claude/settings.json` |
| **Stop hook** | Full `ruff check` + `ruff format --check` + `check_vetinari_rules.py` + `pytest` | Session end | `.claude/settings.json` |
| **Pre-commit hook** | `ruff check` + `ruff format` + `check_vetinari_rules.py --errors-only` | `git commit` | `.pre-commit-config.yaml` |

### Custom Rules (`scripts/check_vetinari_rules.py`)

31 rules across 9 categories:

| Category | Rules | What They Catch |
|----------|-------|-----------------|
| Import Canonicalization | VET001-006 | Wrong import sources, duplicate enums, wildcard imports |
| Future Annotations | VET010 | Missing `from __future__ import annotations` |
| Error Handling | VET020-022 | Bare except, empty except blocks |
| Completeness | VET030-036 | TODOs, stubs, print(), commented-out code |
| Security | VET040-041 | Hardcoded credentials, localhost URLs |
| Logging | VET050-051 | Root logger usage, f-strings in logger calls |
| Robustness | VET060-063 | Missing encoding, debug code, long sleeps, os.path |
| Integration | VET070-072 | Hallucinated imports, unwired code, fake implementations |
| Organization | VET080-082 | Missing __init__.py, non-snake_case filenames |
| Documentation | VET090-096 | Missing/short docstrings, missing Args/Returns/Raises sections |
| Markdown Quality | VET100-102 | Missing headings, empty sections, sparse content |

Inline suppression: `# noqa: VETxxx` on any line to skip a specific rule.

---

*This file describes the Vetinari project to AI coding agents and human developers alike. Keep it accurate and up to date when the architecture changes.*
