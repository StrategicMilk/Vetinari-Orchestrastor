# Vetinari Code Style Guide

Code style rules for agents generating or reviewing Python code. When creating or modifying Python files, follow these rules exactly.

## Python Version

Target Python 3.10+. Always use modern syntax:
- `X | Y` union syntax, not `Union[X, Y]`
- `list[str]`, not `List[str]`
- `dict[str, Any]`, not `Dict[str, Any]`
- `match/case` statements where appropriate
- `from __future__ import annotations` at the top of every file

## Module File Organization

When creating or modifying a Python file, organize in this order:
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

## Naming Conventions

- Variables and functions: `snake_case` (e.g., `process_task`, `agent_result`)
- Classes: `PascalCase` (e.g., `ForemanAgent`, `StatusEnum`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- Private members: single underscore prefix `_` (e.g., `_internal_state`)
- Boolean variables: use `is_`, `has_`, `can_`, `should_` prefixes
- Avoid single-letter names except in comprehensions and short lambdas

## PEP 8 Compliance

- Line length: soft limit 88 characters, hard limit 120 characters
- 4-space indentation, never tabs
- Two blank lines between top-level definitions
- One blank line between methods within a class
- No trailing whitespace

## String Formatting

- f-strings for general string formatting: `f"Task {task_id} completed"`
- %-style ONLY inside logging calls: `logger.info("Task %s completed", task_id)`
- Never use `.format()` — f-strings are always preferred
- Never use `+` for string concatenation in loops — use `"".join()` or `io.StringIO`

## Boolean Expressions

- Use `is None` / `is not None`, never `== None` / `!= None`
- Use truthiness for collections: `if items:` not `if len(items) > 0:`
- Use truthiness for strings: `if name:` not `if name != "":`
- Explicit boolean comparison when `flag` could be truthy non-boolean: `if flag is True:`

## File and Directory Naming

- Python source files: `snake_case.py`
- Test files: `test_<source_module>.py` — must mirror source module name
- Config files: `snake_case.yaml` or `snake_case.json`
- Documentation: `UPPER_CASE.md` for root-level, `kebab-case.md` for subdirectories
- Scripts: `snake_case.py` in `scripts/`

## Directory Structure

- New Python modules go in `vetinari/` — never create top-level packages
- Active agent implementations follow the 3-agent pipeline: Foreman in `vetinari/agents/planner_agent.py`, Worker in `vetinari/agents/consolidated/worker_agent.py`, and Inspector in `vetinari/agents/consolidated/quality_agent.py`. Other agent-layout references are historical unless a current ADR and runtime import prove otherwise.
- Tests mirror source structure: `vetinari/analytics/cost.py` -> `tests/test_cost.py`
- Config files in `config/` — never embed config in source files
- Every Python directory must have an `__init__.py`
- Never nest directories more than 3 levels deep under `vetinari/`

## Configuration Standards

- YAML keys: `snake_case` with 2-space indentation; `true`/`false` for booleans
- TOML: follow PEP 621 for project metadata
- JSON: 2-space indentation, `camelCase` keys

## Audit Prevention Standards

These rules exist because repeated audits found technically green code paths that did not prove the behavior they claimed.

### Success Must Prove Effect

- A command, route, save, migration, model download, training lifecycle action, or rule update may report success only after verifying the durable side effect it claims.
- If the effect is asynchronous, the response must say queued/degraded/unknown instead of success unless a durable job record exists.
- Do not treat zero selected cases, missing handlers, skipped imports, missing optional dependencies, or broad `status == "error"` assertions as proof.
- A success, passed, OK, deployed, saved, registered, archived, or completed result must include a checked postcondition from the real side effect path. Command invocation, handler reachability, return-code-only success, skipped zero-case input, default metric values, or stale artifact presence is not proof.

### Read Paths Must Not Mutate

- Functions, routes, or commands named like `get`, `list`, `report`, `stats`, `search`, `render`, `why`, `how`, `dry_run`, or `preview` must not create databases, append rows, initialize persistent state, or rewrite files.
- If initialization on read is intentional, the name, docs, response body, and tests must say so explicitly.
- GET routes and read-named CLI commands must be idempotent under repeated calls. If a read path intentionally materializes state, the route/command name, response schema, docs, and tests must classify it as materializing, and the second identical read must prove no additional mutation.

### Filesystem Boundaries Are Resolved Boundaries

- Resolve user/model/LLM-supplied paths against an allowlisted root before reading, writing, copying, zipping, moving, or deleting.
- Reject symlinks unless the feature explicitly supports them and proves resolved containment.
- Never recursively delete a caller-supplied directory unless the code proves it created and owns that directory.

### Cache And Retrieval Keys Need Scope

- Cache keys for prompts, tool results, model registries, few-shot examples, classifier outputs, memory retrieval, RAG chunks, and checkpoints must include the relevant project/model/agent/task/version/provenance scope.
- Retrieval keys must be literal identifiers. Do not feed user/model-provided keys into glob patterns or substring scans that can enumerate unrelated entries.

### Supply Chain Inputs Must Be Pinned Or Classified

- Do not use `uvx ...@latest`, raw `pip install <name>`, `from_pretrained(...)` without immutable revision/hash policy, or `joblib.load`/`pickle.load` from operator-writable roots in release/operator paths.
- If a path is intentionally mutable or local-only, classify it as such and keep it out of release-proof claims.
- Model, dataset, and tool downloads are supply-chain inputs. Release/operator paths must use an operator-owned cache root, immutable revision or digest, provenance metadata, license classification, and checksum or equivalent integrity proof. Project/install-tree writes are forbidden unless the artifact is vendored and reviewed.

### Package Metadata Must Have One Authority

- `pyproject.toml` is the release metadata authority unless a current ADR explicitly says otherwise.
- If `setup.py` is retained, it must be a thin compatibility shim or have an automated parity check against `pyproject.toml`; it must not duplicate dependencies, classifiers, scripts, or package-data scope by hand.
- Python version classifiers, console scripts, extras, package-data globs, and checked-in `*.egg-info` files are release claims. Do not advertise a Python version, wrapper, or packaged asset unless the release gate builds and smokes that exact surface.
- Generated package manifests such as `SOURCES.txt` must be regenerated from the current tree or demoted as historical. Missing files, root junk artifacts, model shards, `node_modules`, probe output, or stale generated UI bundles in release metadata are blockers.

### Route Auth And Redaction Are Explicit

- Every mounted route must have an explicit auth classification.
- Mutating routes require admin auth by default.
- Sensitive read routes for adapters, models, traces, logs, replay, config, credentials, filesystem paths, eval scores, deployment state, or internal provider metadata require admin auth unless they return a documented public projection with central redaction and pagination.

### Local Paths Are Sensitive Output

- Route responses, logs, telemetry, audit records, and exported inventories must not expose raw local filesystem paths, usernames, cache roots, model roots, adapter roots, or install-tree locations.
- Return stable IDs, display names, or redacted public projections unless an admin-only contract explicitly requires raw paths.

### Background Work Has A Shutdown Contract

- Every executor, background thread, async worker, queue consumer, SSE generator, timer, or freshness checker must have an owner, stop signal, cancellation path, join/drain timeout, and shutdown test.
- Shutdown must not report clean while work can continue, resurrect, block forever, or keep side effects running after app shutdown.

### Identity Fields Are Immutable

- Fields named `id`, `*_id`, `session_id`, `model_id`, `plan_id`, `task_id`, `subtask_id`, `adr_id`, or `contract_id` are identity fields.
- Update APIs must reject changes to identity fields or perform an atomic rekey that updates every persistence index.
- Load, delete, and overwrite predicates must include the full identity tuple.

### Proof Hooks Must Bind Evidence To Claims

- Evidence hooks must match the full referenced file path or symbol, not only an extension, substring, or unrelated recent tool result.
- Regexes used with `findall()` must use non-capturing groups unless the captured subgroup is the intended value. If a hook expects paths, tests must include a different file with the same extension and prove it is rejected.
- Promotion or prevention scripts must fail closed when every candidate is malformed, skipped, or invalid. "No valid candidates" is not a successful promotion unless the command is explicitly advisory and excluded from release proof.

### Prompt And Agent Config Authority Must Be Runtime-True

- Prompt-loader paths must preserve mandatory global/project standards when selecting per-mode prompts; mode-specific prompts may add constraints but must not silently drop base safety, logging, docstring, and no-placeholder rules.
- Prompt/config frontmatter is a release claim. Model aliases, tool grants, write authority, and delegation depth must either be consumed by runtime routing or demoted as documentation-only with a validator.
- Read-only roles may not list write tools in active prompt/config authority unless an ADR and runtime guard prove the exception.

### Motion Must Honor User Preferences

- Retained UI surfaces with animations, transitions, infinite pulses, or chart animations must honor `prefers-reduced-motion` or a live equivalent setting.
- Tests for removed/dead reduced-motion preferences are not enough; active packaged CSS/JS/Svelte motion paths need direct static or runtime proof.
