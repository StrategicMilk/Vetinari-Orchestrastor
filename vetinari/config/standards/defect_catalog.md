# Vetinari Defect Catalog

Known failure modes ranked by frequency. Agents should review relevant sections before starting work to avoid common mistakes.

## Builder — Common Defects

### 1. Unwired Features (40% of rejections)
New code that is never imported or called from anywhere. A new function, class, module, CLI command, or config option exists but nothing references it.
- **Prevention**: Before marking complete, search for usages of every new symbol. If not called, wire it in.
- **Example**: Creating `vetinari/utils/helpers.py` with a `format_output()` function that no other module imports.

### 2. Missing Tests (25% of rejections)
New public functions or methods without corresponding test coverage.
- **Prevention**: For every new public function, write at least one test in `tests/test_<module>.py`.
- **Example**: Adding `validate_config()` to `vetinari/config/settings.py` without a `test_validate_config_*` test.

### 3. Import Errors (15% of rejections)
Importing packages not in `pyproject.toml`, redefining enums locally, or using wrong import paths.
- **Prevention**: Check `pyproject.toml` before importing any third-party package. Always import enums from `vetinari.types`.
- **Example**: `from vetinari.types import AgentType` is correct. `class AgentType(Enum): ...` in your module is wrong.

### 4. Incomplete Error Handling (10% of rejections)
Bare `except:` clauses, swallowed exceptions, or missing exception chaining.
- **Prevention**: Always catch specific exceptions. Always chain with `from exc`. Always log or re-raise.
- **Example**: `except Exception: pass` swallows the error. Use `except ValueError as exc: raise ConfigError("...") from exc`.

### 5. Missing Type Annotations (5% of rejections)
Functions without return type annotations or parameter types.
- **Prevention**: Annotate every function signature before implementation.

### 6. Placeholder Code (5% of rejections)
`TODO` comments, `pass` bodies, `NotImplementedError`, or `...` in non-abstract methods.
- **Prevention**: Implement fully or don't create the function. No stubs in production code.

## Quality — Common Defects

### 1. Rubber-Stamp Approval (35% of rejections)
Approving code without substantive review. No specific findings, no file/line references.
- **Prevention**: Every review must cite at least 3 specific observations with file paths and line numbers.

### 2. Missing Security Checks (25% of rejections)
Not checking OWASP Top 10 categories, especially SQLi, XSS, and path traversal.
- **Prevention**: Systematically check each OWASP category. Document which were checked and what was found.

### 3. Vague Findings (20% of rejections)
Findings without file references or actionable remediation steps.
- **Prevention**: Every finding must include: file path, line number, severity, and specific fix suggestion.

### 4. False Positives (20% of rejections)
Flagging correct code as problematic, or misunderstanding project conventions.
- **Prevention**: Read the quality criteria and style guide before reviewing. Verify findings against actual code.

## Researcher — Common Defects

### 1. Hallucinated References (30% of rejections)
File paths, function names, or API methods that don't exist in the codebase.
- **Prevention**: Verify every file path and function name against the actual codebase before including in output.

### 2. Unsourced Claims (25% of rejections)
Factual statements without citations or evidence.
- **Prevention**: Every factual claim must cite a source: file path + line number, URL, or explicit reasoning chain.

### 3. Single-Perspective Analysis (20% of rejections)
Only considering one approach or viewpoint.
- **Prevention**: Always consider at least 2-3 alternatives or perspectives. Document trade-offs.

### 4. Stale Information (15% of rejections)
Referencing code or APIs that have been updated or removed.
- **Prevention**: Always read the current version of files, not cached or remembered versions.

### 5. Unstructured Output (10% of rejections)
Research findings in unstructured prose that downstream agents can't easily parse.
- **Prevention**: Use structured format: bullet points, tables, or JSON for findings.

## Oracle — Common Defects

### 1. Insufficient Alternatives (30% of rejections)
Architecture decisions with fewer than 3 evaluated alternatives.
- **Prevention**: Always propose at least 3 alternatives with documented trade-offs before deciding.

### 2. ADR Quality Issues (25% of rejections)
ADRs missing context, consequences, or alternatives sections.
- **Prevention**: Every ADR must have: context (problem + constraints), decision (explicit choice), consequences (positive + negative).

### 3. Contradicting Existing ADRs (20% of rejections)
Making decisions that conflict with accepted ADRs without formal supersession.
- **Prevention**: Query accepted ADRs before making decisions. If contradicting, create a supersession ADR.

### 4. Vague Recommendations (25% of rejections)
Advice that isn't actionable by Builder (e.g., "consider improving the architecture").
- **Prevention**: Every recommendation must be specific enough for Builder to implement without interpretation.

## Planner — Common Defects

### 1. Missing Dependencies (35% of rejections)
Task DAGs with implicit dependencies not declared, causing execution order issues.
- **Prevention**: For every task, explicitly list what it depends on. Verify the DAG is acyclic.

### 2. Unbounded Scope (25% of rejections)
Tasks without clear boundaries (e.g., "research everything about X").
- **Prevention**: Every task must have specific, measurable acceptance criteria and a bounded scope.

### 3. Wrong Agent Assignment (20% of rejections)
Tasks assigned to agents that lack the capability or jurisdiction.
- **Prevention**: Check agent capabilities before assignment. Builder writes code, Quality reviews, Researcher investigates.

### 4. Missing Acceptance Criteria (20% of rejections)
Tasks without testable conditions for completion.
- **Prevention**: Every task must have at least one acceptance criterion that can be verified programmatically or by inspection.

## Operations — Common Defects

### 1. Documentation Drift (40% of rejections)
Documentation that doesn't match current code behavior.
- **Prevention**: Read the actual code before writing documentation. Verify examples compile and run.

### 2. Empty Sections (25% of rejections)
Markdown sections with headers but no content.
- **Prevention**: Every section must have meaningful content. If nothing to say, remove the section.

### 3. Broken Cross-References (20% of rejections)
Links to files or sections that don't exist.
- **Prevention**: Verify every cross-reference path exists before including it.

### 4. Inaccurate Cost Estimates (15% of rejections)
Cost projections more than 2x off from actuals.
- **Prevention**: Use measured token counts, not estimates. Include confidence intervals.
