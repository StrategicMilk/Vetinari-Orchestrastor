# Vetinari Quality Criteria

What "good work" and "bad work" look like for each agent and mode. Quality uses these criteria when reviewing agent output.

## Builder — build mode

### Good work looks like:
- All functions have type hints and Google-style docstrings
- Tests cover happy path, edge cases, and error cases
- Canonical imports from `vetinari.types` and `vetinari.agents.contracts`
- No TODO/FIXME/placeholder code anywhere
- `encoding='utf-8'` on all file I/O operations
- Error handling with specific exceptions chained via `from exc`
- Every new function is called from at least one place (wired in)
- Module-level docstring explaining purpose

### Bad work looks like:
- Missing type hints or docstrings on public functions
- Tests that only check the happy path
- Redefining enums locally instead of importing from `vetinari.types`
- Using `print()` instead of `logging`
- Leaving `NotImplementedError` stubs in non-abstract methods
- Hardcoded file paths, credentials, or magic numbers
- New functions that are never imported or called (unwired code)
- Bare `except:` clauses that swallow errors silently

## Builder — image_generation mode

### Good work looks like:
- Image matches the description specifications exactly
- Format and dimensions are as requested
- SVG fallback documented when Stable Diffusion is unavailable
- Generated image is free of text artifacts

### Bad work looks like:
- Image doesn't match the description
- Wrong format or dimensions
- No fallback plan when generation fails

## Quality — code_review mode

### Good work looks like:
- Every finding includes file path, line number, and severity
- Security patterns (OWASP Top 10) checked systematically
- Code style compliance verified against project standards
- Binary pass/fail gate decision with clear justification
- Actionable remediation suggestions for each finding

### Bad work looks like:
- Findings without file references ("there might be an issue somewhere")
- Rubber-stamp approval without substantive review
- Missing OWASP categories in security checks
- Subjective opinions without evidence

## Quality — security_audit mode

### Good work looks like:
- All OWASP Top 10 categories explicitly addressed
- CWE references for each vulnerability found
- Severity classified (Critical/High/Medium/Low/Info)
- Attack vectors described with exploit scenarios
- Remediation steps that are specific and implementable

### Bad work looks like:
- Critical vulnerabilities (SQLi, RCE, path traversal) missed
- Generic advice without specific code references
- Missing severity classifications
- No CWE references for identified issues

## Researcher — code_discovery mode

### Good work looks like:
- All file paths verified to exist in the codebase
- Function signatures accurately captured with types
- Dependencies and callers identified with line references
- Confidence scores on each finding
- Related patterns and utilities noted

### Bad work looks like:
- Hallucinated file paths or function names
- Function signatures that don't match the actual code
- Missing dependency analysis
- No confidence indication on uncertain findings

## Researcher — domain_research mode

### Good work looks like:
- Sources cited for every factual claim
- Multiple perspectives considered and compared
- Findings structured for downstream agent consumption
- Clear distinction between facts and inferences

### Bad work looks like:
- Unsourced claims presented as certain
- Single-perspective analysis
- Unstructured prose that's hard for other agents to use
- Fabricated references or citations

## Oracle — architecture mode

### Good work looks like:
- At least 3 alternatives evaluated with pros/cons
- Trade-offs quantified where possible (token cost, latency, complexity)
- ADR created with context, decision, and consequences
- Existing ADRs checked for conflicts before deciding
- Decision is actionable by Builder

### Bad work looks like:
- Single approach presented without alternatives
- ADR missing context or consequences section
- Decision contradicts an accepted ADR without formal supersession
- Vague recommendations ("consider using a better approach")

## Planner — plan mode

### Good work looks like:
- Task DAG is acyclic with clear dependencies
- Each task has measurable acceptance criteria
- Agent assignments match task requirements and capabilities
- Scope is bounded — no unbounded exploration tasks
- Resource estimates included for each task

### Bad work looks like:
- Circular dependencies in the task DAG
- Tasks without acceptance criteria
- Tasks assigned to non-existent agent types
- Unbounded scope ("research everything about X")
- Missing dependency declarations between related tasks

## Operations — documentation mode

### Good work looks like:
- Google-style docstrings with Args/Returns/Raises sections
- Module-level docstring explains purpose and responsibilities
- Cross-references use correct, verified file paths
- Code examples are syntactically correct and follow project conventions
- No empty sections (heading with no content)

### Bad work looks like:
- Documentation that contradicts actual code behavior
- Docstrings that just restate the function name
- Broken cross-references to non-existent files
- Code examples with syntax errors
- Empty placeholder sections

## Operations — cost_analysis mode

### Good work looks like:
- Token counts from actual measurements, not estimates
- Cost projections include confidence intervals
- Comparison with at least 2 alternatives
- Recommendations tied to specific budget constraints

### Bad work looks like:
- Estimated token counts without measurement
- Cost projections without uncertainty ranges
- Single-option analysis without alternatives
- Estimates more than 2x off from actuals
