---
name: Git Archaeology
description: Trace code history using blame, log, and bisect to understand why code exists in its current form
mode: git_workflow
agent: worker
version: "1.0.0"
capabilities:
  - git_archaeology
  - code_discovery
tags:
  - research
  - git
  - history
  - provenance
---

# Git Archaeology

## Purpose

Git Archaeology uses version control history to answer "why does this code exist in this form?" It applies git blame, log, bisect, and diff to trace the evolution of specific code sections, identify the commits and authors behind key decisions, and recover the reasoning that led to the current implementation. This is essential when modifying code that appears unusual, overly complex, or potentially wrong -- the history often reveals constraints or bugs that the current form was designed to address.

## When to Use

- Before modifying code that looks unusual or overly complex -- it may be a deliberate workaround
- When investigating a bug to determine when and why the regression was introduced
- When a code section has a comment like "do not change" or "workaround for X" and you need context
- When understanding the design evolution of a module before proposing architectural changes
- When attributing a decision to a specific commit for ADR documentation
- When looking for a known-good state to revert to after a regression

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What history to investigate and why                                |
| files           | list[string]    | No       | Specific files to investigate                                      |
| symbols         | list[string]    | No       | Functions or classes whose history to trace                        |
| date_range      | dict            | No       | Time window: {since: "2024-01-01", until: "2024-06-01"}           |
| commit_range    | string          | No       | Commit range to investigate (e.g., "abc123..def456")               |
| bisect_good     | string          | No       | Known-good commit for bisect operations                            |
| bisect_bad      | string          | No       | Known-bad commit for bisect operations                             |

## Process Steps

1. **Target identification** -- Determine exactly which code sections need historical investigation. Map file paths and line ranges for the code in question. If the target is a function, use LSP to identify its exact location.

2. **Blame analysis** -- Run git blame on the target files to identify which commit last modified each line. Group consecutive lines by commit to identify logical change units. Note the author, date, and commit message for each change.

3. **Commit message analysis** -- For each significant commit identified by blame, read the full commit message. Look for: issue references (e.g., #123), ADR references, "fix" or "workaround" language, and cross-references to other files changed in the same commit.

4. **Change context recovery** -- For key commits, examine the full diff (not just the target file). Understanding what else changed in the same commit reveals the broader context: was this part of a refactor? A bug fix? A feature addition?

5. **Evolution timeline** -- Build a timeline of significant changes to the target code. For each change, record: what changed, why (from commit message), and what the code looked like before. This reveals the "geological layers" of the code.

6. **Decision point identification** -- Identify commits where the code took a particular design direction. These are the decision points that may need ADR documentation. Note any commits where the approach fundamentally changed.

7. **Regression bisect** (if bisect_good and bisect_bad provided) -- Use git bisect to find the exact commit that introduced a regression. This narrows the search from hundreds of commits to a single commit, making root cause analysis tractable.

8. **Pattern inference** -- From the history, infer why the code is in its current form. Common patterns: workaround for upstream bug (check if bug is still present), performance optimization (check if it's still needed), compatibility shim (check if old version is still supported).

9. **Findings synthesis** -- Compile the historical analysis into actionable findings: what the code does, why it does it that way, whether the original constraints still apply, and whether it's safe to modify.

## Output Format

The skill produces a historical analysis report:

```json
{
  "success": true,
  "output": {
    "target": "vetinari/orchestration/replan_engine.py:_compute_salvageable",
    "timeline": [
      {
        "commit": "abc1234",
        "date": "2024-03-15",
        "author": "dev-team",
        "summary": "Initial implementation of replan engine",
        "significance": "Original design used simple retry without salvage analysis"
      },
      {
        "commit": "def5678",
        "date": "2024-06-22",
        "author": "dev-team",
        "summary": "fix: replan engine was discarding completed work on partial failure",
        "significance": "Added salvage analysis after production incident where 80% of completed work was thrown away"
      }
    ],
    "key_findings": [
      "The _compute_salvageable function was added in response to a production incident (def5678)",
      "The complex dependency walking logic is intentional -- simple approaches missed transitive invalidation",
      "The 3-retry limit was chosen based on observed failure patterns (see commit message in def5678)"
    ],
    "safe_to_modify": false,
    "reason": "The salvage logic addresses a real production failure mode. Modifications should preserve the transitive invalidation check."
  },
  "provenance": [
    {"tool": "git blame", "file": "vetinari/orchestration/replan_engine.py", "lines": "45-89"},
    {"tool": "git log", "range": "--follow replan_engine.py", "commits_reviewed": 12}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files

## Examples

### Example: Understanding a complex workaround before simplifying

**Input:**
```
task: "Investigate why base_agent.py has a double-dispatch pattern for mode execution instead of a simple dict lookup"
files: ["vetinari/agents/base_agent.py"]
symbols: ["BaseAgent.execute"]
```

**Output (abbreviated):**
```
timeline:
  - 2024-01: Simple dict lookup {mode: handler}
  - 2024-03: Refactored to double-dispatch after adding modes that need pre/post hooks
  - 2024-05: Added fallback handling for unknown modes (graceful degradation)

key_findings:
  - "The double-dispatch pattern was introduced because some modes need pre-processing (context assembly) and post-processing (output validation) that varies by mode."
  - "A simple dict lookup was tried first (commit abc123) but led to duplicated pre/post logic in every handler."
  - "The current pattern centralizes cross-cutting concerns while allowing mode-specific behavior."

safe_to_modify: "Yes, but preserve the pre/post hook points. A strategy pattern or decorator chain could simplify without losing functionality."
```
