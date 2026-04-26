---
name: Performance Review
description: Detect O(n^2) algorithms, N+1 queries, static complexity analysis, and memory leak patterns
mode: code_review
agent: inspector
version: "1.0.0"
capabilities:
  - performance_review
  - complexity_analysis
tags:
  - quality
  - performance
  - complexity
  - optimization
---

# Performance Review

## Purpose

Performance Review performs static analysis of code to identify performance anti-patterns before they cause problems in production. It detects quadratic algorithms hidden in nested loops, N+1 query patterns in database access, unnecessary memory allocations, string concatenation in loops, and other patterns that degrade performance as data grows. Unlike profiling (which requires runtime measurement), this skill catches performance issues at review time, before the code is deployed.

## When to Use

- As part of comprehensive code review when performance is a concern
- When code processes collections that could grow unboundedly
- When code interacts with databases, files, or network resources in loops
- When the monitoring skill detects performance degradation trends
- Before adding code to hot paths (request handling, event processing)
- When reviewing algorithms that process user-variable-sized inputs

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| code            | string          | Yes      | Code to analyze for performance issues                             |
| mode            | string          | No       | "code_review" with performance focus                               |
| context         | dict            | No       | Expected data sizes, access patterns, performance requirements     |
| focus_areas     | list[string]    | No       | Specific areas: "algorithmic", "io", "memory", "database"          |
| thinking_mode   | string          | No       | Thinking budget tier                                               |

## Process Steps

1. **Algorithmic complexity analysis** -- Identify the time complexity of each function:
   - Nested loops over the same collection: O(n^2) or worse
   - Recursion without memoization on overlapping subproblems: potentially exponential
   - Linear search where a set/dict lookup would give O(1)
   - Sorting where a partial sort or heap would suffice
   - Repeated list operations (insert at front, contains check) that should use deque or set

2. **N+1 query detection** -- Identify database access patterns in loops:
   - Querying inside a loop when a single batch query would work
   - Loading related objects one at a time instead of joining or prefetching
   - Multiple round-trips when a single transaction would suffice
   - Missing pagination on potentially large result sets

3. **String operation analysis** -- Detect string concatenation anti-patterns:
   - `result += string` in a loop (O(n^2) total copying)
   - Repeated f-string or format() in a loop (should accumulate parts, join at end)
   - Building strings when a list with join() is more efficient
   - Regular expression compilation inside loops (should compile once outside)

4. **Memory analysis** -- Identify memory issues:
   - Unbounded collections (lists that grow without limit)
   - Loading entire files into memory when streaming would work
   - Deep copying large objects unnecessarily
   - Circular references preventing garbage collection
   - Global mutable state that accumulates entries

5. **I/O efficiency analysis** -- Check I/O patterns:
   - File opens without context managers (resource leak risk)
   - Unbuffered reads/writes for large files
   - Missing connection pooling for database/HTTP connections
   - Synchronous I/O where async would prevent blocking
   - Missing timeouts on network operations

6. **Caching analysis** -- Evaluate caching opportunities:
   - Pure functions called repeatedly with the same arguments
   - Expensive computations whose results are discarded
   - Configuration loading performed on every request
   - LSP lookups that could be cached within a session

7. **Data structure appropriateness** -- Check if the right data structures are used:
   - List for frequent membership testing (should be set)
   - Dict where a specialized structure (Counter, defaultdict) is cleaner
   - Mutable where immutable would be safer (tuple vs list for read-only data)
   - Nested dicts where a dataclass or namedtuple would be clearer

8. **Quantified impact estimation** -- For each finding, estimate the impact:
   - Current input size and expected growth
   - Time complexity difference between current and suggested approach
   - Memory difference between current and suggested approach
   - At what data size the issue becomes noticeable

## Output Format

The skill produces a performance review report:

```json
{
  "passed": true,
  "grade": "B",
  "score": 0.78,
  "issues": [
    {
      "severity": "high",
      "category": "performance",
      "description": "O(n^2) algorithm: nested loop checking task dependencies. For 50 tasks, this performs 2500 comparisons.",
      "file": "vetinari/orchestration/plan_generator.py",
      "line": 78,
      "suggestion": "Build a dependency set first (O(n)), then check membership in O(1). Total: O(n) instead of O(n^2)."
    },
    {
      "severity": "medium",
      "category": "performance",
      "description": "String concatenation in loop: builds error message with += for each validation failure",
      "file": "vetinari/validation/goal_verifier.py",
      "line": 45,
      "suggestion": "Accumulate parts in a list, join at the end: '\\n'.join(error_parts)"
    },
    {
      "severity": "low",
      "category": "performance",
      "description": "Re-compiling regex pattern on every function call",
      "file": "vetinari/prompts/assembler.py",
      "line": 112,
      "suggestion": "Compile at module level: PATTERN = re.compile(r'...')"
    }
  ],
  "metrics": {
    "functions_analyzed": 15,
    "hotspots_identified": 3,
    "estimated_improvement": "40% reduction in plan generation time for 50-task plans"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-INS-001**: Code review MUST check all 5 dimensions (performance is one of the five)
- **STD-INS-005**: Every issue MUST have a severity level and actionable description
- **STD-INS-006**: Inspector MUST NOT modify code -- only report findings and suggestions
- **STD-INS-007**: Gate decision MUST be based on objective criteria
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-INS-001**: Inspector is READ-ONLY -- cannot modify production files
- **GDL-INS-003**: Group issues by severity for efficient triage

## Examples

### Example: Reviewing a data processing function

**Input:**
```
code: |
  def find_affected_tasks(tasks, changed_file):
      affected = []
      for task in tasks:
          for dep in task.dependencies:
              for other_task in tasks:
                  if other_task.id == dep and changed_file in other_task.files:
                      affected.append(task)
      return affected
context: {expected_task_count: "up to 50", expected_files_per_task: "up to 20"}
```

**Output (abbreviated):**
```
grade: D
issues:
  - [critical/performance] "Triple nested loop: O(n^2 * m) where n=tasks, m=dependencies. For 50 tasks with 5 deps each, this is 12,500 iterations."
    fix: "Build a lookup dict: task_by_id = {t.id: t for t in tasks}. Then look up deps in O(1)."

  - [high/correctness] "Same task can be appended to affected multiple times (once per matching dependency). Use a set."
    fix: "Use affected = set() and affected.add(task.id)"

  - [medium/performance] "'changed_file in other_task.files' is O(f) list scan. If files is large, use a set."
    fix: "Convert task.files to frozenset for O(1) membership test"

estimated_after_fix: "O(n * d) where d=avg deps per task. For 50 tasks with 5 deps: 250 lookups instead of 12,500."
```
