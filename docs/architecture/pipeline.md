# Vetinari Pipeline

This page is the current public quick-start reference for the runtime pipeline.
Vetinari is the orchestration engine inside AM Workbench.

## Runtime Shape

Vetinari uses a 3-agent factory pipeline:

```text
Foreman -> Worker -> Inspector
```

The live enum is `vetinari.types.AgentType` with only `FOREMAN`, `WORKER`, and `INSPECTOR`.

## Agents And Modes

| Agent | Responsibility | Modes |
|---|---|---|
| Foreman | Plans, clarifies, consolidates context, owns the task graph | 6 |
| Worker | Executes research, architecture, build, operations, and recovery work | 24 |
| Inspector | Reviews quality, security, tests, and simplification | 4 |

Total active modes: 34.

## Common Paths

```text
Express:
  Worker(task-specific mode) -> Inspector(review mode)

Standard:
  Foreman(plan) -> Worker(research/build as needed) -> Inspector -> Worker(documentation/synthesis as needed)

Custom:
  Foreman(clarify -> plan) -> Worker(research -> architecture -> build as needed) -> Inspector -> Worker(documentation/synthesis/improvement as needed)
```

Legacy names such as `Researcher`, `Oracle`, `Builder`, `Quality`, and `Operations` may appear in historical plans or as internal implementation class names. They are not current public runtime agent identities.

## Prompt And Rules Note

Runtime prompts can come from more than one path: agent markdown, prompt evolution, `PromptAssembler`, rules, learned failure patterns, and recalled examples. Frontmatter fields in `vetinari/config/agents/*.md` are metadata, not runtime model or tool enforcement.
