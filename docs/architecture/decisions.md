# Architecture Decisions

## Naming

The repository and user-facing product name is **AM Workbench**.

The orchestration engine, Python package, CLI command, and most runtime logs
remain **Vetinari**. This preserves compatibility and the existing engineering
language while giving the broader project a cleaner public name.

## Runtime Pipeline

Vetinari uses a three-agent factory pipeline:

```text
Foreman -> Worker -> Inspector
```

Foreman owns planning and clarification, Worker owns execution, and Inspector
owns quality review. Legacy implementation class names may still exist, but the
public runtime identity is the three-agent pipeline.

## UI Direction

`ui/svelte` is the canonical UI source.

`ui/static/svelte` is generated output from the Svelte build. The repository
does not keep a separate legacy template UI.

## Release Boundary

Python release artifacts are package-first. They include the `vetinari` package,
runtime config data, public metadata, license files, and public docs needed for
installation and review. They exclude UI workspaces, tests, local state,
generated dependency trees, and model files.

## Model Artifacts

Model weights are operator assets. They belong in local model directories or
remote model stores, not in the repository or Git history.

## Public Documentation

Public docs should describe how to install, run, test, operate, and understand
the system.
