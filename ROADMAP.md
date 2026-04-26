# AM Workbench Roadmap

AM Workbench is the public project name. Vetinari remains the orchestration
engine, Python package, CLI command, and internal architecture name.

This roadmap is intentionally public-facing. It does not include private
maintainer audit notes, AI-assistant work logs, or working decision corpora.

## Current Release Line

### 0.6.x

- Keep the shipped Python package bounded to the `vetinari` runtime package,
  public docs, configuration, and license files.
- Keep the Svelte application as the canonical UI source under `ui/svelte`.
- Keep generated Svelte bundles and legacy HTML templates out of Python release
  artifacts until the web server mounts a single supported UI shell.
- Preserve local-first inference with native `vllm` or NIM endpoints preferred
  and GGUF llama.cpp fallback available.
- Maintain the three-agent factory pipeline: Foreman, Worker, Inspector.

## Near-Term Work

### Public Packaging

- Rename the GitHub project and public docs to AM Workbench while preserving
  `vetinari` as the import package and CLI for compatibility.
- Add CI for the public export branch: publication-boundary gate, package build,
  release-artifact inspection, focused unit tests, and optional frontend build.
- Publish a concise release checklist that can be run by contributors without
  access to private maintainer artifacts.

### UI Direction

- Treat `ui/svelte` as the real UI path.
- Decide whether the Litestar server should mount the Svelte build directly or
  whether the Svelte dev/build workflow remains a separate frontend surface.
- Retire or quarantine `ui/templates` once the Svelte shell is mounted and the
  old HTML entry points are no longer needed for historical reference.
- Keep generated bundles under `ui/static/svelte` rebuildable from source and
  excluded from Python package artifacts until their release contract is clear.

### Documentation

- Keep public docs limited to install, operation, architecture, API/reference,
  security, and troubleshooting material.
- Maintain public architecture decisions in
  `docs/architecture/decisions.md` instead of shipping private working ADRs.
- Maintain known limitations in `docs/status/known-limitations.md`.
- Maintain license and provenance notes in `docs/security/license-notes.md`.

### Developer Workflow

- Keep script entry points stable while grouping their purpose in
  `scripts/README.md`.
- Move only scripts that are not referenced by tests, pre-commit, or public docs.
- Remove one-off repair helpers after their tests and references are retired.

## Later Work

- Split the public UI, runtime service, and local inference operator workflows
  into clearer installation profiles.
- Add route-level public documentation for the Litestar API.
- Add reproducible frontend build proof for the Svelte app.
- Add model provenance guidance for user-downloaded model assets without
  committing model files to the repository.
- Decide whether the Python distribution should eventually rename from
  `vetinari` to an AM Workbench package name, with a compatibility shim if that
  migration becomes worth the cost.

## Naming Policy

Use **AM Workbench** for the repository, product, public documentation, and
release presentation.

Use **Vetinari** for the orchestration engine, Python package, CLI command,
runtime logs, and day-to-day shorthand inside the project.

That gives the project a broader public name without forcing every developer
conversation to distinguish between "the project" and "the brain" when the
current code, package, and mental model still revolve around Vetinari.
