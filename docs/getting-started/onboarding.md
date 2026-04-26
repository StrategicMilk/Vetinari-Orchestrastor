# AM Workbench Developer Onboarding

AM Workbench is the public project. Vetinari is the orchestration engine and
the current Python package/CLI name.

## Repository Tour

```text
vetinari/          Python runtime package and CLI implementation
tests/             Pytest suite and focused contract tests
config/            Runtime configuration and standards
docs/              Public documentation
scripts/           Developer, release, and maintenance utilities
ui/svelte/         Canonical Svelte UI source
ui/static/svelte/  Generated Svelte bundles, rebuildable from ui/svelte
ui/legacy/templates/      Legacy HTML templates retained for reference only
```

The Python release artifact is intentionally package-first. UI workspaces,
tests, generated bundles, local state, model files, and maintainer-only assets
are excluded from wheel/sdist artifacts unless a future release explicitly
changes that boundary.

## Install

```bash
git clone <repo-url>
cd am-workbench

python -m venv .venv312
source .venv312/bin/activate          # Windows: .venv312\Scripts\activate

pip install -e ".[dev,local,ml,search,notifications]"
```

`uv` users can substitute:

```bash
uv pip install -e ".[dev,local,ml,search,notifications]"
```

Add heavier optional stacks only when you need them:

```bash
pip install -e ".[training]"
pip install -e ".[vllm]"
```

## Verify

```bash
python -c "import vetinari; print('OK')"
python -m pytest tests/ -x -q
python -m ruff check vetinari/
python scripts/quality/check_vetinari_rules.py
```

On Windows, the repo-root `python.cmd` helper can run the same commands through
the project environment:

```powershell
.\python.cmd scripts/dev/run_tests.py
.\python.cmd -m pytest tests/ -x -q
```

## Run

```bash
python -m vetinari init
python -m vetinari serve --port 5000
python -m vetinari start --goal "Summarise the README"
```

The package and CLI are still named `vetinari` for compatibility.

## Architecture Primer

Vetinari uses a three-agent factory pipeline:

```text
Foreman -> Worker -> Inspector
```

- Foreman plans, clarifies, and prepares the task graph.
- Worker executes research, architecture, build, operations, and recovery work.
- Inspector performs the mandatory quality gate.

See `docs/architecture/pipeline.md` for the current public pipeline summary and
`docs/architecture/decisions.md` for public architecture decisions.

## UI Development

The real UI source lives under `ui/svelte`.

```bash
cd ui/svelte
npm install
npm run dev
npm run build
```

The Svelte build emits generated assets into `ui/static/svelte`. Those files are
workspace artifacts, not Python package contents. `ui/legacy/templates` contains older
HTML shells and should not be used as the active UI path for new work.

## Making A Change

1. Read the nearby code and tests before editing.
2. Wire the call site before adding a new helper or public function.
3. Add or update focused tests for the changed behavior.
4. Run the smallest credible validation set first, then broaden if the change
   touches shared behavior.
5. Keep public docs free of private maintainer paths, local audit corpora, and
   AI-assistant workflow instructions.

## Decision Records

Do not ship private working ADRs or audit corpora. Public architecture decisions
belong in `docs/architecture/decisions.md`.

If a future change needs a detailed decision record, add a concise public
summary there and keep any private investigation notes outside the public
export.

## Useful References

| Topic | Public location |
|---|---|
| Quick start | `docs/getting-started/quick-start.md` |
| Runtime pipeline | `docs/architecture/pipeline.md` |
| Public architecture decisions | `docs/architecture/decisions.md` |
| Known limitations | `docs/status/known-limitations.md` |
| License/provenance notes | `docs/security/license-notes.md` |
| Script categories | `scripts/README.md` |
