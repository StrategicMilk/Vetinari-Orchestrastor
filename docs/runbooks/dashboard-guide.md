# AM Workbench UI And Dashboard Guide

The current public UI direction is Svelte-first.

## Supported Surfaces

| Surface | Path | Status |
|---|---|---|
| Svelte UI source | `ui/svelte` | Canonical UI source for new work |
| Generated Svelte assets | `ui/static/svelte` | Rebuildable output from `ui/svelte` |
| Litestar API | `vetinari/web/litestar_app.py` | Runtime API server |
| Legacy HTML templates | `ui/templates` | Historical/dormant reference |

`ui/templates` is not the real UI path for new work. Do not add features there.

## Running The Backend

```bash
python -m vetinari serve --port 5000
```

This starts the Litestar API server. The public route set is API-first; the
current package boundary tests intentionally keep old `/` and `/dashboard`
template routes dormant until the Svelte shell has a clear mounted-server
contract.

## Running The Svelte App

```bash
cd ui/svelte
npm install
npm run dev
```

The Vite dev server runs the Svelte app and proxies API calls to the Litestar
backend on `http://localhost:5000`.

## Building The Svelte App

```bash
cd ui/svelte
npm run build
```

The build writes generated assets to `ui/static/svelte`. Those generated files
are workspace artifacts and are excluded from Python package artifacts.

## Frontend Tests

```bash
cd ui/svelte
npm run test:e2e:install
npm run test:e2e
```

The Playwright tests run against the Svelte dev server with API calls mocked.
Python tests also include source-level UI contract checks under `tests/`.

## Release Boundary

Before publishing:

- keep `ui/svelte/models/` and all model weights out of Git history
- keep `node_modules/` out of the repository
- rebuild generated Svelte assets from source when they are needed
- do not mount legacy templates as public routes without a license, CSP, and
  attribution review
- do not include UI workspaces in Python package artifacts unless the package
  contract is intentionally changed

The eventual clean target is one public UI shell, built from `ui/svelte`, with
the backend serving either the compiled app or a documented separate frontend
deployment.
