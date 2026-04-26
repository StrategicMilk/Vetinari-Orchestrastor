# Vetinari UI smoke suite

Playwright tests for components that need browser-rendered verification.
The Python `pytest` suite covers the API contract these components consume;
this suite covers the actual DOM that ships to users.

## Why Playwright

`AttentionTrack` and `ProjectReceiptStrip` are covered by source-text
tripwires in the Python tests, but those cannot prove the components
actually render their two states (empty / populated). SHARD-03 task 3.4
explicitly required browser-rendered tests for the empty-state copy
("No work recorded yet" / "No attention required"), and
PLAN 2026-04-25-roadmap-03-followups Item 2 closes that gap here.

## One-time setup

```bash
cd ui/svelte
npm install
npm run test:e2e:install   # downloads the Chromium browser (~150MB)
```

## Running the suite

```bash
cd ui/svelte
npm run test:e2e
```

The Playwright config (`playwright.config.js`) starts `vite dev --port 5174`
as the test web server. The dev server uses `ui/svelte/index.html` directly,
which is the only self-contained way to serve the SPA without the Litestar
backend running — the production build emits chunks to `../static/svelte/`
(no standalone `index.html`; Litestar normally serves the HTML shell).

Port 5174 (dev default + 1) avoids colliding with a developer's running
`npm run dev` on 5173.

## How the tests stay backend-free

Every API call is mocked via `page.route()` — no Litestar server is
required. The tests install a permissive fallback mock for every `/api/`
endpoint the Dashboard / Projects view touches, then layer specific
canned responses for `/api/attention` and `/api/projects/*/receipts` to
drive the empty / populated states.

## Out of scope

- CI integration (separate infra task — config is local-runnable).
- Visual-regression / screenshot diffing — these tests assert DOM text and
  attributes only.
- Mounting components in isolation — tests navigate to the real Dashboard
  / Projects view to keep the production wiring honest.
