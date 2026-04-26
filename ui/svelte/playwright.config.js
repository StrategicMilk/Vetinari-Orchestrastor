// Playwright config for the Vetinari UI smoke suite.
//
// The tests run against `vite preview` on a fixed port (4173) so the same
// build that ships in CI is the one being smoke-tested. Backend API calls
// are mocked per-test via `page.route()` so no Litestar server is needed.
//
// First-time setup on a developer machine:
//   npm install
//   npm run test:e2e:install   # downloads the Chromium binary
//
// Running the suite:
//   npm run test:e2e

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  expect: {
    timeout: 5_000,
  },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',

  use: {
    baseURL: 'http://127.0.0.1:5174',
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    // Smoke against `vite dev`, not `vite preview` — the production build
    // emits chunks to ../static/svelte/ (no standalone index.html, Litestar
    // serves the HTML shell). The dev server uses ui/svelte/index.html
    // directly and bundles src/main.js on demand, which is the only
    // self-contained way to serve the SPA without the backend running.
    //
    // Port 5174 (dev default + 1) avoids colliding with a developer's
    // running `npm run dev` on 5173.
    //
    // /api/* calls are mocked via page.route() in each spec, so the dev
    // server's /api proxy (configured for the Litestar backend) is never
    // exercised.
    command: 'npx vite dev --host 127.0.0.1 --port 5174 --strictPort',
    url: 'http://127.0.0.1:5174',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
});
