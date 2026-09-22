import { defineConfig, devices } from '@playwright/test';

/*
  BROWSER TESTS -- issue #21.

  These do not replace the vitest suite and must not grow into it. The unit
  suite is fast, deterministic and covers the maths; it stays the primary one.
  This layer answers the one question it structurally cannot: does the BUILT
  page work in a real browser? Every frontend defect that has actually reached
  a rendered page here needed a real DOM, a real Leaflet, a real build or a
  real network to exist at all -- the first-paint race, the temporal dead zone,
  `VIRIDIS` never being exported, the watermarked basemap, the Pages base path.
  Each one was invisible to code that only checked its own expectations.

  A PRODUCTION BUILD, NOT THE DEV SERVER. `webServer` below builds and previews
  with VITE_BASE set exactly as .github/workflows/deploy.yml sets it. The dev
  server would miss the entire base-path class of failure, which is the one
  that renders a blank page with nothing on it to explain why -- and a blank
  page is the failure a test suite is least likely to be written against,
  because there is nothing on screen to write an assertion about.

  NO PIXEL COMPARISON, DELIBERATELY. Snapshot diffing is the standard tool here
  and it is the wrong one for this page: the particle layers animate on
  requestAnimationFrame, so a full-frame comparison flakes. A suite that cries
  wolf gets ignored, and an ignored suite is worse than no suite -- the same
  reasoning that made a CI pipeline which had never gone green worse than
  having none, because it reads as evidence. So the assertions are on DOM state
  and on a cheap "is anything painted" pixel sample. Pictures for a human to
  look at are `tools/shoot.mjs`, which asserts nothing.

  ARTEFACTS ARE OFF. Traces and videos run to tens of megabytes per failing
  run. Screenshots are kept only on failure, where they earn their keep.
*/
export default defineConfig({
  testDir: './tests/e2e',

  // vitest owns tests/**/*.test.js; these are .spec.js so the two runners
  // cannot pick up each other's files.
  testMatch: '**/*.spec.js',

  // One browser, one worker. The suite is small and its cost should stay
  // proportional to what it buys.
  workers: 1,
  fullyParallel: false,

  // A retry masks a flaky test, and a flaky test here means a wrong assertion
  // rather than a busy machine: everything it touches is local and offline.
  retries: 0,

  timeout: 30_000,
  expect: { timeout: 10_000 },

  reporter: process.env.CI ? [['github'], ['list']] : [['list']],

  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://localhost:4173/MaritimeSAROptimisation/',
    screenshot: 'only-on-failure',
    video: 'off',
    trace: 'off',
  },

  webServer: {
    /*
      Build and serve exactly what deploy.yml publishes. VITE_DATA_BASE is set
      to an address that is NEVER reachable: every test intercepts it with
      page.route() and answers from tests/e2e/fixtures/. Pointing it at a real
      origin would make an un-intercepted request silently succeed, and the
      suite would then depend on a public server being up while appearing not
      to. An unroutable host turns that mistake into an immediate failure.
    */
    command: 'npm run build && npm run preview -- --port 4173 --strictPort',
    url: 'http://localhost:4173/MaritimeSAROptimisation/',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      VITE_BASE: '/MaritimeSAROptimisation/',
      VITE_DATA_BASE: 'https://e2e.invalid/archive',
    },
  },
});
