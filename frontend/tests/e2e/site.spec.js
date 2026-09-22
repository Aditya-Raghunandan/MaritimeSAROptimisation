import { readFile } from 'node:fs/promises';
import { join, dirname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test, expect } from '@playwright/test';

/*
  Does the built page actually work in a browser? Five checks, and each one
  exists because the failure it catches has already shipped here at least once.

  They are deliberately about WIRING and PAINTING, never about values. What the
  numbers are is the vitest and pytest suites' job, against a golden fixture and
  against the real archive; asking this layer to check arithmetic as well would
  make it slow, duplicated and the first thing anyone disables.
*/

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');

/*
  Serve the committed fixture in place of the published archive.

  The suite must be offline: a browser test that reaches Hugging Face fails
  when a public server is slow, which teaches everyone to re-run it until it
  passes. The config points the build at an unroutable host so that anything
  this function fails to intercept fails immediately and visibly, rather than
  quietly succeeding against the real thing.
*/
async function serveFixture(page) {
  await page.route('**/e2e.invalid/**', async (route) => {
    const url = new URL(route.request().url());
    // Everything after the archive root is a path inside the fixture. normalize
    // collapses any `..` before it is joined, so a malformed request cannot
    // walk out of the fixture directory and read the repository.
    const rel = normalize(url.pathname.replace(/^\/archive\/?/, '')).replace(/^(\.\.[/\\])+/, '');
    try {
      const body = await readFile(join(FIXTURES, rel));
      await route.fulfill({
        status: 200,
        contentType: rel.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        body,
      });
    } catch {
      // A 404 is the honest answer and is what the real store would give. The
      // client's own fallback path then runs, which is worth exercising.
      await route.fulfill({ status: 404, body: '' });
    }
  });
}

/** Basemap tiles are third-party and not under test; stub them so nothing leaves. */
async function stubTiles(page) {
  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    'base64',
  );
  await page.route(/(basemaps\.cartocdn\.com|server\.arcgisonline\.com|tile\.openstreetmap\.org)/,
    (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
}

/*
  Everything the page complained about, so a test can assert on all of it.

  `Failed to load resource` is dropped, and only that. Opening a Zarr v3 store
  begins with a probe for the Zarr v2 names -- `.zarray` and `.zattrs` -- which
  do not exist, so every array logs a 404 before it opens correctly by its v3
  `zarr.json`. That is protocol negotiation working, not a defect, and it
  happens against the real published store too. The message carries no URL, so
  it cannot be filtered more precisely than by text; what makes that safe is
  that failed requests are asserted separately and exactly, below.

  Uncaught exceptions are NOT filtered. They are the signal that matters: the
  first-paint race and the temporal dead zone were both a throw that escaped
  `start()` and left half the page wired to nothing.
*/
const RESOURCE_404 = /Failed to load resource/;

function collectProblems(page) {
  const problems = [];
  page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !RESOURCE_404.test(m.text())) {
      problems.push(`console: ${m.text()}`);
    }
  });
  return problems;
}

/*
  Ready means the data path got all the way through, not that a timer expired.
  The provenance line is written only once a tier has been chosen and its
  manifest read. Waiting on a fixed sleep instead would paper over exactly the
  first-paint ordering bug that this file exists to catch.
*/
async function ready(page) {
  await page.waitForFunction(
    () => document.querySelector('#provenance')?.textContent?.trim().length > 0,
    null, { timeout: 20_000 },
  );
}

/** How many sampled pixels a canvas has actually painted. */
async function paintedOn(page, selector) {
  return page.evaluate((sel) => {
    const c = document.querySelector(sel);
    if (!c) return -1;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4 * 37) if (d[i] > 8) n += 1;
    return n;
  }, selector);
}

test.beforeEach(async ({ page }) => {
  await serveFixture(page);
  await stubTiles(page);
});

/*
  #19: `addTo(map)` calls Leaflet's `onAdd` synchronously, the renderers painted
  from there, and the only `await` sat below them. The throw escaped `start()`
  and left the layer control, slider, play, ruler, rings, click handler and
  provenance line unwired -- a map that rendered with dead controls. Nothing
  crashed visibly and the page looked plausible.
*/
test('the page loads with no uncaught errors', async ({ page }) => {
  const problems = collectProblems(page);
  await page.goto('');
  await ready(page);
  expect(problems, problems.join('\n')).toEqual([]);
});

/*
  The Pages base path. A project site is served from /<repo>/, so without the
  prefix every asset 404s and the page renders blank with nothing on it to say
  why. It exists only in a production build, which is why this suite builds one.
*/
test('every asset the page ships is served', async ({ page }) => {
  /*
    SAME-ORIGIN ONLY, and the restriction is the point rather than a weakening.
    The failure being caught is the page's OWN assets 404ing under /<repo>/,
    which blanks the site. Requests to the archive origin are a different
    matter: opening a Zarr v3 store legitimately probes for v2 metadata that is
    not there, so demanding 200 from everything would assert that a working
    client is broken -- and the first person to see this test fail over a
    harmless probe would delete it.
  */
  const origin = new URL(test.info().project.use.baseURL).origin;
  const missing = [];
  page.on('response', (r) => {
    if (r.status() >= 400 && r.url().startsWith(origin)) {
      missing.push(`${r.status()} ${r.url()}`);
    }
  });
  await page.goto('');
  await ready(page);
  expect(missing, missing.join('\n')).toEqual([]);
});

/*
  The field must actually paint. `ZarrSource.vector` throwing on a non-resident
  frame is correct and stays -- zeros would paint a calm, plausible, wrong map --
  but the consequence is that "the page rendered" and "the data arrived" are
  genuinely different states, and only one of them is worth shipping.
*/
test('the wind field paints onto the canvas', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="wind"]');
  await page.waitForTimeout(600);

  const painted = await paintedOn(page, 'canvas.leaflet-raster-layer');
  expect(painted).toBeGreaterThan(100);
});

/*
  Controls that exist but do nothing. The temporal dead zone left every control
  below it wired to nothing, and the page still rendered: the slider moved, it
  just did not drive anything. So assert the EFFECT, not the element.
*/
test('the time slider drives the clock', async ({ page }) => {
  await page.goto('');
  await ready(page);

  const before = await page.textContent('#stamp');
  await page.$eval('#time', (el) => {
    el.value = String(Math.min(6, Number(el.max)));
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.waitForTimeout(400);
  const after = await page.textContent('#stamp');

  expect(after).not.toEqual(before);
});

/*
  Clicking the map must answer with the place that was clicked. This is the
  path the resultant read-out sits on, so it is the one that must not be
  silently dead when a number beside it changes.
*/
test('clicking the map fills the point panel', async ({ page }) => {
  await page.goto('');
  await ready(page);

  const box = await page.locator('#map').boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForTimeout(600);

  const where = await page.textContent('#point-where');
  expect(where).toMatch(/\d+\.\d+\s*[NS],\s*\d+\.\d+\s*[EW]/);
});
