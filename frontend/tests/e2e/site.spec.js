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

/*
  The drift arrows must paint once their data ARRIVES, not only when the clock
  next moves. They were handed the new frame straight from the clock, before the
  current's chunk had landed; they found nothing to draw, and nothing told them
  again. Paused, or switched on while paused, the Drift view stayed blank until
  the slider moved. Found on the live site, 23 Sep.

  Reproduced offline by failing the current's first fetch -- the start-up
  warm-up, as a dropped connection would -- and delaying the rest, so the current
  is NOT resident when Drift is switched on. That is the state a new day's chunk
  is in on the live site for the ~2.7 s Hugging Face takes to serve one.
*/
test('the drift arrows paint once their data arrives, even while paused', async ({ page }) => {
  let failCurrent = true;
  await page.route(/current_3-hourly\.zarr\/water_[uv]\/c\//, async (route) => {
    if (failCurrent) return route.abort('connectionfailed');
    await new Promise((r) => setTimeout(r, 800));
    return route.fallback();
  });

  await page.goto('');
  await ready(page);
  failCurrent = false;

  await page.click('button[data-preset="drift"]');
  await page.waitForTimeout(2500);

  expect(await paintedOn(page, 'canvas.leaflet-quiver-layer')).toBeGreaterThan(0);
});

/*
  A click on land must say it is land. It used to answer "If someone were in the
  water here ... leeway alone would carry them 208 m NNE in an hour" for a point
  in North Carolina. The fixture's south-west corner, 26.00-26.20 N by
  79.00-78.84 W, is land in its current field, so the click goes inside it.
*/
test('clicking on land says it is land, and gives no drift', async ({ page }) => {
  await page.goto('');
  await ready(page);

  // The dashed study-box outline is drawn at the data bounds, 26-27 N by 79-78 W.
  // Linear in latitude is close enough over one degree for a target 0.2 deg wide.
  const box = await page.locator('path[stroke-dasharray="6,5"]').boundingBox();
  const x = box.x + (-78.92 - -79.0) * box.width;
  const y = box.y + (27.0 - 26.08) * box.height;
  await page.mouse.click(x, y);
  await page.waitForTimeout(800);

  // The click landed where intended, or the rest of the test proves nothing. The
  // read-out snaps to the nearest 0.25 deg wind cell, so 26.08 N 78.92 W shows as
  // the corner cell; the drift sample itself is taken at the exact click.
  expect(await page.textContent('#point-where')).toBe('26.00 N, 79.00 W');
  expect(await page.textContent('#dr-head')).toBe('On land');
  expect(await page.textContent('#dr-lead')).toMatch(/land in the current model/);
  expect(await page.textContent('#dr-1h')).toBe('—');
});

/*
  The drift arrows must keep up with playback past the first chunk. A sixteen-day
  archive makes the page open on its 6-hourly tier and switch to hourly for the
  default day. The drift layer had copied the store it was built with and kept
  reading it with hourly frame numbers, so it painted the wrong day and froze at
  hour 48 -- the first chunk nobody was fetching -- while the clock ran on. Seen on
  the live archive at hour 49, 23 Sep. The fixture is sixteen days long so the suite
  reaches this at all.
*/
test('the drift arrows keep moving with the clock past hour 48', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drift"]');
  await page.waitForTimeout(800);

  const fingerprint = () => page.evaluate(() => {
    const c = document.querySelector('canvas.leaflet-quiver-layer');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let h = 0;
    for (let i = 3; i < d.length; i += 4 * 37) {
      if (d[i] > 8) h = (h * 31 + d[i - 3] + d[i - 2] * 7 + i) % 1e9;
    }
    return h;
  });

  await page.click('#play');
  // Past hour 48 of the archive, which opens at 2021-01-05: the stamp reads the 7th.
  await page.waitForFunction(
    () => /2021-01-(0[7-9]|1\d)/.test(document.querySelector('#stamp')?.textContent ?? ''),
    null, { timeout: 30_000 },
  );
  const prints = new Set();
  const stamps = new Set();
  for (let k = 0; k < 8; k += 1) {
    prints.add(await fingerprint());
    stamps.add(await page.textContent('#stamp'));
    await page.waitForTimeout(250);
  }
  await page.click('#play');

  expect(stamps.size).toBeGreaterThan(2);   // the clock was moving
  expect(prints.size).toBeGreaterThan(2);   // and the arrows moved with it
});
