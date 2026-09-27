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

/*
  THE DRIFTER LAYER (#50). Three invented buoys in the fixture, written by the real
  exporter: E2E-A loses its drogue part-way, E2E-B is sealed, E2E-C starts on the
  15th, outside the default one-day window, so only the search can reach it.
*/
test('the drifters preset shows the buoys in the window, and the list says how many', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.waitForTimeout(800);

  // The default window is 2021-01-05; only E2E-A is seen that day.
  expect(await page.locator('path.drifter-dot').count()).toBe(1);
  expect(await page.textContent('.dp-status')).toMatch(/1 buoy in this window/);
  expect(await page.locator('.dp-list li').count()).toBe(1);
});

test('hovering a dot says what the buoy is', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.waitForTimeout(800);

  await page.locator('path.drifter-dot').first().hover();
  await page.waitForTimeout(300);
  const tip = await page.textContent('.drifter-tip');
  expect(tip).toMatch(/Buoy E2E-A/);
  expect(tip).toMatch(/loses its drogue/);
});

test('search reaches a buoy outside the window, and picking it time-travels', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.fill('.dp-search', 'E2E-C');
  await page.waitForTimeout(300);
  expect(await page.textContent('.dp-status')).toMatch(/1 buoy match/);

  await page.click('.dp-list li >> nth=0');
  await page.waitForFunction(() => /2021-01-15/.test(document.querySelector('#stamp')?.textContent ?? ''),
    null, { timeout: 10_000 });
  // Its 30 h record crosses a UTC day, so the span steps up to hold all of it.
  expect(await page.$eval('#span', (el) => el.selectedOptions[0].textContent)).toMatch(/3 days/);
  expect(await page.isVisible('#lifetime')).toBe(true);
});

test('a sealed buoy is badged in the list', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.fill('.dp-search', 'E2E-B');
  await page.waitForTimeout(300);
  expect(await page.locator('.dp-list li .dp-badge').count()).toBe(1);
});

test('playing draws the picked buoy\'s path, and it changes colour when the drogue goes', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.click('.dp-list li >> nth=0');                // E2E-A
  await page.waitForTimeout(1500);
  expect(await page.locator('path.drifter-trail').count()).toBe(0);   // nothing drawn yet

  await page.click('#play');
  // E2E-A loses its drogue at 06:00 on the 6th, a day into its record.
  await page.waitForFunction(() => /2021-01-06 (0[7-9]|1\d|2\d)/.test(document.querySelector('#stamp')?.textContent ?? ''),
    null, { timeout: 30_000 });
  await page.click('#play');
  await page.waitForTimeout(500);

  const colours = await page.$$eval('path.drifter-trail', (els) => [...new Set(els.map((e) => e.getAttribute('stroke')))]);
  expect(colours).toHaveLength(2);                          // drogued, then undrogued
});

/*
  Picking a buoy whose record needs a coarser stride must still land on its first
  fix. `setTime` left the window behind, and the tier switch then clamped the moment
  back into the old window: a buoy first seen in March 2021, picked from a view of
  1 Jan 2019, landed the clock on 1 Jan 2019. And the click must stay in the panel:
  rebuilding the list mid-click let it through to the map ("Outside the data box.").
*/
test('picking a long-lived buoy lands on its first fix, and the click stays in the panel', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="drifters"]');
  await page.fill('.dp-search', 'E2E-D');
  await page.waitForTimeout(300);
  await page.click('.dp-list li >> nth=0');
  await page.waitForTimeout(2500);

  expect(await page.textContent('#stamp')).toMatch(/2021-01-10 12:00/);
  // Ten days needs more than a week, so the stride really did change.
  expect(await page.$eval('#span', (el) => el.selectedOptions[0].textContent)).not.toMatch(/hour steps/);
  expect(await page.textContent('#status')).not.toMatch(/Outside the data box/);
  expect(await page.$eval('#point', (el) => el.classList.contains('visible'))).toBe(false);
});

/*
  THE SEARCH (#64): a Coast Guard pattern flown from a placed base to find a real buoy.
  The whole sequence on the fixture -- pick the buoy, take it as the report, place the
  base near it, fly at the quick-look speed -- ending in a result either way.
*/
test('the search view flies a pattern from a placed base and reports a result', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(500);
  await page.click('.dp-list li >> nth=0');                 // E2E-A: jumps to its first fix
  await page.waitForTimeout(1500);

  await page.click('.sp-use');
  await expect(page.locator('.sp-target')).toHaveText(/Buoy E2E-A, reported/);
  const lkp = await page.locator('path.search-lkp').boundingBox();
  expect(lkp).not.toBeNull();

  await page.click('.sp-place');
  await page.mouse.click(lkp.x - 60, lkp.y + 40);           // a base a little way off
  await expect(page.locator('.sp-base')).toHaveText(/NM out/);
  expect(await page.$eval('#point', (el) => el.classList.contains('visible'))).toBe(false);

  await page.selectOption('.sp-speed', '600');
  await page.click('.sp-fly');
  await page.waitForSelector('.search-heli', { timeout: 15_000 });
  await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
  expect(await page.locator('path.search-trail').count()).toBeGreaterThan(0);
  expect(await page.locator('path.search-marker').count()).toBe(1);
  expect(await page.textContent('.sp-result')).toMatch(/datum error/);
});

/** Pick E2E-A, take it as the report, and place a base a little way off. */
async function readySearch(page) {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(500);
  await page.click('.dp-list li >> nth=0');
  await page.waitForTimeout(1500);
  await page.click('.sp-use');
  await expect(page.locator('.sp-target')).toHaveText(/Buoy E2E-A, reported/);
  const lkp = await page.locator('path.search-lkp').boundingBox();
  await page.click('.sp-place');
  await page.mouse.click(lkp.x - 60, lkp.y + 40);
  await expect(page.locator('.sp-base')).toHaveText(/NM out/);
}

/*
  ONE CLOCK (#69). A search owns the time bar while it is loaded: the band shows its
  phases, the label says where it is, Play pauses it, and the buoy is drawn once -- the
  drifter layer, which moved on the site clock, is out of the way. Reset hands it back.
*/
test('a search takes over the time bar, draws the buoy once, and hands the bar back', async ({ page }) => {
  await readySearch(page);
  expect(await page.locator('path.drifter-live').count()).toBe(0);   // the list is out of the way
  expect(await page.locator('.drifter-panel').count()).toBe(0);

  await page.selectOption('.sp-speed', '45');
  await page.click('.sp-fly');
  await page.waitForSelector('.search-heli', { timeout: 15_000 });
  expect(await page.isVisible('#search-phases')).toBe(true);
  await expect(page.locator('#window-label')).toHaveText(/T\+/);
  expect(Number(await page.$eval('#time', (el) => el.max))).toBeGreaterThan(2700);

  // The bottom Play pauses the search rather than the site.
  await page.click('#play');
  const held = await page.$eval('#time', (el) => el.value);
  await page.waitForTimeout(600);
  expect(await page.$eval('#time', (el) => el.value)).toBe(held);
  expect(await page.locator('path.search-target').count()).toBeLessThanOrEqual(1);

  await page.click('.sp-reset');
  expect(await page.isVisible('#search-phases')).toBe(false);
  await expect(page.locator('#window-label')).not.toHaveText(/T\+/);
});

test('the panel keeps its choices when the view is left and opened again', async ({ page }) => {
  await readySearch(page);
  await page.click('button[data-preset="wind"]');
  await page.waitForTimeout(300);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(300);
  await expect(page.locator('.sp-base')).toHaveText(/NM out/);
  await expect(page.locator('.sp-target')).toHaveText(/Buoy E2E-A/);
});

/*
  FLOWN BY HAND (#68): spawn a helicopter with a click, steer it with the keys, and it
  draws its strip and ends with what it flew.
*/
test('a spawned helicopter flies on the keyboard and reports what it flew', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(500);
  await page.click('.sp-tab[data-tab="fly"]');
  await page.click('.sp-spawn');
  const box = await page.locator('#map').boundingBox();
  await page.mouse.click(box.x + box.width * 0.6, box.y + box.height * 0.5);
  await page.waitForSelector('.search-heli', { timeout: 5_000 });
  await expect(page.locator('.sp-phase')).toHaveText(/take off/);

  await page.selectOption('.sp-speed', '600');
  await page.keyboard.down('KeyD');                // WASD
  await page.waitForTimeout(700);
  await page.keyboard.up('KeyD');
  await page.keyboard.down('ArrowUp');             // and the arrow keys
  await page.waitForTimeout(700);
  await page.keyboard.up('ArrowUp');
  expect(await page.locator('path.search-trail').count()).toBeGreaterThan(0);

  await expect(page.locator('.sp-result')).toHaveText(/You flew/, { timeout: 15_000 });
});

/** How far the search panel would have to scroll to show everything, in px. */
async function panelOverflow(page) {
  return page.$eval('.sp-body', (el) => el.scrollHeight - el.clientHeight);
}

/*
  IT HAS TO FIT (#71). "The menus still clip": on a laptop the panel was taller than the
  map. The screenshots showed it and it was judged acceptable because it scrolled; this
  makes a panel that has to scroll a failure instead of a judgement.
*/
for (const size of [{ width: 1440, height: 900 }, { width: 1366, height: 768 }]) {
  test(`the search panel fits without scrolling at ${size.width}x${size.height}`, async ({ page }) => {
    await page.setViewportSize(size);
    await readySearch(page);
    expect(await panelOverflow(page)).toBeLessThanOrEqual(1);                            // the set-up

    await page.selectOption('.sp-speed', '600');
    await page.click('.sp-fly');
    await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
    expect(await panelOverflow(page)).toBeLessThanOrEqual(1);                            // the result
    await expect(page.locator('.sp-summary-text')).toHaveText(/Buoy E2E-A/);
  });
}

test('the buoy leaves a trace, and the prediction is drawn as a path to the datum', async ({ page }) => {
  await readySearch(page);
  await page.selectOption('.sp-speed', '600');
  await page.click('.sp-fly');
  await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
  expect(await page.locator('path.search-predicted').count()).toBe(1);
  expect(await page.locator('path.search-buoy-path').count()).toBe(1);
  await expect(page.locator('.search-tag-predicted')).toHaveText(/predicted drift/);
});

/*
  ONE PLAY CONTROL (#73). In the Search view the time bar's ▶ is the search's: it flies
  it, pauses it and resumes it, exactly as the panel's one button, whose label follows.
*/
test('in the Search view ▶ flies the search, and the one button follows it', async ({ page }) => {
  await readySearch(page);
  await page.selectOption('.sp-speed', '45');
  await expect(page.locator('.sp-fly')).toHaveText('Fly the search');
  await page.click('#play');
  await page.waitForSelector('.search-heli', { timeout: 15_000 });
  await expect(page.locator('.sp-fly')).toHaveText('Pause');
  await page.click('#play');
  await expect(page.locator('.sp-fly')).toHaveText('Resume');
  await page.click('.sp-fly');
  await expect(page.locator('.sp-fly')).toHaveText('Pause');
  expect(await page.isVisible('.search-compass')).toBe(true);
  await expect(page.locator('.search-compass .cp-lines')).toContainText('Heading');
});

test('▶ in the Search view never plays the site hours', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(400);
  const before = await page.textContent('#stamp');
  await page.click('#play');
  await page.waitForTimeout(900);
  expect(await page.textContent('#stamp')).toBe(before);
  await expect(page.locator('#status')).toHaveText(/flies the search/);
  // The close-up's canvas is part of the Search view (#79; the 2-D sea of #73 before it).
  expect(await page.locator('canvas.closeup-life').count()).toBe(1);
});

/*
  THE DRIFT MODEL LAYS THESE OUT (#75). Parallel Track and Trackline are flown along the
  predicted drift; the pattern still to fly is drawn faint ahead of the helicopter (only
  ahead since #83), and a Parallel Track's area is outlined.
*/
for (const [kind, area] of [['parallel_track', 1], ['trackline_return', 0]]) {
  test(`a ${kind.replace('_', ' ')} flies to a result, drawn ahead of the helicopter`, async ({ page }) => {
    await readySearch(page);
    await page.click(`.sp-pill:has(input[value="${kind}"])`);
    await page.selectOption('.sp-speed', '600');
    await page.click('.sp-fly');
    await page.waitForSelector('path.search-planned', { timeout: 10_000 });   // ahead, while it flies
    await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
    expect(await page.locator('path.search-planned').count()).toBeLessThanOrEqual(1);
    expect(await page.locator('path.search-area').count()).toBe(area);
    await expect(page.locator('.sp-result')).toHaveText(/datum line|as set/);
  });
}

/* A north arrow in every view, beside the scale bar (#75). */
test('the map always carries a north arrow', async ({ page }) => {
  await page.goto('');
  await ready(page);
  await expect(page.locator('.north-arrow')).toBeVisible();
  await page.click('button[data-preset="search"]');
  await expect(page.locator('.north-arrow')).toBeVisible();
});

/** How far the top-right stack runs into the bottom-right one, px (negative is clear). */
async function cornerOverlap(page) {
  return page.evaluate(() => {
    const top = document.querySelector('.leaflet-top.leaflet-right').getBoundingClientRect();
    const bottom = document.querySelector('.leaflet-bottom.leaflet-right').getBoundingClientRect();
    return top.bottom - bottom.top;
  });
}

/*
  THE RIGHT-HAND CORNERS NEVER MEET (#76). On 25 Sep the drifter list lay over the
  legends and the compass over the Surface current key.
*/
for (const size of [{ width: 1440, height: 900 }, { width: 1366, height: 768 }]) {
  test(`the right-hand controls do not overlap at ${size.width}x${size.height}`, async ({ page }) => {
    await page.setViewportSize(size);
    await page.goto('');
    await ready(page);
    for (const preset of ['drifters', 'both', 'drift']) {
      await page.click(`button[data-preset="${preset}"]`);
      await page.waitForTimeout(500);
      expect(await cornerOverlap(page), preset).toBeLessThanOrEqual(0);
    }
    await readySearch(page);
    await page.selectOption('.sp-speed', '600');
    await page.click('.sp-fly');
    await page.waitForSelector('.search-compass');
    await page.waitForTimeout(300);
    expect(await cornerOverlap(page), 'search').toBeLessThanOrEqual(0);
  });
}

/** Spawn a helicopter to fly by hand: the view follows it close up, at zoom 15. */
async function spawnCloseUp(page) {
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(500);
  await page.click('.sp-tab[data-tab="fly"]');
  await page.click('.sp-spawn');
  const box = await page.locator('#map').boundingBox();
  await page.mouse.click(box.x + box.width * 0.6, box.y + box.height * 0.5);
  await page.waitForSelector('.search-heli', { timeout: 5_000 });
}

/*
  CLOSE UP (#79). From zoom 13.5 the Search view draws its own sea, hides the colour field
  and its key, shows a key of its own and goes to satellite; zooming out puts it all back.
*/
test('close up, the Search view draws its own sea and key, and zooming out puts the page back', async ({ page }) => {
  // CI draws WebGL in software: the sea is slow there until it steps its resolution down,
  // and a dozen zoom animations take their time. The work is bounded; the limit says so.
  test.setTimeout(60_000);
  await page.goto('');
  await ready(page);
  await page.click('button[data-preset="search"]');
  await page.waitForTimeout(500);
  const legendBefore = await page.isVisible('.current-legend');
  await spawnCloseUp(page);

  await expect(page.locator('#map')).toHaveClass(/closeup-on/);
  await expect(page.locator('.closeup-key')).toBeVisible();
  await expect(page.locator('.closeup-key')).toHaveText(/cloud shadows/i);
  // Four rows at most (#83): the sea, the weed, night if it is, and that none of it is data.
  expect(await page.locator('.closeup-key .ck-body > span').count()).toBeLessThanOrEqual(4);
  // The streak switch (#85): drift by default, and a click changes it. On a short screen
  // the corner guard folds the key for room, so open it first, as a person would.
  if (await page.locator('.closeup-key.collapsed').count()) await page.click('.closeup-key .legend-toggle');
  await expect(page.locator('.closeup-key button[data-flow="drift"]')).toHaveClass(/on/);
  await page.click('.closeup-key button[data-flow="current"]');
  await expect(page.locator('.closeup-key button[data-flow="current"]')).toHaveClass(/on/);
  expect(await page.locator('.closeup-key .ck-body > span').count()).toBeLessThanOrEqual(4);
  // No arrows at the buoy any more (#83).
  expect(await page.locator('path.search-why-gap').count()).toBe(0);
  expect(await page.isVisible('.current-legend')).toBe(false);
  // The shader where WebGL runs, the 2-D texture where it does not: one of them is there.
  expect(await page.locator('.closeup-sea, .ocean-canvas').count()).toBeGreaterThan(0);
  await expect(page.locator('.leaflet-control-attribution')).toHaveText(/Maxar/);
  // The compass gives knots beside m/s, so the rough-sea threshold can be checked (#81).
  await expect(page.locator('.search-compass .cp-lines')).toHaveText(/m\/s \(\d+ kt\)/, { timeout: 10_000 });

  // One click at a time: a click during Leaflet's zoom animation is dropped. 14 eighths of a
  // zoom take 15 below 13.5, where the close-up ends.
  for (let k = 0; k < 14; k += 1) {
    await page.click('.leaflet-control-zoom-out');
    await page.waitForTimeout(350);
  }
  await page.waitForTimeout(800);
  await expect(page.locator('#map')).not.toHaveClass(/closeup-on/);
  expect(await page.locator('.closeup-key').count()).toBe(0);
  expect(await page.isVisible('.current-legend')).toBe(legendBefore);
  await expect(page.locator('.leaflet-control-attribution')).not.toHaveText(/Maxar/);
});

for (const size of [{ width: 1440, height: 900 }, { width: 1366, height: 768 }]) {
  test(`close up, the right-hand controls still do not overlap at ${size.width}x${size.height}`, async ({ page }) => {
    await page.setViewportSize(size);
    await page.goto('');
    await ready(page);
    await spawnCloseUp(page);
    await expect(page.locator('.closeup-key')).toBeVisible();
    await page.waitForTimeout(300);
    expect(await cornerOverlap(page)).toBeLessThanOrEqual(0);
  });
}

/* WHY IT MISSED (#80): the result says why the buoy left the prediction, with the reasoning folded. */
test('the result says why the buoy left the prediction', async ({ page }) => {
  await readySearch(page);
  await page.selectOption('.sp-speed', '600');
  await page.click('.sp-fly');
  await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
  await expect(page.locator('.sp-result')).toHaveText(/Why it left the prediction/);
  await page.click('.sp-why summary');
  await expect(page.locator('.sp-why')).toHaveText(/Buoy.*Model.*Missing/s);
  await expect(page.locator('.sp-why summary')).toHaveAttribute('title', /15 m down/);
});

/* Open or closed, the reasoning must not make the panel scroll (#71, #80). */
for (const size of [{ width: 1440, height: 900 }, { width: 1366, height: 768 }]) {
  test(`the reasoning fits in the panel when opened at ${size.width}x${size.height}`, async ({ page }) => {
    await page.setViewportSize(size);
    await readySearch(page);
    await page.selectOption('.sp-speed', '600');
    await page.click('.sp-fly');
    await expect(page.locator('.sp-result')).toHaveText(/Why it left the prediction/, { timeout: 25_000 });
    await page.click('.sp-why summary');
    await page.waitForTimeout(200);
    expect(await panelOverflow(page)).toBeLessThanOrEqual(1);
  });
}

/*
  LABELS NEVER OVERLAP (#86). When the buoy, the marker and the datum bunch up, their chips
  used to land on each other. Scrub through a whole search and check every moment.
*/
test('the search labels never sit on each other, at any moment of a search', async ({ page }) => {
  test.setTimeout(60_000);
  await readySearch(page);
  await page.selectOption('.sp-speed', '600');
  await page.click('.sp-fly');
  await expect(page.locator('.sp-result')).toHaveText(/Found|Not found/, { timeout: 25_000 });
  const hits = await page.evaluate(async () => {
    const s = document.querySelector('input[type=range]');
    const max = Number(s.max);
    const found = [];
    for (let f = 0; f <= 1.0001; f += 0.05) {
      s.value = String(Math.round(max * f));
      s.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise((r) => setTimeout(r, 50));
      const r = [...document.querySelectorAll('.search-tag span')].map((e) => e.getBoundingClientRect());
      for (let i = 0; i < r.length; i += 1) {
        for (let j = i + 1; j < r.length; j += 1) {
          const a = r[i];
          const b = r[j];
          if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) found.push(f.toFixed(2));
        }
      }
    }
    return found;
  });
  expect(hits).toEqual([]);
});
