import { readFile } from 'node:fs/promises';
import { dirname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

/*
  The scenario page and the open-day game, in the built site (D031). As in site.spec.js,
  these check wiring and painting, never arithmetic: the referee's numbers are held to
  Python by referee.test.js and playback.test.js. The one number checked here is that the
  page shows the paper's own figure from the bundle, and that replaying the flight to the
  end lands on it, which is the whole promise of the page.

  The fixture is one scenario at the 2 h arrival, 200 particles, constant forcing, written by
  scripts/make_e2e_scenarios.py with the same exporter as the published bundles: scenarios/v2,
  flown by the D032 helicopter that turns at 7 deg/s. scenarios/v1 beside it is the older
  format, flown by the helicopter that turned at once; one test hides v2 to check the site
  still plays from a data root that only has v1.
*/

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');
const LAPTOP = { width: 1366, height: 768 };

async function serveFixture(page, { hide = null } = {}) {
  await page.route('**/e2e.invalid/**', async (route) => {
    const url = new URL(route.request().url());
    const rel = normalize(url.pathname.replace(/^\/archive\/?/, '')).replace(/^(\.\.[/\\])+/, '');
    if (hide && url.pathname.includes(hide)) {
      await route.fulfill({ status: 404, body: '' });
      return;
    }
    try {
      const body = await readFile(join(FIXTURES, rel));
      await route.fulfill({
        status: 200,
        contentType: rel.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        body,
      });
    } catch {
      await route.fulfill({ status: 404, body: '' });
    }
  });
  const png = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    'base64',
  );
  await page.route(/(basemaps\.cartocdn\.com|server\.arcgisonline\.com|tile\.openstreetmap\.org)/,
    (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));
}

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errors.push(m.text());
  });
  return errors;
}

/** The page holds no scrollbar at this size: Aditya's rule, a panel that scrolls is clipped. */
async function expectFits(page) {
  const f = await page.evaluate(() => ({
    h: document.documentElement.scrollHeight, w: document.documentElement.scrollWidth,
    ih: window.innerHeight, iw: window.innerWidth,
  }));
  expect(f.h).toBeLessThanOrEqual(f.ih);
  expect(f.w).toBeLessThanOrEqual(f.iw);
}

/** Some pixels of a canvas are painted. */
async function painted(page, selector) {
  return page.evaluate((sel) => {
    const c = document.querySelector(sel);
    if (!c) return 0;
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n += 1;
    return n;
  }, selector);
}

async function fixtureFlight(noise, searcher, version = 'v2') {
  const meta = JSON.parse(await readFile(join(FIXTURES, `scenarios/${version}/S01`, `${noise}_2h.json`), 'utf8'));
  return meta.flights[searcher].python;
}

test.describe('the scenario page', () => {
  test.use({ viewport: LAPTOP });

  test('shows the paper\'s number, paints the cloud, and replays to the same number', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('bench.html');
    const want = await fixtureFlight('rv', 'expanding-square');
    await expect(page.locator('#paper')).toHaveText(`${(100 * want.pos).toFixed(1)} %`);
    expect(await painted(page, 'canvas.probability-layer')).toBeGreaterThan(0);
    await expectFits(page);

    await page.locator('#time').evaluate((el) => {
      el.value = '2700';
      el.dispatchEvent(new Event('input'));
    });
    await expect(page.locator('#pos')).toHaveText(`${(100 * want.pos).toFixed(1)} %`);
    await expect(page.locator('#buoy')).toContainText(/real buoy/);
    expect(errors).toEqual([]);
  });

  test('flies by hand', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('bench.html');
    await expect(page.locator('#paper')).not.toHaveText('–');
    await page.locator('input[value=self]').check();
    await page.selectOption('#speed', '120');
    await page.keyboard.down('ArrowUp');
    await expect(page.locator('#clock')).not.toHaveText(/^0:00 /, { timeout: 5000 });
    await page.keyboard.up('ArrowUp');
    expect(errors).toEqual([]);
  });
});

test.describe('the open-day game', () => {
  test.use({ viewport: LAPTOP });

  test('plays from the start screen to the results, and every screen fits', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=20');
    await expect(page.locator('#attract')).toHaveClass(/on/);
    await expectFits(page);

    await page.click('#startBtn');
    await page.click('#surprise');
    await page.fill('#name', 'E2E');
    await expectFits(page);
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    expect(await painted(page, '#flyMap canvas.probability-layer')).toBeGreaterThan(0);
    await expectFits(page);

    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expectFits(page);

    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    await expectFits(page);
    const cg = await fixtureFlight('rv', 'expanding-square');
    await expect(page.locator('#scoreTable tr')).toHaveCount(4);
    await expect(page.locator('#scoreTable')).toContainText(`${(100 * cg.pos).toFixed(1)} %`);
    await expect(page.locator('#verdict')).not.toBeEmpty();
    await expect(page.locator('#boardNow')).toContainText('E2E');
    await expect(page.locator('#benchTable tr')).toHaveCount(7);
    expect(errors).toEqual([]);
  });

  test('flies with the mouse: toward the pointer, turning at the rate, with the bend drawn ahead', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=4');
    await page.click('#startBtn');
    await page.click('#surprise');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await expect(page.locator('#hint')).toContainText(/mouse/);
    const box = await page.locator('#flyMap').boundingBox();
    // Click well to one side to take off, then hold the pointer there.
    await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.5);
    await page.mouse.click(box.x + box.width * 0.15, box.y + box.height * 0.5);
    await expect(page.locator('#hint')).toBeHidden();
    await expect(page.locator('#hudTime')).not.toHaveText('45:00', { timeout: 5000 });
    // The dashed line ahead of the helicopter is drawn while it flies.
    await expect(page.locator('#flyMap path[stroke-dasharray]')).toHaveCount(1);
    await expect(page.locator('#flyMap .scene-heli svg')).toHaveAttribute('style', /rotate\(/);
    await page.mouse.move(box.x + box.width * 0.85, box.y + box.height * 0.3);
    await page.waitForTimeout(800);
    expect(errors).toEqual([]);
  });

  test('still plays from a data root that only has the older v1 bundles', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page, { hide: '/scenarios/v2/' });
    await page.goto('game.html?speed=20');
    await page.click('#startBtn');
    await page.click('#surprise');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    const cg = await fixtureFlight('rv', 'expanding-square', 'v1');
    await expect(page.locator('#scoreTable')).toContainText(`${(100 * cg.pos).toFixed(1)} %`);
    expect(errors).toEqual([]);
  });

  test('the currents-only mode draws the drift field instead of the cloud', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html');
    await page.click('#startBtn');
    await page.click('#surprise');
    await page.click('[data-mode=currents]');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    expect(await painted(page, '#flyMap canvas.field-layer')).toBeGreaterThan(0);
    await expect(page.locator('#flyMap canvas.probability-layer')).toHaveCount(0);
    await expect(page.locator('#hudPosCard')).toBeHidden();
    expect(errors).toEqual([]);
  });
});
