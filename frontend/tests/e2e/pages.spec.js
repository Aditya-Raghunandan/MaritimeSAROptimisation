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

  v3 (the trained agent's flight beside the six) is not a fixture on disk: `agent: true` serves
  it as v2 with the Sector's flight renamed "ppo". It checks the wiring for a seventh searcher,
  so its flight only has to differ from greedy's, which plays the AI where there is no agent.
*/

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, 'fixtures');
const LAPTOP = { width: 1366, height: 768 };

/** v2's file as v3 would hold it: the Sector's flight and rows renamed as the agent's. The 2 h
 * window is also listed as a 30 min one (served from the same files), so the game's default
 * arrival, which the trained model flies, has a window (v4 has 15 min to 3 h). */
function asV3(rel, body) {
  if (!rel.endsWith('.json')) return body;
  const j = JSON.parse(body);
  if (j.scenarios) j.searchers = [...j.searchers, 'ppo'];
  if (j.scenarios) {
    j.scenarios = j.scenarios.map((s) => ({
      ...s, windows: [...s.windows, ...s.windows.filter((w) => w.endsWith('_2h')).map((w) => w.replace('_2h', '_0.5h'))],
    }));
  }
  if (j.flights) j.flights.ppo = { ...j.flights.sector, layout: 'e2e: the Sector, as the agent' };
  for (const key of ['scenarios55', 'dev']) {
    if (Array.isArray(j[key])) {
      j[key] = [...j[key], ...j[key].filter((r) => r.searcher === 'sector').map((r) => ({ ...r, searcher: 'ppo' }))];
      j[key] = [...j[key], ...j[key].filter((r) => r.arrival_h === 2).map((r) => ({ ...r, arrival_h: 0.5 }))];
    }
  }
  return Buffer.from(JSON.stringify(j));
}

async function serveFixture(page, { hide = null, agent = false } = {}) {
  await page.route('**/e2e.invalid/**', async (route) => {
    const url = new URL(route.request().url());
    let rel = normalize(url.pathname.replace(/^\/archive\/?/, '')).replace(/^(\.\.[/\\])+/, '');
    if (hide && url.pathname.includes(hide)) {
      await route.fulfill({ status: 404, body: '' });
      return;
    }
    const v3 = /^scenarios[/\\]v3[/\\]/.test(rel);
    if (v3 && agent) rel = rel.replace(/^scenarios([/\\])v3/, 'scenarios$1v2').replace(/_0\.5h\.(json|f32)$/, '_2h.$1');
    try {
      let body = await readFile(join(FIXTURES, rel));
      if (v3 && agent) body = asV3(rel, body);
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
    const want = await fixtureFlight('rvc', 'expanding-square');
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

  test('keys the probability map, recolours it on a pick, and remembers the pick', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('bench.html');
    await expect(page.locator('#paper')).not.toHaveText('–');
    const legend = page.locator('.prob-legend');
    await expect(legend).toContainText('half of what is left');
    await expect(legend.locator('button[data-palette=viridis]')).toHaveAttribute('aria-pressed', 'true');
    const pixels = () => page.evaluate(() => {
      const c = document.querySelector('canvas.probability-layer');
      return Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).join(',');
    });
    const before = await pixels();
    await legend.locator('button[data-palette=heat]').click();
    await expect(legend.locator('button[data-palette=heat]')).toHaveAttribute('aria-pressed', 'true');
    expect(await pixels()).not.toEqual(before);
    await expectFits(page);
    await page.reload();
    await expect(page.locator('.prob-legend button[data-palette=heat]')).toHaveAttribute('aria-pressed', 'true');
    expect(errors).toEqual([]);
  });

  test('lists the trained AI only when the files carry it, and replays its flight', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('bench.html');
    await expect(page.locator('#paper')).not.toHaveText('–');
    await expect(page.locator('#searchers input')).toHaveCount(7);   // six and "fly it yourself"
    await expect(page.locator('input[value=ppo]')).toHaveCount(0);
    await expect(page.locator('#searchers')).not.toContainText('coming');
    expect(errors).toEqual([]);
  });

  test('replays the trained AI\'s flight to the paper\'s number', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page, { agent: true });
    await page.goto('bench.html');
    await expect(page.locator('#paper')).not.toHaveText('–');
    await expect(page.locator('#searchers input')).toHaveCount(8);
    await expect(page.locator('#searchers')).toContainText('Telemachus');
    await page.locator('input[value=ppo]').check();
    const want = await fixtureFlight('rvc', 'sector');
    await expect(page.locator('#paper')).toHaveText(`${(100 * want.pos).toFixed(1)} %`);
    await page.locator('#time').evaluate((el) => {
      el.value = '2700';
      el.dispatchEvent(new Event('input'));
    });
    await expect(page.locator('#pos')).toHaveText(`${(100 * want.pos).toFixed(1)} %`);
    await expectFits(page);
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

    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.fill('#name', 'E2E');
    await expectFits(page);
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    expect(await painted(page, '#flyMap canvas.probability-layer')).toBeGreaterThan(0);
    await expect(page.locator('#flyMap .prob-legend')).toBeVisible();
    await expectFits(page);

    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expectFits(page);

    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    await expectFits(page);
    const cg = await fixtureFlight('rvc', 'expanding-square');
    await expect(page.locator('#scoreTable tr')).toHaveCount(4);
    await expect(page.locator('#scoreTable')).toContainText(`${(100 * cg.pos).toFixed(1)} %`);
    await expect(page.locator('#verdict')).not.toBeEmpty();
    await expect(page.locator('#boardNow')).toContainText('E2E');
    await expect(page.locator('#benchTable tr')).toHaveCount(7);
    // Points are scored against the cloud's size, and the page says so in numbers.
    await expect(page.locator('#pointsWhy')).toContainText(/covered \d+ km²/);
    await expect(page.locator('#pointsWhy')).not.toContainText(/1\d\d %/);
    await expect(page.locator('#aiWords')).toContainText('greedy');
    expect(errors).toEqual([]);
  });

  test('three missions, then the real search: each one plays, and each fits the laptop', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=20');
    await page.click('#startBtn');
    await expect(page.locator('#tut')).toHaveClass(/on/);
    await expect(page.locator('#tutTitle')).toHaveText('Go with the flow');
    await expect(page.locator('#tutResult')).toContainText('2 hours', { timeout: 15_000 });
    await expectFits(page);
    await page.click('text=Fly the helicopter');
    await expect(page.locator('#tutTitle')).toHaveText('Catch the drifting buoy');
    await expect(page.locator('#tutMap .scene-heli')).toBeVisible();
    await page.click('#tutSkip');
    await expect(page.locator('#tutTitle')).toHaveText('Read the sea between the dots');
    await expect(page.locator('#tutText')).toContainText('Click on the map', { timeout: 15_000 });
    await page.mouse.click(700, 420);
    await expect(page.locator('#tutResult')).toContainText('off', { timeout: 15_000 });
    await expectFits(page);
    await page.click('text=Mission 3');
    await expect(page.locator('#tutTitle')).toHaveText('10,000 guesses');
    await page.locator('#tutSigma').fill('66');
    await page.click('text=Release the drifters');
    await expect(page.locator('#tutResult')).toContainText('Just right', { timeout: 15_000 });
    await expectFits(page);
    await page.click('text=Now the real search');
    await expect(page.locator('#pick')).toHaveClass(/on/);
    expect(errors).toEqual([]);
  });

  test('the video plays on demand, stays put when the mouse moves, and its button closes it', async ({ page }) => {
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=20');
    await page.click('#demoBtn');
    await expect(page.locator('#demo')).toHaveClass(/on/);
    await page.mouse.move(300, 300);
    await page.mouse.move(320, 330);
    await expect(page.locator('#demo')).toHaveClass(/on/);
    await expectFits(page);
    await page.click('#demoClose');
    await expect(page.locator('#attract')).toHaveClass(/on/);
    await page.click('#demoBtn');
    await page.keyboard.press('Escape');
    await expect(page.locator('#attract')).toHaveClass(/on/);
    expect(errors.filter((e) => !/media|video|webm|mp4/i.test(e))).toEqual([]);
  });

  test('from an hour on the computer flies greedy, even where the files carry the model\'s flight', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page, { agent: true });
    await page.goto('game.html?speed=20');
    await page.click('#searchBtn');
    await page.click('#surprise');
    await expect(page.locator('#arrivalSeg button[aria-pressed=true]')).toHaveText('30 min');
    await page.locator('#arrivalSeg button', { hasText: '2 h' }).click();
    await expect(page.locator('#arrivalNote')).toContainText('greedy');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    const greedy = await fixtureFlight('rvc', 'greedy');
    await expect(page.locator('#scoreTable tr').last()).toContainText(`${(100 * greedy.pos).toFixed(1)} %`);
    await expect(page.locator('#aiWords')).toContainText('within 45 minutes');
    await expect(page.locator('#benchNote')).toContainText('2 h after the call');
    await expectFits(page);
    expect(errors).toEqual([]);
  });

  test('emails the result: the numbers, Telemachus by name and a picture of the flights', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page, { agent: true });
    let posted = null;
    await page.route('**/mailer.e2e.invalid/**', async (route) => {
      if (route.request().method() === 'POST') posted = route.request().postDataJSON();
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    });
    await page.goto('game.html?speed=20');
    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.fill('#name', 'Mail');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    await expect(page.locator('#mailForm')).toBeVisible();
    await expectFits(page);
    await page.fill('#mailTo', 'player@example.com');
    await page.click('#mailBtn');
    await expect(page.locator('#mailNote')).toContainText('Sent!');
    expect(posted.email).toBe('player@example.com');
    expect(posted.name).toBe('Mail');
    expect(posted.ai.name).toBe('Telemachus');
    const agent = await fixtureFlight('rvc', 'sector');
    expect(posted.ai.pos).toBeCloseTo(agent.pos, 9);
    expect(posted.points).toBeGreaterThanOrEqual(0);
    expect(posted.areaKm2).toBeGreaterThan(0);
    expect(posted.image).toMatch(/^data:image\/jpeg;base64,/);
    expect(posted.image.length).toBeLessThan(900_000);
    expect(errors).toEqual([]);
  });

  test('the ML model is the trained agent where the files carry its flight', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page, { agent: true });
    await page.goto('game.html?speed=20');
    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await page.keyboard.down('ArrowRight');
    await expect(page.locator('#reveal')).toHaveClass(/on/, { timeout: 20_000 });
    await page.keyboard.up('ArrowRight');
    await expect(page.locator('#results')).toHaveClass(/on/, { timeout: 20_000 });
    const agent = await fixtureFlight('rvc', 'sector');
    await expect(page.locator('#scoreTable tr').last()).toContainText(`${(100 * agent.pos).toFixed(1)} %`);
    await expect(page.locator('#aiWords')).toContainText('machine-learning model');
    await expect(page.locator('#benchTable')).toContainText('Telemachus');
    await expect(page.locator('#benchTable tr')).toHaveCount(8);
    await expectFits(page);
    expect(errors).toEqual([]);
  });

  test('flies with the mouse: toward the pointer, turning at the rate, with the bend drawn ahead', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=4');
    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await expect(page.locator('#hint')).toContainText(/mouse/);
    await expect(page.locator('#hint')).toContainText('cannot stop or turn on the spot');
    // The helicopter is drawn as one, its rotor and searchlight included.
    await expect(page.locator('#flyMap .scene-heli svg rect')).not.toHaveCount(0);
    await expect(page.locator('#flyMap .scene-heli svg linearGradient')).toHaveCount(1);
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

  test('ends the search early on its button, and says so', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page);
    await page.goto('game.html?speed=4');
    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.fill('#name', 'Quitter');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    await expect(page.locator('#endBtn')).toBeHidden();             // not before take-off
    await page.keyboard.down('ArrowUp');
    await expect(page.locator('#endBtn')).toBeVisible();
    await expect(page.locator('#hudTime')).not.toHaveText(/^4[45]:/, { timeout: 10_000 });
    await page.keyboard.up('ArrowUp');
    await expectFits(page);
    await page.click('#endBtn');
    await expect(page.locator('#reveal')).toHaveClass(/on/);
    await expect(page.locator('#revealNote')).toContainText('You ended your search with');
    await page.click('#skipBtn');
    await expect(page.locator('#results')).toHaveClass(/on/);
    await expect(page.locator('#verdictWhy')).toContainText('left on the clock');
    await expect(page.locator('#boardNow')).toContainText('Quitter');
    expect(errors).toEqual([]);
  });

  test('the start buttons wait for the buoys, and the kiosk has no links off the game', async ({ page }) => {
    const errors = collectErrors(page);
    let release;
    const held = new Promise((r) => { release = r; });
    await serveFixture(page);
    // Registered last, so asked first: hold the list of buoys back, then hand it on.
    await page.route(/scenarios\/v\d\/index\.json/, async (route) => { await held; await route.fallback(); });
    await page.goto('game.html?kiosk=1');
    await expect(page.locator('#attract')).toHaveClass(/on/);
    await expect(page.locator('#startBtn')).toBeDisabled();
    await expect(page.locator('#searchBtn')).toBeDisabled();
    await expect(page.locator('#status')).toHaveText('loading the buoys…');
    await expect(page.locator('header nav')).toBeHidden();
    release();
    await expect(page.locator('#searchBtn')).toBeEnabled();
    await expect(page.locator('#status')).toHaveText('');
    await page.click('#searchBtn');
    await expect(page.locator('#pick')).toHaveClass(/on/);
    expect(errors).toEqual([]);
  });

  test('still plays from a data root that only has the older v1 bundles', async ({ page }) => {
    test.setTimeout(60_000);
    const errors = collectErrors(page);
    await serveFixture(page, { hide: '/scenarios/v2/' });
    await page.goto('game.html?speed=20');
    await page.click('#searchBtn');
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
    await page.click('#searchBtn');
    await page.click('#surprise');
    await page.click('[data-mode=currents]');
    await page.click('#goBtn');
    await expect(page.locator('#fly')).toHaveClass(/on/);
    expect(await painted(page, '#flyMap canvas.field-layer')).toBeGreaterThan(0);
    await expect(page.locator('#flyMap canvas.probability-layer')).toHaveCount(0);
    await expect(page.locator('#flyMap .prob-legend')).toHaveCount(0);
    await expect(page.locator('#hudPosCard')).toBeHidden();
    expect(errors).toEqual([]);
  });
});
