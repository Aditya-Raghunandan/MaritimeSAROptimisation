/*
  shoot.mjs -- pose the built site and photograph it.

  This is NOT a test. It asserts nothing and it can never fail a build. It
  exists because the suites in `tests/` answer "did my assertions hold", and
  that is a different question from "what does the page actually look like".
  Every frontend defect that has reached a rendered page here -- the first-paint
  race, the temporal dead zone, the unexported VIRIDIS, the watermarked
  basemap -- was visible instantly to anyone looking at the page and invisible
  to code that was only checking its own expectations. A green suite written by
  whoever also wrote the bug is not an independent witness. A picture is.

  WHAT IT SHOOTS. The five view presets and the point panel, at a fixed frame,
  into fixed filenames.

  STORAGE. Fixed filenames, overwritten every run: six files, about 2 MB, no
  matter how many times it is run. There are no timestamps and no run ids, on
  purpose -- an accumulating directory of near-identical screenshots is how a
  tool like this stops being used. Output defaults to the system temp
  directory, NEVER the repository and never OneDrive. A picture worth keeping
  is copied into `figures/` deliberately, with a row in the vault's MAP, which
  is a decision rather than a side effect.

  WHICH DATA. `--data real` (the default) points the build at the published
  Hugging Face archive, which is what `deploy.yml` sets and therefore what a
  visitor to the live site actually sees. `--data local` uses
  `frontend/public/data/`, which is WIND ONLY and has no current in it -- so
  the resultant renders in its leeway-only mode and the current presets come up
  empty. That is a fine way to work offline as long as you know that is what
  you are looking at; it is not the site.

  It also runs against a PRODUCTION build served by `vite preview`, with the
  same base path the deploy sets, because a whole class of failure -- every
  asset 404ing under `/<repo>/` and the page rendering blank with nothing on it
  to say why -- exists only in a built site and cannot be reproduced by
  `npm run dev`.

  SHOOTING THE DEPLOYED SITE. `--url <address>` skips the build and the local
  server and photographs whatever is already at that address -- normally the
  live GitHub Pages site. That is a different question from the local build and
  worth being able to ask separately, because some things differ between them
  by design: the CARTO key is domain-restricted, so a keyed basemap that works
  in production is SUPPOSED to fail from localhost. Without this flag there is
  no way to tell that apart from a key that has simply expired.

  Run:
      npm run shots                       # real data, temp directory
      npm run shots -- --data local       # offline, wind only
      npm run shots -- --out some/dir     # somewhere else
      npm run shots -- --url https://aditya-raghunandan.github.io/MaritimeSAROptimisation/
*/

import { mkdir, readdir, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const HF = 'https://huggingface.co/datasets/AdityaRugs/MaritimeSARoperations/resolve/main';

// The repository name, which is what GitHub Pages serves the project site
// under. Hard-coded here rather than read from git so the local shoot and the
// deployed site cannot disagree about it.
const BASE = '/MaritimeSAROptimisation/';

const VIEWPORT = { width: 1440, height: 900 };

/*
  A frame far enough into the archive to be past any first-chunk special case,
  and a position in the Gulf Stream rather than in the middle of the box. The
  position is given as a fraction of the map element because the Leaflet map
  instance is deliberately not exposed on `window` -- exporting it so a script
  could reach it would be production code changed for the convenience of a
  tool, which is how test hooks end up shipping. The panel reports the latitude
  and longitude it actually resolved and this script prints them, so the guess
  is checked rather than trusted.
*/
const FRAME = 12;
const CLICK = { fx: 0.42, fy: 0.30 };

const SHOTS = [
  ['01-wind', 'wind'],
  ['02-current', 'current'],
  ['03-both', 'both'],
  ['04-drift', 'drift'],
  ['05-clean', 'clean'],
];

function parseArgs(argv) {
  const args = { data: 'real', out: null, port: 4173, build: true, url: null };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--data') args.data = argv[++i];
    else if (a === '--out') args.out = argv[++i];
    else if (a === '--port') args.port = Number(argv[++i]);
    else if (a === '--no-build') args.build = false;
    else if (a === '--url') args.url = argv[++i];
    else if (a === '--help') args.help = true;
    else throw new Error(`unknown argument ${a}`);
  }
  if (!['real', 'local'].includes(args.data)) {
    throw new Error(`--data must be real or local, got ${args.data}`);
  }
  return args;
}

/** Where the pictures go. Temp by default, and it says so out loud. */
function defaultOut() {
  return join(tmpdir(), 'sar-shots');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log('npm run shots -- [--data real|local] [--out dir] [--port n] [--no-build]');
    return;
  }

  const out = args.out ?? defaultOut();
  await mkdir(out, { recursive: true });

  const { chromium } = await import('@playwright/test');

  /*
    Two modes. With `--url` nothing is built and nothing is served: the target
    already exists and is photographed as found. Otherwise a production build
    is made here and served locally, which is the mode that can see build-time
    failures at all.
  */
  let url = args.url;
  let server = null;

  if (!url) {
    /*
      Set before vite is imported for the build: `vite.config.js` reads
      VITE_BASE at config load, and `import.meta.env.VITE_DATA_BASE` is
      statically replaced at build time. This is the same mechanism deploy.yml
      uses, which is the point -- a different mechanism here would mean the
      thing photographed is not the thing deployed.
    */
    process.env.VITE_BASE = BASE;
    if (args.data === 'real') process.env.VITE_DATA_BASE = HF;
    else delete process.env.VITE_DATA_BASE;

    const { build, preview } = await import('vite');
    if (args.build) {
      console.log(`building (base ${BASE}, data ${args.data})...`);
      await build({ logLevel: 'warn' });
    }
    server = await preview({ preview: { port: args.port, strictPort: true } });
    url = server.resolvedUrls.local[0];
    console.log(`serving ${url}`);
  } else {
    console.log(`shooting deployed site ${url}`);
  }

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: 1 });

  // Collected and printed at the end rather than thrown on. This tool's job is
  // to come back with pictures; an error on the page is something to SEE in
  // the picture as well as read, and throwing here would lose both.
  const problems = [];
  page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`console: ${m.text()}`);
  });
  page.on('requestfailed', (r) => problems.push(`request failed: ${r.url()}`));

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded' });

    /*
      Wait on a real readiness signal, not a sleep. The provenance line is
      written only once a tier has been chosen and its manifest read, so a
      non-empty one means the data path got all the way through. A fixed sleep
      would be both slower and a liar: it would paper over exactly the
      first-paint ordering bug that made this whole exercise necessary.
    */
    await page.waitForFunction(
      () => document.querySelector('#provenance')?.textContent?.trim().length > 0,
      null,
      { timeout: 60_000 },
    );
    console.log('provenance:', (await page.textContent('#provenance'))?.trim());

    await setFrame(page, FRAME);

    for (const [name, preset] of SHOTS) {
      await page.click(`button[data-preset="${preset}"]`);
      await settle(page);
      await page.screenshot({ path: join(out, `${name}.png`) });
      console.log(`  ${name}.png`);
    }

    // The point panel, over the drift view, because that is the combination
    // the resultant is actually read in.
    await page.click('button[data-preset="drift"]');
    await settle(page);
    const box = await page.locator('#map').boundingBox();
    await page.mouse.click(box.x + box.width * CLICK.fx, box.y + box.height * CLICK.fy);
    await settle(page);
    const where = (await page.textContent('#point-where'))?.trim();
    console.log(`  clicked ${where}`);
    await page.screenshot({ path: join(out, '06-panel.png') });
    console.log('  06-panel.png');
  } finally {
    await browser.close();
    if (server) await server.close();
  }

  await report(out, problems);
}

/** Move the clock without reaching into the app: set the input, fire its event. */
async function setFrame(page, index) {
  await page.$eval('#time', (el, i) => {
    el.value = String(i);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }, index);
  await settle(page);
}

/*
  Let the canvases catch up. `networkidle` covers the chunk fetch a frame
  change can trigger; the short fixed wait afterwards is for the animation
  frame that paints it, which no event reports. It is 400 ms and it is a
  timing guess -- acceptable in a tool whose output a person looks at, and
  which is why the assertions in tests/e2e/ wait on conditions instead.
*/
async function settle(page) {
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.waitForTimeout(400);
}

async function report(out, problems) {
  const files = (await readdir(out)).filter((f) => f.endsWith('.png')).sort();
  let total = 0;
  for (const f of files) total += (await stat(join(out, f))).size;

  console.log(`\n${files.length} files, ${(total / 1e6).toFixed(1)} MB in ${out}`);
  console.log('These are overwritten by the next run. Nothing accumulates.');

  if (problems.length) {
    console.log(`\n${problems.length} problem(s) on the page:`);
    for (const p of [...new Set(problems)].slice(0, 20)) console.log(`  - ${p}`);
  } else {
    console.log('\nNo page errors, console errors or failed requests.');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
