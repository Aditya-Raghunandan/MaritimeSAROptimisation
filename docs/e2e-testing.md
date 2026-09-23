# Browser tests, and seeing the page

Two tools, one Playwright install, opposite jobs. Issue #21.

| | `npm run test:e2e` | `npm run shots` |
|---|---|---|
| Asks | did the page behave as asserted? | what does the page look like? |
| Data | committed fixture, offline | the real published archive |
| Output | pass/fail | six PNGs in a temp directory |
| Fails a build | yes (non-blocking in CI for now) | never |

They are separate because they answer different questions, and conflating them
is how each does the other badly. A test suite reports that *its own
assertions* held — which is not the same as the page being right, and it is
written by whoever also wrote the code, so it is not an independent witness. A
screenshot is: it shows what is there, including what nobody thought to assert.

## Why a browser layer exists at all

The unit suites are the primary ones and stay that way: 292 pytest and 228
vitest, fast, deterministic, none touching the network, covering the maths.

But every frontend defect that has actually reached a rendered page here was an
**integration** defect, invisible to a unit test because it needed a real DOM,
a real Leaflet, a real build or a real network to exist at all:

| What shipped | Why the unit tests were silent |
|---|---|
| The first-paint race (#19). `addTo(map)` → Leaflet's `onAdd` → paint, before anything was fetched. The throw escaped `start()` and left the layer control, slider, play, ruler, rings, click handler and provenance line unwired | `ZarrSource` and `RasterLayer` are each correct alone. The bug is in the **order** two correct modules run in, which only exists once Leaflet is real |
| The temporal dead zone. `axis` referenced above its declaration; every control below the throw silently never wired | Needs the whole module to execute against a DOM |
| `VIRIDIS` never exported — caught only by `npm run build`, because no test imports `raster.js` | A unit suite tests what it imports |
| `leaflet-velocity` painting copies of the study box across the Pacific | Needs a real map at a real zoom |
| CARTO serving watermarked tiles instead of failing | Needs real network requests to be observable |
| The Pages base path. Without the prefix every asset 404s and the page renders blank with nothing on it to say why | Exists only in a **production build** |

The shape is identical every time, and it is this project's named worst failure
mode: **the page renders and is wrong.** Nothing crashes, the map looks
plausible, and the defect is visible only to something that opens the page and
looks at it.

## What the suite asserts

Five checks, in `tests/e2e/site.spec.js`. Each maps to a failure above.

1. **No uncaught errors.** Uncaught exceptions must be empty — that is the
   first-paint race and the dead zone. Console `Failed to load resource` is
   filtered, and only that: opening a Zarr **v3** store begins by probing for
   the **v2** names `.zarray` and `.zattrs`, which 404 before each array opens
   correctly by its `zarr.json`. That is negotiation working, and it happens
   against the real store too.
2. **Every asset the page ships is served** — same-origin only, which is the
   base-path failure precisely. Archive-origin 404s are the legitimate probe
   above; demanding 200 from those would assert that a working client is
   broken, and the first person to see that fail would delete the test.
3. **The wind field paints.** Counts non-transparent pixels on the raster
   canvas. `ZarrSource.vector` throwing on a non-resident frame is correct and
   stays — zeros would paint a calm, plausible, wrong map — but that means "the
   page rendered" and "the data arrived" are different states.
4. **The slider drives the clock.** Asserts the *effect* — the stamp changes —
   not that the element exists. The dead zone left controls that existed and
   did nothing.
5. **Clicking the map fills the point panel.** The path the resultant read-out
   sits on.

**No pixel-diff snapshots, deliberately.** It is the standard tool and the
wrong one here: the particle layers animate on `requestAnimationFrame`, so
full-frame comparison flakes, and a suite that cries wolf gets ignored. An
ignored suite is worse than none, for the same reason a CI pipeline that had
never gone green was worse than none — it reads as evidence.

## The fixture

`tests/e2e/fixtures/` — 24 files, 20 kB, committed. Built by
`scripts/make_e2e_fixture.py`, which runs **the real export pipeline over fake
input**: `make_demo_forcing.py` writes correctly shaped NetCDF (D020
conventions, both grids' real steps, a NaN land wedge), and
`sar.viz.archive.export_archive` — the same function that published the real
archive — turns it into manifests and Zarr stores.

Hand-writing the fixture would have been a second, unverified opinion about
what the published archive looks like, and the first time the exporter changed
shape the suite would go on passing against a store the site no longer reads.
Generating it through the exporter means it cannot drift silently. Same
argument as `export_resultant_golden.py`.

`.gitignore` ignores `*.zarr/` everywhere, so there is a **narrowly scoped
negation** for this one path. Without it CI has no data and the suite passes
while testing nothing.

The values are meaningless — a smooth invented field over a 1° box. The suite
asserts wiring and painting, never numbers. Numbers are the unit suites' job.

Rebuild with `python scripts/make_e2e_fixture.py`. It refuses to write more
than 2 MB.

## Seeing the page

```
npm run shots                    # real archive, five presets + the point panel
npm run shots -- --data local    # offline; frontend/public/data/ is WIND ONLY
npm run shots -- --url https://aditya-raghunandan.github.io/MaritimeSAROptimisation/
```

**Storage is flat by construction.** Six fixed filenames, overwritten every
run, ~2.8 MB total, into the system temp directory — never the repository,
never OneDrive. No timestamps and no run ids, because an accumulating pile of
near-identical screenshots is how a tool like this stops being used. A picture
worth keeping is copied into `figures/` deliberately, with a row in the vault's
MAP.

`--url` matters more than it looks: some things differ between a local build
and production by design — a domain-restricted CARTO key is *supposed* to fail
from localhost — and without being able to shoot both, that is
indistinguishable from a key that has expired.

## Running it

```
npm run test:e2e          # builds, previews, runs the five checks (~8 s)
npx playwright test --ui  # same, with the inspector
```

The browser is pinned: `@playwright/test` is an **exact** version, not a caret
range, because the package and its Chromium build are coupled and a floating
minor would download a second browser and miss the CI cache.

**You do not need any of this to work on the project.** `npm test` is unchanged
and runs no browser. Only run these if you are changing the site. On the
cluster they cannot run at all — no browser, and the shared-host rule.

CI runs the suite on every PR as a **non-blocking** job. That is temporary and
the workflow says so: a required check gates both of us, and this one was
written by one of us.
