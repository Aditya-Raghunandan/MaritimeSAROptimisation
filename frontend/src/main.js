/**
 * main.js -- wires the bundle, the clock and the layers onto a map.
 *
 * Load order: manifest first, then each layer's buffer, then the map. The
 * manifest is the only thing that knows what exists, which is what lets a
 * layer be added later by writing its data and one manifest entry, with no
 * change here.
 */

import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Clock } from './clock.js';
import { buildLayer } from './layers.js';
import { ZarrSource, pickTier, residentSpanOf } from './sources.js';
import { quiverLayer } from './quiver.js';
import { currentLegend, resultantLegend, windLegend } from './legend.js';
import { ResultantSource, resultantScale } from './resultant.js';
import { rasterLayer } from './raster.js';
import { MAGMA } from './colormap.js';
import { CURRENT_RAMP, DRIFT_RAMP } from './style.js';
import { particleLayer } from './particles.js';
import { domainLabel, domainMask } from './domain.js';
import { RangeRings, Ruler, addScaleBar, formatDistance } from './measure.js';
import { PointPanel } from './chart.js';
import { TYPICAL_CURRENT_MS } from './geo.js';

const DATA = import.meta.env.VITE_DATA_BASE ?? 'data';

/*
  THE CONTROL PICKS A SPAN. THE TIER FOLLOWS.

  Choosing a published tier was the wrong handle. What a viewer wants is "show
  me a day" or "show me a year"; which stride the archive happens to publish is
  our implementation detail, and picking it left the slider spanning five years
  at one-hour steps -- 43,824 positions, one pixel of travel worth several
  hours, and no way to step through a single day.

  So the span is chosen and `chooseTier` picks the FINEST tier that fits the
  span into a slider you can actually resolve. A slider is a few hundred pixels
  wide, so more than a few hundred positions buys nothing.

    1 day       -> hourly     24 steps
    3 days      -> hourly     72 steps
    1 week      -> 6-hourly   28 steps
    30 days     -> 6-hourly  120 steps
    1 year      -> daily     365 steps
    everything  -> daily    1826 steps

  THE FLOOR IS ONE HOUR, and it is a property of the data rather than of this
  control. ERA5 publishes hourly and HYCOM 3-hourly; there is no sub-hourly
  forcing to show. Offering a 60-minute view would mean interpolating between
  published hours and presenting the result as observation, which is the same
  thing this project refuses to do when filling HYCOM's real gaps with NaN
  rather than with invented current. Minutes arrive with the drift engine,
  which emits roughly every 15 min (D009) on its own axis -- and the clock
  holds a timestamp precisely so that layer can have a finer cadence than the
  forcing underneath it.
*/
const SPANS = [
  { id: 'day', label: '1 day', seconds: 86400 },
  { id: '3days', label: '3 days', seconds: 3 * 86400 },
  { id: 'week', label: '1 week', seconds: 7 * 86400 },
  { id: 'month', label: '30 days', seconds: 30 * 86400 },
  { id: 'year', label: '1 year', seconds: 365 * 86400 },
  { id: 'all', label: 'whole archive', seconds: null },
];

/** More positions than this and the slider cannot resolve them anyway. */
const MAX_SLIDER_STEPS = 400;

/**
 * The finest published tier that fits `spanSeconds` into a usable slider.
 *
 * Finest-first, so a short span gets the most detail the archive actually
 * holds. Falls back to the coarsest tier when nothing fits, which is what the
 * whole-archive span always does.
 */
function chooseTier(tiers, spanSeconds) {
  const byStride = Object.entries(tiers).sort((a, b) => a[1].step_seconds - b[1].step_seconds);
  if (!spanSeconds) return byStride[byStride.length - 1][0];
  for (const [name, tier] of byStride) {
    if (spanSeconds / tier.step_seconds <= MAX_SLIDER_STEPS) return name;
  }
  return byStride[byStride.length - 1][0];
}

// CARTO's raster basemaps now want a key, and without one they serve a
// watermarked tile. It is read from the environment rather than written here:
// the repository is public, and a key in it is a key published. It still ends
// up in the built bundle -- unavoidable for a client-side basemap, and the
// reason CARTO's protection is a DOMAIN RESTRICTION set on their dashboard
// rather than secrecy. Without a key the CARTO layers are simply left out,
// so the map works for anyone who clones this.
const CARTO_KEY = import.meta.env.VITE_CARTO_KEY ?? '';
const carto = (style) => `https://basemaps.cartocdn.com/rastertiles/${style}/{z}/{x}/{y}.png`
  + `?key=${CARTO_KEY}`;
const CARTO_ATTR = '&copy; OpenStreetMap contributors, &copy; CARTO';

/**
 * Esri Ocean is the default because the bathymetry does real work here: the
 * Gulf Stream follows the shelf edge and separates offshore at Cape Hatteras
 * because the shelf turns away there. Over flat blue that is invisible.
 * Mapbox and MapTiler look better and are refused -- they need an API key, and
 * a key in a public repository is a leak. Attribution is required by Esri's
 * terms and stays.
 */
const BASEMAPS = {
  /*
    Dark first, and it is not only taste. The painted speed field is the thing
    being read now, and viridis runs from near-black to bright yellow: over a
    pale ocean its dark end disappears into the basemap and the ramp loses its
    bottom third. On a dark basemap the whole range separates, which is why
    every wind map that leads with a painted field is dark.

    It also stops the chrome fighting the map -- the panels are #1a1a19 and a
    bright blue ocean between them was the loudest thing on screen.

    Esri Ocean stays, because its bathymetry does real work when the arrows are
    the subject: the Gulf Stream follows the shelf edge and separates at Cape
    Hatteras because the shelf turns away there.
  */
  'Ocean (bathymetry)': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 13, attribution: 'Esri, GEBCO, NOAA, National Geographic, and other contributors' },
  ),
  Satellite: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 17, attribution: 'Esri, Maxar, Earthstar Geographics' },
  ),
};

/*
  CARTO basemaps are added ONLY when a key is present, and that is a behaviour
  choice rather than tidiness. Without a key CARTO still serves the tile -- it
  just stamps a watermark across it. Since the dark CARTO map is the one worth
  defaulting to, a missing key would otherwise mean the site opens watermarked,
  which is the worst of both: it looks broken and it is nobody's fault that is
  visible. Omitting them means a fork, a local checkout with no .env.local, or
  a deploy whose secret was never set all open on Esri Ocean and look
  deliberate.

  Voyager carries its own place names, which is what makes it useful here: the
  domain is open water and the landmarks are how anyone orients.
*/
if (CARTO_KEY) {
  Object.assign(BASEMAPS, {
    'Dark (field first)': L.tileLayer(carto('dark_all'),
      { maxZoom: 19, attribution: CARTO_ATTR }),
    'Light (data first)': L.tileLayer(carto('light_all'),
      { maxZoom: 19, attribution: CARTO_ATTR }),
    'Street (named places)': L.tileLayer(carto('voyager'),
      { maxZoom: 19, attribution: CARTO_ATTR }),
  });
}

/** Dark when we can, bathymetry when we cannot. See above. */
const DEFAULT_BASEMAP = CARTO_KEY ? 'Dark (field first)' : 'Ocean (bathymetry)';

/**
 * Place names, drawn OVER the basemap rather than baked into it.
 *
 * Esri Ocean is beautiful bathymetry and almost unlabelled, so the map gave no
 * answer to "where is that?" -- which matters here, because the domain is open
 * water and the few landmarks (Hatteras, the Bahamas, the Florida Straits) are
 * how anyone orients in it. A separate reference layer keeps the labels when
 * the basemap is switched, instead of needing a labelled twin of each one.
 */
const LABELS = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  { maxZoom: 13, opacity: 0.9 },
);

/**
 * Load the published multi-resolution archive, if there is one.
 *
 * Returns the same `{ manifest, layers }` shape as `loadBundle` so nothing
 * downstream knows which it got -- the difference between "one 48 h window in
 * memory" and "five years fetched a chunk at a time" lives entirely in the
 * layer's source.
 *
 * Which TIER is chosen depends on how much time the archive spans, because
 * nobody can perceive hourly detail while scrubbing across a year and the
 * download is 1.32 GB against 0.07 GB for the same five years.
 */
/**
 * Open one published tier of an archive as a field source.
 *
 * Separate from `loadArchive` because it is called again every time the viewer
 * changes the time resolution: the tiers are the same data at different
 * strides, so switching is opening a different store, not reloading the page.
 */
async function openTier(base, archive, tierName) {
  const tier = archive.tiers[tierName];
  if (!tier) throw new Error(`the archive does not publish a "${tierName}" tier`);

  // A tier published with --allow-gaps cannot have its timestamps
  // reconstructed as start + k * step. Refusing is right: the alternative is a
  // map that confidently labels frames with dates up to weeks out.
  if (tier.regular === false) {
    throw new Error(
      `tier "${tierName}" has an irregular time axis (${tier.gaps.length} gap(s)); ` +
      'the clock cannot reconstruct its timestamps. Re-publish without --allow-gaps.',
    );
  }

  return new ZarrSource(`${base}/${tier.path}`, {
    frames: tier.frames,
    chunks: tier.chunks,
    variables: archive.variables,
    grid: archive.grid,
  }).open();
}

/**
 * Load one published product as a field layer, or null if it is not there.
 *
 * Null rather than throwing: the wind archive is what the clock is built from
 * and the map is useless without it, but the current archive is an overlay.
 * A site that refuses to open because one of two datasets is missing is worse
 * than one that opens and says which it has -- and during the five days
 * between publishing wind and publishing current, that was the live state.
 */
async function loadProduct(base, product, valueRange) {
  let archive;
  try {
    const res = await fetch(`${base}/${product}_archive.json`);
    if (!res.ok) return null;
    archive = await res.json();
  } catch {
    return null;
  }

  const anyTier = Object.values(archive.tiers)[0];
  const spanDays = (new Date(anyTier.end) - new Date(anyTier.start)) / 86400000;
  const tierName = pickTier(archive.tiers, spanDays);
  const tier = archive.tiers[tierName];
  const source = await openTier(base, archive, tierName);

  const layer = buildLayer({
    id: archive.product,
    type: 'field',
    label: archive.label,
    units: archive.units,
    grid: archive.grid,
    value_range: valueRange,
  }, source);

  return {
    archive,
    tierName,
    tier,
    layer,
    axis: {
      start: new Date(tier.start),
      stepSeconds: tier.step_seconds,
      frames: tier.frames,
    },
  };
}

async function loadArchive(base) {
  const res = await fetch(`${base}/wind_archive.json`);
  if (!res.ok) throw new Error(`no archive manifest at ${base}/wind_archive.json`);
  const archive = await res.json();

  const anyTier = Object.values(archive.tiers)[0];
  const spanDays = (new Date(anyTier.end) - new Date(anyTier.start)) / 86400000;
  const tierName = pickTier(archive.tiers, spanDays);
  const tier = archive.tiers[tierName];

  const source = await openTier(base, archive, tierName);

  const [latMin, lonMin, latMax, lonMax] = archive.bbox;
  const endMs = new Date(tier.start).getTime() + tier.frames * tier.step_seconds * 1000;
  const manifest = {
    generated: archive.generated,
    bbox: [latMin, lonMin, latMax, lonMax],
    clock: {
      start: tier.start,
      end: new Date(endMs).toISOString(),
      step_seconds: tier.step_seconds,
      frames: tier.frames,
    },
    layers: [{
      id: archive.product,
      type: 'field',
      label: archive.label,
      units: archive.units,
      grid: archive.grid,
      // Read from the data once the first chunk lands; a fixed ramp would
      // either clip the Gulf Stream or wash out a calm day.
      value_range: [0, 25],
    }],
    provenance: {
      source_file: `${tier.path} (${tierName}, ${tier.compression}, ` +
        `${(tier.bytes / 1e6).toFixed(0)} MB)`,
      arrival_convention: archive.longitude_convention,
      script: 'sar.viz.archive',
    },
    _tier: tierName,
    _archive: archive,
    _base: base,
  };

  /*
    The surface current, as a second field.

    Its own grid (476 x 238 against wind's 77 x 77), its own cadence (3-hourly
    against hourly) and its own tiers -- which is exactly why nothing here
    merges the two onto a common grid. The clock maps a shared moment to each
    layer's own nearest frame, and the resultant samples the current by nearest
    neighbour at each wind cell centre. What must agree is the conventions, not
    the grids.

    Top of scale is 2.5 m/s, not wind's 25: the Gulf Stream core runs about
    1.8 m/s, so on the wind ramp every current arrow would be invisible.
  */
  const current = await loadProduct(base, 'current', [0, 2.5]);

  return { manifest, layers: [buildLayer(manifest.layers[0], source)], current };
}

async function loadBundle(base) {
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const layers = [];
  for (const spec of manifest.layers) {
    let buffer = null;
    if (spec.source) {
      const bytes = await (await fetch(`${base}/${spec.source}`)).arrayBuffer();
      // One call, no parse. This is the whole reason the exporter writes raw
      // float32 rather than JSON: 2.28 MB and an instant view, against roughly
      // 4 MB of text that has to be walked.
      buffer = new Float32Array(bytes);
    }
    layers.push(buildLayer(spec, buffer));
  }
  return { manifest, layers };
}

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

/** "one hour", "three hours", "a day" -- a step length anyone can read. */
function describeStep(stepSeconds) {
  const hours = stepSeconds / 3600;
  if (hours >= 24) return hours === 24 ? 'a day' : `${hours / 24} days`;
  if (hours === 1) return 'one hour';
  return `${hours} hours`;
}

/**
 * The provenance line, which must be rewritten whenever the tier changes --
 * it names the store, its compression and its size, and all three move.
 */
function setProvenance(tierName, tier) {
  const el = document.getElementById('provenance');
  if (!el) return;
  if (!tier) { el.textContent = ''; return; }
  el.textContent = `${tierName} · ${tier.frames.toLocaleString()} frames · `
    + `${describeStep(tier.step_seconds)} per step · ${tier.compression} · `
    + `${(tier.bytes / 1e6).toFixed(0)} MB published`;
}

async function start() {
  // The published Zarr archive is preferred; the flat 48 h bundle is the
  // fallback, and is still the right thing for a single scenario and for
  // working offline.
  let bundle;
  try {
    bundle = await loadArchive(DATA);
  } catch (archiveErr) {
    try {
      bundle = await loadBundle(DATA);
    } catch (bundleErr) {
      setStatus(
        `No data at ${DATA}/. Either publish the archive: ` +
        `python -m sar.viz.archive --data <dir> --out ${DATA}  ` +
        `or export one window: python -m sar.viz.export --data <dir> ` +
        `--start 2021-01-01 --end 2021-01-03 --out ${DATA}`,
      );
      console.error('archive:', archiveErr);
      throw bundleErr;
    }
  }
  const { manifest, layers, current } = bundle;

  const [latMin, lonMin, latMax, lonMax] = manifest.bbox;
  const dataBounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax]);
  const map = L.map('map', {
    center: dataBounds.getCenter(),
    zoom: 5,
    /*
      Fractional zoom. Leaflet steps in whole levels by default, and a whole
      level is a factor of two -- between the 300 km and 100 km views there was
      simply no stop, so the right framing of the domain was not reachable.
      Quarter steps were not enough either -- the useful range for this box
      sits between the 300 km and 200 km views, and a quarter level still
      skipped over it. Eighths give eight stops per level, and the wheel is
      slowed to match so one notch is one eighth rather than a jump.
    */
    zoomSnap: 0.125,
    zoomDelta: 0.125,
    wheelPxPerZoomLevel: 240,
    layers: [BASEMAPS[DEFAULT_BASEMAP]],
    // The study box is 19 deg square. Panning to the Pacific shows nothing and
    // is where every projection problem lives, so the map is held near the
    // data: a generous margin to keep the coastline and the Gulf Stream's exit
    // in frame, and no further.
    // Held close to the study box. A 19 deg domain does not need the globe,
    // and every projection problem we have hit lived out past the edge of it.
    maxBounds: dataBounds.pad(0.12),
    maxBoundsViscosity: 1.0,
    minZoom: 5,
    maxZoom: 11,
    // No wrapped copies of the world, so a layer can never be asked to draw
    // itself at a longitude 360 deg away from its data.
    worldCopyJump: false,
  });
  /*
    Open on a slightly narrower box than the data.

    fitBounds on the full domain leaves a wide screen showing the Gulf of
    Mexico and a lot of Texas either side of a study area that is entirely at
    sea. Trimming 7 % of the longitude on each side for the OPENING view only
    puts the Gulf Stream and the Bahamas across the middle of the frame, which
    is what the map is about. The data, the mask and the domain label are
    untouched -- this is framing, not cropping, and panning still reaches the
    full box.
  */
  const lonInset = (lonMax - lonMin) * 0.07;
  map.fitBounds(L.latLngBounds(
    [latMin, lonMin + lonInset],
    [latMax, lonMax - lonInset],
  ));
  LABELS.addTo(map);

  // Dim the world outside the forcing domain. The painted field is a hard-edged
  // rectangle because that is exactly where the data stops; without this it
  // reads as a loading failure rather than as the study box.
  const mask = domainMask(dataBounds).addTo(map);
  domainLabel(dataBounds, 'Study domain · 17–36 N, 82–63 W').addTo(map);
  addScaleBar(map);

  const clock = Clock.fromManifest(manifest);

  // Declared HERE, before anything reads it. It sat below the layer control
  // and the resultant layer referenced it, which is a temporal dead zone: the
  // ReferenceError threw out of start() and every wiring below that point --
  // the layer control, the slider, the play button, the ruler, the rings, the
  // click handler, the provenance line -- silently never happened. The map and
  // its canvases still rendered, because they are added above the throw, so it
  // looked like a working map with dead controls rather than like a crash.
  const axis = {
    start: new Date(manifest.clock.start),
    stepSeconds: manifest.clock.step_seconds,
    frames: manifest.clock.frames,
  };
  const overlays = {};
  const field = layers.find((l) => l.type === 'field');

  /** The current raster's normal weight, when it is the subject of the view. */
  const CURRENT_RASTER_OPACITY = 0.55;

  /*
    LAYER NAMES THAT SAY WHAT THE LAYER IS.

    The control listed "10 m wind — speed", "— flow" and "— arrows", three times
    over, and those words name the RENDERING rather than the question it
    answers. Somebody who has not read the source cannot tell what "flow" is, or
    why they would want it rather than "arrows", so the list read as three
    arbitrary switches per product and people left it alone. Leaflet writes the
    name straight into the label, so a second line costs only this helper.
  */
  const entry = (name, hint) => `<span class="lc-n">${name}</span><span class="lc-h">${hint}</span>`;

  /*
    SWITCHING A LAYER ON MUST ALSO BRING IT TO THE CURRENT MOMENT.

    The current's chunk is fetched only inside `redraw()`, and `redraw()` ran
    only when the CLOCK changed -- deliberately, because the 3-hourly tier is
    4.3 GB and nothing should download it for a layer that is switched off. But
    that left the other half unhandled: turning a layer ON is also a moment at
    which its data may be missing, and nothing triggered the fetch.

    So a current layer switched on mid-session painted whatever frame it last
    held -- frame 0 from the start-up warm-up -- and went on painting it until
    somebody moved the slider. The map showed 1 January under a clock reading
    13 January, with nothing on screen to say so. The point panel found this the
    day it started reporting the resultant: a stale frame is invisible in a
    field of colour and obvious in a number.
  */
  map.on('overlayadd overlayremove', () => { redraw(); });

  /*
    A LEGEND IS SHOWN ONLY WHILE THE THING IT DESCRIBES IS ON THE MAP.

    The current's key already worked this way and the wind's did not -- it was
    added once and never removed -- so the Clean preset, whose layer list is
    literally empty, still displayed a wind key, and the Drift view displayed
    the WIND key over resultant arrows scaled to 2.5 m/s against its 25 m/s.
    Anyone reading arrow length off it was wrong by a factor of ten, in the one
    view this project exists to produce.
  */
  const bindLegend = (legend, layersOf) => {
    const sync = () => {
      const anyOn = layersOf().some((l) => l && map.hasLayer(l));
      if (anyOn && !legend._map) legend.addTo(map);
      else if (!anyOn && legend._map) map.removeControl(legend);
    };
    map.on('overlayadd overlayremove', sync);
    return sync;
  };

  /*
    THE FIRST FRAME BEFORE THE FIRST PAINT, and this await is load-bearing.

    Every renderer below draws from Leaflet's `onAdd`, which `addTo(map)` calls
    synchronously. On the Zarr path nothing is resident until a chunk has been
    fetched, and `ZarrSource.vector` throws on a non-resident frame rather than
    returning zeros -- correctly, because a calm field is a plausible, wrong
    picture. Without this line that throw escaped `start()` and the layer
    control, the slider, play, the ruler, the rings, the click handler and the
    provenance line were never wired: a map that renders with dead controls.

    It never showed on the flat-bundle path, where `BufferSource` is always
    resident, so making Zarr the default turned a latent ordering assumption
    into a crash. The renderers now also refuse to paint a non-resident frame
    (see `isFrameReady`), but that is the backstop; this is the fix.
  */
  if (field) {
    try {
      await field.ensure(clock.frameOf(axis));
    } catch (err) {
      // Wire the page anyway. A tool that works over a blank field is more
      // use than a dead page, and the status line says what went wrong.
      setStatus(`Could not load the first frame: ${err.message}`);
    }
  }

  // Our own renderer, not leaflet-velocity. That library indexes its grid with
  // floorMod(lon, 360) against raw map bounds, so at low zoom it painted copies
  // of our box across the Pacific, and it rebuilds on a 750 ms debounce, so a
  // pan smeared the previous frame across the new position. Both were visible
  // on 2026-09-18 and neither is reachable from outside the library.
  // src/quiver.js explains why this one cannot do either.
  let quiver = null;
  let raster = null;
  let particles = null;
  let currentRaster = null;
  let currentParticles = null;
  let currentQuiver = null;
  if (field) {
    /*
      Three renderings of the same field, because they answer different
      questions and a good weather map uses all three:

        raster     WHERE the wind is strong -- continuous, shows the shape
        particles  THAT it is moving, and which way it turns
        arrows     WHAT the value is at a real cell centre

      Raster and particles are on by default: together they are what makes
      the map read as a flow field rather than a lattice. The arrow grid is
      off by default now -- it was the only rendering, and as the only one it
      had to carry all three jobs badly.
    */
    /*
      0.38 was a compromise for stacking two rasters, and the presets removed
      the need for it -- no preset paints both, because two full-domain rasters
      is mud. At 0.38 viridis over a dark basemap is not a field, it is a green
      tint over the sea: the ramp's own contrast is thrown away and what reads
      is the average, not the structure. Back up to where the pattern is the
      thing you see.
    */
    raster = rasterLayer(field, { maxSpeed: field.valueRange[1], opacity: 0.62 });
    // Thin, fast, short-lived wisps. More of them, because wind structure is
    // spread across the whole domain rather than concentrated in one jet.
    particles = particleLayer(field, {
      maxSpeed: field.valueRange[1],
      count: 900,
      trail: 7,
      maxAgeMs: 3400,
      alpha: 0.7,
      width: 0.8,
    });
    quiver = quiverLayer(field, { maxSpeed: field.valueRange[1] });

    raster.addTo(map);
    particles.addTo(map);
    quiver.addTo(map);

    overlays[entry(`${field.label} — colour`, 'how strong, painted everywhere')] = raster;
    overlays[entry(`${field.label} — streaks`, 'which way it is blowing, animated')] = particles;
    overlays[entry(`${field.label} — arrows`, 'the value at each real cell centre')] = quiver;

    bindLegend(windLegend({ maxSpeed: field.valueRange[1] }), () => [raster, particles, quiver]);
  }

  /*
    The surface current, rendered the same three ways and OFF by default.

    Off because wind and current painted on top of each other are two flow
    fields in one frame and neither is readable; the viewer turns on the one
    they are asking about. The resultant layer below is the one that shows
    them combined, which is the honest way to see both at once.

    Its renderers are the same functions as wind's -- the whole reason
    FieldLayer carries its own grid is so a renderer never has to know which
    product it was handed, and this is the first time two products prove it.
  */
  if (current) {
    /*
      DELIBERATELY NOT THE SAME LOOK AS THE WIND.

      The two were painted identically and could not be told apart at a glance,
      which on a projector across a room is the only glance anyone gets. They
      are different quantities on different grids at different cadences, and
      they now differ in every channel a viewer reads:

        ramp     magma against wind's viridis -- opposite ends of the
                 sequential-ramp space, and magma's near-black low end lets the
                 slow two thirds of the box recede so the JET GLOWS
        arrows   cyan against wind's amber, and drawn 45 % heavier, because
                 weight survives greyscale and colour-blindness where hue does
                 not -- the same redundancy that already encodes speed as both
                 colour and length
        streaks  cyan, FEWER, SLOWER, longer-lived, longer-trailed. Physically
                 honest rather than decorative: a western-boundary current is
                 slower than the wind above it but far more persistent and
                 laminar, so long coherent streaks are what it looks like.
    */
    const cMax = current.layer.valueRange[1];
    // The current's raster carries a touch more weight than the wind's: most
    // of it is near-black by construction, so what actually paints is the jet.
    currentRaster = rasterLayer(current.layer,
      { maxSpeed: cMax, ramp: MAGMA, opacity: CURRENT_RASTER_OPACITY });
    // Thick, slow, long-lived ribbons -- roughly half as many as the wind's,
    // twice the stroke weight, three times the trail and four times the life.
    // A boundary current is slower than the wind above it and far more
    // persistent, so a long coherent ribbon is what it actually looks like.
    currentParticles = particleLayer(current.layer, {
      maxSpeed: cMax,
      rgb: [140, 240, 255],
      count: 340,
      trail: 26,
      maxAgeMs: 14000,
      alpha: 0.9,
      width: 1.75,
    });
    currentQuiver = quiverLayer(current.layer, {
      maxSpeed: cMax, ramp: CURRENT_RAMP, weight: 1.45,
    });

    // Its own key, shown only while a current layer is on. Two legends stacked
    // permanently would take a quarter of the map to explain a layer that is
    // off by default.
    bindLegend(currentLegend({ maxSpeed: cMax }),
      () => [currentRaster, currentQuiver, currentParticles]);

    overlays[entry(`${current.layer.label} — colour`, 'how fast, painted; land left blank')] = currentRaster;
    overlays[entry(`${current.layer.label} — streaks`, 'which way the water sets, animated')] = currentParticles;
    overlays[entry(`${current.layer.label} — arrows`, 'the value at each real cell centre')] = currentQuiver;

    // Same rule as the wind field: make the first frame resident before any
    // renderer can be added, so switching the layer on never paints a frame
    // that is not there.
    try {
      await current.layer.ensure(clock.frameOf(current.axis));
    } catch (err) {
      setStatus(`Surface current did not load: ${err.message}`);
    }
  }

  // The remaining three types have no data yet. They are listed as disabled so
  // the map says what is coming rather than pretending it is complete.
  overlays[entry('Place names', 'coastlines and towns, over the basemap')] = LABELS;
  overlays[entry('Dim outside the domain', 'shade everything beyond 17-36 N, 82-63 W')] = mask;

  /*
    The resultant drift field: what a person in the water would actually
    follow, rather than what the wind is doing. Issue #10, rendered.

    It is built NOW, with current = null, so it runs as the leeway term alone
    and labels itself that way. When the current archive is published the only
    change is passing a second source -- the arithmetic, the layer, the
    renderer and the legend are already the ones that will be used. A dead
    checkbox reserving the name would have proved nothing.
  */
  let resultant = null;
  let resultantFlow = null;
  // Hoisted: the point panel needs it too. `sampleAt` is the bilinear,
  // golden-tested path and is what the click read-out should answer from,
  // rather than a second copy of the same arithmetic in chart.js.
  let resultantSource = null;
  if (field) {
    /*
      GETTERS, NOT VALUES, and do not "simplify" them back. `applySpan` swaps
      `field.source` and `current.layer.source` onto another published tier
      whenever the span changes, and the page itself does this once at start-up
      (it opens on the daily tier, then switches to hourly for the default day).
      `source: field.source` copied the daily store in and kept it: the drift
      arrows read daily data with hourly frame numbers, painted the wrong day,
      and froze past hour 48 while the clock ran on. Reproduced on the real
      archive 23 Sep; see resultant.js.
    */
    const source = new ResultantSource(
      { get source() { return field.source; }, grid: field.grid, axis },
      // The second argument, at last. Everything else about this layer was
      // already the code that would be used -- passing it is the whole change,
      // and `isPartial` flips to false on its own, so the caveat the UI shows
      // stops saying the current is missing without anyone editing the wording.
      current
        ? { get source() { return current.layer.source; }, grid: current.layer.grid, axis: current.axis }
        : null,
    );
    resultantSource = source;
    const meta = source.describe();
    const scale = resultantScale(field.valueRange[1], !source.isPartial);

    /*
      ITS OWN COLOUR, AND IT HAS TO BE A THIRD ONE. This layer is neither of its
      parents, and drawn in either one's colour it reads as that parent with the
      other quietly added. Green at 148 deg of OKLab hue sits about 85 deg from
      both the wind's amber and the current's cyan. Heavier again than the
      current's arrows: on the Drift view this is the subject, not the context.
    */
    resultant = quiverLayer(
      { grid: field.grid, vector: (f, j, i) => source.vector(f, j, i), source },
      { maxSpeed: scale, ramp: DRIFT_RAMP, weight: 1.7 },
    );
    resultant._resultantMeta = meta;

    /*
      STREAKS FOR THE RESULTANT TOO, because arrows answer "what is the value
      here" and motion answers "where does this go" -- and the second is the
      question the layer exists for. Tuned between its two parents: longer-lived
      and thicker than the wind's wisps, because the resultant is dominated by
      the current and inherits its persistence; fewer and shorter than the
      current's ribbons, so switching Current to Drift reads as a change rather
      than as the same picture in another colour.
    */
    resultantFlow = particleLayer(
      { grid: field.grid, vector: (f, j, i) => source.vector(f, j, i), source },
      {
        maxSpeed: scale,
        rgb: [90, 245, 135],
        count: 460,
        trail: 18,
        maxAgeMs: 9000,
        alpha: 0.85,
        width: 1.4,
      },
    );

    overlays[entry(`${meta.label} — arrows`, 'current + leeway: where a person goes')] = resultant;
    overlays[entry(`${meta.label} — streaks`, 'the path they would be carried along')] = resultantFlow;

    // Its own key at last. Until now the Drift view showed the WIND legend,
    // keyed to 25 m/s over arrows that top out near 2.5 -- wrong by a factor of
    // ten, in the view that is the point of the project.
    bindLegend(resultantLegend({ maxSpeed: scale }), () => [resultant, resultantFlow]);

    // No frame is set here. `redraw()` sets it, and only once both the wind and
    // the current for that moment are resident -- see the end of `redraw()`.
    // Switching the layer on already triggers a redraw (the overlayadd handler
    // above), which is what brings it to the current moment.
    map.on('overlayadd', (e) => {
      if (e.layer !== resultant && e.layer !== resultantFlow) return;
      setStatus(`${meta.label} — ${meta.caveat}`);
    });
    map.on('overlayremove', (e) => {
      if (e.layer === resultant || e.layer === resultantFlow) setStatus('');
    });
  }

  for (const pending of ['Probability map', 'Search tracks']) {
    overlays[entry(pending, 'awaiting the engine')] = L.layerGroup();
  }
  L.control.layers(BASEMAPS, overlays, { collapsed: true }).addTo(map);

  /*
    VIEW PRESETS.

    The layer control is a flat list of eleven checkboxes, three of which are
    disabled placeholders. It is the right tool for "turn that one thing off"
    and the wrong one for the question people actually ask, which is "show me
    the wind" or "show me where a person would drift". Getting from one of
    those to another is four clicks in the right order, and on a touchscreen in
    front of an audience nobody is going to do it.

    So the named combinations are one button each, and the checkbox list stays
    underneath for anything a preset does not cover. Presets do not replace the
    control; they are the shortcut through it.

    Each preset is a SET of layers to end up with, not a diff to apply. Diffs
    accumulate error -- press two presets in a row and the state depends on the
    order -- while a set is idempotent and can be pressed from anywhere.
  */
  const PRESETS = [
    {
      id: 'wind',
      label: 'Wind',
      hint: 'what the air is doing',
      layers: () => [raster, particles, quiver],
    },
    {
      id: 'current',
      label: 'Current',
      hint: 'what the water is doing',
      layers: () => [currentRaster, currentParticles, currentQuiver],
    },
    {
      id: 'both',
      label: 'Wind + water',
      hint: 'the two together',
      // Not every renderer of each: two rasters is mud and two sets of arrows
      // is a lattice. The current carries the colour, the wind carries the
      // motion, and that reads as one picture rather than two competing ones.
      layers: () => [currentRaster, particles, quiver],
    },
    {
      id: 'drift',
      label: 'Drift',
      hint: 'where a person would go',
      // The answer the project exists to give. Current underneath for context,
      // resultant arrows on top; no wind marks, because the wind is already
      // inside the resultant as the leeway term and drawing it twice invites
      // the reader to add it twice.
      layers: () => [currentRaster, resultantFlow, resultant],
    },
    {
      id: 'clean',
      label: 'Map only',
      hint: 'no data layers',
      layers: () => [],
    },
  ];

  const ALL_FIELD_LAYERS = () => [
    raster, particles, quiver,
    currentRaster, currentParticles, currentQuiver,
    resultant, resultantFlow,
  ].filter(Boolean);

  function applyPreset(preset) {
    const want = new Set(preset.layers().filter(Boolean));
    for (const layer of ALL_FIELD_LAYERS()) {
      const on = map.hasLayer(layer);
      if (want.has(layer) && !on) map.addLayer(layer);
      else if (!want.has(layer) && on) map.removeLayer(layer);
    }
    /*
      In the Drift view the current raster is CONTEXT, not the subject, and at
      its normal weight it wins: the resultant's arrows and streaks are thin
      marks over a bright jet and the eye goes to the colour. Dropping it makes
      the green legible without removing the thing it is drifting on. Set here
      rather than baked in, because the same raster IS the subject in the
      Current and Wind + water views and must keep its weight there.
    */
    if (currentRaster) {
      currentRaster.setOpacity(preset.id === 'drift' ? 0.26 : CURRENT_RASTER_OPACITY);
    }

    for (const btn of presetBar.querySelectorAll('button')) {
      btn.classList.toggle('on', btn.dataset.preset === preset.id);
    }
    setStatus(`${preset.label} — ${preset.hint}.`);
  }

  const presetBar = L.DomUtil.create('div', 'preset-bar');
  L.DomEvent.disableClickPropagation(presetBar);
  for (const preset of PRESETS) {
    const btn = L.DomUtil.create('button', '', presetBar);
    btn.type = 'button';
    btn.dataset.preset = preset.id;
    /*
      THE HINT GOES ON THE BUTTON, not in a tooltip. Each preset already carried
      a line saying what it shows and it was reachable only by hovering -- no
      help on the touchscreen these were built for, and none to anyone scanning
      the bar deciding which to press. Five nouns do not say what any will do.
    */
    btn.innerHTML = `<span class="preset-n">${preset.label}</span>`
      + `<span class="preset-h">${preset.hint}</span>`;
    btn.addEventListener('click', () => applyPreset(preset));
  }

  const PresetControl = L.Control.extend({
    options: { position: 'topright' },
    onAdd() { return presetBar; },
  });
  new PresetControl().addTo(map);

  /*
    THE PRESET BAR TAKES THE TOP OF ITS CORNER.

    Both controls live top-right and Leaflet stacks that corner in the order
    controls are added, so the layer list -- added first -- sat above the
    presets and pushed them down whenever it expanded. Once the list carried two
    lines per entry that was far enough to shove the preset bar into the middle
    of the map, over the legend.

    Reordered here, after both exist, rather than by moving the code: both
    `PRESETS` and `PresetControl` are `const`, so building the bar earlier would
    reference them above their declarations -- a ReferenceError that would
    unwire everything below it in `start()`, which is precisely what the
    temporal dead zone bug did to this file once already.
  */
  presetBar.parentNode.insertBefore(presetBar, presetBar.parentNode.firstChild);

  /*
    Touching a checkbox directly means no preset describes what is on screen
    any more, so none of them should look selected. Saying "this is the Wind
    view" over something that is not it is worse than saying nothing.
  */
  map.on('overlayadd overlayremove', () => {
    const current_ = PRESETS.find((preset) => {
      const want = new Set(preset.layers().filter(Boolean));
      return ALL_FIELD_LAYERS().every((l) => want.has(l) === map.hasLayer(l));
    });
    for (const btn of presetBar.querySelectorAll('button')) {
      btn.classList.toggle('on', Boolean(current_) && btn.dataset.preset === current_.id);
    }
  });

  applyPreset(PRESETS[0]);

  const slider = document.getElementById('time');
  clock.setWindowSpan(null);   // replaced below once the span control is wired
  slider.max = String(clock.steps - 1);
  slider.addEventListener('input', () => clock.setIndex(Number(slider.value)));

  /*
    TIME RESOLUTION. The archive is published at several strides of the same
    data, and until now the client picked one from the total span and gave the
    viewer no say -- five years spans 1,826 days, so it always chose `daily`
    and playback jumped a day at a time with no way to look inside one.

    Switching tier is opening a different store, not reloading the page. The
    clock holds a TIMESTAMP, so the moment survives the change and every layer
    re-derives its own frame from it; only the slider's granularity changes.
    An index-based clock would land on 1/24th of the intended date here.

    The cost is stated rather than hidden: the hourly tier is 1.27 GB against
    72 MB for daily, but only the chunks actually scrubbed through are ever
    fetched, so the honest number to show is the chunk size, not the tier size.
  */
  const windowLabel = document.getElementById('window-label');
  const winBack = document.getElementById('win-back');
  const winFwd = document.getElementById('win-fwd');

  /** Re-point the slider at the clock's current window. */
  function syncSlider() {
    slider.max = String(clock.steps - 1);
    slider.value = String(clock.index);
    if (windowLabel) windowLabel.textContent = clock.windowLabel();
    if (winBack) winBack.disabled = !clock.canShift(-1);
    if (winFwd) winFwd.disabled = !clock.canShift(1);
  }

  async function shiftWindow(n) {
    if (!clock.shiftWindow(n)) return;
    syncSlider();
    await redraw();
  }

  if (winBack) winBack.addEventListener('click', () => shiftWindow(-1));
  if (winFwd) winFwd.addEventListener('click', () => shiftWindow(1));

  /*
    THE SPAN CONTROL.

    One handle: how much time the slider covers. The tier follows from it, so
    the viewer never has to know what a "6-hourly tier" is -- they ask for a
    week and get the finest stride the archive can serve a week at.

    Switching is opening a different store, not reloading the page, and the
    moment survives because the clock holds a timestamp. Only the stride and
    the reach change.
  */
  const spanSelect = document.getElementById('span');
  const archive = manifest._archive;

  /** Roughly what one sliderful costs to fetch, in chunks and megabytes. */
  function fetchCost(tier, steps) {
    const chunks = Math.max(1, Math.ceil(steps / tier.chunks.time));
    const perChunk = tier.chunk_bytes_uncompressed / (tier.compression_ratio || 1);
    return { chunks, mb: (chunks * perChunk) / 1e6 };
  }

  async function applySpan(spanSeconds, { quiet = false } = {}) {
    if (!archive || !archive.tiers) return;
    const name = chooseTier(archive.tiers, spanSeconds);
    const tier = archive.tiers[name];

    if (name !== manifest._tier) {
      const source = await openTier(manifest._base, archive, name);
      // Mutated in place, not replaced: the resultant layer and the clock
      // listeners closed over this object when they were wired. The resultant
      // reads `field.source` through a getter for the same reason.
      axis.start = new Date(tier.start);
      axis.stepSeconds = tier.step_seconds;
      axis.frames = tier.frames;
      field.source = source;
      manifest._tier = name;
    }

    /*
      THE CURRENT FOLLOWS THE SPAN TOO. It used to be opened once, at the tier
      suited to the whole five-year archive -- daily -- and never switched, so
      the default one-day view drew hourly wind over a single daily current
      snapshot, and the drift arrows snapped to the next day's current at noon.
      Same rule as the wind: the finest tier that fits the slider, which for the
      current is 3-hourly up to 30 days and daily beyond.
    */
    if (current && current.archive && current.archive.tiers) {
      const cName = chooseTier(current.archive.tiers, spanSeconds);
      if (cName !== current.tierName) {
        const cTier = current.archive.tiers[cName];
        const cSource = await openTier(manifest._base, current.archive, cName);
        current.axis.start = new Date(cTier.start);
        current.axis.stepSeconds = cTier.step_seconds;
        current.axis.frames = cTier.frames;
        current.layer.source = cSource;
        current.tierName = cName;
        current.tier = cTier;
      }
    }

    clock.setStep(tier.step_seconds);
    clock.setWindowSpan(spanSeconds);
    syncSlider();
    setProvenance(name, tier);

    await field.ensure(clock.frameOf(axis));
    await redraw();

    if (!quiet) {
      const cost = fetchCost(tier, clock.steps);
      setStatus(`${clock.windowLabel()} — ${clock.steps} steps of `
        + `${describeStep(tier.step_seconds)}, about `
        + `${cost.mb < 1 ? `${Math.round(cost.mb * 1000)} kB` : `${cost.mb.toFixed(1)} MB`} `
        + `over ${cost.chunks} chunk${cost.chunks === 1 ? '' : 's'}.`);
    }
  }

  if (spanSelect && archive && archive.tiers) {
    const finest = Math.min(...Object.values(archive.tiers).map((t) => t.step_seconds));
    spanSelect.innerHTML = '';
    for (const sp of SPANS) {
      const name = chooseTier(archive.tiers, sp.seconds);
      const stride = archive.tiers[name].step_seconds;
      const opt = document.createElement('option');
      opt.value = String(sp.seconds ?? '');
      opt.textContent = `${sp.label} · ${describeStep(stride)} steps`;
      spanSelect.appendChild(opt);
    }
    // Open on a day: the finest view the archive supports, which is what
    // someone arriving at a drift map is most likely to want to look at.
    spanSelect.value = '86400';

    spanSelect.addEventListener('change', async () => {
      const raw = spanSelect.value;
      const spanSeconds = raw === '' ? null : Number(raw);
      spanSelect.disabled = true;
      setStatus('switching …');
      try {
        await applySpan(spanSeconds);
      } catch (err) {
        setStatus(`could not switch: ${err.message}`);
      } finally {
        spanSelect.disabled = false;
      }
    });

    // The floor is the data's, not the control's, and it is worth saying once.
    if (finest > 3600) {
      console.info(`finest published stride is ${describeStep(finest)}`);
    }
  }

  /*
    Play through the window.
    Each tick AWAITS the redraw rather than firing on a fixed interval, so
    playback slows down when a chunk has to be fetched instead of racing ahead
    of the data and showing stale frames. setTimeout, not setInterval, for the
    same reason: the next tick is scheduled only once this one has drawn.
  */
  const playBtn = document.getElementById('play');
  let playing = false;
  let playTimer = null;

  function setPlaying(on) {
    playing = on;
    playBtn.innerHTML = on ? '&#10073;&#10073;' : '&#9654;';
    playBtn.classList.toggle('on', on);
    playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
    if (on) tick();
  }

  async function tick() {
    if (!playing) return;
    if (clock.index + 1 >= clock.steps) {
      /*
        Off the end of the window. Advance into the next one rather than
        looping, so play means "time passes" at every tier -- an hourly view
        that looped the same 24 h forever would be a toy. At the end of the
        archive there is nowhere to advance to, so it wraps to the start,
        which is the old behaviour where it still applies.
      */
      if (!clock.shiftWindow(1)) clock.setIndex(0);
      else clock.setIndex(0);
      syncSlider();
    } else {
      clock.setIndex(clock.index + 1);
      slider.value = String(clock.index);
    }
    await redraw();
    if (playing) playTimer = setTimeout(tick, 110);
  }

  playBtn.addEventListener('click', () => setPlaying(!playing));
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
      e.preventDefault();
      setPlaying(!playing);
    }
  });

  let pinned = null;

  // The clicked CELL, not the clicked pixel. The data is a 0.25 deg average,
  // so showing a pinpoint would imply a precision the field does not have.
  const highlight = L.layerGroup().addTo(map);
  function markCell(cell) {
    highlight.clearLayers();
    if (!cell || !field) return;
    const g = field.grid;
    const half = [g.dlat / 2, g.dlon / 2];
    const c = [g.lat(cell.j), g.lon(cell.i)];
    L.rectangle(
      [[c[0] - half[0], c[1] - half[1]], [c[0] + half[0], c[1] + half[1]]],
      { color: '#3987e5', weight: 2, fillColor: '#3987e5', fillOpacity: 0.16 },
    ).addTo(highlight);
    L.circleMarker(c, { radius: 3, color: '#fff', weight: 1.5, fillColor: '#3987e5', fillOpacity: 1 })
      .addTo(highlight);
  }

  const panel = new PointPanel({
    root: document.getElementById('point'),
    chart: document.getElementById('chart'),
    oceanChart: document.getElementById('chart-ocean'),
    onClose: () => { pinned = null; highlight.clearLayers(); setStatus(''); },
  });

  // Scrubbing can outrun the network, so only the newest request may draw.
  // Without this, chunks landing out of order repaint an older frame over a
  // newer one and the map runs backwards under the slider.
  let drawToken = 0;
  let driftToken = 0;

  async function redraw() {
    const mine = ++drawToken;
    const frame = clock.frameOf(axis);
    document.getElementById('stamp').textContent = clock.label();

    /*
      The current is on its own axis, so it gets its own frame from the shared
      moment rather than reusing the wind's index. Wind is hourly and current
      3-hourly; reusing the index would run the current at a third speed and
      three times behind, and it would look entirely plausible while doing it.

      Only fetched when a layer that needs it is on the map. The 3-hourly tier
      is 4.3 GB, and nobody should download a chunk of it to render a layer that
      is switched off.

      AND ITS FETCH STARTS NOW, alongside the wind's, not after it. One chunk is
      about 2.7 s from Hugging Face (measured 23 Sep: ~1.4 s of that before the
      first byte), and awaiting the two one after the other made every jump a
      ~5 s wait. An error is caught into a value here and reported below, so a
      rejected fetch cannot go unhandled while the wind is still loading.
    */
    const shown = current
      ? [currentRaster, currentQuiver, currentParticles].filter((l) => l && map.hasLayer(l))
      : [];
    const drift = [resultant, resultantFlow].filter((l) => l && map.hasLayer(l));
    const needCurrent = Boolean(current && (shown.length || drift.length));
    const cFrame = current ? clock.frameOf(current.axis) : null;
    const currentJob = needCurrent && !current.layer.isResident(cFrame)
      ? current.layer.ensure(cFrame).then(() => null, (err) => err)
      : Promise.resolve(null);

    if (field && !field.isResident(frame)) {
      setStatus(`loading …`);
      try {
        await field.ensure(frame);
      } catch (err) {
        setStatus(`could not load that time: ${err.message}`);
        return;
      }
      if (mine !== drawToken) return;      // the clock moved on; let the newer draw win
      setStatus('');
    }

    if (field) {
      if (map.hasLayer(raster)) raster.setFrame(frame);
      if (map.hasLayer(quiver)) quiver.setFrame(frame);
      particles.setFrame(frame);
    }

    if (needCurrent) {
      const err = await currentJob;
      if (mine !== drawToken) return;
      if (err) setStatus(`current: ${err.message}`);
      for (const l of shown) l.setFrame(cFrame);
    }

    /*
      THE DRIFT LAYERS DRAW LAST, once BOTH of their inputs are resident. They
      used to be handed the new frame straight from the clock, before either
      chunk had arrived: the arrows found nothing to draw, kept the previous
      hour, and nothing told them again when the data landed. Paused on a new
      day, or switched on while paused, the drift view stayed blank or stale
      until somebody moved the slider. It is the bug #37 fixed for the current
      layers, one list over. Found on the live site, 23 Sep.
    */
    if (resultantSource && drift.length && resultantSource.isResident(frame)) {
      for (const l of drift) l.setFrame(frame);
    }

    if (playing) prefetchAhead(frame, needCurrent ? cFrame : null);
    if (pinned) showSeries(pinned);
  }

  /*
    FETCH THE NEXT CHUNK BEFORE PLAYBACK REACHES IT. Playback awaits each
    redraw, so without this it stalled ~2.7 s at every chunk boundary. A chunk
    is 48 frames of wind (two days at hourly) and 8 frames of current (one day
    at 3-hourly, eight at daily), so at the default span the current is the
    one crossing boundaries daily. Only while playing: while scrubbing it would
    fetch chunks nobody reaches and evict ones they will.
  */
  function prefetchAhead(frame, cFrame) {
    if (field) field.source?.prefetchNext?.(frame);
    if (current && cFrame !== null) current.layer.source?.prefetchNext?.(cFrame);
  }
  clock.onChange(() => {
    slider.value = String(clock.index);
    if (windowLabel) windowLabel.textContent = clock.windowLabel();
    redraw();
  });

  function showSeries(latlng) {
    if (!field) return;
    const cell = field.grid.cellAt(latlng.lat, latlng.lng);
    if (!cell) {
      setStatus('Outside the data box.');
      return;
    }
    pinned = latlng;
    markCell(cell);
    const frame = clock.frameOf(axis);
    if (!field.isResident(frame)) return;

    const [u, v] = field.vector(frame, cell.j, cell.i);

    /*
      Plot what is actually loaded, not the whole tier.

      This used to ask for `axis.frames` -- 1,826 at the daily tier, 43,824 at
      hourly -- when only the cached chunks have values. The chart drew a
      sliver of real data against four empty years, and the panel reported a
      window mean "over 192 frames" without saying which 192. Both were honest
      and neither was legible.

      The series is still free: whatever is resident already holds this cell at
      every one of its timesteps, so it is the same memory read along a
      different axis. What changed is that the x-range now matches it.
    */
    /*
      Sample the current at the same place, on its own grid and its own frame.

      Nearest neighbour at the wind cell centre, the same rule ResultantSource
      uses, so the panel and the resultant arrow can never disagree about what
      the current is doing here. Null when the archive is absent or the frame
      has not been fetched -- the panel says which, rather than showing a zero.
    */
    let currentAt = null;
    let currentSeries = null;
    let currentAxis = null;
    let currentCursor = 0;
    if (current) {
      const cFrame = clock.frameOf(current.axis);
      const cCell = current.layer.grid.cellAt(field.grid.lat(cell.j), field.grid.lon(cell.i));
      if (cCell && current.layer.isResident(cFrame)) {
        const [cu, cv] = current.layer.vector(cFrame, cCell.j, cCell.i);
        currentAt = { u: cu, v: cv };

        // The current's own series, over ITS resident span on ITS axis. Not
        // the wind's: the two are on different cadences and different chunk
        // boundaries, so what is loaded of one says nothing about the other.
        const cSpan = residentSpanOf(current.layer, cFrame, current.axis.frames);
        currentSeries = current.layer.seriesAt(
          cCell.j, cCell.i, cSpan.to - cSpan.from, cSpan.from,
        );
        currentAxis = {
          start: new Date(current.axis.start.getTime()
            + cSpan.from * current.axis.stepSeconds * 1000),
          stepSeconds: current.axis.stepSeconds,
          frames: cSpan.to - cSpan.from,
        };
        currentCursor = cFrame - cSpan.from;
      }
    }

    const span = residentSpanOf(field, frame, axis.frames);
    const series = field.seriesAt(cell.j, cell.i, span.to - span.from, span.from);
    const spanAxis = {
      start: new Date(axis.start.getTime() + span.from * axis.stepSeconds * 1000),
      stepSeconds: axis.stepSeconds,
      frames: span.to - span.from,
    };

    /*
      SYNCHRONOUS IF THE DATA IS ALREADY HERE, FETCHED IF IT IS NOT.

      `sampleAt` reads the current at the frame the RESULTANT maps this moment
      to, which is not always the chunk the visible layer happens to hold, so it
      can legitimately throw `not resident` on a page where the arrows are drawn
      and the ocean column is populated. That is what it did the first time this
      panel ran, and the honest reading is that a click may need data the map
      did not.

      So: try it now, show the panel immediately either way, and if it was not
      resident, fetch and fill the block in when it lands. Blocking the panel on
      a round trip would make every click feel broken; leaving it blank lies.
      `ensure` is ResultantSource's own, so it fetches the frames it will
      actually read rather than the ones this file would have guessed at.
    */
    const mine = ++driftToken;
    let drift = null;
    if (resultantSource && resultantSource.isResident(frame)) {
      try {
        drift = resultantSource.sampleAt(frame, latlng.lat, latlng.lng);
      } catch {
        drift = null;
      }
    }

    panel.show({
      drift,
      lat: field.grid.lat(cell.j),
      lon: field.grid.lon(cell.i),
      u,
      v,
      series,
      axis: spanAxis,
      cursor: frame - span.from,
      when: clock.label(),
      // The MEASURED current at this cell if the archive is loaded, falling
      // back to the Gulf Stream typical. Comparing leeway against a constant
      // 1.8 m/s was right while nothing better existed and is wrong now that
      // the real field is one nearest-neighbour lookup away -- the whole point
      // of the comparison is whether leeway matters HERE.
      currentSpeed: currentAt && Number.isFinite(currentAt.u)
        ? Math.hypot(currentAt.u, currentAt.v) : TYPICAL_CURRENT_MS,
      currentAt,
      currentSeries,
      currentAxis,
      currentCursor,
    });
    setStatus('');

    // Not resident: fetch, then fill the block in -- but only if this click is
    // still the one on screen, so an older answer cannot land on a newer point.
    if (resultantSource && !drift) {
      resultantSource.ensure(frame)
        .then(() => {
          if (mine !== driftToken) return;
          panel.showDrift(resultantSource.sampleAt(frame, latlng.lat, latlng.lng));
        })
        .catch(() => {});
    }
  }

  const ruler = new Ruler(map, {
    onChange: ({ legs, total, driftHours }) => {
      if (!legs.length) {
        setStatus('Ruler — click a second point to measure a leg.');
        return;
      }
      const last = legs[legs.length - 1];
      setStatus(
        `last leg ${formatDistance(last.distance)} on ${last.bearing.toFixed(0)}° `
        + `· total ${formatDistance(total)} `
        + `· ${driftHours.toFixed(1)} h adrift at 1.8 m/s`,
      );
    },
  });

  /*
    THE RULER AND THE RINGS CAN NOW BE ON TOGETHER.

    They were mutually exclusive, and the reason was real: each registered its
    own map click handler, so with both on a single click added a ruler point
    AND moved the rings AND was swallowed before the ruler's readout ran --
    three owners for one click. Exclusivity fixed the click by throwing away
    the case people actually want, which is measuring a leg against rings that
    stay on the map while you do it.

    What was missing is the state between on and off: DRAWN BUT NOT LISTENING.
    Both tools can now be on; at most one is ARMED, and arming is what owns the
    click. Turning a tool on arms it and disarms the other without erasing it.
    Turning the armed tool off hands the click back to the other if it is still
    on, rather than leaving a visible tool that quietly ignores you.

    The rings keep working while disarmed -- the centre handle drags on its own
    mousedown and +/- resize on a keydown, neither of which is the map click --
    so they act as a scale reference while the ruler owns the pointer.
  */
  const ringControls = document.getElementById('ring-controls');
  const ringRadius = document.getElementById('ring-radius');

  const rings = new RangeRings(map, {
    onChange: ({ centre, radiiKm }) => {
      if (!centre) return;
      ringRadius.textContent = `${radiiKm[radiiKm.length - 1]} km`;
      setStatus(
        `Range rings at ${Math.abs(centre.lat).toFixed(2)} ${centre.lat >= 0 ? 'N' : 'S'}, `
        + `${Math.abs(centre.lng).toFixed(2)} ${centre.lng >= 0 ? 'E' : 'W'} `
        + `— ${radiiKm.join(', ')} km. Click to move the datum, drag the dot, `
        + `or use −/+ . The Gulf Stream covers ~155 km/day.`,
      );
    },
  });

  const rulerBtn = document.getElementById('ruler');
  const ringsBtn = document.getElementById('rings');

  // Which tool owns a map click: 'ruler', 'rings', or null for the point panel.
  let armed = null;

  function arm(which) {
    armed = which;
    ruler.setArmed(which === 'ruler');
    rings.setArmed(which === 'rings');
    // The button that owns the click reads as active; a tool that is on but
    // not listening is shown as merely present, so the map never looks like it
    // is ignoring a control that appears pressed.
    rulerBtn.classList.toggle('armed', armed === 'ruler');
    ringsBtn.classList.toggle('armed', armed === 'rings');
  }

  /** Whichever tool is still on takes the click back. */
  function rearmSurvivor(justTurnedOff) {
    if (justTurnedOff !== armed) return;
    if (justTurnedOff !== 'ruler' && ruler.active) arm('ruler');
    else if (justTurnedOff !== 'rings' && rings.active) arm('rings');
    else arm(null);
  }

  function toolStatus() {
    if (armed === 'ruler') {
      return 'Ruler has the click — click two or more points. '
        + (rings.active ? 'Rings stay on the map; drag the dot or use −/+.' : '');
    }
    if (armed === 'rings') {
      return 'Rings have the click — click to move the datum, drag the dot, or use −/+. '
        + (ruler.active ? 'The ruler stays drawn; press Ruler to measure again.' : '');
    }
    return '';
  }

  rulerBtn.addEventListener('click', () => {
    const on = ruler.toggle();
    rulerBtn.classList.toggle('on', on);
    if (on) arm('ruler'); else rearmSurvivor('ruler');
    setStatus(toolStatus());
  });

  ringsBtn.addEventListener('click', () => {
    const on = rings.toggle();
    ringsBtn.classList.toggle('on', on);
    ringControls.hidden = !on;
    if (on) arm('rings'); else rearmSurvivor('rings');
    setStatus(toolStatus());
  });
  document.getElementById('ring-bigger').addEventListener('click', () => rings.rescale(1.25));
  document.getElementById('ring-smaller').addEventListener('click', () => rings.rescale(0.8));

  map.on('click', (e) => {
    // The ARMED tool owns the click and handles its own readout. A tool that
    // is on but disarmed is drawn only, so the point panel is still what a
    // click means when nothing is armed.
    if (armed) return;
    showSeries(e.latlng);
  });

  setProvenance(manifest._tier, manifest._archive && manifest._archive.tiers
    ? manifest._archive.tiers[manifest._tier] : null);

  // Open on one day at the finest stride the archive serves, rather than on
  // five years of daily steps -- the default view should be the one that shows
  // the data at its real resolution.
  try {
    await applySpan(86400, { quiet: true });
  } catch (err) {
    setStatus(`could not open the default span: ${err.message}`);
  }

  syncSlider();
  await redraw();
  setStatus('Click anywhere for a time series at that cell.');
}

/*
  A throw inside start() used to leave a map that rendered and a UI that did
  nothing, with the reason only in the console. Anything that stops setup part
  way now says so on the page, because a half-wired interface looks like a
  design decision rather than a crash.
*/
start().catch((err) => {
  console.error(err);
  const el = document.getElementById('status');
  if (el) {
    el.textContent = `Setup failed: ${err.message}. The map may be partly wired — `
      + 'see the browser console.';
    el.style.color = '#f08a89';
  }
});
