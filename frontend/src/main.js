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
import { ZarrSource, pickTier } from './sources.js';
import { quiverLayer } from './quiver.js';
import { windLegend } from './legend.js';
import { ResultantSource, resultantScale } from './resultant.js';
import { rasterLayer } from './raster.js';
import { particleLayer } from './particles.js';
import { domainLabel, domainMask } from './domain.js';
import { RangeRings, Ruler, addScaleBar, formatDistance } from './measure.js';
import { PointPanel } from './chart.js';
import { TYPICAL_CURRENT_MS } from './geo.js';

const DATA = import.meta.env.VITE_DATA_BASE ?? 'data';

// CARTO's raster basemaps now want a key, and without one they serve a
// watermarked tile. It is read from the environment rather than written here:
// the repository is public, and a key in it is a key published. It still ends
// up in the built bundle -- unavoidable for a client-side basemap, and the
// reason CARTO's protection is a DOMAIN RESTRICTION set on their dashboard
// rather than secrecy. Without a key the CARTO layers are simply left out,
// so the map works for anyone who clones this.
const CARTO_KEY = import.meta.env.VITE_CARTO_KEY ?? '';
const carto = (style) => `https://basemaps.cartocdn.com/rastertiles/${style}/{z}/{x}/{y}.png`
  + (CARTO_KEY ? `?key=${CARTO_KEY}` : '');
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
  'Dark (field first)': L.tileLayer(carto('dark_all'),
    { maxZoom: 19, attribution: CARTO_ATTR }),
  'Ocean (bathymetry)': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 13, attribution: 'Esri, GEBCO, NOAA, National Geographic, and other contributors' },
  ),
  'Light (data first)': L.tileLayer(carto('light_all'),
    { maxZoom: 19, attribution: CARTO_ATTR }),
  // Voyager carries its own place names, which is what makes it useful here:
  // the domain is open water and the landmarks are how anyone orients.
  'Street (named places)': L.tileLayer(carto('voyager'),
    { maxZoom: 19, attribution: CARTO_ATTR }),
  Satellite: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 17, attribution: 'Esri, Maxar, Earthstar Geographics' },
  ),
};

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
async function loadArchive(base) {
  const res = await fetch(`${base}/wind_archive.json`);
  if (!res.ok) throw new Error(`no archive manifest at ${base}/wind_archive.json`);
  const archive = await res.json();

  const anyTier = Object.values(archive.tiers)[0];
  const spanDays = (new Date(anyTier.end) - new Date(anyTier.start)) / 86400000;
  const tierName = pickTier(archive.tiers, spanDays);
  const tier = archive.tiers[tierName];

  // A tier published with --allow-gaps cannot have its timestamps
  // reconstructed as start + k * step. Refusing is right: the alternative is a
  // map that confidently labels frames with dates up to weeks out.
  if (tier.regular === false) {
    throw new Error(
      `tier "${tierName}" has an irregular time axis (${tier.gaps.length} gap(s)); ` +
      'the clock cannot reconstruct its timestamps. Re-publish without --allow-gaps.',
    );
  }

  const source = await new ZarrSource(`${base}/${tier.path}`, {
    frames: tier.frames,
    chunks: tier.chunks,
    variables: archive.variables,
    grid: archive.grid,
  }).open();

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
  };

  return { manifest, layers: [buildLayer(manifest.layers[0], source)] };
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
  const { manifest, layers } = bundle;

  const [latMin, lonMin, latMax, lonMax] = manifest.bbox;
  const dataBounds = L.latLngBounds([latMin, lonMin], [latMax, lonMax]);
  const map = L.map('map', {
    center: dataBounds.getCenter(),
    zoom: 5,
    layers: [BASEMAPS['Dark (field first)']],
    // The study box is 19 deg square. Panning to the Pacific shows nothing and
    // is where every projection problem lives, so the map is held near the
    // data: a generous margin to keep the coastline and the Gulf Stream's exit
    // in frame, and no further.
    // Held close to the study box. A 19 deg domain does not need the globe,
    // and every projection problem we have hit lived out past the edge of it.
    maxBounds: dataBounds.pad(0.25),
    maxBoundsViscosity: 1.0,
    minZoom: 5,
    maxZoom: 11,
    // No wrapped copies of the world, so a layer can never be asked to draw
    // itself at a longitude 360 deg away from its data.
    worldCopyJump: false,
  });
  map.fitBounds(dataBounds);
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
  if (field) windLegend({ maxSpeed: field.valueRange[1] }).addTo(map);

  // Our own renderer, not leaflet-velocity. That library indexes its grid with
  // floorMod(lon, 360) against raw map bounds, so at low zoom it painted copies
  // of our box across the Pacific, and it rebuilds on a 750 ms debounce, so a
  // pan smeared the previous frame across the new position. Both were visible
  // on 2026-09-18 and neither is reachable from outside the library.
  // src/quiver.js explains why this one cannot do either.
  let quiver = null;
  let raster = null;
  let particles = null;
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
    raster = rasterLayer(field, { maxSpeed: field.valueRange[1] });
    particles = particleLayer(field, { maxSpeed: field.valueRange[1] });
    quiver = quiverLayer(field, { maxSpeed: field.valueRange[1] });

    raster.addTo(map);
    particles.addTo(map);
    quiver.addTo(map);

    overlays[`${field.label} — speed`] = raster;
    overlays[`${field.label} — flow`] = particles;
    overlays[`${field.label} — arrows`] = quiver;
  }

  // The remaining three types have no data yet. They are listed as disabled so
  // the map says what is coming rather than pretending it is complete.
  overlays['Place names'] = LABELS;
  overlays['Dim outside the domain'] = mask;

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
  if (field) {
    const source = new ResultantSource(
      { source: field.source, grid: field.grid, axis },
      null,                       // currents: awaiting the HYCOM publish
    );
    const meta = source.describe();
    const scale = resultantScale(field.valueRange[1], !source.isPartial);
    resultant = quiverLayer(
      { grid: field.grid, vector: (f, j, i) => source.vector(f, j, i), source },
      { maxSpeed: scale },
    );
    resultant._resultantMeta = meta;
    overlays[meta.label] = resultant;

    // Keep it on the same clock as everything else even while hidden, so
    // switching it on shows the current moment rather than frame zero.
    clock.onChange(() => { if (map.hasLayer(resultant)) resultant.setFrame(clock.frameOf(axis)); });

    map.on('overlayadd', (e) => {
      if (e.layer !== resultant) return;
      resultant.setFrame(clock.frameOf(axis));
      setStatus(`${meta.label} — ${meta.caveat}`);
    });
    map.on('overlayremove', (e) => { if (e.layer === resultant) setStatus(''); });
  }

  for (const pending of ['Drift particles', 'Probability map', 'Search tracks']) {
    overlays[`${pending} (awaiting the engine)`] = L.layerGroup();
  }
  L.control.layers(BASEMAPS, overlays, { collapsed: true }).addTo(map);

  const slider = document.getElementById('time');
  slider.max = String(clock.steps - 1);
  slider.addEventListener('input', () => clock.setIndex(Number(slider.value)));

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
    const next = clock.index + 1 >= clock.steps ? 0 : clock.index + 1;
    clock.setIndex(next);
    slider.value = String(next);
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
    onClose: () => { pinned = null; highlight.clearLayers(); setStatus(''); },
  });

  // Scrubbing can outrun the network, so only the newest request may draw.
  // Without this, chunks landing out of order repaint an older frame over a
  // newer one and the map runs backwards under the slider.
  let drawToken = 0;

  async function redraw() {
    const mine = ++drawToken;
    const frame = clock.frameOf(axis);
    document.getElementById('stamp').textContent = clock.label();

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
    if (pinned) showSeries(pinned);
  }
  clock.onChange(() => { slider.value = String(clock.index); redraw(); });

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
    // Still free: whatever is resident already holds this cell at every one of
    // its timesteps, so the series is the same memory read along a different
    // axis. Unloaded frames come back NaN and are drawn as a gap, not joined.
    const series = field.seriesAt(cell.j, cell.i, axis.frames);

    panel.show({
      lat: field.grid.lat(cell.j),
      lon: field.grid.lon(cell.i),
      u,
      v,
      series,
      axis,
      cursor: frame,
      when: clock.label(),
      currentSpeed: TYPICAL_CURRENT_MS,
    });
    setStatus('');
  }

  const ruler = new Ruler(map);

  document.getElementById('ruler').addEventListener('click', (e) => {
    const on = ruler.toggle();
    e.target.classList.toggle('on', on);
    setStatus(on
      ? 'Ruler on — click two or more points. Each leg is labelled on the map.'
      : '');
  });

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

  document.getElementById('rings').addEventListener('click', (e) => {
    const on = rings.toggle();
    e.target.classList.toggle('on', on);
    // Resizing lives on buttons, not the wheel: scroll already means zoom on a
    // map, and taking it over stopped the map zooming at all while rings were
    // on -- swapping one gesture for another instead of adding one.
    ringControls.hidden = !on;
    if (!on) setStatus('');
  });
  document.getElementById('ring-bigger').addEventListener('click', () => rings.rescale(1.25));
  document.getElementById('ring-smaller').addEventListener('click', () => rings.rescale(0.8));

  map.on('click', (e) => {
    // The ruler and the rings each own the click while they are on, so a
    // measurement does not also fire the point panel underneath it.
    if (rings.active) return;
    if (ruler.active) {
      const { legs, total, driftHours } = ruler.summary();
      // The distance goes ON THE MAP, beside the leg it measures. Reporting it
      // only into the status line meant the one number the tool exists to
      // produce was the easiest thing on the page to overlook.
      ruler.label(legs, total);
      if (legs.length) {
        const last = legs[legs.length - 1];
        setStatus(
          `last leg ${formatDistance(last.distance)} on ${last.bearing.toFixed(0)}° `
          + `· total ${formatDistance(total)} `
          + `· ${driftHours.toFixed(1)} h adrift at 1.8 m/s`,
        );
      } else {
        setStatus('Ruler — click a second point to measure a leg.');
      }
      return;
    }
    showSeries(e.latlng);
  });

  document.getElementById('provenance').textContent =
    `${manifest.provenance.source_file} | ${manifest.clock.frames} frames | ` +
    `built ${manifest.generated.slice(0, 10)}` +
    (manifest.provenance.git_sha ? ` | ${manifest.provenance.git_sha}` : '');

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
