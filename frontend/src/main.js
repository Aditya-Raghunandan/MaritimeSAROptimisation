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
import 'leaflet-velocity';
import 'leaflet-velocity/dist/leaflet-velocity.css';

import { Clock } from './clock.js';
import { buildLayer } from './layers.js';
import { ZarrSource, pickTier } from './sources.js';
import { Ruler, addScaleBar, formatDistance, rangeRings } from './measure.js';
import { PointChart } from './chart.js';

const DATA = import.meta.env.VITE_DATA_BASE ?? 'data';

/**
 * Esri Ocean is the default because the bathymetry does real work here: the
 * Gulf Stream follows the shelf edge and separates offshore at Cape Hatteras
 * because the shelf turns away there. Over flat blue that is invisible.
 * Mapbox and MapTiler look better and are refused -- they need an API key, and
 * a key in a public repository is a leak. Attribution is required by Esri's
 * terms and stays.
 */
const BASEMAPS = {
  'Ocean (bathymetry)': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 13, attribution: 'Esri, GEBCO, NOAA, National Geographic, and other contributors' },
  ),
  'Light (data first)': L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors, &copy; CARTO' },
  ),
  Satellite: L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 17, attribution: 'Esri, Maxar, Earthstar Geographics' },
  ),
};

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
  const map = L.map('map', {
    center: [(latMin + latMax) / 2, (lonMin + lonMax) / 2],
    zoom: 5,
    layers: [BASEMAPS['Ocean (bathymetry)']],
  });
  map.fitBounds([[latMin, lonMin], [latMax, lonMax]]);
  addScaleBar(map);

  const clock = Clock.fromManifest(manifest);
  const overlays = {};
  const field = layers.find((l) => l.type === 'field');

  // leaflet-velocity draws the animated particle streaks. If it ever becomes a
  // problem -- it is not actively maintained -- a canvas quiver over the same
  // buffer is about eighty lines and we would own it. The data does not change
  // either way, which is why that stayed a late decision rather than an early one.
  let velocity = null;
  if (field) {
    velocity = L.velocityLayer({
      displayValues: true,
      displayOptions: {
        velocityType: field.label,
        displayPosition: 'bottomleft',
        displayEmptyString: 'no data here',
        speedUnit: 'm/s',
      },
      data: field.velocityFrame(0),
      minVelocity: field.valueRange[0],
      maxVelocity: field.valueRange[1],
      velocityScale: 0.01,
    });
    velocity.addTo(map);
    overlays[field.label] = velocity;
  }

  // The remaining three types have no data yet. They are listed as disabled so
  // the map says what is coming rather than pretending it is complete.
  for (const pending of ['Surface current', 'Drift particles', 'Probability map', 'Search tracks']) {
    overlays[`${pending} (awaiting data)`] = L.layerGroup();
  }
  L.control.layers(BASEMAPS, overlays, { collapsed: false }).addTo(map);

  const axis = {
    start: new Date(manifest.clock.start),
    stepSeconds: manifest.clock.step_seconds,
    frames: manifest.clock.frames,
  };

  const slider = document.getElementById('time');
  slider.max = String(clock.steps - 1);
  slider.addEventListener('input', () => clock.setIndex(Number(slider.value)));

  const chart = new PointChart(document.getElementById('chart'));
  let pinned = null;

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

    if (field && velocity) velocity.setData(field.velocityFrame(frame));
    if (pinned) showSeries(pinned);
  }
  clock.onChange(() => { redraw(); });

  function showSeries(latlng) {
    if (!field) return;
    const cell = field.grid.cellAt(latlng.lat, latlng.lng);
    if (!cell) {
      setStatus('Outside the data box.');
      return;
    }
    pinned = latlng;
    // Still free: whatever is resident already holds this cell at every one of
    // its timesteps, so the series is the same memory read along a different
    // axis. Over the flat bundle that is the whole window; over Zarr it is the
    // loaded chunks, and the rest comes back NaN so the chart draws a gap
    // rather than a flat line through zero, which would read as calm weather.
    const frame = clock.frameOf(axis);
    const series = field.seriesAt(cell.j, cell.i, axis.frames);
    chart.show(series, axis, frame);

    const known = Array.from(series).filter((x) => Number.isFinite(x));
    const now = series[frame];
    setStatus(
      `${field.label} at ${field.grid.lat(cell.j).toFixed(2)} N, ` +
      `${Math.abs(field.grid.lon(cell.i)).toFixed(2)} W - ` +
      `now ${Number.isFinite(now) ? `${now.toFixed(1)} m/s` : 'not loaded'}, ` +
      `max ${Math.max(...known).toFixed(1)} m/s over the ` +
      `${known.length} frame(s) in memory`,
    );
  }

  const ruler = new Ruler(map);
  let rings = null;

  document.getElementById('ruler').addEventListener('click', (e) => {
    const on = ruler.toggle();
    e.target.classList.toggle('on', on);
    setStatus(on ? 'Ruler on - click points on the map. Click the button again to clear.' : '');
  });

  document.getElementById('rings').addEventListener('click', (e) => {
    if (rings) {
      map.removeLayer(rings);
      rings = null;
      e.target.classList.remove('on');
      return;
    }
    rings = rangeRings(map, map.getCenter());
    e.target.classList.add('on');
    setStatus('Range rings at the map centre. The Gulf Stream covers ~155 km/day.');
  });

  map.on('click', (e) => {
    if (ruler.active) {
      const { legs, total, driftHours } = ruler.summary();
      if (legs.length) {
        const last = legs[legs.length - 1];
        setStatus(
          `leg ${formatDistance(last.distance)} bearing ${last.bearing.toFixed(0)} deg | ` +
          `total ${formatDistance(total)} | ${driftHours.toFixed(1)} h of drift at 1.8 m/s`,
        );
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

start();
