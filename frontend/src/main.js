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
  let bundle;
  try {
    bundle = await loadBundle(DATA);
  } catch (err) {
    setStatus(`No bundle at ${DATA}/. Run: python -m sar.viz.export --data <dir> ` +
      `--start 2021-01-01 --end 2021-01-03 --out frontend/public/data`);
    throw err;
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

  function redraw() {
    document.getElementById('stamp').textContent = clock.label();
    if (field && velocity) velocity.setData(field.velocityFrame(clock.frameOf(axis)));
    if (pinned) showSeries(pinned);
  }
  clock.onChange(redraw);

  function showSeries(latlng) {
    if (!field) return;
    const cell = field.grid.cellAt(latlng.lat, latlng.lng);
    if (!cell) {
      setStatus('Outside the data box.');
      return;
    }
    pinned = latlng;
    // Free: the loaded buffer already holds this cell at every timestep, so
    // the series is the same array read along a different axis. No request.
    const series = field.seriesAt(cell.j, cell.i, axis.frames);
    chart.show(series, axis, clock.frameOf(axis));
    setStatus(
      `${field.label} at ${field.grid.lat(cell.j).toFixed(2)} N, ` +
      `${Math.abs(field.grid.lon(cell.i)).toFixed(2)} W - ` +
      `now ${series[clock.frameOf(axis)].toFixed(1)} m/s, ` +
      `window max ${Math.max(...series).toFixed(1)} m/s`,
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

  redraw();
  setStatus('Click anywhere for a time series at that cell.');
}

start();
