/**
 * bench.js -- the scenario page: pick a buoy, a drift model, an arrival and a searcher, and
 * watch the probability map drain while the paper's referee scores the flight (D031, Stage 4).
 *
 * Every number on the page is computed here, in the browser, by referee.js replaying the
 * flight over the published cloud, and the paper's own number for the same flight sits
 * beside it. They agree to about 1e-15 (frontend/tests/referee.test.js); if they ever did
 * not, the page would show it.
 *
 * "Fly it yourself" steers a helicopter with the keyboard over the same cloud; the flight
 * is scored by the same referee.
 */

import '@fontsource-variable/inter';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import { PlayerFlight, RecordedFlight } from './playback.js';
import { Scene } from './scene.js';
import { NOISE_INFO, SEARCHER_INFO, loadIndex, loadWindow } from './scenarioData.js';
import { headingFromKeys, keyDirection } from './searchRun.js';
import { esriTiles } from './tiles.js';

const WINDOW_S = 45 * 60;
const ARRIVALS = [1, 2, 3];
const SELF = 'self';
const $ = (id) => document.getElementById(id);

const state = {
  index: null, scenario: null, noise: 'rv', hours: 2, searcher: 'expanding-square',
  window: null, flight: null, t: 0, playing: false, last: 0, held: new Set(),
};

// Whole zoom levels: a fractional zoom scales tiles by a non-integer and leaves hairline seams.
const map = L.map('map', { zoomSnap: 1, zoomControl: true, attributionControl: true });
// Dark grey, not the ocean basemap: viridis separates over a dark sea (main.js), and the
// open ocean is plain at every zoom, so there are no seams where tiles run out.
esriTiles('Canvas/World_Dark_Gray_Base', {
  maxZoom: 16, maxNativeZoom: 16, attribution: 'Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
}).addTo(map);
map.setView([26.5, -72.5], 5);
const scene = new Scene(map);

function percent(x, digits = 0) {
  return `${(100 * x).toFixed(digits)} %`;
}

function clock(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

function straightWords(s) {
  if (s === null || !Number.isFinite(s)) return 'its path is unknown';
  if (s >= 0.99) return 'drifted in an almost straight line';
  if (s >= 0.95) return 'drifted in a gentle curve';
  if (s >= 0.9) return 'turned through about a quarter circle';
  if (s >= 0.7) return 'turned sharply';
  return 'looped back on itself';
}

function segmented(el, options, current, onPick) {
  el.innerHTML = '';
  for (const [value, text] of options) {
    const b = document.createElement('button');
    b.textContent = text;
    b.setAttribute('aria-pressed', String(value === current));
    b.addEventListener('click', () => onPick(value));
    el.append(b);
  }
}

function drawControls() {
  segmented($('noise'), Object.entries(NOISE_INFO).map(([k, v]) => [k, v.label]), state.noise,
    (v) => { state.noise = v; load(); });
  segmented($('arrival'), ARRIVALS.map((h) => [h, `${h} h`]), state.hours,
    (v) => { state.hours = v; load(); });
  const box = $('searchers');
  box.innerHTML = '';
  const options = [...Object.entries(SEARCHER_INFO), ['ml', { label: 'ML agent (coming)', colour: '#555' }],
    [SELF, { label: 'Fly it yourself', colour: '#fff' }]];
  for (const [key, info] of options) {
    const label = document.createElement('label');
    if (key === 'ml') label.className = 'off';
    label.innerHTML = `<input type="radio" name="searcher" value="${key}" ${key === state.searcher ? 'checked' : ''}`
      + `${key === 'ml' ? ' disabled' : ''}><span class="swatch" style="background:${info.colour}"></span>${info.label}`;
    label.querySelector('input').addEventListener('change', () => { state.searcher = key; start(); });
    box.append(label);
  }
  $('keys').hidden = state.searcher !== SELF;
}

let plot = null;

function cumulative(drain) {
  const out = [0];
  for (const d of drain) out.push(out[out.length - 1] + d);
  return out;
}

function drawChart() {
  const minutes = Array.from({ length: 46 }, (_, i) => i);
  const series = [{}];
  const data = [minutes];
  for (const [key, info] of Object.entries(SEARCHER_INFO)) {
    const f = state.window.flights[key];
    data.push(cumulative(f.python.drain_rate).map((v) => 100 * v));
    series.push({ label: info.label, stroke: info.colour, width: key === state.searcher ? 2 : 1, alpha: 0.5 });
  }
  data.push(minutes.map(() => null));
  series.push({ label: 'this flight', stroke: '#ffffff', width: 3 });
  const el = $('chart');
  if (plot) plot.destroy();
  const box = el.getBoundingClientRect();
  plot = new uPlot({
    width: Math.max(200, Math.floor(box.width)),
    height: Math.max(110, Math.floor(box.height)),
    legend: { show: false },
    scales: { x: { time: false, range: [0, 45] }, y: { range: [0, 100] } },
    axes: [
      { stroke: '#8a8a80', grid: { stroke: '#2a2a28' }, label: 'minutes searching', labelSize: 16, size: 30 },
      { stroke: '#8a8a80', grid: { stroke: '#2a2a28' }, values: (u, v) => v.map((x) => `${x}%`), size: 46 },
    ],
    series,
  }, data, el);
}

function updateChart() {
  if (!plot) return;
  const m = state.flight.metrics();
  const live = cumulative(m.drainRate).map((v) => 100 * v);
  const data = plot.data.slice(0, -1);
  data.push(plot.data[0].map((_, i) => (i < live.length ? live[i] : null)));
  plot.setData(data);
}

function render() {
  const { flight, t } = state;
  if (!flight) return;
  flight.advanceTo(t);
  scene.update(t, flight);
  const m = flight.metrics();
  const k = m.drainRate.length;
  $('pos').textContent = percent(m.pos, 1);
  $('drain').textContent = k ? (100 * m.drainRate[k - 1]).toFixed(1) : '0.0';
  $('clock').textContent = `${clock(t)} / ${clock(WINDOW_S)}`;
  $('time').value = String(Math.round(t));
  const done = flight.done;
  const target = m.target;
  if (done && target) {
    $('buoy').textContent = target.found
      ? `Found the real buoy ${clock(target.foundS)} into the search.`
      : `Missed the real buoy: the closest pass was ${(target.closestM / 1000).toFixed(1)} km away.`;
    scene.showBuoy(t);
  } else {
    $('buoy').textContent = done ? '' : 'Where is the real buoy? Revealed at the end.';
    if ($('showBuoy').checked) scene.showBuoy(t); else scene.hideBuoy();
  }
  updateChart();
}

function tick(now) {
  if (!state.playing) return;
  const dt = Math.max(0, Math.min(0.1, (now - state.last) / 1000));
  state.last = now;
  if (state.searcher === SELF) {
    const heading = headingFromKeys(state.held);
    if (heading !== null) state.flight.steer(heading);
  }
  state.t = Math.min(WINDOW_S, state.t + dt * Number($('speed').value));
  render();
  if (state.t >= WINDOW_S) {
    state.playing = false;
    $('play').textContent = 'Again';
    return;
  }
  requestAnimationFrame(tick);
}

function play() {
  if (state.t >= WINDOW_S) start();
  state.playing = !state.playing;
  $('play').textContent = state.playing ? 'Pause' : 'Play';
  state.last = performance.now();
  if (state.playing) requestAnimationFrame(tick);
}

/** A fresh flight over the loaded window, from the helicopter's arrival. */
function start() {
  if (!state.window) return;
  state.playing = false;
  $('play').textContent = 'Play';
  state.t = 0;
  scene.setWindow(state.window, { fit: false });
  if (state.searcher === SELF) {
    state.flight = new PlayerFlight(state.window, { heading: state.window.meta.drift_bearing_deg ?? 0 });
    $('paper').textContent = '–';
    $('time').disabled = true;
  } else {
    state.flight = new RecordedFlight(state.window, state.searcher);
    $('paper').textContent = percent(state.flight.python.pos, 1);
    $('time').disabled = false;
  }
  $('keys').hidden = state.searcher !== SELF;
  drawChart();
  render();
}

let loading = 0;

async function load() {
  // Nothing from the previous choice stays on screen while the next cloud downloads (3.7 MB
  // from Hugging Face), and a download overtaken by a newer choice is dropped when it lands.
  const token = ++loading;
  state.playing = false;
  state.window = null;
  state.flight = null;
  scene.probability.clear();
  scene.heli.remove();
  scene.strip.setLatLngs([]);
  scene.path.setLatLngs([]);
  scene.hideBuoy();
  $('paper').textContent = '–';
  $('pos').textContent = '–';
  $('play').textContent = 'Play';
  drawControls();
  $('status').textContent = 'loading the cloud…';
  let loaded;
  try {
    loaded = await loadWindow(state.scenario.scenario, state.noise, state.hours);
  } catch (err) {
    if (token === loading) $('status').textContent = `could not load this scenario: ${err.message}`;
    return;
  }
  if (token !== loading) return;
  state.window = loaded;
  $('status').textContent = `${state.window.meta.particles.toLocaleString()} possible positions, `
    + `${NOISE_INFO[state.noise].note}`;
  scene.setWindow(state.window);
  start();
}

function pickScenario(name) {
  state.scenario = state.index.scenarios.find((s) => s.scenario === name);
  const s = state.scenario;
  const when = s.start_utc.replace('T', ' ');
  $('facts').textContent = `${s.water} water · called in ${when} UTC · over 4 h the buoy `
    + `${straightWords(s.straightness_4h)}`;
  load();
}

async function main() {
  try {
    state.index = await loadIndex();
  } catch (err) {
    $('status').textContent = `the scenario files are not reachable: ${err.message}`;
    return;
  }
  const select = $('scenario');
  for (const s of state.index.scenarios) {
    const o = document.createElement('option');
    o.value = s.scenario;
    o.textContent = `${s.scenario} · ${s.water} water · ${s.start_utc.slice(0, 10)}`;
    select.append(o);
  }
  select.addEventListener('change', () => pickScenario(select.value));
  $('play').addEventListener('click', play);
  $('time').addEventListener('input', (e) => {
    if (state.searcher === SELF) return;
    const t = Number(e.target.value);
    if (t < state.t) {
      state.flight = new RecordedFlight(state.window, state.searcher);
      scene.setWindow(state.window, { fit: false });
    }
    state.t = t;
    render();
  });
  $('showBuoy').addEventListener('change', render);
  window.addEventListener('keydown', (e) => {
    const d = keyDirection(e.code);
    if (d && state.searcher === SELF) {
      state.held.add(d);
      e.preventDefault();
      if (!state.playing && state.t === 0) play();
    }
  });
  window.addEventListener('keyup', (e) => {
    const d = keyDirection(e.code);
    if (d) state.held.delete(d);
  });
  pickScenario(state.index.scenarios[0].scenario);
}

main();

window.addEventListener('resize', () => {
  if (state.window) {
    drawChart();
    updateChart();
  }
});
