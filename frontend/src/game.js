/**
 * game.js -- "Can you survive the Bermuda Triangle?", the open-day game (D031, Stage 5).
 *
 * A visitor picks a real buoy (one of the 55 scenarios), chooses to see the probability map
 * or only the currents, and flies one helicopter for the 45-minute window, the clock
 * compressed so the search takes about 90 seconds. Then the same sea is searched again,
 * side by side, by the Coast Guard's Expanding Square and by the AI (the greedy search until
 * the trained agent exists, and labelled so), and the real buoy is revealed.
 *
 * FLYING IT. The helicopter turns as the real one does, at most 7 degrees a second (D032),
 * so it swings round on a 379 m curve instead of snapping to a new heading. Three ways to
 * say where to go, whichever was used last wins:
 *   the mouse      it flies toward the pointer, and circles it once there (a click takes off);
 *   a controller   the left stick or the d-pad points the way (the Gamepad API: a PS5 or Xbox
 *                  pad, or any home-made stick that shows up as a USB game controller);
 *   the keys       arrows or WASD, a compass direction each, two for a diagonal.
 *
 * FAIR BY CONSTRUCTION. All three are scored by referee.js, the browser's copy of the
 * paper's referee, over the same published cloud: the rivals' flights are the paper's own,
 * replayed, and the player's flight is a heading record Python's replay_policy scores the
 * same way (frontend/tests/playback.test.js). Nothing here is tuned to make anyone win.
 */

import '@fontsource-variable/inter';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { FieldLayer } from './fieldLayer.js';
import { add, load, nickname, score, top, verdict } from './leaderboard.js';
import { PlayerFlight, RecordedFlight, SAMPLE_S } from './playback.js';
import { Scene } from './scene.js';
import {
  SEARCHER_INFO, SHOWCASE_ROOT, bestNoise, display, loadBenchmark, loadField, loadIndex,
  loadWindow,
} from './scenarioData.js';
import { bearing } from './kinematics.js';
import { M_PER_DEG_LAT } from './geo.js';
import { headingFromKeys, keyDirection } from './searchRun.js';
import { esriTiles } from './tiles.js';

// The drift model the game flies over: the newest a scenario has (rvc, sized by the current,
// D033, once published; rv until then). Chosen per scenario in fly().
const ARRIVAL_H = 2;
const WINDOW_S = 45 * 60;
// ?speed=10 plays everything ten times faster: for rehearsing the stand and for the tests.
const SPEED_UP = Math.max(1, Number(new URLSearchParams(window.location.search).get('speed')) || 1);
const FLY_SPEED = 30 * SPEED_UP;          // 45 minutes in 90 seconds
const REVEAL_SPEED = 90 * SPEED_UP;       // the side-by-side replay in 30 seconds
const RIVALS = { cg: 'expanding-square', ai: 'greedy' };
const MODES = { map: 'with the probability map', currents: 'with currents only' };
const $ = (id) => document.getElementById(id);

const game = {
  index: null, scenario: null, mode: 'map', name: 'Player', window: null, field: null,
  player: null, held: new Set(), t: 0, last: 0, running: false, result: null,
  input: 'keys', pointer: null, pad: null,
};

const HINT = 'Point with the mouse where you want to fly and click to take off. The helicopter '
  + 'turns like the real one, 7\u00b0 a second, so plan your turns. '
  + 'Arrow keys, WASD or a game controller work too.';

/* ---------------------------------------------------------------- steering */

/** The compass heading from the helicopter now to a point on the map, [lat, display lon]. */
function headingTo(latlng) {
  const [lat, lon] = game.player.position(game.t);
  const dLon = latlng.lng - display(lon);
  const east = dLon * M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180);
  const north = (latlng.lat - lat) * M_PER_DEG_LAT;
  return Math.hypot(east, north) < 1 ? null : bearing(east, north);
}

/** The first connected game controller's stick or d-pad as a heading, or null. */
function padHeading() {
  const pads = typeof navigator.getGamepads === 'function' ? navigator.getGamepads() : [];
  for (const pad of pads) {
    if (!pad) continue;
    const [x = 0, y = 0] = pad.axes;
    if (Math.hypot(x, y) > 0.45) return bearing(x, -y);
    const b = (i) => Boolean(pad.buttons[i]?.pressed);
    const held = new Set([b(12) && 'up', b(13) && 'down', b(14) && 'left', b(15) && 'right'].filter(Boolean));
    const h = headingFromKeys(held);
    if (h !== null) return h;
    if (pad.buttons.some((btn) => btn.pressed)) return undefined;   // pressed, but no direction
  }
  return null;
}

/** What the player is asking for now, from whichever input they used last. */
function wanted() {
  const pad = padHeading();
  if (pad !== null) {
    game.input = 'pad';
    return pad ?? null;
  }
  if (game.input === 'mouse' && game.pointer) return headingTo(game.pointer);
  if (game.input === 'keys') return headingFromKeys(game.held);
  return null;
}

function takeOff() {
  if (game.running || game.t !== 0 || !$('fly').classList.contains('on')) return;
  game.running = true;
  game.last = performance.now();
  $('hint').style.display = 'none';
  requestAnimationFrame(flyTick);
}

/** Before take-off, a controller is only seen by asking it, so ask every frame. */
function waitForPad() {
  if (!$('fly').classList.contains('on') || game.running || game.t !== 0) return;
  if (padHeading() !== null) {
    game.input = 'pad';
    takeOff();
    return;
  }
  requestAnimationFrame(waitForPad);
}

function basemap() {
  return esriTiles('Canvas/World_Dark_Gray_Base', {
    maxZoom: 16, maxNativeZoom: 16, attribution: 'Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
  });
}

function newMap(id, options = {}) {
  // Whole zoom levels: a fractional zoom scales tiles by a non-integer and leaves hairline seams.
  const map = L.map(id, { zoomSnap: 1, attributionControl: false, ...options });
  basemap().addTo(map);
  return map;
}

function show(id) {
  for (const s of document.querySelectorAll('.screen')) s.classList.toggle('on', s.id === id);
}

function clock(s) {
  const r = Math.max(0, Math.round(s));
  return `${Math.floor(r / 60)}:${String(r % 60).padStart(2, '0')}`;
}

const pct = (x) => `${(100 * x).toFixed(1)} %`;

/* ---------------------------------------------------------------- attract */

function drawBoards() {
  const list = load();
  for (const [mode, el] of [['map', $('boardMap')], ['currents', $('boardCurrents')]]) {
    const best = top(list, mode, 5);
    el.innerHTML = best.length
      ? best.map((e) => `<li><span>${e.name}</span><span>${e.score}</span></li>`).join('')
      : '<li class="muted">nobody yet</li>';
  }
}

function attract() {
  drawBoards();
  // The two 3D loops (scripts/render_showcase.py), one after the other.
  const video = $('showcase');
  const loops = ['mountain', 'cube'].map((n) => `${SHOWCASE_ROOT}/${n}.webm`);
  let i = 0;
  video.loop = false;
  video.onended = () => { i = (i + 1) % loops.length; video.src = loops[i]; video.play().catch(() => {}); };
  video.onerror = () => { video.style.visibility = 'hidden'; };
  video.src = loops[0];
  show('attract');
}

/* ---------------------------------------------------------------- pick */

let pickMap = null;
let pickDots = null;

function pick() {
  show('pick');
  if (!pickMap) {
    pickMap = newMap('pickMap');
    pickDots = L.layerGroup().addTo(pickMap);
    const colours = { jet: '#eb6834', moderate: '#e8b23d', quiet: '#3987e5' };
    const pts = [];
    for (const s of game.index.scenarios) {
      const dot = L.circleMarker([s.lat, display(s.lon)], {
        radius: 7, color: '#fff', weight: 1.5, fillColor: colours[s.water] ?? '#999', fillOpacity: 0.9,
      }).addTo(pickDots);
      dot.bindTooltip(`${s.scenario}: ${s.water} water, ${s.start_utc.slice(0, 10)}`);
      dot.on('click', () => choose(s));
      pts.push([s.lat, display(s.lon)]);
    }
    pickMap.fitBounds(L.latLngBounds(pts).pad(0.1));
  }
  setTimeout(() => pickMap.invalidateSize(), 0);
}

function choose(s) {
  game.scenario = s;
  const words = { jet: 'in the Gulf Stream, the fastest water', moderate: 'in moderate current', quiet: 'in quiet water' };
  $('pickInfo').textContent = `${s.start_utc.replace('T', ' ')} UTC, ${words[s.water] ?? ''}. `
    + 'The helicopter gets there 2 hours after the call.';
  $('goBtn').disabled = false;
}

/* ---------------------------------------------------------------- fly */

let flyMap = null;
let flyScene = null;
let fieldLayer = null;

async function fly() {
  game.name = nickname($('name').value);
  $('status').textContent = 'loading the sea…';
  $('goBtn').disabled = true;            // one flight per click, however slow the download
  try {
    game.noise = bestNoise(game.scenario);
    game.window = await loadWindow(game.scenario.scenario, game.noise, ARRIVAL_H);
    game.field = await loadField(game.scenario.scenario, ARRIVAL_H, game.index);
  } catch (err) {
    $('status').textContent = `could not load this buoy: ${err.message}`;
    $('goBtn').disabled = false;
    return;
  }
  $('goBtn').disabled = false;
  $('status').textContent = '';
  show('fly');
  if (!flyMap) {
    // The mouse steers, so it must not also drag the map about.
    flyMap = newMap('flyMap', { keyboard: false, dragging: false, doubleClickZoom: false, boxZoom: false });
    fieldLayer = new FieldLayer();
    flyMap.on('mousemove', (e) => {
      game.pointer = e.latlng;
      if (game.running) game.input = 'mouse';
    });
    flyMap.on('click', (e) => {
      game.pointer = e.latlng;
      game.input = 'mouse';
      takeOff();
    });
  }
  flyMap.invalidateSize();
  if (flyScene) flyScene.destroy();
  const showCloud = game.mode === 'map';
  flyScene = new Scene(flyMap, { colour: '#ffffff', showCloud });
  flyScene.setWindow(game.window);
  if (showCloud) fieldLayer.remove(); else fieldLayer.addTo(flyMap);
  $('hudPosCard').style.display = showCloud ? '' : 'none';
  // It arrives pointing along the drift, as every searcher in the benchmark does.
  game.player = new PlayerFlight(game.window);
  game.t = 0;
  game.running = false;
  game.pointer = null;
  $('hint').style.display = '';
  $('hint').textContent = HINT;
  drawFly();
  requestAnimationFrame(waitForPad);
}

function drawFly() {
  flyScene.update(game.t, game.player);
  if (game.mode === 'currents') {
    const c = game.window.meta.field_centre;
    fieldLayer.setState({ field: game.field, centre: [c.lat, c.lon], t: game.t });
  }
  $('hudTime').textContent = clock(WINDOW_S - game.t);
  $('hudPos').textContent = pct(game.player.metrics().pos);
}

function flyTick(now) {
  if (!game.running) return;
  const dt = Math.max(0, Math.min(0.1, (now - game.last) / 1000));
  game.last = now;
  const heading = wanted();
  if (heading !== null) game.player.steer(heading);
  game.t = Math.min(WINDOW_S, game.t + dt * FLY_SPEED);
  game.player.advanceTo(game.t);
  drawFly();
  if (game.t >= WINDOW_S) {
    game.running = false;
    reveal();
    return;
  }
  requestAnimationFrame(flyTick);
}

/* ---------------------------------------------------------------- reveal */

let panels = null;

function reveal() {
  show('reveal');
  if (!panels) {
    panels = ['mapYou', 'mapCg', 'mapAi'].map((id) => newMap(id, { zoomControl: false }));
    const [a, ...rest] = panels;
    for (const m of rest) {
      a.on('move', () => m.setView(a.getCenter(), a.getZoom(), { animate: false }));
    }
  }
  for (const m of panels) m.invalidateSize();
  if (game.scenes) for (const s of game.scenes) s.destroy();
  const flights = [
    rewind(game.player),
    new RecordedFlight(game.window, RIVALS.cg),
    new RecordedFlight(game.window, RIVALS.ai),
  ];
  const colours = ['#ffffff', SEARCHER_INFO[RIVALS.cg].colour, SEARCHER_INFO[RIVALS.ai].colour];
  game.scenes = panels.map((m, i) => {
    const s = new Scene(m, { colour: colours[i] });
    s.setWindow(game.window, { fit: i === 0 });
    return s;
  });
  game.flights = flights;
  $('revealNote').textContent = `${game.scenario.water} water, the helicopter arriving 2 hours after the call.`;
  game.t = 0;
  game.last = performance.now();
  game.revealing = true;
  requestAnimationFrame(revealTick);
}

/** The player's flight again from the start, so it can be watched beside the others. */
function rewind(player) {
  const again = new PlayerFlight(game.window);
  again.script = player.record();
  return again;
}

/** Fly a flight on to t: a recorded searcher as recorded, the player's legs as steered. */
function advance(flight, t) {
  if (!flight.script) {
    flight.advanceTo(t);
    return;
  }
  const headings = flight.script.heading_deg;
  while (!flight.done && flight.t + SAMPLE_S <= t + 1e-9) {
    flight.steer(headings[Math.min(Math.round(flight.t / SAMPLE_S), headings.length - 1)]);
    flight.advanceTo(flight.t + SAMPLE_S);
  }
}

function revealTick(now) {
  if (!game.revealing) return;
  const dt = Math.max(0, Math.min(0.1, (now - game.last) / 1000));
  game.last = now;
  game.t = Math.min(WINDOW_S, game.t + dt * REVEAL_SPEED);
  game.flights.forEach((f, i) => {
    advance(f, game.t);
    game.scenes[i].update(game.t, f);
  });
  $('posYou').textContent = pct(game.flights[0].metrics().pos);
  $('posCg').textContent = pct(game.flights[1].metrics().pos);
  $('posAi').textContent = pct(game.flights[2].metrics().pos);
  if (game.t >= WINDOW_S) {
    for (const s of game.scenes) s.showBuoy(WINDOW_S);
    game.revealing = false;
    setTimeout(results, 2500 / SPEED_UP);
    return;
  }
  requestAnimationFrame(revealTick);
}

/* ---------------------------------------------------------------- results */

async function results() {
  game.revealing = false;
  for (const f of game.flights) advance(f, WINDOW_S);
  const [you, cg, ai] = game.flights.map((f) => f.metrics());
  const v = verdict({ pos: you.pos }, [{ name: 'the Coast Guard', pos: cg.pos }, { name: 'the AI', pos: ai.pos }]);
  const points = score(you.pos, you.target?.found);
  $('verdict').textContent = v.won ? 'You win! Claim your prize.' : 'The search goes on…';
  $('verdict').className = `verdict${v.won ? ' win' : ''}`;
  $('verdictWhy').textContent = v.won
    ? `You searched more of the likely sea than both the Coast Guard and the AI. ${points} points.`
    : `${v.beaten.length ? `You beat ${v.beaten.join(' and ')}, but not` : 'You did not beat'} `
      + `${['the Coast Guard', 'the AI'].filter((n) => !v.beaten.includes(n)).join(' or ')}. ${points} points.`;
  const row = (name, m) => {
    const t = m.target;
    const buoy = t?.found ? `found at ${clock(t.foundS)}` : `missed by ${(t.closestM / 1000).toFixed(1)} km`;
    return `<tr><td>${name}</td><td class="n">${pct(m.pos)}</td><td>${buoy}</td></tr>`;
  };
  $('scoreTable').innerHTML = '<tr><th></th><th>Searched</th><th>The real person</th></tr>'
    + row(`You (${game.name})`, you) + row('Coast Guard', cg) + row('The AI', ai);

  if (!game.saved) {
    const entry = {
      name: game.name, mode: game.mode, scenario: game.scenario.scenario, pos: you.pos,
      found: Boolean(you.target?.found), score: points, won: v.won, when: new Date().toISOString(),
      record: game.player.record(),
    };
    const { list, saved } = add(entry);
    game.saved = true;
    game.board = list;
    if (!saved) $('status').textContent = 'this browser would not save the board';
  }
  $('boardTitle').textContent = `The board, ${MODES[game.mode]}`;
  $('boardNow').innerHTML = top(game.board ?? load(), game.mode, 8)
    .map((e) => `<li><span>${e.name}</span><span>${e.score}</span></li>`).join('');
  show('results');
  benchmark();
}

async function benchmark() {
  try {
    const b = await loadBenchmark();
    const rows = (b.scenarios55 ?? b).filter((r) => r.noise === game.noise && r.arrival_h === ARRIVAL_H)
      .sort((x, y) => y.pos - x.pos);
    $('benchNote').textContent = `Over all ${rows[0]?.n_groups ?? 55} buoys, the helicopter arriving 2 hours after the call:`;
    $('benchTable').innerHTML = '<tr><th>Searcher</th><th>Searched, on average</th><th>Found the person</th></tr>'
      + rows.map((r) => `<tr><td>${SEARCHER_INFO[r.searcher]?.label ?? r.searcher}</td>`
        + `<td class="n">${pct(r.pos)}</td><td class="n">${pct(r.found)}</td></tr>`).join('');
  } catch {
    $('benchNote').textContent = 'The benchmark across every buoy is not published yet.';
  }
}

function exportBoard() {
  const blob = new Blob([JSON.stringify(load(), null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `leaderboard-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------------------------------------------------------------- wiring */

async function main() {
  try {
    game.index = await loadIndex();
  } catch (err) {
    $('status').textContent = `the scenario files are not reachable: ${err.message}`;
    return;
  }
  $('startBtn').addEventListener('click', pick);
  $('surprise').addEventListener('click', () => {
    const all = game.index.scenarios;
    choose(all[Math.floor(Math.random() * all.length)]);
  });
  for (const b of $('modes').querySelectorAll('button')) {
    b.addEventListener('click', () => {
      game.mode = b.dataset.mode;
      for (const o of $('modes').querySelectorAll('button')) o.setAttribute('aria-pressed', String(o === b));
    });
  }
  $('goBtn').addEventListener('click', () => { game.saved = false; fly(); });
  $('skipBtn').addEventListener('click', () => {
    for (const s of game.scenes ?? []) s.showBuoy(WINDOW_S);
    results();
  });
  $('againBtn').addEventListener('click', attract);
  $('exportBtn').addEventListener('click', exportBoard);
  window.addEventListener('keydown', (e) => {
    const d = keyDirection(e.code);
    if (!d || !$('fly').classList.contains('on')) return;
    e.preventDefault();
    game.held.add(d);
    game.input = 'keys';
    takeOff();
  });
  window.addEventListener('keyup', (e) => {
    const d = keyDirection(e.code);
    if (d) game.held.delete(d);
  });
  attract();
}

main();
