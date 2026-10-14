/**
 * game.js -- "Can you survive the Bermuda Triangle?", the open-day game (D031, Stage 5).
 *
 * A visitor picks a real buoy (one of the 55 scenarios), chooses to see the probability map
 * or only the currents, and flies one helicopter for the 45-minute window, the clock
 * compressed so the search takes a minute (or less: "End the search" stops it early). Then the same sea is searched again,
 * side by side, by the Coast Guard's Expanding Square and by our ML model, and the real buoy is
 * revealed. The model is Telemachus, Suné's trained agent, where the published files carry its
 * flight (v3), and the greedy search where they do not; the results say which one it was.
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
import {
  AREA_CELL_M, add, cloudAreaKm2, load, nickname, par, score, standing, top, verdict,
} from './leaderboard.js';
import { densityGrid } from './probabilityMap.js';
import { trackAt } from './referee.js';
import { PlayerFlight, RecordedFlight, SAMPLE_S } from './playback.js';
import { ProbabilityLegend } from './probabilityLayer.js';
import { drawResultCard } from './resultCard.js';
import { Scene } from './scene.js';
import {
  SEARCHER_INFO, SHOWCASE_ROOT, aiOf, bestNoise, display, loadBenchmark, loadField, loadIndex,
  arrivalLabel,
  arrivalsOf,
  loadWindow,
  ML_UNTIL_H,
} from './scenarioData.js';
import { bearing } from './kinematics.js';
import { M_PER_DEG_LAT } from './geo.js';
import { SEARCH_SPEED_MS, TURN_RADIUS_M, TURN_RATE_DEG_S } from './platform.js';
import { headingFromKeys, keyDirection } from './searchRun.js';
import { DIRECTIONS, PadEvents, firstPad, nearestThatWay, padHeading, randomName } from './gamepad.js';
import { addSea } from './sea.js';
import { Tutorial } from './tutorial.js';

// The drift model the game flies over: the newest a scenario has (rvc, sized by the current,
// D033, once published; rv until then). Chosen per scenario in fly().
// How long after the call the helicopter arrives, chosen on the pick screen from what the
// scenario's files carry. 30 min unless the player picks another (v4 has 15 min to 3 h).
const DEFAULT_ARRIVAL_H = 0.5;
const WINDOW_S = 45 * 60;
// ?speed=10 plays everything ten times faster: for rehearsing the stand and for the tests.
const PARAMS = new URLSearchParams(window.location.search);
const SPEED_UP = Math.max(1, Number(PARAMS.get('speed')) || 1);
// ?kiosk=1 is the open-day stand: no links off the game (frontend/tools/kiosk.cmd opens it).
const KIOSK = PARAMS.has('kiosk');
// Faster after open-day testers found 90 s long (Aditya, 14 Oct).
const FLY_SPEED = 45 * SPEED_UP;          // 45 minutes in 60 seconds
const REVEAL_SPEED = 135 * SPEED_UP;      // the side-by-side replay in 20 seconds
const REVEAL_HOLD_MS = 5000;              // the real buoy left up this long before the results
const CG = 'expanding-square';
// What the results say the computer's searcher was, by who flew it (aiOf).
const AI_WORDS = {
  // Telemachus 2 (scenarios/v4): paid plain POS, gamma 1, two headings a minute; the first
  // Telemachus was also paid for finding it sooner, and these words described it until 14 Oct.
  ppo: '<b>Telemachus</b> is our machine-learning model: a small neural network trained by '
    + 'reinforcement learning (PPO). It learned where to fly by searching five years of simulated '
    + 'drift, choosing a new heading every 30 seconds, rewarded for every possible position its strip passed over.',
  greedy: '<b>The computer</b> here flies a greedy search, with no learning: always to where the most '
    + 'probability is left. The trained model takes its place once its flights are published.',
  // The trained model's flight is there, but the helicopter arrives too late for it (aiOf).
  late: '<b>The computer</b> here flies a greedy search, with no learning: always to where the most '
    + 'probability is left. When the helicopter arrives within 45 minutes, our machine-learning '
    + 'model, Telemachus, flies instead; from an hour on, greedy finds more on our test buoys.',
};

/** The words under the results for whoever flew the computer's side of this window. */
function aiWords(window, hours) {
  const ai = aiOf(window, hours);
  return AI_WORDS[ai === 'greedy' && window?.flights?.ppo ? 'late' : ai];
}
/** What the computer's searcher is called, by who flew it: Telemachus is the trained model (named
 * 12 Oct). Machine learning, not "AI" (Suné, 13 Oct); greedy does not learn, so it is not called ML. */
const AI_NAME = { ppo: 'Telemachus', greedy: 'the greedy search' };
const AI_LABEL = { ppo: 'Telemachus (ML model)', greedy: 'Greedy search (no learning)' };
// The mailer (mailer/ in this repo): a Vercel function for the public site, or a laptop at the
// stand. Unset, the game offers no email.
const MAILER_URL = String(import.meta.env.VITE_MAILER_URL ?? '');
// The open-day video plays when nobody has touched the stand for a while (Aditya, 12 Oct):
// on these screens, after this long, counted from the last touch or the last change of screen.
// Never in the middle of a flight or a replay, and never on the results: the clock used to run
// on from take-off, so the board vanished ~20 s after it appeared, before anyone had read it.
// It stays until someone presses a button (Aditya, 14 Oct).
const IDLE_MS = { attract: 30_000, pick: 45_000, tut: 60_000 };
const DEMO_VIDEO = String(import.meta.env.VITE_DEMO_VIDEO ?? `${SHOWCASE_ROOT}/openday.mp4`);
const MODES = { map: 'with the probability map', currents: 'with currents only' };
const $ = (id) => document.getElementById(id);

const game = {
  index: null, scenario: null, mode: 'map', name: 'Player', window: null, field: null,
  player: null, held: new Set(), t: 0, last: 0, running: false, result: null,
  input: 'keys', pointer: null, pad: null, hours: DEFAULT_ARRIVAL_H,
};

// The real helicopter's numbers (D032), in words for a visitor: why it will not simply go
// where it is pointed. A tester on 11 Oct did not know it banks round like a plane.
const SPEED_KMH = Math.round(SEARCH_SPEED_MS * 3.6);
const TURN_DEG = Math.round(TURN_RATE_DEG_S);
const U_TURN_S = Math.round(180 / TURN_RATE_DEG_S);
const CIRCLE_M = Math.round((2 * TURN_RADIUS_M) / 10) * 10;
const HINT = '<b>Fly it like a real search helicopter.</b> Point with the mouse where you want to '
  + `go and click to take off. At ${SPEED_KMH}\u00a0km/h it cannot stop or turn on the spot: it banks `
  + `round at ${TURN_DEG}\u00b0 a second, so a U-turn takes ${U_TURN_S}\u00a0s and swings ${CIRCLE_M}\u00a0m wide. `
  + 'Point well ahead of it. Arrow keys, WASD or a game controller work too.';
const PAD_HINT = '<br />🎮 <strong>Controller:</strong> push the stick (or the d-pad) the way to fly, or press ✕ to take off. '
  + '△ ends the search.';
const hint = () => HINT + (firstPad() ? PAD_HINT : '');
/** How far off the asked-for heading it must be before the game says it is still turning. */
const COACH_OFF_DEG = 25;

/* ---------------------------------------------------------------- steering */

/** East and north metres from the helicopter now to a point on the map, [lat, display lon]. */
function offsetTo(latlng) {
  const [lat, lon] = game.player.position(game.t);
  const dLon = latlng.lng - display(lon);
  return [dLon * M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180), (latlng.lat - lat) * M_PER_DEG_LAT];
}

/** The compass heading from the helicopter now to a point on the map. */
function headingTo(latlng) {
  const [east, north] = offsetTo(latlng);
  return Math.hypot(east, north) < 1 ? null : bearing(east, north);
}

/**
 * What to tell a player whose helicopter is not yet going where they asked, or null: still
 * swinging round, or pointed at a spot inside the circle it turns on, which it can only orbit.
 */
function coach() {
  const off = Math.abs(((game.player.command - game.player.headingAt(game.t) + 540) % 360) - 180);
  if (off <= COACH_OFF_DEG) return null;
  if (game.input === 'mouse' && game.pointer && Math.hypot(...offsetTo(game.pointer)) < 2 * TURN_RADIUS_M) {
    return 'Too close to turn to: it circles round that spot. Point further away.';
  }
  return `Turning, ${TURN_DEG}\u00b0 a second: at search speed a helicopter banks round like a plane.`;
}

/** What the player is asking for now, from whichever input they used last. */
function wanted() {
  const pad = padHeading();
  if (pad !== null) {
    game.input = 'pad';
    return pad;
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
  $('hint').classList.remove('intro');
  $('endBtn').hidden = false;
  requestAnimationFrame(flyTick);
}

/**
 * "End the search": the player has seen enough. The minute in progress is flown out, so the
 * referee scores it whole, and the search ends there; the rivals still fly all 45 minutes.
 */
function endSearch() {
  if (!game.running) return;
  game.running = false;
  game.player.stop();
  game.t = game.player.until;
  reveal();
}

/** Before take-off, a controller is only seen by asking it, so ask every frame. A stick or the
 * d-pad takes off this way; ✕ takes off as it is pointing (onPad). */
function waitForPad() {
  if (!$('fly').classList.contains('on') || game.running || game.t !== 0) return;
  if (padHeading() !== null) {
    game.input = 'pad';
    takeOff();
    return;
  }
  requestAnimationFrame(waitForPad);
}

function newMap(id, options = {}) {
  // Whole zoom levels: a fractional zoom scales tiles by a non-integer and leaves hairline seams.
  const map = L.map(id, { zoomSnap: 1, attributionControl: false, ...options });
  addSea(map, { maxZoom: 16, maxNativeZoom: 16, attribution: 'Esri, HERE, Garmin, &copy; OpenStreetMap contributors' });
  return map;
}

function show(id) {
  if (game.screen === 'tut' && id !== 'tut') stopTutorial();
  if (game.screen === 'demo' && id !== 'demo') $('demoVideo').pause();
  game.screen = id;
  touched();
  setPadFocus(null);
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
  $('attract').style.backgroundImage = `url(${import.meta.env.BASE_URL}img/hero.jpg)`;
  show('attract');
}

/* ---------------------------------------------------------------- the video, when idle */

/**
 * The open-day video on a loop, full screen. If it cannot be loaded (offline, or not
 * published yet) the two 3D loops (scripts/render_showcase.py) play instead.
 */
function demo() {
  show('demo');
  const video = $('demoVideo');
  if (!video.dataset.ready) {
    video.dataset.ready = '1';
    const loops = ['mountain', 'cube'].map((n) => `${SHOWCASE_ROOT}/${n}.webm`);
    let i = 0;
    video.loop = true;
    video.onerror = () => {
      video.loop = false;
      video.onerror = null;
      video.onended = () => { i = (i + 1) % loops.length; video.src = loops[i]; video.play().catch(() => {}); };
      video.src = loops[0];
      video.play().catch(() => {});
    };
    video.src = DEMO_VIDEO;
  }
  video.play().catch(() => {});
}

// Any input keeps the stand awake; only the video's own buttons (or Esc) close it, so it can
// be watched, paused and photographed without a stray mouse move ending it.
let lastInput = performance.now();
function touched() {
  lastInput = performance.now();
}

function watchIdle() {
  for (const ev of ['pointermove', 'pointerdown', 'keydown', 'wheel', 'touchstart']) {
    window.addEventListener(ev, touched, { passive: true });
  }
  setInterval(() => {
    const limit = IDLE_MS[game.screen];
    if (limit && performance.now() - lastInput > limit) demo();
  }, 1000);
}

/* ---------------------------------------------------------------- the three missions */

let tutorial = null;

function stopTutorial() {
  tutorial?.stop();
  tutorial = null;
}

function missions() {
  show('tut');
  stopTutorial();
  tutorial = new Tutorial({ speedUp: SPEED_UP, onDone: () => pick() });
  tutorial.start();
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
  $('pickInfo').textContent = `${s.start_utc.replace('T', ' ')} UTC, ${words[s.water] ?? ''}.`;
  drawArrivals();
  $('goBtn').disabled = false;
}

/** The arrival buttons for the chosen buoy: only the times its files carry. */
function drawArrivals() {
  const have = arrivalsOf(game.scenario, bestNoise(game.scenario));
  if (!have.includes(game.hours)) {
    game.hours = have.includes(DEFAULT_ARRIVAL_H) ? DEFAULT_ARRIVAL_H
      : have.reduce((a, h) => (Math.abs(h - DEFAULT_ARRIVAL_H) < Math.abs(a - DEFAULT_ARRIVAL_H) ? h : a), have[0] ?? 2);
  }
  const box = $('arrivalSeg');
  box.innerHTML = '';
  for (const h of have) {
    const b = document.createElement('button');
    b.textContent = arrivalLabel(h);
    b.setAttribute('aria-pressed', String(h === game.hours));
    b.addEventListener('click', () => { game.hours = h; drawArrivals(); });
    box.append(b);
  }
  $('arrivalNote').textContent = game.hours <= ML_UNTIL_H
    ? `Arriving ${arrivalLabel(game.hours)} after the call: the cloud is still small, and you are up against Telemachus, our machine-learning model.`
    : `Arriving ${arrivalLabel(game.hours)} after the call: the cloud has spread, and you are up against the greedy search.`;
}

/* ---------------------------------------------------------------- fly */

let flyMap = null;
let flyScene = null;
let fieldLayer = null;
const flyLegend = new ProbabilityLegend({ compact: true });

async function fly() {
  game.name = nickname($('name').value);
  $('status').textContent = 'loading the sea…';
  $('goBtn').disabled = true;            // one flight per click, however slow the download
  try {
    game.noise = bestNoise(game.scenario);
    game.window = await loadWindow(game.scenario.scenario, game.noise, game.hours);
    game.field = await loadField(game.scenario.scenario, game.hours, game.index);
  } catch (err) {
    $('status').textContent = `could not load this buoy: ${err.message}`;
    $('goBtn').disabled = false;
    return;
  }
  $('goBtn').disabled = false;
  $('status').textContent = '';
  show('fly');
  if (!flyMap) {
    // Dragging pans the map; steering waits while it does, so a pan is never a turn.
    flyMap = newMap('flyMap', { keyboard: false, doubleClickZoom: false, boxZoom: false });
    fieldLayer = new FieldLayer();
    flyMap.on('dragstart', () => { game.panning = true; });
    flyMap.on('dragend', () => {
      game.panning = false;
      game.pannedAt = performance.now();
    });
    flyMap.on('mousemove', (e) => {
      if (game.panning) return;
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
  flyScene = new Scene(flyMap, { colour: '#ffffff', showCloud, heliSize: 60 });
  // The cloud where the search starts, one zoom level out: room round it for the helicopter to
  // turn in. In fast water the cloud then drifts several km, and the camera follows.
  flyScene.setWindow(game.window, { startOnly: true, zoomOut: 1 });
  if (showCloud) {
    fieldLayer.remove();
    flyLegend.addTo(flyMap);
  } else {
    fieldLayer.addTo(flyMap);
    flyLegend.remove();
  }
  $('hudPosCard').style.display = showCloud ? '' : 'none';
  // It arrives pointing along the drift, as every searcher in the benchmark does.
  game.player = new PlayerFlight(game.window);
  game.t = 0;
  game.running = false;
  game.pointer = null;
  $('endBtn').hidden = true;
  $('hint').style.display = '';
  $('hint').innerHTML = hint();
  $('hint').classList.add('intro');
  drawFly();
  requestAnimationFrame(waitForPad);
}

/** Seconds after a pan before the camera follows the helicopter again. */
const FOLLOW_AFTER_PAN_MS = 4000;

function drawFly() {
  flyScene.update(game.t, game.player);
  // If the helicopter flies off the screen the camera follows it, unless the player has just
  // panned away to look at something.
  if (!game.panning && performance.now() - (game.pannedAt ?? 0) > FOLLOW_AFTER_PAN_MS) {
    const [lat, lon] = game.player.position(game.t);
    flyMap.panInside([lat, display(lon)], { padding: [60, 60], animate: false });
  }
  if (game.mode === 'currents') {
    const c = game.window.meta.field_centre;
    fieldLayer.setState({ field: game.field, centre: [c.lat, c.lon], t: game.t });
  }
  $('hudTime').textContent = clock(WINDOW_S - game.t);
  $('hudPos').textContent = pct(game.player.metrics().pos);
  if (game.running) drawCoach(performance.now());
}

/** The coaching line, kept up a moment after it stops applying so it does not flicker. */
function drawCoach(now) {
  const say = coach();
  if (say) {
    game.coachUntil = now + 900;
    if ($('hint').textContent !== say) $('hint').textContent = say;
  }
  $('hint').style.display = say || now < (game.coachUntil ?? 0) ? '' : 'none';
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
  loadBenchmark().catch(() => {});     // fetched while the replay plays, so the results open whole
  if (game.scenes) for (const s of game.scenes) s.destroy();
  const ai = aiOf(game.window, game.hours);
  $('aiTag').textContent = AI_LABEL[ai];
  const flights = [
    rewind(game.player),
    new RecordedFlight(game.window, CG),
    new RecordedFlight(game.window, ai),
  ];
  const colours = ['#ffffff', SEARCHER_INFO[CG].colour, SEARCHER_INFO[ai].colour];
  game.scenes = panels.map((m, i) => {
    const s = new Scene(m, { colour: colours[i], heliSize: 46 });
    s.setWindow(game.window, { fit: i === 0 });
    return s;
  });
  game.flights = flights;
  $('revealNote').textContent = `${game.scenario.water} water, the helicopter arriving ${arrivalLabel(game.hours)} after the call.`
    + (endedEarly() ? ` You ended your search with ${clock(WINDOW_S - game.player.until)} left.` : '');
  game.t = 0;
  game.last = performance.now();
  game.revealing = true;
  requestAnimationFrame(revealTick);
}

/** The player's flight again from the start, so it can be watched beside the others. */
function rewind(player) {
  const again = new PlayerFlight(game.window);
  again.script = player.record();
  again.endS = player.endS;
  return again;
}

/** Did the player press "End the search" before the 45 minutes were up? */
function endedEarly() {
  return game.player.until < WINDOW_S;
}

/** Fly a flight on to t: a recorded searcher as recorded, the player's legs as steered. */
function advance(flight, t) {
  if (!flight.script) {
    flight.advanceTo(t);
    return;
  }
  const headings = flight.script.heading_deg;
  const end = Math.min(t, flight.until);
  while (!flight.done && flight.t + SAMPLE_S <= end + 1e-9) {
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
    // The real buoy is often kilometres outside the cloud: zoom out until it is in view (the
    // other panels follow the first), and leave it up long enough to look at.
    game.scenes[0].fitAll(WINDOW_S);
    game.revealing = false;
    game.holdTimer = setTimeout(results, REVEAL_HOLD_MS / SPEED_UP);
    return;
  }
  requestAnimationFrame(revealTick);
}

/* ---------------------------------------------------------------- results */

/** The cloud's size when the helicopter arrives: the smallest area holding 90 % of it, km². */
function arrivalAreaKm2(window) {
  const { cloud, marker } = window;
  const n = cloud.particles;
  const [mlat, mlon] = trackAt(marker, 0);
  const grid = densityGrid(cloud.lat.subarray(0, n), cloud.lon.subarray(0, n), cloud.weight, mlat, mlon,
    { cellM: AREA_CELL_M });
  return grid ? cloudAreaKm2(grid.mass, AREA_CELL_M) : NaN;
}

const WATER = { jet: 'The Gulf Stream', moderate: 'Moderate current', quiet: 'Quiet water' };

/** 1st, 2nd, 3rd, 4th … 11th, 12th, 13th … 21st. */
function ordinal(n) {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

/** The big line on the results, and the sentence under it, by who the player beat. */
function headline(v, aiName) {
  const ai = `<span class="ai">${aiName === 'Telemachus' ? 'Telemachus' : 'the computer'}</span>`;
  if (v.won) {
    return { html: 'You win!', cls: 'win', why: `You searched more of the likely sea than both the Coast Guard and ${aiName}. Claim your prize!` };
  }
  if (v.beaten.includes(aiName)) {
    return { html: `You beat ${ai}`, cls: 'close', why: 'But the Coast Guard’s pattern searched more. Beat both to win the prize.' };
  }
  if (v.beaten.length) {
    return { html: 'So close!', cls: 'close', why: `You beat the Coast Guard, but not ${aiName}. Rematch?` };
  }
  return { html: `${ai} wins`, cls: '', why: `The Coast Guard and ${aiName} both searched more of the sea this time. Rematch?` };
}

/** 0 up to the score, quickly, the way a scoreboard does. */
function countUp(el, to, ms = 1200) {
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, Math.max(0, (now - t0) / ms));
    el.textContent = Math.round(to * (1 - (1 - k) ** 3)).toLocaleString('en-GB');
    if (k < 1 && game.screen === 'results') requestAnimationFrame(step);
    else el.textContent = to.toLocaleString('en-GB');
  };
  requestAnimationFrame(step);
}

function confetti() {
  const box = document.createElement('div');
  box.className = 'confetti';
  const colours = ['#04c951', '#ffc24a', '#ff5a1f', '#ff6fb5', '#3fd0ff', '#ffffff'];
  for (let i = 0; i < 90; i += 1) {
    const c = document.createElement('i');
    c.style.left = `${Math.random() * 100}%`;
    c.style.background = colours[i % colours.length];
    c.style.animationDuration = `${2.4 + 2 * Math.random()}s`;
    c.style.animationDelay = `${0.8 * Math.random()}s`;
    box.append(c);
  }
  $('results').append(box);
  setTimeout(() => box.remove(), 5500);
}

/** The board for this mode: the top `shown`, and the player's own line below them if they are further down. */
function renderBoard(shown) {
  const s = standing(game.board ?? load(), game.mode, game.name, shown);
  const line = (l, i) => `<li class="${l.you ? 'me' : ''}" style="--i:${i}"><span class="rank">${l.rank}</span>`
    + `<span class="nm">${l.name}</span><span class="sc">${Number(l.score).toLocaleString('en-GB')}</span></li>`;
  let html = s.lines.slice(0, shown).map(line).join('');
  if (s.rank > shown) html += `${s.rank > shown + 1 ? '<li class="gap">⋯</li>' : ''}${line(s.lines[shown], shown + 1)}`;
  $('boardNow').innerHTML = html || '<li class="empty">nobody yet</li>';
  $('boardTitle').textContent = MODES[game.mode];
  const where = KIOSK ? 'today' : 'on this computer';
  $('boardYou').textContent = !s.rank ? ''
    : s.rank === 1 ? (s.of > 1 ? `🏆 Top of the board, ahead of ${s.of - 1} ${s.of === 2 ? 'other' : 'others'}!` : '🏆 First on the board!')
      : `${ordinal(s.rank)} of ${s.of} ${where}`;
}

/**
 * As many of the top as fit beside everything else (nine on a tall screen, six on a 1366 x 768
 * laptop), measured rather than guessed: the results never scroll. A phone scrolls, so it gets ten.
 */
function drawBoardNow() {
  const res = $('results');
  const last = res.querySelector('.res-side > :last-child');
  const scrolls = window.matchMedia('(max-width: 900px)').matches;
  for (let shown = scrolls ? 10 : 9; shown >= 3; shown -= 1) {
    renderBoard(shown);
    if (scrolls || last.offsetTop + last.offsetHeight <= res.clientHeight - 12) return;
  }
}

async function results() {
  clearTimeout(game.holdTimer);
  game.revealing = false;
  for (const f of game.flights) advance(f, WINDOW_S);
  const [you, cg, ai] = game.flights.map((f) => f.metrics());
  const aiKey = aiOf(game.window, game.hours);
  const aiName = AI_NAME[aiKey];
  const v = verdict({ pos: you.pos }, [{ name: 'the Coast Guard', pos: cg.pos }, { name: aiName, pos: ai.pos }]);
  const area = arrivalAreaKm2(game.window);
  const points = score(you.pos, area);
  game.result = { points, area };
  const h = headline(v, aiName);
  $('resKicker').textContent = `${WATER[game.scenario.water] ?? game.scenario.water} · the helicopter ${arrivalLabel(game.hours)} after the call`;
  $('verdict').innerHTML = h.html;
  $('verdict').className = `verdict ${h.cls}`;
  $('verdictWhy').textContent = h.why
    + (endedEarly() ? ` You ended the search with ${clock(WINDOW_S - game.player.until)} left on the clock.` : '');
  const best = Math.max(you.pos, cg.pos, ai.pos);
  const row = (name, m, colour, cls = '') => {
    const t = m.target;
    const buoy = !t ? '' : t.found ? `✓ found them at ${clock(t.foundS)}` : `✗ missed by ${(t.closestM / 1000).toFixed(1)} km`;
    return `<tr class="${cls}"><td class="who">${name}${m.pos === best ? '<span class="best">BEST</span>' : ''}</td>`
      + `<td class="barcell"><div class="bar"><i style="width:${(100 * m.pos).toFixed(1)}%;background:${colour}"></i></div></td>`
      + `<td class="pos">${pct(m.pos)}</td><td class="found${t?.found ? ' yes' : ''}">${buoy}</td></tr>`;
  };
  $('scoreTable').innerHTML = '<tr><th></th><th colspan="2">Searched</th><th>The real person</th></tr>'
    + row(`You (${game.name})`, you, '#ffffff', 'you')
    + row('Coast Guard', cg, SEARCHER_INFO[CG].colour)
    + row(AI_LABEL[aiKey], ai, SEARCHER_INFO[aiKey].colour);
  $('pointsWhy').textContent = `This cloud covered ${Math.round(area)} km² when the helicopter arrived. On a cloud `
    + `that big the Coast Guard's pattern searches about ${Math.round(100 * par(area))} % on average: that is 1,000 points, `
    + 'and you score the same share of 1,000 for what you searched. A bigger cloud is harder, so it pays more.';
  $('aiWords').innerHTML = aiWords(game.window, game.hours);

  if (!game.saved) {
    const entry = {
      name: game.name, mode: game.mode, scenario: game.scenario.scenario, pos: you.pos,
      found: Boolean(you.target?.found), area_km2: area, score: points, won: v.won,
      arrival_h: game.hours, ended_s: endedEarly() ? game.player.until : null,
      when: new Date().toISOString(),
      record: game.player.record(),
    };
    const { list, saved } = add(entry);
    game.saved = true;
    game.board = list;
    if (!saved) $('status').textContent = 'this browser would not save the board';
  }
  const wasOn = game.screen === 'results';
  show('results');
  drawBoardNow();
  countUp($('pointsBig'), points);
  if (v.won && !wasOn) confetti();
  benchmark().then(drawBoardNow);      // its rows change the room the board has
  offerEmail();
}

/* ---------------------------------------------------------------- email */

function offerEmail() {
  const form = $('mailForm');
  form.hidden = !MAILER_URL;
  $('mailBtn').disabled = false;
  $('mailNote').textContent = 'We use your address once, to send this, and keep nothing.';
  $('mailNote').className = 'muted note';
}

/** One searcher's numbers for the email. */
function forEmail(m) {
  const t = m.target ?? {};
  return { pos: m.pos, found: Boolean(t.found), foundS: t.foundS ?? 0, missedKm: (t.closestM ?? NaN) / 1000 };
}

/** The picture of the three flights over the cloud, as a JPEG data URL. */
function flightPicture() {
  const ai = aiOf(game.window, game.hours);
  const flights = [
    { label: `You (${game.name})`, colour: '#ffffff', trail: game.flights[0].trail(WINDOW_S) },
    { label: 'Coast Guard', colour: SEARCHER_INFO[CG].colour, trail: game.flights[1].trail(WINDOW_S) },
    { label: AI_LABEL[ai], colour: SEARCHER_INFO[ai].colour, trail: game.flights[2].trail(WINDOW_S) },
  ];
  return drawResultCard(document.createElement('canvas'), game.window, flights, WINDOW_S)
    .toDataURL('image/jpeg', 0.88);
}

async function sendEmail(e) {
  e.preventDefault();
  const note = $('mailNote');
  $('mailBtn').disabled = true;
  note.className = 'muted note';
  note.textContent = 'Sending…';
  const [you, cg, ai] = game.flights.map((f) => f.metrics());
  const body = {
    email: $('mailTo').value.trim(),
    name: game.name,
    points: game.result.points,
    areaKm2: game.result.area,
    you: forEmail(you),
    cg: forEmail(cg),
    ai: { ...forEmail(ai), name: AI_NAME[aiOf(game.window, game.hours)] === 'Telemachus' ? 'Telemachus' : 'the greedy search' },
    bench: (game.benchRows ?? []).slice(0, 4).map((r) => ({ label: SEARCHER_INFO[r.searcher]?.label ?? r.searcher, pos: r.pos })),
    arrival: arrivalLabel(game.hours),
    // The board on this computer (at the stand, everyone who played there today).
    board: { mode: game.mode, ...standing(game.board ?? load(), game.mode, game.name) },
  };
  try {
    body.image = flightPicture();
  } catch { /* the email goes without the picture */ }
  try {
    const res = await fetch(MAILER_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const out = await res.json().catch(() => ({}));
    if (!res.ok || !out.ok) throw new Error(out.error ?? `the mail server answered ${res.status}`);
    note.className = 'note ok';
    note.textContent = 'Sent! Check your inbox, and your spam folder just in case.';
    $('mailTo').value = '';
  } catch (err) {
    note.className = 'note bad';
    note.textContent = `Not sent: ${err.message}. Try again in a moment.`;
    $('mailBtn').disabled = false;
  }
}

async function benchmark() {
  try {
    const b = await loadBenchmark();
    const rows = (b.scenarios55 ?? b).filter((r) => r.noise === game.noise && r.arrival_h === game.hours)
      .sort((x, y) => y.pos - x.pos);
    game.benchRows = rows;
    $('benchNote').textContent = `all ${rows[0]?.n_groups ?? 55} buoys, the helicopter ${arrivalLabel(game.hours)} after the call`;
    $('benchTable').innerHTML = '<tr><th>Searcher</th><th>Searched, on average</th><th class="n">Found them</th></tr>'
      + rows.map((r) => {
        const info = SEARCHER_INFO[r.searcher];
        return `<tr><td>${info?.label ?? r.searcher}</td><td><div class="pbar"><div class="bar sm">`
          + `<i style="width:${(100 * r.pos).toFixed(1)}%;background:${info?.colour ?? '#8a8a80'}"></i></div>`
          + `<span>${pct(r.pos)}</span></div></td><td class="n">${pct(r.found)}</td></tr>`;
      }).join('');
  } catch {
    $('benchNote').textContent = 'not published yet';
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

/* ---------------------------------------------------------------- the controller */

// A PS5 or Xbox pad plays the whole game (Aditya, 14 Oct: it did nothing on the menus or in
// mission 1). The stick or the d-pad moves a highlight between the buttons, ✕ presses, ○ goes
// back; in flight the stick steers, ✕ takes off and △ ends the search; a mission can claim the
// pad for itself (steering in mission 1, the buoy in mission 2). The first touch only shows the
// highlight, so nobody presses a button they have not seen.
let padFocus = null;
let tutButtons = 0;
const padEvents = new PadEvents((name) => onPad(name));

/** Where the highlight starts on each screen: the button a visitor most likely wants. */
const PAD_START = { attract: 'startBtn', pick: 'surprise', results: 'againBtn', demo: 'demoPlay' };

const padMode = () => document.body.classList.contains('pad-mode');

/** Everything on this screen the highlight can sit on. */
function padTargets() {
  const screen = $(game.screen);
  if (!screen) return [];
  return [...screen.querySelectorAll('button, input')].filter((el) => !el.disabled && el.getClientRects().length > 0);
}

function defaultPadTarget() {
  const els = padTargets();
  if (game.screen === 'tut') return els.find((el) => $('tutControls').contains(el)) ?? $('tutSkip');
  return $(PAD_START[game.screen]) ?? els[0] ?? null;
}

function setPadFocus(el) {
  padFocus?.classList.remove('pad-focus');
  padFocus = el;
  if (!el) return;
  el.classList.add('pad-focus');
  // A text box takes the keyboard once highlighted; anything else gives it back.
  if (el.tagName === 'INPUT') el.focus({ preventScroll: true });
  else if (document.activeElement?.tagName === 'INPUT') document.activeElement.blur();
}

function movePadFocus(dir) {
  if (!padFocus) {
    setPadFocus(defaultPadTarget());
    return;
  }
  // On a slider, left and right slide it; up and down leave it.
  if (padFocus.type === 'range' && (dir === 'left' || dir === 'right')) {
    const r = padFocus;
    const step = (Number(r.max) - Number(r.min)) / 25;
    r.value = String(Number(r.value) + (dir === 'right' ? step : -step));
    r.dispatchEvent(new Event('input', { bubbles: true }));
    return;
  }
  const next = nearestThatWay(padFocus, padTargets(), dir);
  if (next) setPadFocus(next);
}

function pressPadFocus() {
  const el = padFocus;
  if (!el) {
    setPadFocus(defaultPadTarget());
    return;
  }
  if (el.id === 'name') {
    el.value = randomName();                 // a name on the board without a keyboard
    return;
  }
  if (el.tagName === 'INPUT') return;
  el.click();
  // "Surprise me" picked a buoy: the next thing a pad player wants is Fly.
  if (el.id === 'surprise' && !$('goBtn').disabled) setPadFocus($('goBtn'));
}

function onPad(name) {
  const first = !padMode();
  document.body.classList.add('pad-mode');
  $('padChip').hidden = false;
  const s = game.screen;
  if (s === 'fly') {
    if (name === 'cross' && !game.running && game.t === 0) {
      game.input = 'pad';
      takeOff();
    }
    if ((name === 'triangle' || name === 'options') && game.running) endSearch();
    return;
  }
  if (s === 'reveal') {
    if (name === 'cross' || name === 'options') $('skipBtn').click();
    return;
  }
  if (s === 'tut' && tutorial?.pad(name)) return;
  if (first) {
    setPadFocus(defaultPadTarget());
    return;
  }
  if (name === 'circle') {
    if (s === 'pick' || s === 'demo') attract();
    else if (s === 'tut') tutorial?.skip();       // even mid-flight in mission 1, where the stick is the helicopter's
    return;
  }
  if (DIRECTIONS.includes(name)) movePadFocus(name);
  else if (name === 'cross' || name === 'options') pressPadFocus();
}

/** Every frame: read the pad, and keep the highlight on something that is there. */
function padLoop(now) {
  if (padEvents.poll(now)) touched();
  if (padMode() && game.screen !== 'fly' && game.screen !== 'reveal') {
    const els = padTargets();
    if (game.screen === 'tut') {
      // A mission's next button has just appeared: go to it, unless the player is on the slider.
      const inside = els.filter((el) => $('tutControls').contains(el));
      if (inside.length > tutButtons && !(padFocus && $('tutControls').contains(padFocus))) setPadFocus(inside.at(-1));
      tutButtons = inside.length;
    }
    if (!padFocus || !els.includes(padFocus)) setPadFocus(defaultPadTarget());
  }
  requestAnimationFrame(padLoop);
}

/* ---------------------------------------------------------------- wiring */

async function main() {
  if (KIOSK) {
    document.body.classList.add('kiosk');
    $('againBtn').textContent = 'Next player';
  }
  // The start screen shows at once, but its two play buttons wait (disabled in the HTML) for
  // the list of buoys: a click before it arrived used to be lost (14 Oct, live: up to a second
  // on good Wi-Fi, longer at a venue).
  $('status').textContent = 'loading the buoys…';
  $('demoBtn').addEventListener('click', () => demo());
  $('demoPlay').addEventListener('click', attract);
  $('demoClose').addEventListener('click', attract);
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && game.screen === 'demo') attract(); });
  watchIdle();
  // The controller: a mouse click puts the highlight away until the pad is touched again.
  window.addEventListener('gamepadconnected', () => { $('padChip').hidden = false; });
  window.addEventListener('gamepaddisconnected', () => { $('padChip').hidden = !firstPad(); });
  window.addEventListener('pointerdown', () => {
    document.body.classList.remove('pad-mode');
    setPadFocus(null);
  });
  requestAnimationFrame(padLoop);
  attract();
  try {
    game.index = await loadIndex();
  } catch (err) {
    $('status').textContent = `the scenario files are not reachable: ${err.message}`;
    return;
  }
  $('status').textContent = '';
  $('startBtn').addEventListener('click', missions);
  $('searchBtn').addEventListener('click', pick);
  $('tutSkip').addEventListener('click', () => tutorial?.skip());
  $('tutSkipAll').addEventListener('click', () => pick());
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
  $('endBtn').addEventListener('click', endSearch);
  $('skipBtn').addEventListener('click', () => {
    for (const s of game.scenes ?? []) s.showBuoy(WINDOW_S);
    game.scenes?.[0]?.fitAll(WINDOW_S);
    results();
  });
  $('againBtn').addEventListener('click', attract);
  $('exportBtn').addEventListener('click', exportBoard);
  $('nameDice').addEventListener('click', () => { $('name').value = randomName(); });
  $('mailForm').addEventListener('submit', sendEmail);
  // A free server sleeps when idle: wake it while the player flies, not when they press Send.
  if (MAILER_URL) fetch(MAILER_URL, { method: 'GET' }).catch(() => {});
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
  $('startBtn').disabled = false;
  $('searchBtn').disabled = false;
}

main();
